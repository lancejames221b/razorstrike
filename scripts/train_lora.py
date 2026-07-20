"""Phase 2 - Training script (the core artifact) for RazorStrike v2.

RazorStrike v2 = clean Qwen/Qwen3.6-35B-A3B base + this multi-domain LoRA.

Loads a PRE-QUANTIZED AWQ checkpoint (QuantTrio/Qwen3.6-35B-A3B-AWQ, 4-bit,
1.09M downloads, non-abliterated - a straight AWQ quant of the same clean base;
uncensoring is done via the `uncensor` data family, not weights). AWQ weights
are already 4-bit ON DISK, so loading is a direct read with NO on-the-fly
quantize-during-load step - this sidesteps an entire class of transformers v5
bugs (huggingface/transformers#43032 and related) that make bitsandbytes
on-the-fly 4-bit loading of this 35B MoE OOM on a single 40GB GPU regardless
of monkeypatching (confirmed empirically: two separate OOM sites patched, a
third remained in the MoE expert weight-merge path). No downgrade path exists
either - the qwen3_5_moe architecture does not exist before the v5 release
that introduced the bug (confirmed empirically against transformers==4.57.1).

The AWQ checkpoint's own quantization_config keeps linear_attn/self_attn/
shared_expert/mlp.gate at full precision (not quantized) - exactly the layers
this LoRA targets - so adapting them gets full-precision gradients while the
bulk MoE expert parameters (which dominate the 35B count) stay compact.

Reads env: BASE_REPO (default the AWQ repo above), DATA_REPO, OUT_DIR,
  ADAPTER_REPO, HF_TOKEN, MAXLEN(4096), TARGET_MLP(0), LORA_R(32),
  LORA_ALPHA(64), SAVE_STEPS(50), EVAL_STEPS(250), RESUME.
"""

import os, torch
from transformers import (AutoModelForImageTextToText,
                          AutoTokenizer,
                          Trainer, TrainingArguments, DataCollatorForSeq2Seq)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

# Pre-quantized AWQ checkpoint of the clean base (see module docstring for why).
BASE   = os.environ.get("BASE_REPO", "QuantTrio/Qwen3.6-35B-A3B-AWQ")
DATA   = os.environ["DATA_REPO"]           # lancejames221b/razorstrike-v2-sft
OUT    = os.environ.get("OUT_DIR", "/content/adapter")
MAXLEN = int(os.environ.get("MAXLEN", "4096"))

tok = AutoTokenizer.from_pretrained(BASE)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


def to_features(ex):
    msgs = ex["messages"]
    prompt = tok.apply_chat_template(msgs[:-1], add_generation_prompt=True, tokenize=True)
    full   = tok.apply_chat_template(msgs,       add_generation_prompt=False, tokenize=True)
    if len(full) > MAXLEN or len(prompt) >= len(full):
        return {"input_ids": None, "attention_mask": None, "labels": None}
    labels = [-100] * len(prompt) + full[len(prompt):]
    return {"input_ids": full, "attention_mask": [1] * len(full), "labels": labels}


ds = load_dataset(DATA)
ds = ds.map(to_features, remove_columns=ds["train"].column_names)
ds = ds.filter(lambda r: r["input_ids"] is not None)

# AWQ weights are already 4-bit on disk; quantization_config lives in the
# checkpoint's own config.json and is auto-applied. No BitsAndBytesConfig,
# no on-the-fly quantize step, no monkeypatch needed.
_load_kw = dict(device_map={"": 0}, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
model = AutoModelForImageTextToText.from_pretrained(BASE, **_load_kw)
model.config.use_cache = False
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

# Grounded targets: BOTH attention types (full + linear/SSM). Vision leaves
# (qkv/proj/fc) differ, so these never touch the vision tower. Confirmed these
# module names match the AWQ checkpoint too (its own quantization_config lists
# linear_attn/self_attn by these same names as kept-unquantized).
targets = ["q_proj", "k_proj", "v_proj", "o_proj",
           "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"]
if os.environ.get("TARGET_MLP", "0") == "1":
    targets += ["gate_proj", "up_proj", "down_proj"]   # 256 experts -> large adapter

lora = LoraConfig(
    r=int(os.environ.get("LORA_R", "32")),
    lora_alpha=int(os.environ.get("LORA_ALPHA", "64")),
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=targets)
model = get_peft_model(model, lora)
model.print_trainable_parameters()

# Fail fast: PEFT silently drops non-matching target names. If almost nothing
# matched, the adapter would "train" for hours and learn nothing.
_tp = sum(p.numel() for p in model.parameters() if p.requires_grad)
_tot = sum(p.numel() for p in model.parameters())
_pct = 100.0 * _tp / max(_tot, 1)
print(f"[guard] trainable params: {_tp:,} ({_pct:.4f}% of {_tot:,})")
assert _pct > 0.01, (f"LoRA matched almost nothing ({_pct:.4f}%). target_modules "
                     f"do not match {BASE}; inspect model.named_modules().")

# ADAPTER_REPO is REQUIRED: durability depends on pushing checkpoints to the
# Hub during training, not just at the very end. A VM reclamation mid-run
# would otherwise wipe /content and lose everything.
ADAPTER_REPO = os.environ["ADAPTER_REPO"]
HF_TOKEN = os.environ.get("HF_TOKEN")

args = TrainingArguments(
    output_dir=OUT, num_train_epochs=2,
    per_device_train_batch_size=1, gradient_accumulation_steps=16,
    learning_rate=2e-4, lr_scheduler_type="cosine", warmup_ratio=0.03,
    bf16=True, gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    max_grad_norm=1.0, logging_steps=5,
    save_steps=int(os.environ.get("SAVE_STEPS", "50")), save_total_limit=3,
    eval_strategy="steps", eval_steps=int(os.environ.get("EVAL_STEPS", "250")), optim="paged_adamw_8bit",
    report_to="none", dataloader_num_workers=2,
    # Durable checkpointing: push the latest checkpoint to the Hub every
    # save_steps, so a VM reclamation loses at most one save interval, not
    # the whole run. Resumable via `last-checkpoint` in ADAPTER_REPO.
    push_to_hub=True, hub_model_id=ADAPTER_REPO, hub_token=HF_TOKEN,
    hub_private_repo=True, hub_strategy="checkpoint")

trainer = Trainer(model=model, args=args,
    train_dataset=ds["train"], eval_dataset=ds["validation"],
    data_collator=DataCollatorForSeq2Seq(tok, label_pad_token_id=-100, padding=True))

resume_path = None
if os.environ.get("RESUME"):
    # Pull the last checkpoint back from the Hub (local OUT may be gone after
    # a VM churn) before resuming.
    from huggingface_hub import snapshot_download
    try:
        ck_dir = snapshot_download(ADAPTER_REPO, allow_patterns="last-checkpoint/*",
                                   token=HF_TOKEN, local_dir=OUT)
        resume_path = os.path.join(OUT, "last-checkpoint")
        print(f"[resume] pulled checkpoint from hub -> {resume_path}")
    except Exception as e:
        print(f"[resume] no hub checkpoint found ({type(e).__name__}: {e}); starting fresh")

trainer.train(resume_from_checkpoint=resume_path)

model.save_pretrained(OUT)
tok.save_pretrained(OUT)
model.push_to_hub(ADAPTER_REPO, private=True, token=HF_TOKEN)
tok.push_to_hub(ADAPTER_REPO, private=True, token=HF_TOKEN)

print("TRAINING_COMPLETE")
