"""Phase 2 - Training script (the core artifact) for RazorStrike v2.

RazorStrike v2 = clean Qwen/Qwen3.6-35B-A3B (35B MoE, 3B activated) base +
this multi-domain LoRA. Uncensoring is done via the `uncensor` data family,
not weight abliteration.

Loads in PLAIN bf16 (~70GB) - no quantization at all. On a 40GB GPU this
doesn't fit and forces on-the-fly 4-bit quantization, which hits a confirmed,
still-open transformers v5 bug class (huggingface/transformers#43032: MoE
expert weight-merge conversion materializes at full precision on GPU before
quantizing, OOMs even after patching two separate call sites, and the patched
load still bloated to ~39.7GB - a broken/non-functional quantization). On an
80-96GB GPU (Colab's G4 = RTX PRO 6000 Blackwell, ~96GB), plain bf16 LoRA
sidesteps that entire bug class - no BitsAndBytesConfig, no monkeypatches,
no on-the-fly quantize step.

Reads env: BASE_REPO (default Qwen/Qwen3.6-35B-A3B), DATA_REPO, OUT_DIR,
  ADAPTER_REPO, HF_TOKEN, MAXLEN(4096), TARGET_MLP(0), LORA_R(32),
  LORA_ALPHA(64), SAVE_STEPS(50), EVAL_STEPS(250), RESUME.
"""

import os, torch
from transformers import (AutoModelForImageTextToText, AutoModelForCausalLM,
                          AutoTokenizer,
                          Trainer, TrainingArguments, DataCollatorForSeq2Seq)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

BASE   = os.environ.get("BASE_REPO", "Qwen/Qwen3.6-35B-A3B")
DATA   = os.environ["DATA_REPO"]           # lancejames221b/razorstrike-v2-sft
OUT    = os.environ.get("OUT_DIR", "/content/adapter")
MAXLEN = int(os.environ.get("MAXLEN", "3072"))  # 4096 tail (0.2% of rows >4096, 1.1% >3072) OOMs on 96GB G4; verified via row-length sampling

tok = AutoTokenizer.from_pretrained(BASE)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


def to_features(ex):
    msgs = ex["messages"]
    prompt = tok.apply_chat_template(msgs[:-1], add_generation_prompt=True, tokenize=True)["input_ids"]
    full   = tok.apply_chat_template(msgs,       add_generation_prompt=False, tokenize=True)["input_ids"]
    if len(full) > MAXLEN or len(prompt) >= len(full):
        return {"input_ids": None, "attention_mask": None, "labels": None}
    labels = [-100] * len(prompt) + full[len(prompt):]
    return {"input_ids": full, "attention_mask": [1] * len(full), "labels": labels}


ds = load_dataset(DATA)
ds = ds.map(to_features, remove_columns=ds["train"].column_names)
ds = ds.filter(lambda r: r["input_ids"] is not None)

# Plain bf16 load (~70GB) - no quantization, no bnb, no monkeypatch. Needs an
# 80GB+ GPU (Colab G4 = RTX PRO 6000 Blackwell, ~96GB). ImageTextToText is
# the correct class for this *ForConditionalGeneration multimodal MoE arch
# (confirmed empirically earlier); CausalLM as fallback only if that class
# genuinely can't resolve the checkpoint.
_load_kw = dict(device_map={"": 0}, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
try:
    model = AutoModelForImageTextToText.from_pretrained(BASE, **_load_kw)
except Exception as e:
    print(f"[load] ImageTextToText failed ({type(e).__name__}); trying CausalLM")
    model = AutoModelForCausalLM.from_pretrained(BASE, **_load_kw)
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()

# Grounded targets: BOTH attention types (full + linear/SSM), verified against
# the real weight-map (40/40 text layers, zero vision collisions). MoE expert
# MLPs (256 experts) are opt-in via TARGET_MLP - large adapter, only needed if
# you want to adapt the expert weights directly.
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
    max_steps=int(os.environ.get("MAX_STEPS", "-1")),  # -1 = full 2 epochs; positive caps for validation runs
    per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=16,
    learning_rate=2e-4, lr_scheduler_type="cosine", warmup_ratio=0.03,
    bf16=True, gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    max_grad_norm=1.0, logging_steps=5,
    save_steps=int(os.environ.get("SAVE_STEPS", "50")), save_total_limit=3,
    eval_strategy="steps", eval_steps=int(os.environ.get("EVAL_STEPS", "250")), optim="adamw_torch",
    report_to="none", dataloader_num_workers=2,
    # Durable checkpointing: push the latest checkpoint to the Hub every
    # save_steps, so a VM reclamation loses at most one save interval, not
    # the whole run. Resumable via `last-checkpoint` in ADAPTER_REPO.
    push_to_hub=True, hub_model_id=ADAPTER_REPO, hub_token=HF_TOKEN,
    hub_private_repo=True, hub_strategy="all_checkpoints", prediction_loss_only=True)

trainer = Trainer(model=model, args=args,
    train_dataset=ds["train"], eval_dataset=ds["validation"],
    data_collator=DataCollatorForSeq2Seq(tok, label_pad_token_id=-100, padding=True))

resume_path = None
if os.environ.get("RESUME"):
    # With hub_strategy="all", checkpoints push to the Hub repo root as
    # checkpoint-N/ dirs. Pull the highest-numbered one and resume from it.
    from huggingface_hub import list_repo_files, snapshot_download
    try:
        files = list_repo_files(ADAPTER_REPO, token=HF_TOKEN)
        ckpt_dirs = sorted({f.split("/")[0] for f in files if f.startswith("checkpoint-")},
                            key=lambda s: int(s.split("-")[1]))
        if ckpt_dirs:
            latest = ckpt_dirs[-1]
            snapshot_download(ADAPTER_REPO, allow_patterns=[f"{latest}/*"],
                               token=HF_TOKEN, local_dir=OUT)
            resume_path = os.path.join(OUT, latest)
            print(f"[resume] pulled {latest} from hub -> {resume_path}")
        else:
            print("[resume] no checkpoint dirs on hub; starting fresh")
    except Exception as e:
        print(f"[resume] no hub checkpoint found ({type(e).__name__}: {e}); starting fresh")

trainer.train(resume_from_checkpoint=resume_path)

model.save_pretrained(OUT)
tok.save_pretrained(OUT)
model.push_to_hub(ADAPTER_REPO, private=True, token=HF_TOKEN)
tok.push_to_hub(ADAPTER_REPO, private=True, token=HF_TOKEN)

print("TRAINING_COMPLETE")
