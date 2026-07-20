"""Phase 2 - Training script (the core artifact) for RazorStrike v2.

RazorStrike v2 = clean Qwen/Qwen3.6-35B-A3B base + this multi-domain QLoRA.

The base is a 35B MULTIMODAL MoE (Qwen3_5MoeForConditionalGeneration): a hybrid
text stack (full_attention every 4th layer, linear-attention/SSM elsewhere) plus
a vision tower. We train a TEXT QLoRA only.

Grounded from the real weight-map (Qwen/Qwen3.6-35B-A3B):
  - full-attention layers: self_attn.{q,k,v,o}_proj
  - linear-attention layers: linear_attn.{in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj}
  - MoE MLP (256 experts): mlp.{gate_proj,up_proj,down_proj}  <- opt-in (TARGET_MLP=1), huge
  - vision tower uses different leaves (qkv/proj/fc) so text targets never touch it
  - router (mlp.gate) is NOT adapted

Loads in 4-bit (QLoRA) so the 35B fits a single 40GB A100 (Colab Pro+).
Reads env: BASE_REPO, DATA_REPO, OUT_DIR, ADAPTER_REPO, HF_TOKEN,
  MAXLEN(4096), TARGET_MLP(0), LORA_R(32), LORA_ALPHA(64), RESUME.
"""

import os, torch, inspect
from transformers import (AutoModelForImageTextToText,
                          AutoTokenizer, BitsAndBytesConfig,
                          Trainer, TrainingArguments, DataCollatorForSeq2Seq)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset


def _patch_transformers_bnb_oom():
    """Runtime monkeypatch for a confirmed, still-open transformers v5 regression
    (huggingface/transformers#43032, reproduced against installed 5.14.1): during
    core_model_loading, parameters that WILL be quantized are still materialized
    on the target GPU at full precision BEFORE the quantize step runs, causing an
    OOM on GPUs that would otherwise easily fit the 4-bit model (a 35B model needs
    ~70GB to materialize in bf16 vs ~20GB once quantized to 4-bit).

    Verified fix (from the issue reporter, confirmed against our exact installed
    source at the `materialize_device = param_device` line): materialize
    quantized params on CPU first; the quantizer's own convert() step still
    places the final compressed tensor on the target GPU device correctly.

    Patches transformers.core_model_loading's source IN-MEMORY at process start
    (not the on-disk package), so it re-applies automatically on every launch -
    including a fresh VM after a reclaim/RESUME - with no persistent VM state.
    """
    import transformers.core_model_loading as cml
    target_fn = cml.convert_and_load_state_dict_in_model
    src = inspect.getsource(target_fn)
    anchor = "            materialize_device = param_device\n"
    n = src.count(anchor)
    assert n == 1, (
        f"[patch] transformers#43032 workaround FAILED: expected exactly 1 anchor "
        f"match for 'materialize_device = param_device' inside "
        f"convert_and_load_state_dict_in_model, found {n}. transformers version "
        f"has drifted from the pinned 5.14.1 this patch was verified against - "
        f"loading now would silently reproduce the 35B QLoRA OOM after a long "
        f"download. Re-verify the fix against the installed version "
        f"(transformers.__version__: {__import__('transformers').__version__}) "
        f"before proceeding.")
    patched = src.replace(
        anchor,
        anchor + "            if mapping.quantization_operation is not None:\n"
                 "                materialize_device = 'cpu'  # patched: avoid pre-quant GPU OOM, transformers#43032\n",
        1)
    # Compile ONLY this one function's source (dedented, since inspect.getsource
    # returns it indented as a class/module member) and rebind it in the
    # module's namespace. Crucially this does NOT re-execute the rest of the
    # file - imports, class definitions, decorators, and any other module-level
    # side effects run exactly once (via the normal `import transformers`), so
    # nothing gets double-registered. (An earlier version of this patch
    # exec()'d the WHOLE module source a second time, which silently
    # double-registered a weight-conversion mapping and broke MoE checkpoint
    # loading with an unrelated "many-to-many" ValueError - do not regress to
    # that approach.)
    # The original file has `from __future__ import annotations` (PEP 563),
    # so its type annotations (e.g. `model: PreTrainedModel`) are lazy strings
    # never evaluated at runtime. Extracting just this function's text loses
    # that directive, so plain compile() would eagerly evaluate the
    # annotation and NameError on names only imported under TYPE_CHECKING
    # (confirmed: PreTrainedModel is exactly such a name). Prepend the same
    # future-import to restore the original file's compilation semantics.
    import textwrap
    dedented = "from __future__ import annotations\n" + textwrap.dedent(patched)
    local_ns = {}
    exec(compile(dedented, cml.__file__, "exec"), cml.__dict__, local_ns)
    cml.convert_and_load_state_dict_in_model = local_ns["convert_and_load_state_dict_in_model"]
    print("[patch] applied transformers#43032 workaround (quantized params materialize on CPU first)")
    return True


_patch_transformers_bnb_oom()

BASE   = os.environ["BASE_REPO"]           # Qwen/Qwen3.6-35B-A3B
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

# 4-bit QLoRA load so the 35B base fits a single 40GB A100.
bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)

# ImageTextToText is the correct class for this *ForConditionalGeneration
# multimodal arch (confirmed empirically: it resolves cleanly, the only
# failure was the OOM patched above). No CausalLM fallback - a genuine
# ImageTextToText failure is virtually always the same root cause, and a
# blind fallback on the same GPU risked leaking the first attempt's memory.
_load_kw = dict(quantization_config=bnb, device_map={"": 0},
                torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
model = AutoModelForImageTextToText.from_pretrained(BASE, **_load_kw)
model.config.use_cache = False
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

# Grounded targets: BOTH attention types (full + linear/SSM). Vision leaves
# (qkv/proj/fc) differ, so these never touch the vision tower.
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

# ADAPTER_REPO is now REQUIRED (not optional): durability depends on pushing
# checkpoints to the Hub during training, not just at the very end. A VM
# reclamation mid-run would otherwise wipe /content and lose everything.
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
