"""Phase 2 — Training script (the core artifact).

Runs identically on Colab (CLI or backup notebook). Reads env:
  BASE_REPO, DATA_REPO, OUT_DIR, ADAPTER_REPO, HF_TOKEN.
"""

import os, torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          Trainer, TrainingArguments, DataCollatorForSeq2Seq)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

BASE   = os.environ["BASE_REPO"]           # lancejames221b/razorstrike-v1-bf16
DATA   = os.environ["DATA_REPO"]           # lancejames221b/razorstrike-offsec-v1
OUT    = os.environ.get("OUT_DIR","/content/adapter")
MAXLEN = int(os.environ.get("MAXLEN","4096"))

tok = AutoTokenizer.from_pretrained(BASE)
if tok.pad_token is None: tok.pad_token = tok.eos_token

def to_features(ex):
    msgs = ex["messages"]
    prompt = tok.apply_chat_template(msgs[:-1], add_generation_prompt=True, tokenize=True)
    full   = tok.apply_chat_template(msgs,       add_generation_prompt=False, tokenize=True)
    if len(full) > MAXLEN or len(prompt) >= len(full):
        return {"input_ids": None, "attention_mask": None, "labels": None}
    labels = [-100]*len(prompt) + full[len(prompt):]
    return {"input_ids": full, "attention_mask":[1]*len(full), "labels": labels}

ds = load_dataset(DATA)
ds = ds.map(to_features, remove_columns=ds["train"].column_names)
ds = ds.filter(lambda r: r["input_ids"] is not None)

model = AutoModelForCausalLM.from_pretrained(
    BASE, dtype=torch.bfloat16, device_map={"":0},
    attn_implementation="sdpa", low_cpu_mem_usage=True)
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()

lora = LoraConfig(
    r=32, lora_alpha=64, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "in_proj_qkv","in_proj_z","in_proj_a","in_proj_b","out_proj",
                    "gate_proj","up_proj","down_proj"])
model = get_peft_model(model, lora)
model.print_trainable_parameters()

args = TrainingArguments(
    output_dir=OUT, num_train_epochs=2,
    per_device_train_batch_size=1, gradient_accumulation_steps=16,
    learning_rate=2e-4, lr_scheduler_type="cosine", warmup_ratio=0.03,
    bf16=True, gradient_checkpointing=True, max_grad_norm=1.0,
    logging_steps=5, save_steps=250, save_total_limit=3,
    eval_strategy="steps", eval_steps=250, optim="adamw_torch",
    report_to="none", dataloader_num_workers=2)

trainer = Trainer(model=model, args=args,
    train_dataset=ds["train"], eval_dataset=ds["validation"],
    data_collator=DataCollatorForSeq2Seq(tok, label_pad_token_id=-100, padding=True))
trainer.train(resume_from_checkpoint=bool(os.environ.get("RESUME")))

model.save_pretrained(OUT); tok.save_pretrained(OUT)
if os.environ.get("ADAPTER_REPO"):
    model.push_to_hub(os.environ["ADAPTER_REPO"], private=True)
    tok.push_to_hub(os.environ["ADAPTER_REPO"], private=True)

print("TRAINING_COMPLETE")
