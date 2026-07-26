#!/usr/bin/env python3
"""One-time offline 4-bit quantization of the HAWQ-v1 base model.

Why this exists: on-the-fly 4-bit quantization during from_pretrained OOMs
on a single 40GB A100 for this MoE arch (transformers' loader materializes
the source tensor at full bf16 precision before quantizing it, confirmed
empirically: OOM at 58-62% through weight loading, GPU at 39.35/39.49GiB).
Spreading that same peak-materialization step across 2 GPUs via
DEVICE_MAP=auto (80GB combined) avoids the OOM. The RESULT is then saved in
its already-quantized form - reloading it later needs no quantization step
at all (no full-precision materialization), so every DDP rank in the real
multi-GPU training run can load it directly onto its own single GPU with
device_map={"": rank} and no OOM risk. train_lora.py auto-detects an
already-quantized checkpoint via its saved quantization_config and skips
re-quantization.

Usage (on a 2x40GB+ A100 host):
    BASE_REPO=lancejames221b/HAWQ-v1 OUT_DIR=/content/hawq-4bit \
        python3 scripts/quantize_base_4bit.py
"""
import os
import torch
from transformers import (AutoModelForImageTextToText, AutoModelForCausalLM,
                          AutoTokenizer, BitsAndBytesConfig)

BASE = os.environ.get("BASE_REPO", "lancejames221b/HAWQ-v1")
OUT = os.environ.get("OUT_DIR", "/content/hawq-4bit")
HF_TOKEN = os.environ.get("HF_TOKEN")

print(f"[quantize] loading {BASE} in 4-bit across device_map=auto (spreads the "
      f"OOM-prone full-precision materialization step across all visible GPUs)", flush=True)

tok = AutoTokenizer.from_pretrained(BASE, token=HF_TOKEN)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)

_max_mem_gib = os.environ.get("MAX_MEMORY_GIB", "36").strip()  # headroom under 40GB/card
max_memory = {i: f"{_max_mem_gib}GiB" for i in range(torch.cuda.device_count())} if _max_mem_gib else None

load_kw = dict(device_map="auto", quantization_config=bnb_cfg, low_cpu_mem_usage=True, token=HF_TOKEN)
if max_memory:
    load_kw["max_memory"] = max_memory

if os.environ.get("FORCE_CAUSAL_LM", "0") == "1":
    model = AutoModelForCausalLM.from_pretrained(BASE, **load_kw)
else:
    try:
        model = AutoModelForImageTextToText.from_pretrained(BASE, **load_kw)
    except Exception as e:
        print(f"[quantize] ImageTextToText failed ({type(e).__name__}); trying CausalLM", flush=True)
        model = AutoModelForCausalLM.from_pretrained(BASE, **load_kw)

print(f"[quantize] loaded OK, device map: {getattr(model, 'hf_device_map', None)}", flush=True)

os.makedirs(OUT, exist_ok=True)
model.save_pretrained(OUT, safe_serialization=True, max_shard_size="5GB")
tok.save_pretrained(OUT)
print(f"[quantize] saved 4-bit checkpoint -> {OUT}", flush=True)

# Sanity check: confirm the saved config actually records the quantization,
# so a downstream train_lora.py load can auto-detect it.
import json
with open(os.path.join(OUT, "config.json")) as f:
    cfg = json.load(f)
assert "quantization_config" in cfg, "saved config.json missing quantization_config - save didn't take"
print("[quantize] QUANTIZE_COMPLETE - config.json carries quantization_config, "
      "downstream loads will auto-detect and skip re-quantization", flush=True)
