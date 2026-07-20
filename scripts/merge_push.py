"""Phase 4 - Merge the RazorStrike v2 text QLoRA into the base and publish.

Loads clean Qwen/Qwen3.6-35B-A3B in bf16, applies the trained adapter, merges,
and pushes the merged model. This is a WEIGHT op, not a GPU op: 35B in bf16 is
~70GB, which does NOT fit the 40GB training GPU. Run it on CPU with high RAM
(Colab A100 high-RAM runtime ~83GB system RAM fits it) or a big-RAM box.

Env: BASE_REPO (default Qwen/Qwen3.6-35B-A3B), ADAPTER_DIR (/content/adapter),
     MERGED_DIR (/content/merged), MERGED_REPO (lancejames221b/razorstrike-v2), HF_TOKEN.
"""

import os, torch
from transformers import AutoModelForImageTextToText, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE        = os.environ.get("BASE_REPO", "Qwen/Qwen3.6-35B-A3B")
ADAPTER_DIR = os.environ.get("ADAPTER_DIR", "/content/adapter")
MERGED_DIR  = os.environ.get("MERGED_DIR", "/content/merged")
MERGED_REPO = os.environ.get("MERGED_REPO", "lancejames221b/razorstrike-v2")

# bf16 on CPU: full-precision merge needs the RAM, not GPU VRAM.
_kw = dict(torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="cpu")
try:
    base = AutoModelForImageTextToText.from_pretrained(BASE, **_kw)
except Exception as e:
    print(f"[load] ImageTextToText failed ({type(e).__name__}); trying CausalLM")
    base = AutoModelForCausalLM.from_pretrained(BASE, **_kw)

m = PeftModel.from_pretrained(base, ADAPTER_DIR)
m = m.merge_and_unload()
m.save_pretrained(MERGED_DIR, safe_serialization=True, max_shard_size="5GB")
AutoTokenizer.from_pretrained(BASE).save_pretrained(MERGED_DIR)

token = os.environ.get("HF_TOKEN")
m.push_to_hub(MERGED_REPO, private=True, token=token)
AutoTokenizer.from_pretrained(MERGED_DIR).push_to_hub(MERGED_REPO, private=True, token=token)
print("MERGE_PUSHED")
