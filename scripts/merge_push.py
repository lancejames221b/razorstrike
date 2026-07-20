"""Phase 4 — Merge adapter into bf16 and publish (on the same G4 kernel).

Push script `merge_push.py` (new) and run it (blocking --timeout 3600 is fine; ~10–20 min).
"""

import os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("lancejames221b/razorstrike-v1-bf16",
        dtype=torch.bfloat16, device_map={"":0}, low_cpu_mem_usage=True)
m = PeftModel.from_pretrained(base, "/content/adapter")
m = m.merge_and_unload()
m.save_pretrained("/content/merged", safe_serialization=True, max_shard_size="5GB")
AutoTokenizer.from_pretrained("lancejames221b/razorstrike-v1-bf16").save_pretrained("/content/merged")
m.push_to_hub("lancejames221b/razorstrike-v1-offsec", private=True)
AutoTokenizer.from_pretrained("/content/merged").push_to_hub("lancejames221b/razorstrike-v1-offsec", private=True)
print("MERGE_PUSHED")
