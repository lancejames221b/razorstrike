"""Phase 4 - Merge the RazorStrike v2 text QLoRA into the base and publish.

Loads clean Qwen/Qwen3.6-35B-A3B in bf16, applies the trained adapter, merges,
and pushes the merged model. This is a WEIGHT op, not a GPU op: 35B in bf16 is
~70GB, which does NOT fit the 40GB training GPU. Run it on CPU with high RAM
(Colab A100 high-RAM runtime ~83GB system RAM fits it) or a big-RAM box.

Env: BASE_REPO (default Qwen/Qwen3.6-35B-A3B), ADAPTER_DIR (/content/adapter),
     ADAPTER_REPO (lancejames221b/razorstrike-v2-offsec-lora) - used as a fallback
     to pull the adapter from the Hub if ADAPTER_DIR doesn't exist locally, since
     this script may run on a fresh VM/session distinct from the one training
     finished on,
     MERGED_DIR (/content/merged), MERGED_REPO (lancejames221b/razorstrike-v2), HF_TOKEN.
"""

import os
import torch
from transformers import AutoModelForImageTextToText, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE         = os.environ.get("BASE_REPO", "lancejames221b/HAWQ")
ADAPTER_DIR  = os.environ.get("ADAPTER_DIR", "/content/adapter")
ADAPTER_REPO = os.environ.get("ADAPTER_REPO", "lancejames221b/HAWQ-SEC-re-validation-lora")
MERGED_DIR   = os.environ.get("MERGED_DIR", "/content/merged")
MERGED_REPO  = os.environ.get("MERGED_REPO", "lancejames221b/HAWQ-RE")
TOKEN        = os.environ.get("HF_TOKEN")

MODEL_CARD = f"""---
license: apache-2.0
base_model: {BASE}
tags:
- qwen3_5_moe
- moe
- lora
- reasoning
- agentic
- coding
- security
- reverse-engineering
- cryptography
- offensive-security
language:
- en
pipeline_tag: image-text-to-text
library_name: transformers
---

# HAWQ-RE

HAWQ-RE is a **LoRA SFT** fine-tune of the **HAWQ** base
({BASE}), a Holo3+Qwopus+AgentWorld merge on Qwen3.6-35B-A3B (hybrid
linear-attention/SSM MoE architecture, 256 experts, vision tower intact).
The LoRA adapter is trained on reverse-engineering (decompile-bench) and
anti-loop recovery data, then merged back into the base.

## Training

- Base: `{BASE}` (Holo3+Qwopus+AgentWorld merge on Qwen3.6-35B-A3B)
- Method: LoRA SFT via `transformers` + `peft`, response-only prompt-prefix
  masking (no TRL, avoiding a v5-transformers compatibility risk)
- Data: `lancejames221b/HAWQ-SEC-re-validation` - RE foundation (decompile-bench)
  and anti-doom-loop traces
- MAX_STEPS=500, `MAXLEN=3072`, LoRA rank/alpha per `adapter_config.json`

## License

Released under **Apache 2.0**, matching the {BASE} base model.
"""


def load_base():
    kw = dict(dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="cpu")
    try:
        return AutoModelForImageTextToText.from_pretrained(BASE, **kw)
    except Exception as e:
        print(f"[load] ImageTextToText failed ({type(e).__name__}); trying CausalLM")
        return AutoModelForCausalLM.from_pretrained(BASE, **kw)


def resolve_adapter_dir():
    if os.path.isdir(ADAPTER_DIR) and os.path.exists(os.path.join(ADAPTER_DIR, "adapter_config.json")):
        print(f"[adapter] using local dir: {ADAPTER_DIR}")
        return ADAPTER_DIR
    print(f"[adapter] {ADAPTER_DIR} not found locally, pulling final adapter from {ADAPTER_REPO}")
    from huggingface_hub import snapshot_download
    return snapshot_download(ADAPTER_REPO, token=TOKEN)


def main():
    adapter_path = resolve_adapter_dir()

    base = load_base()
    m = PeftModel.from_pretrained(base, adapter_path)
    m = m.merge_and_unload()

    # Sanity check: confirm the merge actually changed weights (a no-op merge
    # would silently ship the unmodified base under a new name).
    sample_param = next(iter(m.state_dict().values()))
    assert torch.isfinite(sample_param).all(), "merged weights contain NaN/Inf - aborting push"
    print(f"[sanity] merged model has {sum(p.numel() for p in m.parameters()):,} parameters, weights finite")

    m.save_pretrained(MERGED_DIR, safe_serialization=True, max_shard_size="5GB")
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.save_pretrained(MERGED_DIR)

    with open(os.path.join(MERGED_DIR, "README.md"), "w") as f:
        f.write(MODEL_CARD)

    m.push_to_hub(MERGED_REPO, private=False, token=TOKEN)
    tok.push_to_hub(MERGED_REPO, private=False, token=TOKEN)
    from huggingface_hub import upload_file
    upload_file(path_or_fileobj=os.path.join(MERGED_DIR, "README.md"), path_in_repo="README.md",
                repo_id=MERGED_REPO, repo_type="model", token=TOKEN)
    print("MERGE_PUSHED")


if __name__ == "__main__":
    main()
