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
import sys
import torch
from transformers import AutoModelForImageTextToText, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_card_defaults import MODEL_CARD_SAMPLING_SECTION, write_generation_config

BASE         = os.environ.get("BASE_REPO", "lancejames221b/HAWQ-v1")
ADAPTER_DIR  = os.environ.get("ADAPTER_DIR", "/content/adapter")
ADAPTER_REPO = os.environ.get("ADAPTER_REPO", "lancejames221b/HAWQ-SEC-RE-lora-v3")
MERGED_DIR   = os.environ.get("MERGED_DIR", "/content/merged")
MERGED_REPO  = os.environ.get("MERGED_REPO", "lancejames221b/HAWQ-SEC-RE")
TOKEN        = os.environ.get("HF_TOKEN")

MODEL_CARD = f"""---
license: apache-2.0
base_model: {BASE}
tags:
- qwen3_5_moe
- moe
- lora
- reasoning
- reverse-engineering
- decompilation
- security
language:
- en
pipeline_tag: text-generation
library_name: transformers
---

# HAWQ-SEC-RE

LoRA SFT fine-tune of HAWQ-v1 teaching faithful reverse-engineering **analysis** of
x86-64 assembly (purpose, I/O, algorithm, security-relevant behavior).

## Training

- Base: `{BASE}` (Holo3+Qwopus+AgentWorld merge on Qwen3.6-35B-A3B, hybrid
  linear-attention/SSM MoE architecture, 256 experts, text-only CausalLM)
- Method: LoRA SFT via `transformers` + `peft`, response-only prompt-prefix
  masking (no TRL, avoiding a v5-transformers compatibility risk)
- Data: `hawq-re-v4` + omp-advisory/clean-control DPO (v1.1) - RE-analysis + decompile families from
  LLM4Binary/decompile-bench with frontier-generated gold analyses
- 4x A100 FSDP, 2 epochs, MAXLEN=4096, r=64/alpha=128, attention+SSM target
  modules

{MODEL_CARD_SAMPLING_SECTION}
## License

Released under **Apache 2.0**, matching the {BASE} base model.
"""


def load_base():
    kw = dict(dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="cpu")
    if os.environ.get("FORCE_CAUSAL_LM", "0") == "1":
        return AutoModelForCausalLM.from_pretrained(BASE, **kw)
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


def _patch_adapter_dir(src_dir):
    """Mirror eval_peft_direct.py's _patch_adapter_dir: strip any
    `.language_model.` segment from the adapter's saved LoRA keys before
    loading. Safe no-op if the segment isn't present (str.replace on a
    missing substring is a no-op) - always applied defensively since this
    script, like the eval harness, loads a text-only CausalLM base whose
    key names may not match what the adapter was trained against."""
    import shutil, tempfile
    from safetensors.torch import safe_open, save_file
    dst_dir = tempfile.mkdtemp(prefix="adapter_patched_")
    for fname in os.listdir(src_dir):
        fpath = os.path.join(src_dir, fname)
        if fname != "adapter_model.safetensors" and os.path.isfile(fpath):
            shutil.copy(fpath, os.path.join(dst_dir, fname))
    src_path = os.path.join(src_dir, "adapter_model.safetensors")
    tensors = {}
    with safe_open(src_path, framework="pt") as f:
        for k in f.keys():
            tensors[k.replace(".language_model.", ".")] = f.get_tensor(k)
    save_file(tensors, os.path.join(dst_dir, "adapter_model.safetensors"))
    return dst_dir


def main():
    adapter_path = _patch_adapter_dir(resolve_adapter_dir())

    base = load_base()
    peft_model = PeftModel.from_pretrained(base, adapter_path)

    # Load-bearing no-op-merge check (skill gguf-qwen35moe-lora-deploy-lmstudio):
    # a key-prefix mismatch between the adapter and the base module tree makes
    # merge_and_unload() silently no-op (UserWarning, not an exception) and
    # ship the unmodified base under a new name. Snapshot real q_proj weight
    # slices from every LoRA-wrapped attention layer before merge (discovered
    # from the state dict, not hardcoded layer indices - only 10/40 layers
    # are full-attention here), diff a strided sample after.
    _pre_sd = peft_model.state_dict()
    _q_keys = sorted(k for k in _pre_sd if k.endswith("self_attn.q_proj.base_layer.weight"))
    if not _q_keys:
        # Some peft versions proxy .weight straight through instead of
        # nesting the original weight under .base_layer.
        _q_keys = sorted(k for k in _pre_sd
                          if k.endswith("self_attn.q_proj.weight") and ".lora_" not in k)
    if not _q_keys:
        print("[merge] FATAL: no q_proj target-module keys found in the "
              "pre-merge state dict - can't verify the merge isn't a no-op", flush=True)
        raise SystemExit(4)
    _stride = max(1, len(_q_keys) // 5)
    _sample_keys = _q_keys[::_stride][:5]
    _pre_snap = {k: _pre_sd[k][:2, :2].clone() for k in _sample_keys}
    print(f"[merge] snapshotted {len(_pre_snap)}/{len(_q_keys)} q_proj slices "
          f"pre-merge: {_sample_keys}", flush=True)

    m = peft_model.merge_and_unload()

    _post_sd = m.state_dict()
    _n_changed = 0
    for k, pre_val in _pre_snap.items():
        post_key = k.replace("base_model.model.", "", 1).replace(".base_layer.weight", ".weight")
        post_val = _post_sd.get(post_key)
        if post_val is None:
            print(f"[merge] FATAL: post-merge key {post_key!r} (from {k!r}) not found", flush=True)
            raise SystemExit(4)
        post_val = post_val[:2, :2]
        if not torch.isfinite(post_val).all():
            print(f"[merge] FATAL: post-merge slice for {post_key} contains NaN/Inf", flush=True)
            raise SystemExit(4)
        if not torch.equal(pre_val, post_val):
            _n_changed += 1
    if _n_changed == 0:
        print("[merge] FATAL: merge_and_unload() was a no-op - none of the sampled "
              "q_proj slices changed. Adapter keys likely don't match the base "
              "module tree (see eval_peft_direct.py's _patch_adapter_dir).", flush=True)
        raise SystemExit(4)
    print(f"[merge] no-op-merge check passed: {_n_changed}/{len(_pre_snap)} sampled "
          f"q_proj slices changed, all finite", flush=True)

    # Sanity check: confirm the merge actually changed weights (a no-op merge
    # would silently ship the unmodified base under a new name).
    sample_param = next(iter(m.state_dict().values()))
    assert torch.isfinite(sample_param).all(), "merged weights contain NaN/Inf - aborting push"
    print(f"[sanity] merged model has {sum(p.numel() for p in m.parameters()):,} parameters, weights finite")

    m.save_pretrained(MERGED_DIR, safe_serialization=True, max_shard_size="5GB")
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.save_pretrained(MERGED_DIR)

    gc_path = write_generation_config(MERGED_DIR)
    print(f"[merge] wrote {gc_path} with pinned sampling defaults "
          f"(temp 0.6/top_p .95/top_k 20/repetition_penalty 1.0)", flush=True)

    with open(os.path.join(MERGED_DIR, "README.md"), "w") as f:
        f.write(MODEL_CARD)

    if os.environ.get("SKIP_HF_PUSH", "0") == "1":
        print(f"[merge] SKIP_HF_PUSH=1, leaving merged model at {MERGED_DIR} "
              f"(no HF upload) - push to GCS/HF separately", flush=True)
    else:
        # Push straight from the already-serialized MERGED_DIR instead of
        # model.push_to_hub()/tok.push_to_hub(), which would each re-run
        # save_pretrained() into a fresh system-temp dir (a second ~70GB
        # serialization pass, off SeXternal/HF_HOME, before uploading).
        _pub = os.environ.get("PUBLISH_PUBLIC", "0") == "1"
        from huggingface_hub import create_repo, upload_folder
        create_repo(MERGED_REPO, private=not _pub, exist_ok=True, token=TOKEN)
        upload_folder(folder_path=MERGED_DIR, repo_id=MERGED_REPO, token=TOKEN,
                       commit_message="Merge HAWQ-SEC-RE adapter into HAWQ-v1")
    print("MERGE_PUSHED")


if __name__ == "__main__":
    main()
