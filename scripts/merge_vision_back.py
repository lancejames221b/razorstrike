#!/usr/bin/env python3
"""Graft the Qwen3.6-35B-A3B anchor's vision tower back onto HAWQ-SEC-RE.

HAWQ-v1 (and everything downstream, including HAWQ-SEC-RE) was extracted as a
text-only Qwen3_5MoeForCausalLM: `model.*` / `lm_head.weight`, 693 keys, zero
vision tensors (config.json still carries stale vision_config/image_token_id
metadata from the original ConditionalGeneration lineage, but the safetensors
never had `model.visual.*` weights - see eval_peft_direct.py's docstring on
the `language_model.` module-tree mismatch this exact split caused for the
adapter merge).

The anchor (`Qwen/Qwen3.6-35B-A3B`, local copy at ANCHOR_PATH, same lineage
HAWQ-v1 was built from - see merge_qwen36_uncensored.py/merge_pair_*.py's
_DEFAULT_ANCHOR_PATH) is the original Qwen3_5MoeForConditionalGeneration
multimodal checkpoint: `model.visual.*` (333 keys, vision encoder),
`model.language_model.*` (692 keys, text backbone, prefixed one level deeper
than our CausalLM's `model.*`), `lm_head.weight` (top-level, same key in
both), and `mtp.*` (10 keys, multi-token-prediction head we stripped for
GGUF but keep here verbatim since the wrapper class supports it).

Text-config parameters (hidden_size, num_hidden_layers, num_attention_heads,
num_key_value_heads, head_dim, vocab_size, num_experts,
moe_intermediate_size) are byte-identical between our merged HAWQ-SEC-RE and
the anchor's text_config - verified before running this - so the anchor's
`model.language_model.*` slots are drop-in compatible with our fine-tuned
weights (just re-prefix `model.*` -> `model.language_model.*`).

Streaming, shard-by-shard, memory-safe (same pattern as
merge_qwen36_uncensored.py): never holds more than one shard's tensors in
RAM at a time.

Env: ANCHOR_PATH (Qwen3.6-35B-A3B multimodal checkpoint, vision tower + mtp
     source), MERGED_PATH (HAWQ-SEC-RE text-only checkpoint, fine-tuned
     weights source), OUTPUT_PATH.
"""
import json
import os
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

ANCHOR_PATH = os.environ.get("ANCHOR_PATH", "/Volumes/Scratch/ml-workspace/models/Qwen3.6-35B-A3B")
MERGED_PATH = os.environ.get("MERGED_PATH", "/Volumes/SeXternal/hawq_v3_out/merged")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/Volumes/SeXternal/hawq_v3_out/merged_vlm")


def load_index(model_path: Path) -> dict:
    with open(model_path / "model.safetensors.index.json") as f:
        return json.load(f)["weight_map"]


def merged_key_for(anchor_key: str) -> str:
    """Map an anchor key (multimodal wrapper naming) to the corresponding
    key in our text-only merged checkpoint."""
    if anchor_key == "lm_head.weight":
        return anchor_key
    prefix = "model.language_model."
    assert anchor_key.startswith(prefix), f"unexpected non-language_model key routed as text: {anchor_key}"
    return "model." + anchor_key[len(prefix):]


def main():
    anchor = Path(ANCHOR_PATH)
    merged = Path(MERGED_PATH)
    output = Path(OUTPUT_PATH)
    output.mkdir(parents=True, exist_ok=True)

    print(f"[vlm-merge] anchor (vision/mtp source): {anchor}")
    print(f"[vlm-merge] merged (text weights source): {merged}")
    print(f"[vlm-merge] output: {output}")

    anchor_map = load_index(anchor)
    merged_map = load_index(merged)
    print(f"[vlm-merge] anchor: {len(anchor_map)} keys, merged: {len(merged_map)} keys")

    vision_keys = {k for k in anchor_map if k.startswith("model.visual.")}
    mtp_keys = {k for k in anchor_map if k.startswith("mtp.")}
    text_keys = {k for k in anchor_map if k.startswith("model.language_model.")}
    lm_head_keys = {k for k in anchor_map if k == "lm_head.weight"}
    other = set(anchor_map) - vision_keys - mtp_keys - text_keys - lm_head_keys
    print(f"[vlm-merge] vision={len(vision_keys)} mtp={len(mtp_keys)} "
          f"text={len(text_keys)} lm_head={len(lm_head_keys)} other={len(other)}")
    if other:
        print(f"[vlm-merge] FATAL: unrecognized anchor keys: {sorted(other)[:10]}")
        raise SystemExit(4)

    missing_in_merged = {merged_key_for(k) for k in (text_keys | lm_head_keys)} - set(merged_map)
    if missing_in_merged:
        print(f"[vlm-merge] FATAL: {len(missing_in_merged)} text keys missing from merged "
              f"checkpoint: {sorted(missing_in_merged)[:10]}")
        raise SystemExit(4)

    shard_files = sorted(set(anchor_map.values()))
    written = set()
    total = 0
    for shard_file in shard_files:
        shard_keys = [k for k in anchor_map if anchor_map[k] == shard_file]
        buf = {}
        for key in shard_keys:
            if key in vision_keys or key in mtp_keys:
                src_path = anchor / anchor_map[key]
                with safe_open(str(src_path), framework="pt") as sf:
                    buf[key] = sf.get_tensor(key)
            else:  # text or lm_head - pull the fine-tuned weight
                mkey = merged_key_for(key)
                src_path = merged / merged_map[mkey]
                with safe_open(str(src_path), framework="pt") as sf:
                    buf[key] = sf.get_tensor(mkey)
        out_path = output / shard_file
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_file(buf, str(out_path))
        total += len(buf)
        written.update(buf.keys())
        print(f"[vlm-merge] wrote {shard_file} ({len(buf)} tensors, {total}/{len(anchor_map)} total)", flush=True)
        del buf

    missing = set(anchor_map) - written
    if missing:
        print(f"[vlm-merge] FATAL: {len(missing)} tensors never written: {sorted(missing)[:10]}")
        raise SystemExit(4)

    index_out = {"metadata": {}, "weight_map": {k: v for k, v in anchor_map.items() if k in written}}
    index_out["metadata"]["total_size"] = sum(
        (output / f).stat().st_size for f in sorted(set(index_out["weight_map"].values())))
    with open(output / "model.safetensors.index.json", "w") as f:
        json.dump(index_out, f)

    # Auxiliary files: anchor's config.json already carries the correct
    # multimodal structure (Qwen3_5MoeForConditionalGeneration, vision_config,
    # image/video token ids) with text_config values that byte-match our
    # merged checkpoint (verified before running this script) - copy as-is
    # rather than hand-editing. Same for tokenizer/chat_template/vision
    # preprocessor files - all from the anchor, since that's the canonical
    # multimodal-aware set (our text-only merged/ dir's tokenizer files are
    # a subset with no vision preprocessing).
    skip = {"model.safetensors.index.json"}
    copied = []
    for item in anchor.iterdir():
        if item.is_file() and not item.name.endswith(".safetensors") and item.name not in skip:
            shutil.copy2(item, output / item.name)
            copied.append(item.name)
    print(f"[vlm-merge] copied {len(copied)} auxiliary files from anchor: {sorted(copied)}")

    print(f"[vlm-merge] DONE: {total} tensors written to {output}")


if __name__ == "__main__":
    main()
