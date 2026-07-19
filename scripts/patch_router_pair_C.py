#!/usr/bin/env python3
"""Patch pair_C_siq1_unimath_bf16's router tensors: swap from donor[0] (SIQ-1, the
default since huihui is absent from this pair) to the ANCHOR (Qwen3.6-35B-A3B base).
Router is a direct-copy family, never touched by DARE-TIES, so this is a targeted
shard patch - no need to re-run the full merge. Matches the router source convention
used when huihui is absent (anchor is the natural fallback, already shown inert for
this defect via byte-identical greedy-output swap test on the full 3-donor merge)."""
import json
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file

ANCHOR = Path("/Volumes/Scratch/ml-workspace/models/Qwen3.6-35B-A3B")
MERGED = Path("/Volumes/Scratch/ml-workspace/merged/pair_C_siq1_unimath_bf16")


def main():
    anchor_wm = json.load(open(ANCHOR / "model.safetensors.index.json"))["weight_map"]
    merged_wm = json.load(open(MERGED / "model.safetensors.index.json"))["weight_map"]

    router_keys = sorted(k for k in merged_wm if k.endswith(".mlp.gate.weight") and "layers." in k)
    print(f"Found {len(router_keys)} router tensors to patch")

    by_shard = {}
    for k in router_keys:
        by_shard.setdefault(merged_wm[k], []).append(k)
    print(f"Spread across {len(by_shard)} output shards")

    for shard_file, keys in sorted(by_shard.items()):
        shard_path = MERGED / shard_file
        buf = {}
        with safe_open(str(shard_path), framework="pt") as sf:
            for k in sf.keys():
                buf[k] = sf.get_tensor(k)

        replaced = 0
        for k in keys:
            anchor_file = ANCHOR / anchor_wm[k]
            with safe_open(str(anchor_file), framework="pt") as sf:
                anchor_tensor = sf.get_tensor(k)
            if anchor_tensor.shape != buf[k].shape:
                raise ValueError(f"Shape mismatch on {k}: anchor {tuple(anchor_tensor.shape)} vs merged {tuple(buf[k].shape)}")
            buf[k] = anchor_tensor.to(buf[k].dtype)
            replaced += 1

        save_file(buf, str(shard_path))
        print(f"  Patched {shard_file}: {replaced}/{len(keys)} router tensors replaced ({len(buf)} total tensors in shard)")

    print(f"\n\u2713 Router patch complete. {len(router_keys)} tensors now verbatim from anchor.")


if __name__ == "__main__":
    main()
