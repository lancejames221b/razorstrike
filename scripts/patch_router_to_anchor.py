#!/usr/bin/env python3
"""Patch razorstrike_v2_bf16's router tensors: swap from donor[0] (huihui) to the
ANCHOR (Qwen3.6-35B-A3B base). Router is a direct-copy family, never touched by
DARE-TIES, so this is a targeted shard patch - no need to re-run the full merge.

Root cause: coherence stress-test showed 100%-reproducible CJK contamination in
coding-context generations (both temp 0.6 and greedy temp 0.0) - textbook
router/expert-misalignment symptom. huihui's router was calibrated to ITS OWN
expert weights; applied to the merged (anchor+DARE-TIES) experts, it misroutes
coding-context tokens. Anchor router is the correctly-matched pairing for
anchor-derived merged experts.
"""
import json
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file

ANCHOR = Path("/Volumes/Scratch/ml-workspace/models/Qwen3.6-35B-A3B")
MERGED = Path("/Volumes/Scratch/ml-workspace/merged/razorstrike_v2_bf16")


def main():
    anchor_wm = json.load(open(ANCHOR / "model.safetensors.index.json"))["weight_map"]
    merged_wm = json.load(open(MERGED / "model.safetensors.index.json"))["weight_map"]

    router_keys = sorted(k for k in merged_wm if k.endswith(".mlp.gate.weight") and "layers." in k)
    print(f"Found {len(router_keys)} router tensors to patch")

    # Group by which OUTPUT shard they live in (patch only those shards)
    by_shard = {}
    for k in router_keys:
        by_shard.setdefault(merged_wm[k], []).append(k)
    print(f"Spread across {len(by_shard)} output shards")

    for shard_file, keys in sorted(by_shard.items()):
        shard_path = MERGED / shard_file
        # Load the full existing shard (need to preserve every other tensor in it)
        buf = {}
        with safe_open(str(shard_path), framework="pt") as sf:
            for k in sf.keys():
                buf[k] = sf.get_tensor(k)

        # Overwrite the router tensors with the anchor's version
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
