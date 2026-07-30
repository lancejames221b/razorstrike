#!/usr/bin/env python3
"""Emergency recovery: convert an FSDP-sharded periodic checkpoint (DCP format,
`pytorch_model_fsdp_0/*.distcp`) into a directly-loadable PEFT adapter directory.

Why this exists: train_lora.py's FINAL save does an explicit FULL_STATE_DICT
gather (`accelerator.get_state_dict()` pulling all 34.66B base+adapter params
onto rank 0) before writing a normal PEFT adapter dir. If that gather OOMs at
the very end of a many-hour run, the only surviving artifacts are the
SHARDED_STATE_DICT periodic checkpoints already pushed to CKPT_GCS during
training (`checkpoint-<step>/pytorch_model_fsdp_0/`) - which merge_push.py
cannot consume directly. This script recovers a usable adapter from the LATEST
such checkpoint instead of losing the run.

Confirmed empirically (2026-07-29 session, against a real checkpoint-250 from
this training run): SHARDED_STATE_DICT periodic saves for this codebase
already contain ONLY the 65M trainable LoRA parameters (no base-model weights
mixed in) - 380 keys (190 lora_A + 190 lora_B pairs), already in PEFT's exact
`base_model.model.<module.path>.lora_A/B.weight` naming convention, full
unsharded tensors (not per-rank shards - lora_A shape[0]==r, lora_B
shape[1]==r for every key, verified). No key remapping needed, just:
dcp_to_torch_save -> unwrap the 'model' key -> save_file -> copy/patch
adapter_config.json.

Usage:
    python3 scripts/dcp_checkpoint_to_peft.py \
        --dcp-dir /content/adapter/checkpoint-2750/pytorch_model_fsdp_0 \
        --config-template /path/to/a/known-good/adapter_config.json \
        --base-model-path /content/base_model \
        --out /content/adapter/recovered_final \
        --expected-r 64

This only recovers the LoRA weights - it does NOT restore optimizer/scheduler
state, so training cannot resume from the recovered directory (use the
sibling `optimizer_0/`/`rng_state_*.pth`/`scheduler.pt` DCP files with
accelerate's own resume path for that - this script is for MERGING, not
resuming). The adapter_config.json is not itself checkpointed per-step, so a
template from any other completed run on the SAME hyperparameters (r,
lora_alpha, lora_dropout, target_modules) is required - only
`base_model_name_or_path` is overwritten.
"""
import argparse
import json
import os

import torch
from safetensors.torch import save_file
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dcp-dir", required=True,
                    help="path to checkpoint-<step>/pytorch_model_fsdp_0/")
    ap.add_argument("--config-template", required=True,
                    help="adapter_config.json from any completed run with the "
                         "SAME r/lora_alpha/lora_dropout/target_modules")
    ap.add_argument("--base-model-path", required=True,
                    help="written into the recovered config's base_model_name_or_path")
    ap.add_argument("--out", required=True, help="output adapter directory")
    ap.add_argument("--expected-r", type=int, required=True,
                    help="LoRA rank to validate every lora_A/lora_B shape against "
                         "before trusting the recovered weights")
    args = ap.parse_args()

    tmp_pt = args.out.rstrip("/") + "_recovered.pt.tmp"
    dcp_to_torch_save(args.dcp_dir, tmp_pt)
    loaded = torch.load(tmp_pt, map_location="cpu", weights_only=False)
    os.remove(tmp_pt)

    sd = loaded["model"] if isinstance(loaded, dict) and "model" in loaded else loaded
    if not isinstance(sd, dict):
        raise SystemExit(f"[dcp-recover] unexpected recovered structure: {type(sd)}")

    a_keys = [k for k in sd if "lora_A" in k]
    b_keys = [k for k in sd if "lora_B" in k]
    non_lora = [k for k in sd if "lora_A" not in k and "lora_B" not in k]
    print(f"[dcp-recover] total keys={len(sd)} lora_A={len(a_keys)} "
          f"lora_B={len(b_keys)} non_lora={len(non_lora)}")
    if non_lora:
        print(f"[dcp-recover] WARNING: {len(non_lora)} non-LoRA keys present "
              f"(sample: {non_lora[:5]}) - this checkpoint may include base "
              f"weights; verify before merging on top of the base model again.")

    bad_a = [(k, tuple(sd[k].shape)) for k in a_keys if sd[k].shape[0] != args.expected_r]
    bad_b = [(k, tuple(sd[k].shape)) for k in b_keys if sd[k].shape[1] != args.expected_r]
    if bad_a or bad_b:
        raise SystemExit(
            f"[dcp-recover] FAIL: {len(bad_a)} lora_A and {len(bad_b)} lora_B "
            f"tensors do NOT match expected r={args.expected_r} - likely "
            f"per-rank shards, not fully reconstructed tensors. "
            f"bad_a sample: {bad_a[:3]} bad_b sample: {bad_b[:3]}")
    print(f"[dcp-recover] shape check PASS: all {len(a_keys)} lora_A have "
          f"dim0=={args.expected_r}, all {len(b_keys)} lora_B have dim1=={args.expected_r}")

    os.makedirs(args.out, exist_ok=True)
    sd = {k: v.contiguous() for k, v in sd.items()}
    save_file(sd, os.path.join(args.out, "adapter_model.safetensors"))

    cfg = json.load(open(args.config_template))
    cfg["base_model_name_or_path"] = args.base_model_path
    json.dump(cfg, open(os.path.join(args.out, "adapter_config.json"), "w"))

    from peft import PeftConfig
    pc = PeftConfig.from_pretrained(args.out)
    print(f"[dcp-recover] PeftConfig round-trip OK: r={pc.r} alpha={pc.lora_alpha} "
          f"target_modules={pc.target_modules}")
    print(f"[dcp-recover] RECOVERED_ADAPTER -> {args.out}")


if __name__ == "__main__":
    main()
