#!/usr/bin/env python3
"""CPU-pinned mlx_lm convert for HAWQ-SEC-RE MoE checkpoints.

Plain `mlx_lm.convert` on this 35B-A3B MoE model crashes at save time with a
Metal command-buffer watchdog timeout (RuntimeError: [METAL] Command buffer
execution failed ... kIOGPUCommandBufferCallbackErrorSubmissionsIgnored) -
the whole quantize+eval graph gets materialized as one giant GPU submission,
which exceeds the watchdog on large MoE/hybrid-attention models. Not OOM,
not fixable by killing competing GPU processes or --dtype float32 (see
skill mlx-lm-convert-gpu-watchdog-timeout). Forcing mx.default_device to
CPU sidesteps the watchdog entirely; CPU eval here is I/O-bound (reading
source shards) more than compute-bound, so it stays viable even for
60-70GB source models.

Env: HF_PATH (source HF checkpoint, model_type must be one mlx_lm
     recognizes - e.g. 'qwen3_5_moe', not the text-only extraction's
     'qwen3_5_moe_text'), MLX_PATH (output dir), Q_BITS (default 4),
     Q_GROUP_SIZE (default 64).
"""
import os

import mlx.core as mx
mx.set_default_device(mx.cpu)

from mlx_lm.convert import convert

convert(
    hf_path=os.environ.get("HF_PATH", "/Volumes/SeXternal/hawq_v3_out/merged_mlxfix"),
    mlx_path=os.environ.get("MLX_PATH", "/Volumes/SeXternal/hawq_v3_out/hawq-sec-re-v1-mlx-4bit"),
    quantize=True,
    q_group_size=int(os.environ.get("Q_GROUP_SIZE", "64")),
    q_bits=int(os.environ.get("Q_BITS", "4")),
    q_mode="affine",
)
print("CONVERT_CPU_DONE")
