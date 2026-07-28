#!/usr/bin/env python3
"""CPU-pinned mlx_vlm convert for the vision-restored HAWQ-SEC-RE checkpoint.

Same Metal command-buffer watchdog timeout risk as mlx_convert_cpu.py (see
skill mlx-lm-convert-gpu-watchdog-timeout) applies here too, on a larger
(67GB) multimodal source - force CPU to sidestep it.

Uses mlx_vlm (not mlx_lm): mlx_lm's qwen3_5_moe.py Model.sanitize() drops
every key starting with `vision_tower`/`model.visual` (confirmed by
reading the source) - converting a vision checkpoint through mlx_lm
silently produces a text-only MLX model again. mlx_vlm.models.qwen3_5_moe
is a real VLM handler (language.py + vision.py) for this exact
architecture (linear-attention/SSM hybrid text backbone + Qwen3-VL-style
vision tower).

Env: HF_PATH (source, model_type must be 'qwen3_5_moe' - the merged_vlm/
     output of merge_vision_back.py already has this, copied verbatim
     from the anchor's config.json, no patch needed), MLX_PATH (output
     dir), Q_BITS (default 4), Q_GROUP_SIZE (default 64).
"""
import os

import mlx.core as mx
mx.set_default_device(mx.cpu)

from mlx_vlm.convert import convert

convert(
    hf_path=os.environ.get("HF_PATH", "/Volumes/SeXternal/hawq_v3_out/merged_vlm"),
    mlx_path=os.environ.get("MLX_PATH", "/Volumes/SeXternal/hawq_v3_out/hawq-sec-re-v1-vlm-mlx-4bit"),
    quantize=True,
    q_group_size=int(os.environ.get("Q_GROUP_SIZE", "64")),
    q_bits=int(os.environ.get("Q_BITS", "4")),
)
print("VLM_CONVERT_CPU_DONE")
