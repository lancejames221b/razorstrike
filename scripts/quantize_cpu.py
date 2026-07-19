#!/usr/bin/env python3
"""CPU-pinned mlx_vlm quantize - bypasses the Metal command-buffer watchdog timeout."""
import sys
import mlx.core as mx
mx.set_default_device(mx.cpu)

from mlx_vlm.convert import convert

hf_path = sys.argv[1]
mlx_path = sys.argv[2]

convert(hf_path=hf_path, mlx_path=mlx_path, quantize=True, q_bits=4, q_group_size=64)
print("QUANTIZE_CPU_DONE", flush=True)
