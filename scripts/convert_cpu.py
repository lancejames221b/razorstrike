#!/usr/bin/env python3
"""CPU-pinned mlx_vlm convert - bypasses the Metal command-buffer watchdog timeout
that crashes GPU-path convert on large (35B+) custom-arch models at save time."""
import sys
import mlx.core as mx
mx.set_default_device(mx.cpu)

from mlx_vlm.convert import convert

hf_path = sys.argv[1]
mlx_path = sys.argv[2]

convert(hf_path=hf_path, mlx_path=mlx_path, dtype="bfloat16")
print("CONVERT_CPU_DONE", flush=True)
