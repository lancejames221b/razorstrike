#!/usr/bin/env python3
"""Idempotent VM setup for RazorStrike v2 training on a fresh Colab session.

Run this ONCE after every VM (re)provision - handles HF auth, repo clone/pull,
and ALL deps needed for full-speed bf16 LoRA training on the G4 (RTX PRO 6000
Blackwell). Consolidates lessons from repeated reclaim cycles so a fresh VM
comes up complete in one shot instead of hitting missing/stale deps one at a
time:
  - transformers pinned to 5.14.1 (matches sm_120/torch 2.11+cu128 stack)
  - torchao>=0.16.0 (stock VM image ships 0.10.0, which transformers rejects)
  - flash-linear-attention (fla-core): the SSM/linear-attention layers are 30
    of 40 layers in this model; without this the fast kernel path is skipped
    and training runs at roughly 2x the step time (24s/it vs ~12s/it observed)

Usage (from a `colab exec` script, or interactively):
    python3 /content/razorstrike/scripts/vm_setup.py
"""

import os
import subprocess
import sys


def run(cmd, timeout=300):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout, r.stderr


def main():
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("[setup] WARNING: HF_TOKEN not set in environment")
    else:
        from huggingface_hub import login
        login(hf_token)
        print("[setup] HF_LOGIN_OK")

    # Clone/pull the repo
    if os.path.isdir("/content/razorstrike"):
        rc, out, err = run("cd /content/razorstrike && git fetch origin && git reset --hard origin/main")
        print(f"[setup] repo synced (rc={rc}): {out.strip()[-200:]}")
    else:
        rc, out, err = run("cd /content && git clone -q https://github.com/lancejames221b/razorstrike.git")
        print(f"[setup] repo cloned (rc={rc})")

    # All deps in one pass, idempotent (pip no-ops if already satisfied).
    deps = [
        "transformers==5.14.1",
        "peft>=0.14",
        "accelerate>=1.0",
        "datasets>=3.0",
        "huggingface_hub",
        "torchao>=0.16.0",       # stock image ships 0.10.0; transformers requires >=0.16.0
        "flash-linear-attention",  # ~2x speedup on the 30/40 linear-attention/SSM layers
    ]
    rc, out, err = run(f"pip -q install -U {' '.join(repr(d) if ' ' in d else d for d in deps)}", timeout=300)
    print(f"[setup] deps installed (rc={rc})")
    if rc != 0:
        print("[setup] STDERR tail:", err[-800:])

    # Verify the two version-sensitive pieces that have silently regressed before.
    checks = {
        "transformers": "import transformers; print(transformers.__version__)",
        "torchao": "import torchao; print(torchao.__version__)",
        "fla (flash-linear-attention)": "import fla; print(fla.__version__)",
    }
    for name, code in checks.items():
        rc, out, err = run(f'python3 -c "{code}"', timeout=30)
        status = out.strip() if rc == 0 else f"FAILED: {err.strip()[-150:]}"
        print(f"[setup] {name}: {status}")

    print("[setup] SETUP_COMPLETE")


if __name__ == "__main__":
    main()
