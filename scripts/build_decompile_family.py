#!/usr/bin/env python3
"""Phase 1g - Build the decompilation SFT family: teach HAWQ to reconstruct C
source from x86-64 assembly, using decompile-bench's own `code` column as
FREE gold (no frontier calls). This is the second task family for v3, chosen
specifically because it is also the only task with a public, externally
comparable benchmark (Decompile-Bench-Eval's HumanEval-Decompile /
MBPP-Decompile re-executability splits - see eval_decompile_bench.py).

Shares the exact same corpus sampler as build_re_analysis.py
(`_load_kept_rows`, seeded SEED=42) so the two families are a deterministic,
disjoint index partition of one shuffled decompile-bench permutation - not a
handoff, not an exclusion set. Pass SKIP equal to the RE family's
`N_TASKS + N_EVAL` so this family starts exactly where that one stopped.

Usage:
    N_DECOMPILE=12000 SKIP=12100 LOCAL_OUT=/tmp/hawq_decompile PUSH=0 \
        python3 scripts/build_decompile_family.py
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_re_analysis import _load_kept_rows, MAX_ASM, MAX_CODE, SEED  # noqa: E402
from datasets import DatasetDict, Dataset  # noqa: E402

N_DECOMPILE = int(os.environ.get("N_DECOMPILE", "12000"))
# Default matches the RE family's N_TASKS(12000) + N_EVAL(100) so the two
# builders partition the shuffled stream with zero overlap by construction.
SKIP = int(os.environ.get("SKIP", "12100"))

DECOMPILE_SYSTEM = (
    "You are an expert decompiler. Given x86-64 assembly from a compiled binary, "
    "reconstruct the original C source function. Output only the C code in a single "
    "```c fenced block, with no commentary."
)


def format_decompile_row(asm, code):
    messages = [
        {"role": "system", "content": DECOMPILE_SYSTEM},
        {"role": "user",
         "content": f"# This is the assembly code:\n\n```asm\n{asm}\n```\n\n# What is the source code?"},
        {"role": "assistant", "content": f"```c\n{code}\n```"},
    ]
    return {"source": "LLM4Binary/decompile-bench", "family": "decompile", "messages": messages}


def build_decompile_dataset(n_decompile=N_DECOMPILE, skip=SKIP):
    print(f"[decompile] SEED={SEED} skip={skip} need={n_decompile} (free gold, no frontier calls)")
    kept = _load_kept_rows(n_decompile, skip=skip)
    print(f"[decompile] {len(kept)} rows sampled (skip={skip} guarantees disjointness from the "
          f"RE-analysis family drawing the same SEED={SEED} stream from 0)")

    rows = [format_decompile_row(r["asm"], r["code"]) for r in kept]

    # Guards, mirroring build_re_analysis.py's shape but WITHOUT its
    # _SRC_REF prose-leak assert: the gold here IS source code, so that
    # assert would fire on every row by design.
    for i, row in enumerate(rows):
        msgs = row["messages"]
        if len(msgs) != 3 or msgs[-1]["role"] != "assistant":
            raise ValueError(f"row[{i}] malformed messages")
        assistant_content = msgs[-1]["content"]
        if "```c\n" not in assistant_content or not assistant_content.split("```c\n", 1)[1].strip():
            raise ValueError(f"row[{i}] decompile gold missing c fence")
        if "```c" in msgs[1]["content"]:
            raise ValueError(f"row[{i}] user turn leaks source")

    import random
    random.seed(SEED)
    random.shuffle(rows)
    n_val = max(1, int(len(rows) * 0.02))
    ds = DatasetDict({
        "train": Dataset.from_list(rows[n_val:]),
        "validation": Dataset.from_list(rows[:n_val]),
    })
    return ds


if __name__ == "__main__":
    ds = build_decompile_dataset()
    print(f"Train: {len(ds['train'])} | Validation: {len(ds['validation'])}")

    for i, row in enumerate(ds["train"].select(range(min(3, len(ds["train"]))))):
        print(f"\n--- sample {i} ---")
        print("user:", row["messages"][1]["content"][:200], "...")
        print("assistant:", row["messages"][2]["content"][:300], "...")

    local_out = os.environ.get("LOCAL_OUT", "/tmp/hawq_decompile/hawq-decompile-dataset")
    ds.save_to_disk(local_out)
    print(f"[save] dataset saved locally -> {local_out} (durability checkpoint before push)")

    if os.environ.get("PUSH", "0") == "1":
        data_repo = os.environ.get("DATA_REPO", "lancejames221b/hawq-decompile")
        print(f"Pushing to HF: {data_repo}")
        try:
            ds.push_to_hub(data_repo, private=True)
            print("Decompile dataset pushed to HF")
        except Exception as e:
            print(f"[push] FAILED: {type(e).__name__}: {e}. "
                  f"Dataset is safe on local disk at {local_out} - retry with: "
                  f"python3 -c \"from datasets import load_from_disk; "
                  f"load_from_disk('{local_out}').push_to_hub('{data_repo}', private=True)\"")
            raise
