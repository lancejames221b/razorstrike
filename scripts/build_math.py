#!/usr/bin/env python3
"""Phase 1f - Build math: foundation from NuminaMath-CoT.

Load AI-MO/NuminaMath-CoT (859,494 train rows), shuffle seed=42,
prepend a system turn, reuse the row's messages [user, assistant] verbatim.

Output: datasets.DatasetDict with train (up to 40,000 rows).
"""

import os
import random
from datasets import Dataset, DatasetDict


def build_math_dataset(cap=0):
    """Build the math dataset from NuminaMath-CoT.

    Returns a DatasetDict with a "train" split; every row has
    messages[] (system/user/assistant) and family="math".

    Args:
        cap: If > 0, limit to `cap` rows for smoke-testing.
    """
    from datasets import load_dataset

    if cap > 0:
        ds = load_dataset("AI-MO/NuminaMath-CoT", split=f"train[:{cap}]")
    else:
        ds = load_dataset("AI-MO/NuminaMath-CoT", split="train")
    random.seed(42)
    ds = ds.shuffle(seed=42)

    system_prompt = (
        "You are a rigorous mathematician and quantitative reasoner. "
        "Work step by step, then give the final answer in \\boxed{}."
    )

    rows = []
    for row in ds:
        messages = row["messages"]  # [user, assistant]
        if not messages or len(messages) < 2:
            continue
        # Prepend system turn → [system, user, assistant]
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        if not full_messages[-1]["content"].strip():
            continue
        rows.append({
            "source": "AI-MO/NuminaMath-CoT",
            "family": "math",
            "messages": full_messages,
        })

    # Keep up to 40,000 rows (oversample the source as needed)
    if len(rows) > 40_000:
        rows = rows[:40_000]

    ds_out = DatasetDict({"train": Dataset.from_list(rows)})
    print(f"Built math dataset: {len(rows)} rows")
    return ds_out


if __name__ == "__main__":
    # Allow a tiny cap for smoke-testing (e.g. CAP=10)
    cap = int(os.environ.get("CAP", "0"))
    ds = build_math_dataset(cap=cap)
    print(f"Train: {len(ds['train'])} rows")
    print("math dataset ready (family=\"math\")")
