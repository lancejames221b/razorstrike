#!/usr/bin/env python3
"""Phase 1 - Build mythos: fable / mythos / creative-narrative family.

Blends two VERIFIED apache/open sources into messages[], family="mythos":
  - Gryphe/Opus-WritingPrompts (conversations, Opus-authored) -> primary, high quality
  - euclaise/writingprompts (prompt/story) -> volume

Preserves and strengthens long-form narrative / storytelling capability.

Output: datasets.DatasetDict with train (~8,000 rows), family="mythos".
"""

import os
import random
from datasets import Dataset, DatasetDict

SEED = 42
TARGET = 8_000
TARGET_OPUS = 6_000      # take all of Gryphe (~6k)
TARGET_EUCLAISE = 4_000  # top up from euclaise

SYSTEM = (
    "You are a masterful storyteller steeped in myth, fable, and legend. Write vivid, "
    "coherent narrative prose with a clear arc and a resonant moral or turn."
)


def _norm_conversations(conv):
    """Map a ShareGPT-style conversations list to [ {role,content}, ... ]."""
    out = []
    for turn in conv:
        if not isinstance(turn, dict):
            continue
        who = turn.get("from") or turn.get("role") or ""
        val = turn.get("value") or turn.get("content") or ""
        if not val or not str(val).strip():
            continue
        role = {"human": "user", "gpt": "assistant", "system": "system",
                "user": "user", "assistant": "assistant"}.get(who, "user")
        out.append({"role": role, "content": str(val)})
    return out


def build_mythos_dataset(cap=0):
    """Build the mythos dataset from verified narrative sources."""
    from datasets import load_dataset
    random.seed(SEED)
    rows = []

    # Source 1: Gryphe/Opus-WritingPrompts (conversations)
    d1 = load_dataset("Gryphe/Opus-WritingPrompts", split="train").shuffle(seed=SEED)
    for row in d1:
        msgs = _norm_conversations(row.get("conversations") or [])
        if len(msgs) < 2 or msgs[-1]["role"] != "assistant":
            continue
        if msgs[0]["role"] != "system":
            msgs = [{"role": "system", "content": SYSTEM}] + msgs
        rows.append({"source": "Gryphe/Opus-WritingPrompts", "family": "mythos", "messages": msgs})
        if cap and len(rows) >= cap:
            break

    # Source 2: euclaise/writingprompts (prompt -> story)
    if not cap or len(rows) < (cap or TARGET):
        d2 = load_dataset("euclaise/writingprompts", split="train").shuffle(seed=SEED)
        n_opus = len(rows)
        for row in d2:
            prompt = (row.get("prompt") or "").strip()
            story = (row.get("story") or "").strip()
            if len(prompt) < 10 or len(story) < 200:
                continue
            rows.append({
                "source": "euclaise/writingprompts", "family": "mythos",
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": story},
                ],
            })
            if len(rows) - n_opus >= TARGET_EUCLAISE:
                break
            if cap and len(rows) >= cap:
                break

    if not cap and len(rows) > TARGET:
        random.shuffle(rows)
        rows = rows[:TARGET]

    ds_out = DatasetDict({"train": Dataset.from_list(rows)})
    print(f"Built mythos dataset: {len(rows)} rows")
    return ds_out


if __name__ == "__main__":
    cap = int(os.environ.get("CAP", "0"))
    ds = build_mythos_dataset(cap=cap)
    print(f"Train: {len(ds['train'])} rows")
    print("mythos dataset ready (family=\"mythos\")")
