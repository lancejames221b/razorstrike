#!/usr/bin/env python3
"""Phase 1a — Build RE foundation: format decompile-bench.

Download LLM4Binary/decompile-bench (2,233,092 rows), filter by asm/code length,
take first 40k kept rows, format as {source, family, messages}.

Output: datasets.DatasetDict with train (40k rows) + validation (2% = 816 rows).
Pushed to HF as lancejames221b/razorstrike-offsec-v1 (family="decompile").
"""

import os, random
from datasets import load_dataset, DatasetDict, Dataset


# Constants
MAX_ASM = 6000       # x86-64 assembly length (chars)
MAX_CODE = 4000      # C/C++ source length (chars)
TARGET_ROWS = 40_000  # target number of rows to keep
SEED = 42

# System prompt for RE task
RE_SYSTEM = (
    "You are an elite reverse engineer. You read x86-64 disassembly the way other people read prose. "
    "Given assembly extracted from a compiled binary, reconstruct faithful, compilable C/C++ source. "
    "Preserve control flow, types, and semantics."
)


def format_row(row):
    """Format a single decompile-bench row into the messages schema."""
    asm = row["asm"]
    code = row["code"]

    # Skip rows that are too long (would exceed 4096 tokens when rendered)
    if len(asm) > MAX_ASM or len(code) > MAX_CODE:
        return None

    # Format as single-turn conversation (system + user + assistant)
    messages = [
        {"role": "system", "content": RE_SYSTEM},
        {
            "role": "user",
            "content": f"Reconstruct the original C/C++ source for this x86-64 function.\n\n```asm\n{asm}\n```",
        },
        {
            "role": "assistant",
            "content": f"```cpp\n{code}\n```",
        },
    ]

    return {
        "source": "LLM4Binary/decompile-bench",
        "family": "decompile",
        "messages": messages,
    }


def build_re_dataset(cap=0):
    """Build the RE foundation dataset from decompile-bench.

    Returns a DatasetDict with train (40k rows) and validation (2% = 816 rows).
    """
    print("Downloading decompile-bench...")

    # Download and load the dataset (uses save_to_disk, so we use load_from_disk)
    local_dir = "/Volumes/Scratch/ml-workspace/decompile-bench"

    # Download the dataset (this creates the local directory with arrow files)
    ds_raw = load_dataset(
        "LLM4Binary/decompile-bench",
        split=(f"train[:{cap*5}]" if cap else "train"),
        cache_dir=local_dir,
    )

    print(f"Raw decompile-bench: {len(ds_raw)} rows")
    print(f"Columns: {ds_raw.column_names}")

    # Format each row and filter by length
    formatted_rows = []
    skipped = 0

    for i, row in enumerate(ds_raw):
        formatted = format_row(row)
        if formatted is not None:
            formatted_rows.append(formatted)
        else:
            skipped += 1

    print(f"Formatted rows (after length filter): {len(formatted_rows)}")
    print(f"Skipped (too long): {skipped}")

    # Shuffle and take first 40k rows
    random.seed(SEED)
    random.shuffle(formatted_rows)

    # Take first 40k rows (or all if fewer than 40k)
    kept_rows = formatted_rows[:(cap or TARGET_ROWS)]

    print(f"Kept rows: {len(kept_rows)}")

    # Split into train (98%) and validation (2%)
    n_val = max(1, int(len(kept_rows) * 0.02))
    n_train = len(kept_rows) - n_val

    train_data = kept_rows[:n_train]
    val_data = kept_rows[n_train:]

    # Build DatasetDict (wrap with Dataset.from_list to ensure proper types)
    ds = DatasetDict({
        "train": Dataset.from_list(train_data),
        "validation": Dataset.from_list(val_data),
    })

    print(f"Train: {len(train_data)} rows")
    print(f"Validation: {len(val_data)} rows")

    # Verify all rows have valid messages
    for split_name, split in ds.items():
        for i, row in enumerate(split):
            if "messages" not in row:
                raise ValueError(f"{split_name}[{i}] missing 'messages'")
            if len(row["messages"]) != 3:
                raise ValueError(
                    f"{split_name}[{i}] messages has {len(row['messages'])} turns, expected 3"
                )
            if row["messages"][-1]["role"] != "assistant":
                raise ValueError(
                    f"{split_name}[{i}] last message role is {row['messages'][-1]['role']}, expected 'assistant'"
                )

    return ds


if __name__ == "__main__":
    # Build the RE dataset
    ds = build_re_dataset()

    # Push to HF (if DATA_REPO is set)
    data_repo = os.environ.get("DATA_REPO", "lancejames221b/razorstrike-offsec-v1")
    print(f"Pushing to HF: {data_repo}")

    ds.push_to_hub(data_repo, private=True)
    print("RE dataset pushed to HF")
