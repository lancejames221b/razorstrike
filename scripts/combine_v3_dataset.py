#!/usr/bin/env python3
"""Phase 1h - Step 3a: concatenate the re_analysis + decompile families into
one v3 DatasetDict (train/validation), then stage it for training (GCS by
default; HF public-repo fallback if GCS write access is unavailable - see
Step 0/3 of the v3 scale plan).

Usage:
    RE_DS=/Volumes/SeXternal/hawq_v3/re_analysis_dataset \
    DECOMPILE_DS=/Volumes/SeXternal/hawq_v3/decompile_dataset \
    OUT=/Volumes/SeXternal/hawq_v3/dataset \
        python3 scripts/combine_v3_dataset.py
"""
import os
import random

from datasets import DatasetDict, Dataset, concatenate_datasets, load_from_disk

SEED = 42


def combine(re_ds_path, decompile_ds_path):
    re_ds = load_from_disk(re_ds_path)
    dec_ds = load_from_disk(decompile_ds_path)

    re_rows = []
    for split in re_ds.values():
        re_rows.extend(split.to_list())
    dec_rows = []
    for split in dec_ds.values():
        dec_rows.extend(split.to_list())

    print(f"[combine] {len(re_rows)} re_analysis + {len(dec_rows)} decompile = "
          f"{len(re_rows) + len(dec_rows)} total rows (pre-dedup, pre-split)")

    # Real tripwire: the two families are a deterministic index partition of
    # one shuffled stream by construction (skip param) - CROSS-family
    # collision would mean that partition broke (seed/filter drift), and
    # that's a hard stop, not something to clean up.
    re_asms = {row["messages"][1]["content"] for row in re_rows}
    dec_asms = {row["messages"][1]["content"] for row in dec_rows}
    cross_collisions = re_asms & dec_asms
    if cross_collisions:
        raise ValueError(
            f"{len(cross_collisions)} asm collisions ACROSS families - the "
            f"RE-analysis and decompile builders' skip/need partition did "
            f"not hold. Do NOT train on this data.")
    print("[combine] cross-family disjointness OK (zero collisions between "
          "re_analysis and decompile asm sets)")

    # Benign, separate issue: decompile-bench itself has ~0.7% corpus-level
    # duplicate function bodies (common utility functions repeated across
    # projects), confirmed empirically - this shows up as a handful of
    # WITHIN-family duplicates, not a partition bug. Drop them (keep first
    # occurrence) so the final set has no duplicate training rows to
    # oversample, without touching either builder's stream positions.
    all_rows = re_rows + dec_rows
    seen, deduped = set(), []
    for row in all_rows:
        key = row["messages"][1]["content"]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    n_dropped = len(all_rows) - len(deduped)
    print(f"[combine] dropped {n_dropped} within-family duplicate-asm rows "
          f"(kept first occurrence) -> {len(deduped)} rows")
    all_rows = deduped

    random.seed(SEED)
    random.shuffle(all_rows)
    n_val = max(1, int(len(all_rows) * 0.02))
    ds = DatasetDict({
        "train": Dataset.from_list(all_rows[n_val:]),
        "validation": Dataset.from_list(all_rows[:n_val]),
    })
    return ds


if __name__ == "__main__":
    re_ds_path = os.environ.get("RE_DS", "/Volumes/SeXternal/hawq_v3/re_analysis_dataset")
    decompile_ds_path = os.environ.get("DECOMPILE_DS", "/Volumes/SeXternal/hawq_v3/decompile_dataset")
    out = os.environ.get("OUT", "/Volumes/SeXternal/hawq_v3/dataset")

    ds = combine(re_ds_path, decompile_ds_path)

    from collections import Counter
    fam_counts = Counter(r["family"] for r in ds["train"]) + Counter(r["family"] for r in ds["validation"])
    print(f"[combine] family counts: {dict(fam_counts)}")
    print(f"[combine] train={len(ds['train'])} validation={len(ds['validation'])}")

    ds.save_to_disk(out)
    print(f"[save] combined v3 dataset -> {out}")
