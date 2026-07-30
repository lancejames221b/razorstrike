#!/usr/bin/env python3
"""Phase 1h - Step 3: combine the v3 families with the new crypto_audit +
exploit_poc families into a v4 training mix.

Does NOT reuse combine_v3_dataset.py: that script's collision tripwire and
dedup key are both `row["messages"][1]["content"]` on the assumption the
user turn is x86-64 assembly (a deterministic index partition of one
shuffled asm stream, so cross-family collision is a hard-stop bug). The new
crypto_audit/exploit_poc families' user turns are C/C++/Python/... source
plus task text, not assembly - the "same partition" assumption doesn't hold,
so a cross-family key collision here is unremarkable (two different
builders drawing from CyberNative's DPO corpus and decompile-bench could
each independently land on similar-looking short code) and is reported as a
WARNING, not a hard stop.

Anti-forgetting mix: crypto_audit + exploit_poc must not be allowed to
dominate the v4 mix and degrade the RE-analysis capability HAWQ-SEC-RE
exists for. With N = total row count across the pre-existing (v3) families,
crypto_audit + exploit_poc combined are capped at min(4000, 0.25*N); if the
cap binds, both new families are downsampled proportionally (seed=42) so
neither is preferentially kept over the other.

Usage:
    FAMILY_DIRS="re_analysis=/Volumes/SeXternal/hawq_v3/re_analysis_dataset,\
decompile=/Volumes/SeXternal/hawq_v3/decompile_dataset,\
crypto_audit=/Volumes/SeXternal/hawq_v4/sec_audit_dataset,\
exploit_poc=/Volumes/SeXternal/hawq_v4/sec_audit_dataset" \
    OUT=/Volumes/SeXternal/hawq_v4/dataset \
        python3 scripts/combine_v4_dataset.py
"""
import os
import random
from collections import Counter

from datasets import DatasetDict, Dataset, load_from_disk

SEED = 42
NEW_FAMILIES = ("crypto_audit", "exploit_poc")
NEW_FAMILY_CAP_ABS = 4000
NEW_FAMILY_CAP_FRAC = 0.25


def _parse_family_dirs(spec):
    """'family=path,family=path,...' -> [(family, path), ...]. Multiple
    entries MAY point at the same path (e.g. crypto_audit and exploit_poc
    both live in one sec_audit_dataset dir, distinguished by each row's
    own `family` field, not by directory)."""
    pairs = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"malformed FAMILY_DIRS entry (want family=path): {item!r}")
        family, path = item.split("=", 1)
        pairs.append((family.strip(), path.strip()))
    if not pairs:
        raise ValueError("FAMILY_DIRS is empty - nothing to combine")
    return pairs


def combine(family_dirs):
    """family_dirs: [(family_name, path), ...]. Returns (DatasetDict, stats)."""
    rows_by_family = {}
    loaded_cache = {}  # path -> concatenated row list, avoid reloading the same dir twice
    for family, path in family_dirs:
        if path not in loaded_cache:
            ds = load_from_disk(path)
            all_rows = []
            for split in ds.values():
                all_rows.extend(split.to_list())
            loaded_cache[path] = all_rows
            print(f"[load] {path}: {len(all_rows)} rows "
                  f"(families present: {dict(Counter(r['family'] for r in all_rows))})")
        family_rows = [r for r in loaded_cache[path] if r["family"] == family]
        if not family_rows:
            raise ValueError(f"family={family!r} path={path!r}: 0 rows with that "
                              f"family tag - check FAMILY_DIRS")
        rows_by_family.setdefault(family, []).extend(family_rows)

    print(f"[combine] pre-dedup family counts: "
          f"{ {f: len(r) for f, r in rows_by_family.items()} }")

    # Within-family dedup (keep first occurrence), same rationale as
    # combine_v3_dataset.py: corpus-level duplicate content shouldn't be
    # oversampled as if it were distinct training signal.
    deduped_by_family = {}
    for family, rows in rows_by_family.items():
        seen, kept = set(), []
        for row in rows:
            key = row["messages"][1]["content"]
            if key in seen:
                continue
            seen.add(key)
            kept.append(row)
        n_dropped = len(rows) - len(kept)
        if n_dropped:
            print(f"[dedup] {family}: dropped {n_dropped} within-family "
                  f"duplicate-user-turn rows -> {len(kept)}")
        deduped_by_family[family] = kept

    # Cross-family overlap: reported, not enforced. See module docstring for
    # why this must NOT be combine_v3_dataset.py's hard-stop tripwire here.
    families = list(deduped_by_family)
    for i, fam_a in enumerate(families):
        keys_a = {row["messages"][1]["content"] for row in deduped_by_family[fam_a]}
        for fam_b in families[i + 1:]:
            keys_b = {row["messages"][1]["content"] for row in deduped_by_family[fam_b]}
            overlap = keys_a & keys_b
            if overlap:
                print(f"[warn] cross-family overlap: {fam_a} vs {fam_b}: "
                      f"{len(overlap)} shared user-turn(s) (not a hard stop - "
                      f"these families' user turns are not a partitioned "
                      f"single stream, unlike v3's asm-only families)")

    # Anti-forgetting cap on the NEW families combined.
    existing_total = sum(len(rows) for fam, rows in deduped_by_family.items()
                          if fam not in NEW_FAMILIES)
    new_total = sum(len(rows) for fam, rows in deduped_by_family.items()
                     if fam in NEW_FAMILIES)
    cap = min(NEW_FAMILY_CAP_ABS, int(NEW_FAMILY_CAP_FRAC * existing_total))
    print(f"[anti-forgetting] existing (v3) rows N={existing_total}, "
          f"new (crypto_audit+exploit_poc) rows={new_total}, "
          f"cap=min({NEW_FAMILY_CAP_ABS}, {NEW_FAMILY_CAP_FRAC}*N)={cap}")
    if new_total > cap and new_total > 0:
        ratio = cap / new_total
        rng = random.Random(SEED)
        for fam in NEW_FAMILIES:
            if fam not in deduped_by_family:
                continue
            rows = deduped_by_family[fam][:]
            rng.shuffle(rows)
            keep_n = int(len(rows) * ratio)
            print(f"[anti-forgetting] downsampling {fam}: {len(rows)} -> {keep_n} "
                  f"(ratio={ratio:.3f})")
            deduped_by_family[fam] = rows[:keep_n]

    all_rows = []
    for rows in deduped_by_family.values():
        all_rows.extend(rows)
    random.seed(SEED)
    random.shuffle(all_rows)
    n_val = max(1, int(len(all_rows) * 0.02))
    ds = DatasetDict({
        "train": Dataset.from_list(all_rows[n_val:]),
        "validation": Dataset.from_list(all_rows[:n_val]),
    })
    fam_counts = dict(Counter(r["family"] for r in all_rows))
    return ds, fam_counts


if __name__ == "__main__":
    family_dirs_spec = os.environ.get("FAMILY_DIRS")
    if not family_dirs_spec:
        raise SystemExit(
            "FAMILY_DIRS is required, e.g. "
            "FAMILY_DIRS=\"re_analysis=/path/a,decompile=/path/b,"
            "crypto_audit=/path/c,exploit_poc=/path/c\" "
            "python3 scripts/combine_v4_dataset.py")
    out = os.environ.get("OUT", "/Volumes/SeXternal/hawq_v4/dataset")

    pairs = _parse_family_dirs(family_dirs_spec)
    ds, fam_counts = combine(pairs)

    print(f"[combine] final family counts: {fam_counts}")
    print(f"[combine] train={len(ds['train'])} validation={len(ds['validation'])} "
          f"total={len(ds['train']) + len(ds['validation'])}")

    ds.save_to_disk(out)
    print(f"[save] combined v4 dataset -> {out}")
