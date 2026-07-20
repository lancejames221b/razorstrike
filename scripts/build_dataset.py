#!/usr/bin/env python3
"""Phase 1e - Build ALL families and combine into the RazorStrike v2 SFT set.

RazorStrike v2 = clean Qwen/Qwen3.6-35B-A3B base + this multi-domain LoRA.
Uncensoring is done via the `uncensor` data family (NOT weight abliteration).

Families combined:
  decompile (re), crypto_id, ransomware-crypto, math, cyber,
  loop_recovery, mythos, uncensor

Output: DatasetDict {train, validation}, each row = {source, family, family_value, messages}.
Pushed to HF at env DATA_REPO (default lancejames221b/razorstrike-v2-sft) when HF_TOKEN is set.
"""

import os, sys, json, random

# --- make `scripts.*` importable whether run as module or script ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from datasets import Dataset, DatasetDict  # noqa: E402

SHUFFLE_SEED = 942
VAL_FRACTION = 0.02
DATA_REPO = os.environ.get("DATA_REPO", "lancejames221b/razorstrike-v2-sft")

# stable numeric id per family (for stratified inspection / loss weighting)
FAMILY_VALUES = {
    "decompile": 0,
    "crypto_id": 1,
    "ransomware-crypto": 2,
    "math": 3,
    "cyber": 4,
    "loop_recovery": 5,
    "mythos": 6,
    "uncensor": 7,
}


def _load_builders():
    """Return list of (label, callable, supports_cap)."""
    from scripts.build_re import build_re_dataset
    from scripts.build_crypto_id import build_crypto_id_dataset
    from scripts.build_ransomware_crypto import build_ransomware_crypto_dataset
    from scripts.build_math import build_math_dataset
    from scripts.build_cyber import build_cyber_dataset
    from scripts.build_loop_recovery import build_loop_recovery_dataset
    from scripts.build_mythos import build_mythos_dataset
    from scripts.build_uncensor import build_uncensor_dataset
    return [
        ("decompile", build_re_dataset, True),
        ("crypto_id", build_crypto_id_dataset, False),
        ("ransomware-crypto", build_ransomware_crypto_dataset, True),
        ("math", build_math_dataset, True),
        ("cyber", build_cyber_dataset, True),
        ("loop_recovery", build_loop_recovery_dataset, True),
        ("mythos", build_mythos_dataset, True),
        ("uncensor", build_uncensor_dataset, True),
    ]


# Rebalance: cap the giant families, upsample the two families that define v2
# (anti-spiral + uncensor) so they get real gradient signal at 2 epochs.
CAPS = {"decompile": int(os.environ.get("CAP_DECOMPILE", "25000")),
        "math": int(os.environ.get("CAP_MATH", "22000"))}
UPSAMPLE = {"loop_recovery": int(os.environ.get("UP_LOOP", "3")),
            "uncensor": int(os.environ.get("UP_UNCENSOR", "3"))}


def build_combined(cap=0, push=False):
    """Build every family, tag family_value, shuffle, split, optionally push."""
    all_rows, counts, failures = [], {}, {}
    for label, fn, supports_cap in _load_builders():
        try:
            ds = fn(cap=cap) if (supports_cap and cap) else fn()
            rows = list(ds["train"])
            if label in CAPS and len(rows) > CAPS[label]:
                random.seed(SHUFFLE_SEED)
                rows = random.sample(rows, CAPS[label])
            for r in rows:
                r.setdefault("family", label)
                r["family_value"] = FAMILY_VALUES.get(r.get("family", label),
                                                       FAMILY_VALUES.get(label, -1))
            all_rows.extend(rows)
            counts[label] = len(rows)
            print(f"[ok] {label}: {len(rows)} rows")
        except Exception as e:
            failures[label] = f"{type(e).__name__}: {e}"
            print(f"[FAIL] {label}: {type(e).__name__}: {e}")

    if failures:
        raise RuntimeError(f"family builders failed: {failures}")

    random.seed(SHUFFLE_SEED)
    random.shuffle(all_rows)  # unique rows only at this point (no dup copies yet)

    n_val = max(1, int(len(all_rows) * VAL_FRACTION))
    val_rows, train_rows = all_rows[:n_val], all_rows[n_val:]

    # Upsample the v2-critical families in TRAIN ONLY - never in val (no leakage).
    extra = []
    for r in train_rows:
        f = UPSAMPLE.get(r.get("family"), 1)
        if f > 1:
            extra.extend([r] * (f - 1))
    if extra:
        train_rows = train_rows + extra
        random.seed(SHUFFLE_SEED + 1)
        random.shuffle(train_rows)

    ds_out = DatasetDict({
        "train": Dataset.from_list(train_rows),
        "validation": Dataset.from_list(val_rows),
    })
    print(f"\nCombined: {len(train_rows)} train + {len(val_rows)} val "
          f"({len(all_rows)} total)")
    print("per-family:", json.dumps(counts))

    if push:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("push requested but HF_TOKEN is not set")
        print(f"Pushing to HF: {DATA_REPO}")
        ds_out.push_to_hub(DATA_REPO, private=True, token=token)
        print("Combined dataset pushed to HF")

    return ds_out


if __name__ == "__main__":
    cap = int(os.environ.get("CAP", "0"))
    push = os.environ.get("PUSH", "0") == "1"
    ds = build_combined(cap=cap, push=push)
    print(f"\nTrain: {len(ds['train'])} | Validation: {len(ds['validation'])}")
    print("RazorStrike v2 SFT dataset ready (all families combined)")
