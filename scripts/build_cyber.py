#!/usr/bin/env python3
"""Phase 1g - Build cyber: offensive-security + investigations.

Blend two open datasets into messages[], family="cyber":
  - AlicanKiraz0/Cybersecurity-Dataset-Fenrir-v2.1 (target ~22k)
  - CyberNative/Code_Vulnerability_Security_DPO (target ~8k)

Output: datasets.DatasetDict with train (~30,000 rows).
"""

import os
import random
from datasets import Dataset, DatasetDict

TARGET_FENRIR = 22_000
TARGET_DPO = 8_000
DPO_SYSTEM = (
    "You are an offensive security engineer and code auditor. "
    "Identify the vulnerability, explain how it is exploited, "
    "and give secure, correct code."
)


def build_cyber_dataset(cap=0):
    """Build the cyber dataset by blending two sources.

    Args:
        cap: if > 0, limit EACH source (and the result) to `cap` rows for smoke-testing.
    """
    from datasets import load_dataset

    fenrir_split = f"train[:{cap}]" if cap else "train"
    dpo_split = f"train[:{cap}]" if cap else "train"

    # Source 1: Fenrir - columns are plain strings (system, user, assistant)
    ds_fenrir = load_dataset("AlicanKiraz0/Cybersecurity-Dataset-Fenrir-v2.1",
                             split=fenrir_split).shuffle(seed=42)
    rows_fenrir = []
    for row in ds_fenrir:
        asst = row.get("assistant") or ""
        if not asst.strip():
            continue
        rows_fenrir.append({
            "source": "AlicanKiraz0/Fenrir-v2.1",
            "family": "cyber",
            "messages": [
                {"role": "system", "content": row.get("system") or ""},
                {"role": "user", "content": row.get("user") or ""},
                {"role": "assistant", "content": asst},
            ],
        })

    # Source 2: Code_Vulnerability_Security_DPO - question -> user, chosen -> assistant
    ds_dpo = load_dataset("CyberNative/Code_Vulnerability_Security_DPO",
                          split=dpo_split).shuffle(seed=42)
    rows_dpo = []
    for row in ds_dpo:
        u = (row.get("question") or "").strip()
        a = (row.get("chosen") or "").strip()
        if not u or not a:
            continue
        rows_dpo.append({
            "source": "CyberNative/Code_Vulnerability_Security_DPO",
            "family": "cyber",
            "messages": [
                {"role": "system", "content": DPO_SYSTEM},
                {"role": "user", "content": u},
                {"role": "assistant", "content": a},
            ],
        })

    if not cap:
        rows_fenrir = rows_fenrir[:TARGET_FENRIR]
        rows_dpo = rows_dpo[:TARGET_DPO]

    all_rows = rows_fenrir + rows_dpo
    random.seed(42)
    random.shuffle(all_rows)

    ds_out = DatasetDict({"train": Dataset.from_list(all_rows)})
    print(f"Built cyber dataset: {len(all_rows)} rows (Fenrir={len(rows_fenrir)}, DPO={len(rows_dpo)})")
    return ds_out


if __name__ == "__main__":
    cap = int(os.environ.get("CAP", "0"))
    ds = build_cyber_dataset(cap=cap)
    print(f"Train: {len(ds['train'])} rows")
    print("cyber dataset ready (family=\"cyber\")")
