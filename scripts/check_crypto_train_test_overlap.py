#!/usr/bin/env python3
"""Check crypto_audit training rows for near-duplicate leakage against
scripts/eval_crypto_audit.py's _CRYPTO_ID_CASES snippets.

This intentionally normalizes integer literals to <n> before comparison so the
check focuses on scaffold/code-shape overlap. Shared constants are expected and
are the primitive signal; copying the eval's function/array scaffold is not.
"""
import argparse
import difflib
import re
import sys

from datasets import load_from_disk

import eval_crypto_audit as ev

_INT_RE = re.compile(r"(?<![A-Za-z0-9_])(?:0[xX][0-9a-fA-F]+|\d+)[uUlL]*(?![A-Za-z0-9_])")


def norm(text):
    text = _INT_RE.sub(" <n> ", text.lower())
    text = re.sub(r"[^a-z0-9_<>]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def iter_rows(ds):
    for split_name, split in ds.items():
        for idx, row in enumerate(split):
            yield split_name, idx, row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="/Volumes/SeXternal/hawq_v4/sec_audit_dataset")
    ap.add_argument("--threshold", type=float, default=0.82)
    args = ap.parse_args()

    eval_norm = {name: norm(snippet) for name, (_aliases, snippet) in ev._CRYPTO_ID_CASES.items()}
    ds = load_from_disk(args.dataset)
    hits = []
    checked = 0
    for split_name, idx, row in iter_rows(ds):
        if row.get("family") != "crypto_audit":
            continue
        checked += 1
        user = row["messages"][1]["content"]
        n_user = norm(user)
        for case, n_eval in eval_norm.items():
            ratio = difflib.SequenceMatcher(None, n_user, n_eval).ratio()
            contains = n_eval in n_user
            if contains or ratio >= args.threshold:
                hits.append((split_name, idx, case, ratio, contains, user[:240].replace("\n", " ")))
    print(f"[overlap] checked crypto_audit rows={checked} threshold={args.threshold}")
    if hits:
        for hit in hits[:20]:
            print(f"[overlap] HIT split={hit[0]} idx={hit[1]} case={hit[2]} ratio={hit[3]:.3f} contains={hit[4]} text={hit[5]!r}")
        raise SystemExit(f"FAIL: {len(hits)} crypto_audit rows are near-duplicates of eval snippets")
    print("[overlap] PASS - no crypto_audit row is near-duplicate of eval snippets")


if __name__ == "__main__":
    main()
