#!/usr/bin/env python3
"""Pre-flight MAXLEN survival check for the v4 SFT dataset.

Mirrors scripts/train_lora.py's exact tokenization/filtering rule:
  prompt = apply_chat_template(messages[:-1], add_generation_prompt=True)
  full   = apply_chat_template(messages, add_generation_prompt=False)
  keep iff len(full) <= MAXLEN and len(prompt) < len(full)

Gate: >=90% of the rows from the new crypto_audit + exploit_poc families must
survive. If they don't, abort/adjust before paying for GCE training.
"""
import argparse
from collections import Counter, defaultdict

from datasets import load_from_disk
from transformers import AutoTokenizer

NEW_FAMILIES = {"crypto_audit", "exploit_poc"}


def iter_rows(ds):
    for split_name, split in ds.items():
        for row in split:
            yield split_name, row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="/Volumes/SeXternal/hawq_v4/dataset")
    ap.add_argument("--base", default="Qwen/Qwen3.6-35B-A3B")
    ap.add_argument("--maxlen", type=int, default=3072)
    ap.add_argument("--gate", type=float, default=0.90)
    ap.add_argument("--families", default=",".join(sorted(NEW_FAMILIES)),
                    help="comma-separated families to include in the gate")
    args = ap.parse_args()

    families = {f.strip() for f in args.families.split(",") if f.strip()}
    ds = load_from_disk(args.dataset)
    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    total = 0
    kept = 0
    counts = Counter()
    kept_counts = Counter()
    max_full = 0
    longest = defaultdict(list)  # family -> [(len, split, idx-ish text)]

    for split_name, row in iter_rows(ds):
        fam = row["family"]
        if fam not in families:
            continue
        total += 1
        counts[fam] += 1
        msgs = row["messages"]
        prompt = tok.apply_chat_template(msgs[:-1], add_generation_prompt=True,
                                         tokenize=True)["input_ids"]
        full = tok.apply_chat_template(msgs, add_generation_prompt=False,
                                       tokenize=True)["input_ids"]
        full_len = len(full)
        max_full = max(max_full, full_len)
        ok = full_len <= args.maxlen and len(prompt) < full_len
        if ok:
            kept += 1
            kept_counts[fam] += 1
        item = (full_len, split_name, (msgs[1]["content"][:120].replace("\n", " ")))
        longest[fam].append(item)

    if total == 0:
        raise SystemExit(f"No rows found for families={sorted(families)} in {args.dataset}")

    rate = kept / total
    print(f"[preflight] dataset={args.dataset}")
    print(f"[preflight] base={args.base} maxlen={args.maxlen} gate={args.gate:.1%}")
    print(f"[preflight] families={sorted(families)}")
    print(f"[preflight] survive={kept}/{total} = {rate:.2%} max_full_len={max_full}")
    for fam in sorted(counts):
        fam_rate = kept_counts[fam] / counts[fam]
        print(f"[preflight] {fam}: {kept_counts[fam]}/{counts[fam]} = {fam_rate:.2%}")
        for length, split_name, text in sorted(longest[fam], reverse=True)[:5]:
            print(f"  longest {fam}: len={length} split={split_name} text={text!r}")
    if rate < args.gate:
        print("[preflight] FAIL - new-family survival below gate; adjust MAXLEN or builders before training")
        raise SystemExit(1)
    print("[preflight] PASS")


if __name__ == "__main__":
    main()
