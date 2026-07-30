#!/usr/bin/env python3
"""Step 4 (HAWQ v1.1 DPO plan) - Pre-flight MAXLEN survival check for the
combined DPO corpus (Step 1 omp-advisory pairs + Step 2 clean-control pairs).

DPO tokenizes prompt+chosen and prompt+rejected SEPARATELY, so both must fit
- unlike SFT's single sequence per row. omp session prompts are serialized
multi-turn conversations and are the long tail here (median ~1MB raw text
for the mined corpus).

This calls scripts/dpo_common.py's build_dpo_pair_features - the EXACT
function scripts/train_dpo.py uses to build training tensors (shared prompt
truncation computed once per pair, then applied to both chosen and
rejected) - so the survival rate measured here is the rate training will
actually see. Prompts truncate from the LEFT at whole-turn boundaries;
responses are NEVER truncated (a pair that still overflows MAXLEN after
prompt truncation is dropped outright - see dpo_common.py's docstrings for
why: response right-truncation would inject a length artifact into DPO's
log-prob-sum loss and would cut off exactly the concluding sentence Step 2's
clean-control pairs exist to teach).

"Survives" = build_dpo_pair_features returns a non-None result for BOTH the
chosen and rejected side. Per the plan: "If survival is below 90% ... drop
the pairs that overflow rather than raising MAXLEN" - so a low survival rate
is a real finding to report and act on (drop those pairs), not a signal to
loosen this script's definition of survival.

Usage:
    python3 scripts/preflight_dpo_maxlen.py \
        --dataset /tmp/hawq_dpo/combined_pairs.jsonl \
        --base Qwen/Qwen3.6-35B-A3B --maxlen 2048 --max-prompt-len 1024 --gate 0.90
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dpo_common import parse_prompt_to_messages, build_dpo_pair_features  # noqa: E402


def load_pairs(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="/tmp/hawq_dpo/combined_pairs.jsonl")
    ap.add_argument("--base", default="Qwen/Qwen3.6-35B-A3B")
    ap.add_argument("--maxlen", type=int, default=2048)
    ap.add_argument("--max-prompt-len", type=int, default=1024)
    ap.add_argument("--gate", type=float, default=0.90)
    ap.add_argument("--drop-overflow-out", default="",
                     help="if set, write the surviving-only pairs here")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    pairs = load_pairs(args.dataset)
    total = len(pairs)
    if total == 0:
        raise SystemExit(f"No rows found in {args.dataset}")

    both_survive = 0
    by_source = Counter()
    survive_by_source = Counter()
    max_full = 0
    survivors = []

    for i, row in enumerate(pairs):
        if (i + 1) % 200 == 0:
            print(f"[preflight-dpo] scanned {i + 1}/{total}")
        src = row.get("source", "agent")
        by_source[src] += 1
        messages = parse_prompt_to_messages(row["prompt"])

        chosen_feat, rejected_feat, _dropped = build_dpo_pair_features(
            messages, row["chosen"], row["rejected"], tok,
            args.maxlen, args.max_prompt_len)

        if chosen_feat is not None:
            max_full = max(max_full, len(chosen_feat["input_ids"]))
        if rejected_feat is not None:
            max_full = max(max_full, len(rejected_feat["input_ids"]))

        if chosen_feat is not None and rejected_feat is not None:
            both_survive += 1
            survive_by_source[src] += 1
            survivors.append(row)

    rate = both_survive / total
    print(f"[preflight-dpo] dataset={args.dataset} base={args.base}")
    print(f"[preflight-dpo] maxlen={args.maxlen} max_prompt_len={args.max_prompt_len} "
          f"gate={args.gate:.1%}")
    print(f"[preflight-dpo] survive={both_survive}/{total} = {rate:.2%} "
          f"max_full_len={max_full}")
    for src in sorted(by_source):
        s_rate = survive_by_source[src] / by_source[src]
        print(f"[preflight-dpo] source={src}: {survive_by_source[src]}/{by_source[src]} "
              f"= {s_rate:.2%}")

    if args.drop_overflow_out:
        with open(args.drop_overflow_out, "w") as f:
            for row in survivors:
                f.write(json.dumps(row) + "\n")
        print(f"[preflight-dpo] wrote {len(survivors)} surviving pairs (overflow "
              f"pairs dropped) -> {args.drop_overflow_out}")

    if rate < args.gate:
        print("[preflight-dpo] FAIL - survival below gate; per plan, drop the "
              "overflowing pairs (--drop-overflow-out) rather than raising MAXLEN")
        sys.exit(1)
    print("[preflight-dpo] PASS")


if __name__ == "__main__":
    main()
