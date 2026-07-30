#!/usr/bin/env python3
"""Step 3 (HAWQ v1.1 DPO plan) - Contamination gate (blocking).

The Step 1 omp-advisory pairs are mined from this project's own sessions,
which include eval runs against the deployed model. Their serialized
`prompt` fields can therefore contain the literal eval-probe text, which
would train on the test set.

For each of eval_crypto_audit.py's four probe code constants (_CRYPTO_ID_CASES
- 5 snippets, _MISUSE_CODE, _CLEAN_CODE, _EXPLOIT_CODE), extract a distinctive
normalized window (skipping the #include preamble, which is generic and would
false-positive on unrelated C code) and drop any DPO pair whose concatenated
prompt+chosen+rejected contains it. Reuses check_crypto_train_test_overlap.py's
`norm()` helper so evasion via re-formatting/re-casing doesn't slip through.

Usage:
    python3 scripts/check_dpo_contamination.py \
        --inputs /tmp/hawq_dpo/agent_pairs.jsonl /tmp/hawq_dpo/clean_pairs.jsonl \
        --out /tmp/hawq_dpo/combined_pairs.jsonl
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_crypto_train_test_overlap import norm  # noqa: E402
import eval_crypto_audit as ev  # noqa: E402

WINDOW_LEN = 200
PROMPT_TAIL_CHARS = 20000


def _strip_includes(code):
    lines = [l for l in code.splitlines() if not l.strip().startswith("#include")]
    return "\n".join(lines)


def _window(code, window_len=WINDOW_LEN):
    n = norm(_strip_includes(code))
    return n[:window_len]


def probe_windows():
    windows = {}
    for name, (_aliases, snippet) in ev._CRYPTO_ID_CASES.items():
        windows[f"crypto_id:{name}"] = _window(snippet)
    windows["misuse_enum"] = _window(ev._MISUSE_CODE)
    windows["clean_control"] = _window(ev._CLEAN_CODE)
    windows["exploit_path"] = _window(ev._EXPLOIT_CODE)
    # De-dupe/report if any window is degenerate (shorter than WINDOW_LEN
    # normalized chars would still work as a substring check, just less
    # distinctive) - print lengths for visibility, no hard requirement.
    for name, w in windows.items():
        if len(w) < WINDOW_LEN:
            print(f"[contam] note: {name} window is only {len(w)} normalized "
                  f"chars (< {WINDOW_LEN}); still used as-is")
    return windows


def load_pairs(paths):
    pairs = []
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row["_source_file"] = p
                pairs.append(row)
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-drop-frac", type=float, default=0.05)
    args = ap.parse_args()

    windows = probe_windows()
    pairs = load_pairs(args.inputs)
    print(f"[contam] loaded {len(pairs)} pairs from {len(args.inputs)} file(s)")

    kept = []
    dropped = []
    for i, row in enumerate(pairs):
        # Step 4/5 truncate the prompt to the last MAX_PROMPT_LEN=1024
        # tokens at MESSAGE-turn granularity before it ever reaches the
        # model, so text further back than that can never leak into
        # training regardless of what it contains. PROMPT_TAIL_CHARS is a
        # generous (~5x) upper bound on that surviving window, used only to
        # keep this O(n) normalize-and-scan pass tractable on multi-MB omp
        # session prompts (median ~1MB here) - chosen/rejected are the
        # actual model outputs and are always scanned in full.
        prompt_tail = (row.get("prompt") or "")[-PROMPT_TAIL_CHARS:]
        blob = norm(prompt_tail + " " +
                    (row.get("chosen") or "") + " " +
                    (row.get("rejected") or ""))
        if (i + 1) % 200 == 0:
            print(f"[contam] scanned {i + 1}/{len(pairs)}")
        hit = None
        for name, window in windows.items():
            if window and window in blob:
                hit = name
                break
        if hit:
            dropped.append((row, hit))
        else:
            kept.append(row)

    n_dropped, n_total = len(dropped), len(pairs)
    drop_frac = n_dropped / max(n_total, 1)
    print(f"[contam] dropped={n_dropped} kept={len(kept)}")
    for row, hit in dropped[:20]:
        src = row.get("_source_file", "?")
        snippet = (row.get("prompt") or "")[:120].replace("\n", " ")
        print(f"  DROP probe={hit} source={src} prompt[:120]={snippet!r}")

    for row in kept:
        row.pop("_source_file", None)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for row in kept:
            f.write(json.dumps(row) + "\n")
    print(f"[contam] wrote {len(kept)} surviving pairs -> {args.out}")

    if drop_frac > args.max_drop_frac:
        print(f"[contam] FAIL: drop fraction {drop_frac:.2%} exceeds gate "
              f"{args.max_drop_frac:.2%} - corpus is substantially eval-"
              f"derived; narrow the mining filter before training.")
        sys.exit(1)
    print(f"[contam] PASS: drop fraction {drop_frac:.2%} within gate "
          f"{args.max_drop_frac:.2%}")


if __name__ == "__main__":
    main()
