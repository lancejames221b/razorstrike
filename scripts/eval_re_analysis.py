#!/usr/bin/env python3
"""Phase 5 - Before/after judge eval for the RE-analysis LoRA (HAWQ-SEC-RE).

For each of the held-out `eval_re_tasks.jsonl` rows ({asm, code} from
build_re_analysis.py's disjoint eval split):
  1. Base HAWQ analysis: HTTP call to HAWQ served on generic (ASM-only prompt,
     the same shape as the stored training user turn).
  2. Tuned HAWQ analysis: same call shape against the merged HAWQ-SEC-RE model
     (over HTTP, matching deployment - the default), OR pass --tuned-url to
     point at any other OpenAI-compatible endpoint serving the tuned model.
  3. Judge: a FRONTIER model (same FRONTIER_* env/guards as build_re_analysis.py)
     receives the asm, the ground-truth C, and the two anonymized analyses
     (A/B order randomized per task) and returns which is the better faithful
     RE analysis (A, B, or tie) with a one-line reason.

Success gate: tuned win-rate (wins / (wins + losses), excluding ties) > 0.55.

Usage:
    BASE_URL=http://generic:1234/v1 BASE_MODEL=hawq \
    TUNED_URL=http://generic:1234/v1 TUNED_MODEL=hawq-sec-re \
    FRONTIER_BASE_URL=... FRONTIER_MODEL=... FRONTIER_API_KEY=... \
        python3 scripts/eval_re_analysis.py --tasks eval_re_tasks.jsonl
"""

import os
import re
import sys
import json
import random
import argparse
import urllib.request

RE_ANALYST_SYSTEM = (
    "You are an elite reverse engineer. Given x86-64 assembly from a compiled "
    "binary, produce a precise technical analysis: purpose, inputs/outputs, "
    "algorithm, and security-relevant behavior."
)

JUDGE_SYSTEM = (
    "You are an expert judge of reverse-engineering analyses. You are given "
    "x86-64 assembly, its known-correct C/C++ source (ground truth), and two "
    "candidate analyses (A and B) written from the assembly alone. Decide "
    "which analysis is more faithful, accurate, and complete relative to the "
    "ground truth. Respond with strict JSON only: "
    '{"winner": "A"|"B"|"tie", "reason": "<one line>"}.'
)

JUDGE_USER_TMPL = (
    "ASSEMBLY:\n```asm\n{asm}\n```\n"
    "KNOWN-CORRECT SOURCE (ground truth):\n```cpp\n{code}\n```\n"
    "ANALYSIS A:\n{a}\n\n"
    "ANALYSIS B:\n{b}\n\n"
    "Which analysis is better? Respond with the JSON object only."
)


def _post(base_url, model, messages, api_key=None, max_tokens=4000, temperature=0.3,
           reasoning_effort=None):
    url = f"{base_url.rstrip('/')}/chat/completions"
    body = {"model": model, "messages": messages, "temperature": temperature,
            "max_tokens": max_tokens}
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.load(r)
    msg = d["choices"][0]["message"]
    # Local thinking-mode models (Qwen-style <think> blocks, per
    # eval_re_v2_http_probes.py precedent) put the response in
    # reasoning_content separately from content, and can leave content
    # empty if the completion is cut off mid-thought (finish_reason=
    # "length") before reaching the final answer. Fall back to
    # reasoning_content so a truncated-but-substantive response isn't
    # silently treated as empty - applied identically for base and tuned
    # so the A/B stays fair.
    content = (msg.get("content") or "").strip()
    if not content:
        content = (msg.get("reasoning_content") or "").strip()
    return content


def get_analysis(base_url, model, asm, api_key=None):
    messages = [
        {"role": "system", "content": RE_ANALYST_SYSTEM},
        {"role": "user",
         "content": f"Analyze this x86-64 function and explain what it does.\n\n```asm\n{asm}\n```"},
    ]
    return _post(base_url, model, messages, api_key=api_key)


def _frontier_env():
    base_url = os.environ.get("FRONTIER_BASE_URL")
    model = os.environ.get("FRONTIER_MODEL")
    api_key = os.environ.get("FRONTIER_API_KEY")
    missing = [n for n, v in (("FRONTIER_BASE_URL", base_url),
                               ("FRONTIER_MODEL", model),
                               ("FRONTIER_API_KEY", api_key)) if not v]
    if missing:
        raise SystemExit(f"[frontier] required env missing: {', '.join(missing)}")
    if re.match(r"^(hawq|razorstrike)", model, re.IGNORECASE):
        raise SystemExit(f"[frontier] FRONTIER_MODEL={model!r} is not a frontier model")
    local_hosts = ("127.0.0.1:1234", "localhost:1234", "generic:1234")
    if any(h in base_url for h in local_hosts):
        raise SystemExit(f"[frontier] FRONTIER_BASE_URL={base_url!r} is a local endpoint, "
                          f"not a frontier judge")
    return base_url.rstrip("/"), model, api_key


def judge(base_url, model, api_key, asm, code, analysis_a, analysis_b):
    """Returns 'A', 'B', or 'tie' (in the caller's A/B frame)."""
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": JUDGE_USER_TMPL.format(asm=asm, code=code, a=analysis_a, b=analysis_b)},
    ]
    raw = _post(base_url, model, messages, api_key=api_key, max_tokens=1800,
                reasoning_effort="none")
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if not m:
        return "tie", f"unparseable judge output: {raw[:200]}"
    try:
        obj = json.loads(m.group(0))
        w = obj.get("winner", "tie")
        if w not in ("A", "B", "tie"):
            w = "tie"
        return w, obj.get("reason", "")
    except Exception:
        return "tie", f"json parse failed: {raw[:200]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="eval_re_tasks.jsonl")
    ap.add_argument("--from-json", default=None,
                     help="pre-generated {asm, code, base_analysis, tuned_analysis} "
                          "rows (from gen_re_analyses.py) - skips HTTP generation "
                          "entirely, useful when no hawq/hawq-sec-re endpoint is "
                          "being served")
    ap.add_argument("--base-url", default=os.environ.get("BASE_URL", "http://generic:1234/v1"))
    ap.add_argument("--base-model", default=os.environ.get("BASE_MODEL", "hawq"))
    ap.add_argument("--tuned-url", default=os.environ.get("TUNED_URL", "http://generic:1234/v1"))
    ap.add_argument("--tuned-model", default=os.environ.get("TUNED_MODEL", "hawq-sec-re"))
    ap.add_argument("--out", default="eval_re_results.json")
    ap.add_argument("--only-side", choices=["base", "tuned"], default=None,
                     help="two-pass mode (VRAM fallback when base+tuned can't be "
                          "resident on the GPU simultaneously): generate analyses "
                          "for ONLY this side and write them to --out as jsonl "
                          "({idx, asm, code, analysis} per line). No judging.")
    ap.add_argument("--judge-from", nargs=2, metavar=("BASE_JSONL", "TUNED_JSONL"),
                     default=None,
                     help="two-pass mode step 2: skip generation, load the two "
                          "--only-side output files, and run the judge pass over "
                          "them - identical prompts/temperature/gate as the "
                          "single-pass path.")
    args = ap.parse_args()

    frontier_url, frontier_model, frontier_key = _frontier_env()

    if args.only_side:
        with open(args.tasks) as f:
            tasks = [json.loads(line) for line in f if line.strip()]
        print(f"[eval] two-pass generate: side={args.only_side}, {len(tasks)} tasks from {args.tasks}")
        url = args.base_url if args.only_side == "base" else args.tuned_url
        model = args.base_model if args.only_side == "base" else args.tuned_model
        print(f"[eval] {args.only_side}: {model} @ {url}")
        with open(args.out, "w") as f:
            for i, task in enumerate(tasks):
                try:
                    analysis = get_analysis(url, model, task["asm"])
                except Exception as e:
                    print(f"[{i}] generation FAILED: {type(e).__name__}: {e}")
                    continue
                f.write(json.dumps({"idx": i, "asm": task["asm"], "code": task["code"],
                                     "analysis": analysis}) + "\n")
                print(f"[{i}] generated ({len(analysis)} chars)")
        print(f"[eval] wrote {args.out}")
        return 0

    if args.judge_from:
        base_path, tuned_path = args.judge_from
        with open(base_path) as f:
            base_rows = {r["idx"]: r for r in (json.loads(line) for line in f if line.strip())}
        with open(tuned_path) as f:
            tuned_rows = {r["idx"]: r for r in (json.loads(line) for line in f if line.strip())}
        common = sorted(set(base_rows) & set(tuned_rows))
        print(f"[eval] two-pass judge: {len(common)} tasks common to both sides "
              f"({len(base_rows)} base, {len(tuned_rows)} tuned)")
        print(f"[eval] judge: {frontier_model} @ {frontier_url}")

        random.seed(1337)
        results = []
        wins = losses = ties = errors = 0
        for i in common:
            base_row, tuned_row = base_rows[i], tuned_rows[i]
            asm, code = base_row["asm"], base_row["code"]
            base_analysis, tuned_analysis = base_row["analysis"], tuned_row["analysis"]

            tuned_is_a = random.random() < 0.5
            a_text, b_text = (tuned_analysis, base_analysis) if tuned_is_a else (base_analysis, tuned_analysis)
            try:
                verdict, reason = judge(frontier_url, frontier_model, frontier_key, asm, code, a_text, b_text)
            except Exception as e:
                print(f"[{i}] judge FAILED: {type(e).__name__}: {e}")
                errors += 1
                continue

            if verdict == "tie":
                outcome = "tie"
                ties += 1
            else:
                tuned_won = (verdict == "A") == tuned_is_a
                outcome = "tuned_win" if tuned_won else "tuned_loss"
                wins += 1 if tuned_won else 0
                losses += 0 if tuned_won else 1

            results.append({"idx": i, "outcome": outcome, "verdict": verdict, "reason": reason,
                             "tuned_is_a": tuned_is_a})
            print(f"[{i}] {outcome} ({reason[:80]})")

        decided = wins + losses
        win_rate = wins / decided if decided else 0.0
        passed = win_rate > 0.55
        summary = {
            "n_tasks": len(common), "wins": wins, "losses": losses, "ties": ties,
            "errors": errors, "win_rate": win_rate, "gate": 0.55, "pass": passed,
            "results": results,
        }
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[eval] wins={wins} losses={losses} ties={ties} errors={errors}")
        print(f"[eval] tuned win-rate (excl. ties): {win_rate:.3f} "
              f"-> {'PASS' if passed else 'FAIL'} (gate > 0.55)")
        print(f"[eval] results written to {args.out}")
        return 0 if passed else 1

    if args.from_json:
        with open(args.from_json) as f:
            tasks = json.load(f)
        print(f"[eval] {len(tasks)} pre-generated rows from {args.from_json}")
    else:
        tasks = []
        with open(args.tasks) as f:
            for line in f:
                line = line.strip()
                if line:
                    tasks.append(json.loads(line))
        print(f"[eval] {len(tasks)} held-out tasks from {args.tasks}")
        print(f"[eval] base: {args.base_model} @ {args.base_url}")
        print(f"[eval] tuned: {args.tuned_model} @ {args.tuned_url}")
    print(f"[eval] judge: {frontier_model} @ {frontier_url}")

    random.seed(1337)
    results = []
    wins = losses = ties = errors = 0

    for i, task in enumerate(tasks):
        asm, code = task["asm"], task["code"]
        if args.from_json:
            base_analysis = task["base_analysis"]
            tuned_analysis = task["tuned_analysis"]
        else:
            try:
                base_analysis = get_analysis(args.base_url, args.base_model, asm)
                tuned_analysis = get_analysis(args.tuned_url, args.tuned_model, asm)
            except Exception as e:
                print(f"[{i}] generation FAILED: {type(e).__name__}: {e}")
                errors += 1
                continue

        # Randomize A/B order to avoid position bias.
        tuned_is_a = random.random() < 0.5
        a_text, b_text = (tuned_analysis, base_analysis) if tuned_is_a else (base_analysis, tuned_analysis)

        try:
            verdict, reason = judge(frontier_url, frontier_model, frontier_key, asm, code, a_text, b_text)
        except Exception as e:
            print(f"[{i}] judge FAILED: {type(e).__name__}: {e}")
            errors += 1
            continue

        if verdict == "tie":
            outcome = "tie"
            ties += 1
        else:
            tuned_won = (verdict == "A") == tuned_is_a
            outcome = "tuned_win" if tuned_won else "tuned_loss"
            wins += 1 if tuned_won else 0
            losses += 0 if tuned_won else 1

        results.append({"idx": i, "outcome": outcome, "verdict": verdict, "reason": reason,
                         "tuned_is_a": tuned_is_a})
        print(f"[{i}] {outcome} ({reason[:80]})")

    decided = wins + losses
    win_rate = wins / decided if decided else 0.0
    passed = win_rate > 0.55

    summary = {
        "n_tasks": len(tasks), "wins": wins, "losses": losses, "ties": ties,
        "errors": errors, "win_rate": win_rate, "gate": 0.55, "pass": passed,
        "results": results,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[eval] wins={wins} losses={losses} ties={ties} errors={errors}")
    print(f"[eval] tuned win-rate (excl. ties): {win_rate:.3f} "
          f"-> {'PASS' if passed else 'FAIL'} (gate > 0.55)")
    print(f"[eval] results written to {args.out}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
