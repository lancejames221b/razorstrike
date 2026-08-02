#!/usr/bin/env python3
"""challenge_suite.py - aggregate challenge-battery runner across every
probe family used to rank HAWQ-SEC-RE weaknesses (edit discipline, tool
loop / error recovery / long-CoT, and the four crypto-audit probes).

No existing script aggregates across probes or across k trials:
probe_edit_discipline.py runs exactly one trial per invocation;
eval_re_v2_http_probes.py runs its three probes once each;
eval_crypto_audit.py's own probe_* functions already run k trials but
average that away into a single majority-vote verdict per case, which
is not what a battery runner needs (we need one row per trial so
failures can be mined into DPO pairs later). This script is the
missing per-trial aggregator: it reuses every probe/case-runner
function UNCHANGED from the three sibling files (no grading logic is
reimplemented here) and just drives them k times per family, writing
one JSONL row per trial plus a final aggregate summary.

Reuse map (see each family's row below for the exact function):
  - probe_edit_discipline.py:      probe_edit_discipline, probe_edit_reread
  - eval_re_v2_http_probes.py:     probe_tool_loop, probe_error_recovery,
                                    probe_long_cot
  - eval_crypto_audit.py:          _crypto_id_case_once, _clean_control_once,
                                    _misuse_enum_once, _exploit_path_once
                                    (the per-trial case runners that already
                                    do exactly one HTTP round + grading; the
                                    k-trials-then-majority wrappers in that
                                    file are intentionally NOT called here,
                                    since this script needs one JSONL row
                                    PER trial rather than one pre-averaged
                                    verdict per case).

raw_text/tool_calls capture: none of the reused functions above return
the model's raw completion text or tool calls (they return only
booleans/verdicts), and per project coordination (concurrent edits to
eval_crypto_audit.py's crypto-id scorer) this script must not touch
that file. So raw_text/tool_calls are captured by temporarily
monkeypatching the MODULE-LEVEL `generate` name that each probe module
resolves at call time (probe_edit_discipline.generate and
eval_re_v2_http_probes.generate are two independent bindings even
though they start out pointing at the same function object - each
must be patched separately) - this requires zero edits to any of the
three sibling files. Per explicit steering, this capture is applied
ONLY for the edit_* and RE-probe families; the four crypto families
record raw_text=null/tool_calls=null (later DPO-pair mining targets
only edit_read_first/edit_reread_after_failure, so this is sufficient
and avoids any risk of racing the concurrent eval_crypto_audit.py edit).

CLI:
    python3 scripts/challenge_suite.py --host <HOST:PORT> --model <LMSTUDIO_KEY> \
        --out /tmp/hawq_dpo/battery_<label>.jsonl [--families a,b,c] \
        [--k-override N] [--expect-context 262144]
"""
import argparse
import collections
import functools
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import probe_edit_discipline  # noqa: E402
import eval_re_v2_http_probes  # noqa: E402
import eval_crypto_audit  # noqa: E402

# Per-family k, fixed so results stay comparable across runs (never change a
# family's k here - use --k-override for a one-off test run instead).
FAMILY_K = {
    "edit_read_first": 15,
    "edit_reread_after_failure": 15,
    "tool_loop": 15,
    "error_recovery": 15,
    "long_cot": 15,
    "crypto_id": 9,
    "clean_control": 9,
    "misuse_enum": 9,
    "exploit_path": 9,
}

ERROR_RATE_THRESHOLD = 0.20


# ---------------------------------------------------------------------------
# lms ps precondition check
# ---------------------------------------------------------------------------

def _parse_lms_ps(text):
    """Parse `lms ps` tabular output into a list of {COLUMN_NAME: value}
    dicts. Column-POSITION based (using the header row's token start
    offsets), not whitespace-split: real `lms ps` output has values with
    internal spaces (e.g. SIZE = "18.94 GB"), which a naive .split() would
    misalign against the header."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    header_idx = None
    for i, ln in enumerate(lines):
        if "IDENTIFIER" in ln and "CONTEXT" in ln:
            header_idx = i
            break
    if header_idx is None:
        return []
    cols = [(m.group(0), m.start()) for m in re.finditer(r"\S+", lines[header_idx])]
    rows = []
    for ln in lines[header_idx + 1:]:
        if not ln.strip():
            continue
        row = {}
        for j, (name, start) in enumerate(cols):
            end = cols[j + 1][1] if j + 1 < len(cols) else len(ln)
            row[name] = ln[start:end].strip()
        rows.append(row)
    return rows


def _check_context(model, expect_context, when):
    """Verifies `model` is loaded in LM Studio at exactly --expect-context.
    A silent context downgrade (e.g. a JIT-reload racing an unload) already
    produced one false FAIL in this project, so this is enforced both
    before the first trial and after the last, not just once. Passing
    --expect-context 0 skips this entirely, for non-LM-Studio endpoints."""
    if expect_context == 0:
        print(f"[precondition] --expect-context 0: skipping lms ps check ({when} run)", flush=True)
        return
    print(f"[precondition] checking lms ps ({when} run)...", flush=True)
    try:
        proc = subprocess.run(["lms", "ps"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as e:
        raise SystemExit(f"lms ps failed ({when} run): {e!r}")
    if proc.returncode != 0:
        raise SystemExit(f"lms ps exited {proc.returncode} ({when} run): {proc.stderr.strip()}")
    rows = _parse_lms_ps(proc.stdout)
    row = next((r for r in rows if r.get("IDENTIFIER") == model), None)
    if row is None:
        seen = [r.get("IDENTIFIER") for r in rows]
        raise SystemExit(f"lms ps ({when} run): no loaded model matches --model {model!r} "
                          f"(loaded: {seen})")
    raw_ctx = row.get("CONTEXT", "")
    try:
        actual = int(raw_ctx)
    except ValueError:
        raise SystemExit(f"lms ps ({when} run): unparseable CONTEXT column {raw_ctx!r} "
                          f"for {model!r}")
    if actual != expect_context:
        raise SystemExit(f"context mismatch: expected {expect_context}, saw {actual}")
    print(f"[precondition] lms ps ({when} run): {model} CONTEXT={actual} OK", flush=True)


# ---------------------------------------------------------------------------
# generate() capture (edit_* and RE-probe families only - see module docstring)
# ---------------------------------------------------------------------------

class _CaptureGenerate:
    """Context manager that monkeypatches `module.generate` for its
    duration, recording every call's (reasoning, content, tool_calls) into
    `self.calls`, then restores the original binding on exit regardless of
    how the block exits."""

    def __init__(self, module):
        self.module = module
        self.calls = []
        self._original = None

    def __enter__(self):
        self._original = self.module.generate

        def _wrapper(*args, **kwargs):
            reasoning, content, finish_reason, completion_tokens, tool_calls = self._original(*args, **kwargs)
            self.calls.append({
                "reasoning": reasoning or "",
                "content": content or "",
                "finish_reason": finish_reason,
                "tool_calls": tool_calls or [],
            })
            return reasoning, content, finish_reason, completion_tokens, tool_calls

        self.module.generate = _wrapper
        return self.calls

    def __exit__(self, exc_type, exc, tb):
        self.module.generate = self._original
        return False


def _join_raw_text(calls):
    parts = []
    for i, c in enumerate(calls):
        turn = f"[turn {i}]"
        if c["reasoning"]:
            turn += f"\n<reasoning>\n{c['reasoning']}\n</reasoning>"
        turn += f"\n{c['content']}"
        parts.append(turn)
    return "\n\n".join(parts)


def _run_captured(module, fn, *args, **kwargs):
    """Calls fn(*args, **kwargs) while capturing every generate() call made
    through `module`'s module-level binding; returns
    (result, raw_text, tool_calls)."""
    with _CaptureGenerate(module) as calls:
        result = fn(*args, **kwargs)
    raw_text = _join_raw_text(calls)
    tool_calls = [tc for c in calls for tc in c["tool_calls"]]
    return result, raw_text, tool_calls


# ---------------------------------------------------------------------------
# Per-trial functions, one per family. Each returns
# (metrics_dict, raw_text_or_None, tool_calls_or_None).
# ---------------------------------------------------------------------------

def _trial_edit_read_first(host, model):
    result, raw_text, tool_calls = _run_captured(
        probe_edit_discipline, probe_edit_discipline.probe_edit_discipline, host, model)
    ok, read_before_first_edit, saw_any_read, saw_any_edit = result
    metrics = {
        "task_pass": bool(ok),
        "read_before_first_edit": bool(read_before_first_edit),
        "saw_any_read": bool(saw_any_read),
        "saw_any_edit": bool(saw_any_edit),
    }
    return metrics, raw_text, tool_calls


def _trial_edit_reread_after_failure(host, model):
    result, raw_text, tool_calls = _run_captured(
        probe_edit_discipline, probe_edit_discipline.probe_edit_reread, host, model)
    ok, reread_after_failure, resubmitted_identical = result
    metrics = {
        "task_pass": bool(ok),
        "reread_after_failure": bool(reread_after_failure),
        "resubmitted_identical": bool(resubmitted_identical),
    }
    return metrics, raw_text, tool_calls


def _trial_tool_loop(host, model):
    result, raw_text, tool_calls = _run_captured(
        eval_re_v2_http_probes, eval_re_v2_http_probes.probe_tool_loop, host, model)
    return {"pass": bool(result)}, raw_text, tool_calls


def _trial_error_recovery(host, model):
    result, raw_text, tool_calls = _run_captured(
        eval_re_v2_http_probes, eval_re_v2_http_probes.probe_error_recovery, host, model)
    return {"pass": bool(result)}, raw_text, tool_calls


def _trial_long_cot(host, model):
    result, raw_text, tool_calls = _run_captured(
        eval_re_v2_http_probes, eval_re_v2_http_probes.probe_long_cot, host, model)
    # TRUNCATE ("hit the token cap without repeating") counts as pass, per spec.
    passed = result in ("PASS", "TRUNCATE")
    return {"pass": passed, "verdict": result}, raw_text, tool_calls


# --- crypto families: call eval_crypto_audit's UNMODIFIED per-case runners
# directly (no reimplemented grading); raw_text/tool_calls are null here
# per explicit steering (a sibling task is concurrently editing that file's
# crypto-id scorer, and DPO-pair mining only targets the edit_* families
# anyway, so capturing generate() there is unnecessary risk for no benefit).

def _trial_crypto_id(host, model, case_name, code):
    result = eval_crypto_audit._crypto_id_case_once(host, model, case_name, code)
    return {"case": case_name, "pass": bool(result)}, None, None


def _trial_clean_control(host, model):
    result = eval_crypto_audit._clean_control_once(host, model)
    return {"false_positive": bool(result), "hits": list(result)}, None, None


def _trial_misuse_enum(host, model):
    result = eval_crypto_audit._misuse_enum_once(host, model)
    found = sum(1 for v in result.values() if v)
    metrics = dict(result)
    metrics["found_count"] = found
    # Mirrors probe_misuse_enum's own aggregate threshold (>=2 of 3 labels),
    # applied per-trial instead of via cross-trial majority-per-label.
    metrics["pass"] = found >= 2
    return metrics, None, None


def _trial_exploit_path(host, model):
    bug_hits, conseq_hits = eval_crypto_audit._exploit_path_once(host, model)
    ok = bool(bug_hits) and bool(conseq_hits)
    return {"bug_hits": bug_hits, "conseq_hits": conseq_hits, "pass": ok}, None, None


def _iter_trials(family, host, model, k):
    """Yields (trial_index, zero-arg callable) pairs for `family`. Every
    family is a flat range(k) except crypto_id, which is (case x k) since
    that family reports a pass_rate PER CASE across the 5 crypto-id cases."""
    if family == "edit_read_first":
        for i in range(k):
            yield i, functools.partial(_trial_edit_read_first, host, model)
    elif family == "edit_reread_after_failure":
        for i in range(k):
            yield i, functools.partial(_trial_edit_reread_after_failure, host, model)
    elif family == "tool_loop":
        for i in range(k):
            yield i, functools.partial(_trial_tool_loop, host, model)
    elif family == "error_recovery":
        for i in range(k):
            yield i, functools.partial(_trial_error_recovery, host, model)
    elif family == "long_cot":
        for i in range(k):
            yield i, functools.partial(_trial_long_cot, host, model)
    elif family == "crypto_id":
        idx = 0
        for case_name, (aliases, code) in eval_crypto_audit._CRYPTO_ID_CASES.items():
            for _ in range(k):
                yield idx, functools.partial(_trial_crypto_id, host, model, case_name, code)
                idx += 1
    elif family == "clean_control":
        for i in range(k):
            yield i, functools.partial(_trial_clean_control, host, model)
    elif family == "misuse_enum":
        for i in range(k):
            yield i, functools.partial(_trial_misuse_enum, host, model)
    elif family == "exploit_path":
        for i in range(k):
            yield i, functools.partial(_trial_exploit_path, host, model)
    else:
        raise ValueError(f"unknown family: {family}")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _rate(trial_metrics, key):
    vals = [bool(m[key]) for m in trial_metrics if key in m]
    return (sum(vals) / len(vals)) if vals else None


def _aggregate(family, trial_metrics):
    """Reduces a family's successful per-trial metrics dicts into the
    summary shape from the family table. None if there's nothing to
    aggregate (only reachable if every trial errored, in which case the
    caller already treats the family as threshold-breached anyway)."""
    if not trial_metrics:
        return None
    if family == "edit_read_first":
        return {
            "read_before_first_edit_rate": _rate(trial_metrics, "read_before_first_edit"),
            "task_pass_rate": _rate(trial_metrics, "task_pass"),
        }
    if family == "edit_reread_after_failure":
        return {
            "reread_after_failure_rate": _rate(trial_metrics, "reread_after_failure"),
            "task_pass_rate": _rate(trial_metrics, "task_pass"),
            "resubmitted_identical_rate": _rate(trial_metrics, "resubmitted_identical"),
        }
    if family in ("tool_loop", "error_recovery", "long_cot", "misuse_enum", "exploit_path"):
        return {"pass_rate": _rate(trial_metrics, "pass")}
    if family == "clean_control":
        return {"false_positive_rate": _rate(trial_metrics, "false_positive")}
    if family == "crypto_id":
        by_case = collections.defaultdict(list)
        for m in trial_metrics:
            by_case[m["case"]].append(bool(m["pass"]))
        per_case_rate = {c: sum(v) / len(v) for c, v in by_case.items()}
        mean_rate = (sum(per_case_rate.values()) / len(per_case_rate)) if per_case_rate else None
        return {"per_case_pass_rate": per_case_rate, "mean_pass_rate": mean_rate}
    raise ValueError(f"no aggregator for family {family!r}")


def run_family(family, host, model, k, out_fh):
    """Runs every trial for `family`, writing one JSONL row per trial to
    out_fh as it goes (flushed immediately, so a crash mid-family still
    leaves prior trials on disk). Returns (aggregate_or_None, breached).

    DESIGN CHOICE (explicitly called out per the assignment): a trial that
    raises does NOT abort the family or the run - every configured trial
    for every selected family is always attempted, and only once a family
    is fully done do we compute its error rate and decide whether its
    aggregate becomes null. This is the "safer for capturing partial data"
    branch: a high error rate in one family (e.g. a transient network blip)
    must not prevent the other families' otherwise-good trials from running
    and being recorded. The non-zero exit(2) in main() still surfaces the
    breach loudly after the fact.
    """
    trial_metrics = []
    error_count = 0
    total = 0
    for trial_idx, call in _iter_trials(family, host, model, k):
        total += 1
        try:
            metrics, raw_text, tool_calls = call()
            record = {
                "family": family,
                "trial": trial_idx,
                "metrics": metrics,
                "raw_text": raw_text,
                "tool_calls": tool_calls,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            trial_metrics.append(metrics)
        except Exception as e:
            error_count += 1
            record = {
                "family": family,
                "trial": trial_idx,
                "metrics": None,
                "error": repr(e),
                "raw_text": None,
                "tool_calls": None,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            print(f"[{family}] trial {trial_idx} ERRORED: {e!r}", flush=True)
        out_fh.write(json.dumps(record) + "\n")
        out_fh.flush()

    error_rate = (error_count / total) if total else 1.0
    breached = error_rate > ERROR_RATE_THRESHOLD
    print(f"[{family}] done: {total - error_count}/{total} ok, {error_count}/{total} "
          f"errored ({error_rate:.0%})", flush=True)
    if breached:
        print(f"[{family}] error rate {error_rate:.0%} exceeds {ERROR_RATE_THRESHOLD:.0%} "
              f"threshold -> family aggregate emitted as null", flush=True)
        return None, True

    agg = _aggregate(family, trial_metrics)
    if agg is not None:
        agg = {"k": k, **agg}
    return agg, False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="HAWQ-SEC-RE challenge battery runner")
    ap.add_argument("--host", required=True, help="host:port of the OpenAI-compatible endpoint")
    ap.add_argument("--model", required=True, help="LM Studio model identifier to test")
    ap.add_argument("--out", required=True, help="output JSONL path (one row per trial)")
    ap.add_argument("--families", default=None,
                     help="comma-separated family ids to run (default: all)")
    ap.add_argument("--k-override", type=int, default=None,
                     help="override k for every selected family, this run only "
                          "(never edits FAMILY_K)")
    ap.add_argument("--expect-context", type=int, default=262144,
                     help="required lms ps CONTEXT for --model (default 262144); "
                          "0 skips the check, for non-LM-Studio endpoints")
    args = ap.parse_args()

    if args.k_override is not None and args.k_override < 1:
        ap.error("--k-override must be >= 1")

    if args.families:
        selected = [f.strip() for f in args.families.split(",") if f.strip()]
        unknown = [f for f in selected if f not in FAMILY_K]
        if unknown:
            ap.error(f"unknown family id(s): {unknown}; choices: {sorted(FAMILY_K)}")
    else:
        selected = list(FAMILY_K)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    print(f"\n=== CHALLENGE BATTERY: {args.model} @ {args.host} "
          f"families={selected} ===\n", flush=True)

    t0 = time.time()
    _check_context(args.model, args.expect_context, "before")

    summary = {"model": args.model, "host": args.host, "families": {}}
    any_breach = False
    with open(args.out, "w") as out_fh:
        for family in selected:
            k = args.k_override if args.k_override is not None else FAMILY_K[family]
            print(f"--- {family} (k={k}) ---", flush=True)
            agg, breached = run_family(family, args.host, args.model, k, out_fh)
            summary["families"][family] = agg
            any_breach = any_breach or breached

    print(f"\n=== SUMMARY ({time.time() - t0:.1f}s total) ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)

    summary_path = args.out + ".summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {args.out}", flush=True)
    print(f"Wrote {summary_path}", flush=True)

    # Artifacts above are written BEFORE the "after" precondition check on
    # purpose: if the model's context drifted mid-run (e.g. a JIT-reload
    # raced an unload), the results are suspect, but the operator still gets
    # the full JSONL + summary for forensics rather than nothing.
    _check_context(args.model, args.expect_context, "after")

    if any_breach:
        print("\nOVERALL: one or more families exceeded the error-rate threshold "
              "(see per-family aggregate = null above).", flush=True)
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
