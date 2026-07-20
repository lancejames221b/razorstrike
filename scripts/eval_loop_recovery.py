#!/usr/bin/env python3
"""Post-training gate: does RazorStrike v2 still spiral on an unresolvable error?

Replays the exact failure mode that exposed v1's error-recovery spiral, against
any OpenAI-compatible endpoint (LM Studio serving the merged v2, vLLM, etc.).

Two probes:
  1. multi-turn tool loop with TRUTHFUL success -> must progress, not repeat.
  2. error-recovery generation: feed an unresolvable error -> must decide/stop
     (bounded tokens, low self-repetition), not ruminate/runaway.

Usage:
  EVAL_BASE_URL=http://127.0.0.1:1234/v1 EVAL_MODEL=razorstrike-v2 \
      python3 scripts/eval_loop_recovery.py

Exit code 0 = PASS (no spiral), 1 = FAIL.
"""

import os, json, re, collections, urllib.request

BASE_URL = os.environ.get("EVAL_BASE_URL", "http://127.0.0.1:1234/v1").rstrip("/")
MODEL    = os.environ.get("EVAL_MODEL", "razorstrike-v2")
URL      = f"{BASE_URL}/chat/completions"


def _post(body):
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=400) as r:
        return json.load(r)


def probe_tool_loop():
    """Truthful successes across turns -> must reach DONE without repeating a call."""
    tools = [{"type": "function", "function": {
        "name": "write_file", "description": "Write a file. Returns bytes written.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                       "required": ["path", "content"]}}}]
    msgs = [
        {"role": "system", "content": "You are a coding agent. Create adder.py, rev.py, "
         "fib.py, palin.py via one write_file call each, then reply DONE. Never rewrite a "
         "file that already succeeded."},
        {"role": "user", "content": "Begin."}]
    calls = []
    for _ in range(14):
        d = _post({"model": MODEL, "messages": msgs, "tools": tools,
                   "temperature": 0.6, "top_p": 0.95, "max_tokens": 1200})
        m = d["choices"][0]["message"]
        tcs = m.get("tool_calls") or []
        am = {"role": "assistant", "content": m.get("content") or ""}
        if tcs:
            am["tool_calls"] = tcs
        msgs.append(am)
        if tcs:
            for tc in tcs:
                try:
                    p = json.loads(tc["function"]["arguments"]).get("path", "?")
                except Exception:
                    p = "?"
                calls.append(p)
                msgs.append({"role": "tool", "tool_call_id": tc.get("id", "x"),
                             "content": json.dumps({"ok": True, "bytes": 96})})
        else:
            if "DONE" in (m.get("content") or "").upper():
                break
            msgs.append({"role": "user", "content": "Continue, or DONE if all four exist."})
    worst = max(collections.Counter(calls).values()) if calls else 0
    ok = worst < 3  # no path written 3+ times
    print(f"[tool_loop] calls={calls} max_repeat={worst} -> {'PASS' if ok else 'FAIL'}")
    return ok


def probe_error_recovery():
    """Unresolvable error -> bounded, low-repetition, decisive (no runaway rumination)."""
    trace = ("Traceback (most recent call last):\n"
             "  File \"build.py\", line 41, in build\n"
             "    ds = load_dataset('acme/does-not-exist')\n"
             "datasets.exceptions.DatasetNotFoundError: Dataset 'acme/does-not-exist' "
             "doesn't exist on the Hub.")
    msgs = [
        {"role": "system", "content": "You are a decisive engineer. If a resource is missing, "
         "choose a concrete fallback or stop with a reason. Never spiral."},
        {"role": "user", "content": f"Fix build.py. The run failed:\n\n{trace}\n\nProceed."}]
    d = _post({"model": MODEL, "messages": msgs, "temperature": 0.6, "top_p": 0.95,
               "max_tokens": 4000})
    ch = d["choices"][0]
    m = ch["message"]
    txt = (m.get("reasoning_content") or "") + "\n" + (m.get("content") or "")
    ctoks = d.get("usage", {}).get("completion_tokens", 0)
    sents = [re.sub(r"\s+", " ", s).strip() for s in re.split(r"[.\n]", txt) if len(s.split()) >= 6]
    dup = 1 - len(set(sents)) / max(len(sents), 1)
    ok = ch.get("finish_reason") == "stop" and ctoks < 3500 and dup < 0.3
    print(f"[error_recovery] finish={ch.get('finish_reason')} tokens={ctoks} "
          f"dup_ratio={dup:.2f} -> {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print(f"Evaluating {MODEL} at {BASE_URL}")
    a = probe_tool_loop()
    b = probe_error_recovery()
    ok = a and b
    print(f"\nOVERALL: {'PASS - no spiral' if ok else 'FAIL - spiral behavior present'}")
    raise SystemExit(0 if ok else 1)
