#!/usr/bin/env python3
"""HTTP-endpoint port of eval_peft_direct.py's three probes (tool_loop,
error_recovery, long_cot), retargeted at the DEPLOYED artifact instead of
bf16 HF weights.

Why this exists: eval_peft_direct.py loads base+adapter in bf16 via
AutoModelForCausalLM + PeftModel on a CUDA box (Colab/GCE) -- it tests full
precision, not the IQ4_XS GGUF + q8_0 KV cache actually shipped to
`generic`. Quantization + KV-cache quantization is exactly where
long_cot/error_recovery degeneration would show up, so a bf16 pass doesn't
answer whether the deployed artifact is clean. This script hits the same
OpenAI-compatible endpoint (http://localhost:1235/v1/chat/completions) a
real client would use, against whatever model identifier is currently
loaded there -- exercising the actual deployed weights.

Same gates as the original, ported 1:1:
  tool_loop:       worst repeat < 3, distinct >= 1, DONE reached
  error_recovery:  finish=="stop", completion_tokens < 3500, sentence dup < 0.3
  long_cot:        PASS if finish=="stop"; if capped at max_tokens, FAIL if
                    3gram_dup > 0.3 else TRUNCATE (soft pass, not a loop)

Difference from the original: tool-call parsing prefers the API's own
message.tool_calls (OpenAI-style, already parsed server-side by LM Studio's
native Qwen tool-call handling) over regex-scraping raw content -- this is
MORE faithful to how a real agent harness consumes the deployed model, not
less. Falls back to the original XML/JSON regex parse if tool_calls is
empty but the content looks like it contains an unparsed call.

Usage:
    python3 scripts/eval_re_v2_http_probes.py [--host localhost:1235] [--model hawq-sec-re-v2]

Run this ON the box serving the endpoint (or anywhere with network access to
it) -- default host is localhost:1235, matching how this project's other
verification checks reach the server via `ssh generic`.
"""
import argparse
import collections
import json
import re
import sys
import time
import urllib.request

TC_OPEN = chr(60) + "tool_call" + chr(62)
TC_CLOSE = chr(60) + "/" + "tool_call" + chr(62)


def _post(host, model, messages, max_tokens, temperature=0.6, top_p=0.95, tools=None, timeout=1200):
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    if tools is not None:
        body["tools"] = tools
    req = urllib.request.Request(
        f"http://{host}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        d = json.loads(resp.read().decode("utf-8"))
    elapsed = time.time() - t0
    return d, elapsed


def generate(host, model, messages, max_tokens=4000, tools=None):
    """Returns (reasoning, content, finish_reason, completion_tokens, tool_calls)."""
    d, elapsed = _post(host, model, messages, max_tokens, tools=tools)
    ch = d["choices"][0]
    msg = ch["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    finish_reason = ch.get("finish_reason", "unknown")
    completion_tokens = d.get("usage", {}).get("completion_tokens", 0)
    tool_calls = msg.get("tool_calls") or []
    print(f"    [http] {elapsed:.1f}s completion_tokens={completion_tokens} "
          f"finish={finish_reason}", flush=True)
    return reasoning, content, finish_reason, completion_tokens, tool_calls


def parse_tool_calls_fallback(text):
    """Regex fallback, ported verbatim from eval_peft_direct.py, used only if
    the API's own tool_calls array is empty but raw content looks call-shaped."""
    calls = []
    tc_pat = re.escape(TC_OPEN) + r"\s*(.*?)\s*" + re.escape(TC_CLOSE)
    func_pat = r"<function=([^>]+)>\s*(.*?)\s*</function>"
    param_pat = r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>"
    for tc_m in re.finditer(tc_pat, text, re.DOTALL):
        block = tc_m.group(1)
        func_m = re.search(func_pat, block, re.DOTALL)
        if not func_m:
            continue
        name = func_m.group(1).strip()
        args = {}
        for p_m in re.finditer(param_pat, func_m.group(2), re.DOTALL):
            args[p_m.group(1).strip()] = p_m.group(2)
        calls.append({"name": name, "arguments": args})
    if calls:
        return calls
    pat = re.escape(TC_OPEN) + r"\s*(\{.*?\})\s*" + re.escape(TC_CLOSE)
    for m in re.finditer(pat, text, re.DOTALL):
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict) and "name" in obj:
                calls.append({"name": obj["name"], "arguments": obj.get("arguments", {})})
        except Exception:
            pass
    return calls


def normalize_tool_calls(api_tool_calls, content):
    """Prefer the API's own parsed tool_calls (OpenAI-style: {"function":
    {"name":..., "arguments": "<json str>"}}); fall back to regex on raw
    content only if the API gave us nothing."""
    if api_tool_calls:
        out = []
        for tc in api_tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name")
            raw_args = fn.get("arguments", "{}")
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
            except Exception:
                args = {}
            if name:
                out.append({"name": name, "arguments": args})
        if out:
            return out
    return parse_tool_calls_fallback(content)


def _dup_metrics(full_text):
    sents = [re.sub(r"\s+", " ", s).strip()
             for s in re.split(r"[.\n]", full_text)
             if len(s.split()) >= 6]
    dup = 1 - len(set(sents)) / max(len(sents), 1)
    words = full_text.split()
    phrases_3 = [" ".join(words[i:i + 3]) for i in range(max(len(words) - 2, 0))]
    phrase_dup = 1 - len(set(phrases_3)) / max(len(phrases_3), 1)
    return dup, phrase_dup


def probe_tool_loop(host, model):
    tools = [{"type": "function", "function": {
        "name": "write_file",
        "description": "Write a file. Returns bytes written.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"},
                                      "content": {"type": "string"}},
                       "required": ["path", "content"]}}}]
    msgs = [
        {"role": "system", "content": "You are a coding agent. Create adder.py, "
         "rev.py, fib.py, palin.py via one write_file call each, then reply DONE. "
         "Never rewrite a file that already succeeded."},
        {"role": "user", "content": "Begin."},
    ]
    calls = []
    done = False
    first_raw = None
    for turn in range(14):
        print(f"  [tool_loop] turn {turn+1}/14", flush=True)
        reasoning, content, finish, ctoks, api_tcs = generate(host, model, msgs, max_tokens=4000, tools=tools)
        if first_raw is None:
            first_raw = (content or reasoning)[:800]
        tcs = normalize_tool_calls(api_tcs, content)
        assistant_msg = {"role": "assistant", "content": content}
        if api_tcs:
            assistant_msg["tool_calls"] = api_tcs
        msgs.append(assistant_msg)
        if tcs:
            for i, tc in enumerate(tcs):
                args = tc.get("arguments", {}) or {}
                p = args.get("path", "?")
                calls.append(p)
                call_id = (api_tcs[i].get("id") if i < len(api_tcs) and api_tcs[i].get("id")
                           else f"call_{turn}_{i}")
                msgs.append({"role": "tool", "tool_call_id": call_id,
                             "content": json.dumps({"ok": True, "bytes": 96})})
        else:
            if "DONE" in content.upper():
                done = True
                break
            msgs.append({"role": "user", "content": "Continue, or DONE if all four exist."})
    worst = max(collections.Counter(calls).values()) if calls else 0
    distinct = len(set(calls))
    ok = (worst < 3) and (distinct >= 1) and done
    print(f"[tool_loop] calls={calls} distinct={distinct} max_repeat={worst} "
          f"done={done} -> {'PASS' if ok else 'FAIL'}", flush=True)
    if first_raw and not calls:
        print(f"[tool_loop] DEBUG no calls parsed; first raw completion:\n{first_raw}", flush=True)
    return ok


def probe_error_recovery(host, model):
    trace = ("Traceback (most recent call last):\n"
             "  File \"build.py\", line 41, in build\n"
             "    ds = load_dataset('acme/does-not-exist')\n"
             "datasets.exceptions.DatasetNotFoundError: Dataset 'acme/does-not-exist' "
             "doesn't exist on the Hub.")
    msgs = [
        {"role": "system", "content": "You are a decisive engineer. If a resource "
         "is missing, choose a concrete fallback or stop with a reason. Never spiral."},
        {"role": "user", "content": f"Fix build.py. The run failed:\n\n{trace}\n\nProceed."},
    ]
    reasoning, content, finish, ctoks, _ = generate(host, model, msgs, max_tokens=4000)
    full = (reasoning or "") + "\n" + (content or "")
    dup, _ = _dup_metrics(full)
    ok = (finish == "stop") and (ctoks < 3500) and (dup < 0.3)
    print(f"[error_recovery] finish={finish} tokens={ctoks} dup_ratio={dup:.2f} "
          f"-> {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def probe_long_cot(host, model):
    msgs = [
        {"role": "system", "content": "You are a careful mathematical reasoner. "
         "Think through the problem step by step in detail before giving your "
         "final answer."},
        {"role": "user", "content": "Find all integer solutions to: "
         "x^4 - 2x^3 - 7x^2 + 8x + 12 = 0. Show your full reasoning."},
    ]
    reasoning, content, finish, ctoks, _ = generate(host, model, msgs, max_tokens=81920)
    full = (reasoning or "") + "\n" + (content or "")
    dup, phrase_dup = _dup_metrics(full)
    print(f"[long_cot] finish={finish} tokens={ctoks} sentence_dup={dup:.3f} "
          f"3gram_dup={phrase_dup:.3f}", flush=True)
    if finish == "stop":
        print(f"  PASS: terminated on its own ({ctoks} tokens)", flush=True)
        return "PASS"
    if phrase_dup > 0.3:
        print(f"  FAIL: hit 81920 cap by repeating (3gram_dup={phrase_dup:.3f})", flush=True)
        return "FAIL"
    print(f"  TRUNCATE: novel content cut off at 81920 cap "
          f"(3gram_dup={phrase_dup:.3f}) - inconclusive, not a loop", flush=True)
    return "TRUNCATE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost:1235")
    ap.add_argument("--model", default="hawq-sec-re-v2")
    args = ap.parse_args()

    print(f"\n=== HTTP-DEPLOYED-ARTIFACT EVAL: {args.model} @ {args.host} ===\n", flush=True)
    results = {}
    print("--- tool_loop ---", flush=True)
    results["tool_loop"] = probe_tool_loop(args.host, args.model)
    print("--- error_recovery ---", flush=True)
    results["error_recovery"] = probe_error_recovery(args.host, args.model)
    print("--- long_cot ---", flush=True)
    long_cot = probe_long_cot(args.host, args.model)
    results["long_cot"] = long_cot

    print("\n=== SUMMARY ===", flush=True)
    for k, v in results.items():
        print(f"  {k}: {v}", flush=True)

    tl = results["tool_loop"]
    er = results["error_recovery"]
    if tl and er and long_cot == "PASS":
        print("OVERALL: PASS - no regression", flush=True)
        sys.exit(0)
    if long_cot == "TRUNCATE" and tl and er:
        print("LONG_COT_TRUNCATE: tool_loop + error_recovery PASS, long_cot "
              "truncated (novel content). Soft pass for go/no-go; note budget "
              "cap for deployment target.", flush=True)
        sys.exit(0)
    print("OVERALL: FAIL - regression detected (see per-probe output)", flush=True)
    sys.exit(1)


if __name__ == "__main__":
    main()
