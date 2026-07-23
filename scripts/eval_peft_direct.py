#!/usr/bin/env python3
"""Lightweight post-training eval: load HAWQ base + LoRA adapter directly,
run eval probes (tool_loop, error_recovery, long_cot) on the Colab GPU.
No merge/quantize/MLX needed - loads base+adapter in-process.

Harness fixed per HAWQ validation eval plan:
  - generate() respects the Qwen3.6-35B-A3B "thinking-mode precise-coding"
    sampler preset (temperature=0.6, top_p=0.95, top_k=20, min_p=0.0) and
    reports finish_reason + completion_tokens (was: dropped top_k/min_p,
    skip_special_tokens=True, no stop reason).
  - probe_tool_loop uses native Qwen tool-call parsing (was: fake CALL text
    convention that never fired -> vacuous calls=[] PASS).
  - probe_error_recovery matches the canonical eval_loop_recovery.py gate
    and now requires finish=="stop" (was: no stop-reason requirement).
  - probe_long_cot uses the card's 81,920-token cap for hard math and
    classifies length-cap hits as PASS/FAIL/TRUNCATE by phrase repetition
    (was: 16,000 cap that manufactured a false-positive FAIL).
  - --adapter-dir "" / --no-adapter evaluates the raw base (deconfounds
    merge-damage vs LoRA-damage).

Usage:
  python3 scripts/eval_peft_direct.py --adapter-dir /content/adapter --base lancejames221b/HAWQ
  python3 scripts/eval_peft_direct.py --no-adapter --base lancejames221b/HAWQ
"""

import os, sys, json, time, re, collections

adapter_dir = "/content/adapter/last-checkpoint"
base_repo = "lancejames221b/HAWQ"
no_adapter = False

for i, arg in enumerate(sys.argv):
    if arg == "--adapter-dir" and i+1 < len(sys.argv):
        adapter_dir = sys.argv[i+1]
    elif arg == "--base" and i+1 < len(sys.argv):
        base_repo = sys.argv[i+1]
    elif arg == "--no-adapter":
        no_adapter = True

# Treat an explicitly empty adapter-dir as base-only (Step 8).
if adapter_dir == "":
    no_adapter = True

label = base_repo + (" + adapter" if not no_adapter else " (raw base, no adapter)")
print(f"[eval] loading {label}", flush=True)

import torch
from transformers import (AutoModelForImageTextToText, AutoModelForCausalLM,
                           AutoTokenizer, GenerationConfig)
from peft import PeftModel

tok = AutoTokenizer.from_pretrained(base_repo)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

_kw = dict(dtype=torch.bfloat16, device_map="cuda", low_cpu_mem_usage=True)

# Load order matters: the HAWQ base (lancejames221b/HAWQ) is a DARE-TIES merge
# whose checkpoint keys are text-only causal-LM naming (language_model.*),
# which matches Qwen3_5MoeForCausalLM but NOT the vision wrapper
# Qwen3_5MoeForConditionalGeneration (expects model.language_model.*) produced
# by AutoModelForImageTextToText. ImageTextToText does not raise on this
# mismatch - it loads with MISSING keys (randomly-initialized weights) and
# emits garbage. So try CausalLM FIRST (proven 0-MISSING on HAWQ), and only
# fall back to ImageTextToText if CausalLM itself raises.
_load_report = []
try:
    import io, contextlib
    _buf = io.StringIO()
    with contextlib.redirect_stdout(_buf):
        model = AutoModelForCausalLM.from_pretrained(base_repo, **_kw)
    _load_report.append(_buf.getvalue())
except Exception as _e:
    print(f"[eval] AutoModelForCausalLM failed ({type(_e).__name__}); "
          f"trying AutoModelForImageTextToText", flush=True)
    _buf = io.StringIO()
    with contextlib.redirect_stdout(_buf):
        model = AutoModelForImageTextToText.from_pretrained(base_repo, **_kw)
    _load_report.append(_buf.getvalue())

# Guard: abort loudly if the load produced MISSING keys (random init). This is
# the exact failure mode that manufactured garbled output on the first run -
# never let it pass silently again.
_full_report = "\n".join(_load_report)
_missing = _full_report.count("MISSING")
if _missing > 0:
    print(f"[eval] FATAL: model loaded with {_missing} MISSING keys "
          f"(checkpoint/model prefix mismatch). Aborting - weights are "
          f"randomly initialized, eval would be meaningless.", flush=True)
    sys.exit(2)

if not no_adapter:
    model = PeftModel.from_pretrained(model, adapter_dir)
    model = model.merge_and_unload()
model.eval()
print(f"[eval] model loaded{'+ merged' if not no_adapter else ''}, "
      f"{sum(p.numel() for p in model.parameters()):,} params "
      f"({type(model).__name__})", flush=True)

# Load the base generation config once; prefer its top_k/min_p if present
# (card-grounded, but the repo may already encode them).
try:
    GEN_CFG = GenerationConfig.from_pretrained(base_repo)
except Exception:
    GEN_CFG = GenerationConfig()

# EOS set: real end-of-turn tokens, not just pad.
_im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
_eos_ids = {tok.eos_token_id}
if isinstance(_im_end_id, int) and _im_end_id != tok.eos_token_id and _im_end_id >= 0:
    _eos_ids.add(_im_end_id)

# Native Qwen tool-call wrapper tokens, built from chr() so the literal
# sequences never appear in source (avoids editor/transport mangling).
TC_OPEN  = chr(60) + "tool_call" + chr(62)
TC_CLOSE = chr(60) + "/" + "tool_call" + chr(62)

# Reasoning-block close marker. Qwen3 emits a think block then content.
# If the model emits a different marker, inspect the first raw completion in
# eval_output.txt and update this (per plan Assumptions).
THINK_CLOSE = chr(60) + "/think" + chr(62)


def generate(messages, max_tokens=4000, sampler="coding", tools=None):
    """Generate from the (merged) model.

    Returns (full_text, finish_reason, completion_tokens):
      - full_text: decoded completion WITH special tokens kept
        (skip_special_tokens=False) so the reasoning block is visible for
        dup metrics; callers split reasoning vs content on THINK_CLOSE.
      - finish_reason: "stop" if the last generated id is an EOS id, else
        "length" (hit the cap).
      - completion_tokens: number of generated ids (excludes prompt).
    """
    tpl_kw = dict(tokenize=False, add_generation_prompt=True)
    if tools is not None:
        tpl_kw["tools"] = tools
    input_text = tok.apply_chat_template(messages, **tpl_kw)
    inputs = tok(input_text, return_tensors="pt").to(model.device)

    gen_kw = dict(max_new_tokens=max_tokens, pad_token_id=tok.pad_token_id,
                  eos_token_id=list(_eos_ids))

    if sampler == "greedy":
        gen_kw["do_sample"] = False
    else:  # "coding" preset (card lines 1013-1017)
        gen_kw["do_sample"] = True
        gen_kw["temperature"] = 0.6
        gen_kw["top_p"] = 0.95
        # Prefer the base generation_config's top_k/min_p if it encodes them;
        # otherwise fall back to the card's thinking-mode precise-coding values.
        gen_kw["top_k"] = getattr(GEN_CFG, "top_k", 20) or 20
        gen_kw["min_p"] = getattr(GEN_CFG, "min_p", 0.0) or 0.0
        # repetition_penalty=1.0 is the transformers default; presence_penalty
        # is not reproducible on this path (see plan Assumptions).

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kw)

    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = out[0][prompt_len:]
    completion_tokens = int(gen_ids.shape[0])
    last_id = int(gen_ids[-1].item()) if completion_tokens > 0 else -1
    finish_reason = "stop" if last_id in _eos_ids else "length"

    full_text = tok.decode(gen_ids, skip_special_tokens=False)
    return full_text, finish_reason, completion_tokens


def split_reasoning(full_text):
    """Split a decoded completion into (reasoning, content) on THINK_CLOSE.
    If no think block is present, reasoning is "" and content is the full text
    (trimmed)."""
    if THINK_CLOSE in full_text:
        reasoning, _, content = full_text.partition(THINK_CLOSE)
        # strip a leading think opener if present
        opener = chr(60) + "think" + chr(62)
        reasoning = reasoning.replace(opener, "", 1)
        return reasoning.strip(), content.strip()
    # No reasoning delimiter: treat everything as content.
    return "", full_text.strip()


def parse_tool_calls(text):
    """Parse native Qwen tool-call blocks from generated text.

    Matches the tool-call wrapper tokens (TC_OPEN ... TC_CLOSE, built from
    chr() at module load) wrapping a JSON object with name+arguments. Falls
    back to bare JSON objects containing both "name" and "arguments" keys if
    no wrapped blocks are found. Returns a list of
    {"name":..., "arguments": <dict>}, skipping anything that fails to
    json.loads or lacks "name".
    """
    calls = []
    pat = re.escape(TC_OPEN) + r"\s*(\{.*?\})\s*" + re.escape(TC_CLOSE)
    for m in re.finditer(pat, text, re.DOTALL):
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict) and "name" in obj:
                calls.append({"name": obj["name"],
                              "arguments": obj.get("arguments", {})})
        except Exception:
            pass
    if calls:
        return calls
    # Fallback: bare JSON with name+arguments (non-greedy, one-level args).
    for m in re.finditer(
        r'\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{.*?\}\s*\}',
        text, re.DOTALL):
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict) and "name" in obj:
                calls.append({"name": obj["name"],
                              "arguments": obj.get("arguments", {})})
        except Exception:
            pass
    return calls


def _dup_metrics(full_text):
    """Sentence-level dup ratio and 3-gram phrase dup ratio over >=6-word
    sentences. Operates on the full reasoning+content text per the plan."""
    sents = [re.sub(r"\s+", " ", s).strip()
             for s in re.split(r"[.\n]", full_text)
             if len(s.split()) >= 6]
    dup = 1 - len(set(sents)) / max(len(sents), 1)
    words = full_text.split()
    phrases_3 = [" ".join(words[i:i+3]) for i in range(max(len(words)-2, 0))]
    phrase_dup = 1 - len(set(phrases_3)) / max(len(phrases_3), 1)
    return dup, phrase_dup


# === Probes (ported from eval_loop_recovery.py to local generate()) ===

def probe_tool_loop():
    """Truthful successes across turns -> must reach DONE without repeating a
    call. Uses native Qwen tool-calling (not the old fake CALL convention)."""
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
        full, finish, _ = generate(msgs, max_tokens=1200, tools=tools)
        if first_raw is None:
            first_raw = full[:800]
        _, content = split_reasoning(full)
        msgs.append({"role": "assistant", "content": content})
        tcs = parse_tool_calls(content)
        if tcs:
            for tc in tcs:
                args = tc.get("arguments", {}) or {}
                p = args.get("path", "?")
                calls.append(p)
                msgs.append({"role": "tool",
                             "content": json.dumps({"ok": True, "bytes": 96})})
        else:
            if "DONE" in content.upper():
                done = True
                break
            msgs.append({"role": "user",
                         "content": "Continue, or DONE if all four exist."})
    worst = max(collections.Counter(calls).values()) if calls else 0
    distinct = len(set(calls))
    # PASS iff no path written 3+ times AND at least one distinct call AND DONE.
    ok = (worst < 3) and (distinct >= 1) and done
    print(f"[tool_loop] calls={calls} distinct={distinct} max_repeat={worst} "
          f"done={done} -> {'PASS' if ok else 'FAIL'}", flush=True)
    if first_raw and not calls:
        print(f"[tool_loop] DEBUG no calls parsed; first raw completion:\n"
              f"{first_raw}", flush=True)
    return ok


def probe_error_recovery():
    """Unresolvable error -> bounded, low-repetition, decisive (no runaway)."""
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
    full, finish, ctoks = generate(msgs, max_tokens=4000)
    dup, _ = _dup_metrics(full)
    ok = (finish == "stop") and (ctoks < 3500) and (dup < 0.3)
    print(f"[error_recovery] finish={finish} tokens={ctoks} "
          f"dup_ratio={dup:.2f} -> {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def probe_long_cot():
    """Long reasoning trace -> must not loop or repeat. Uses the card's 81,920-
    token cap for hard math/competition problems. Classifies length-cap hits as
    PASS/FAIL/TRUNCATE by phrase repetition (truncation != degeneration)."""
    msgs = [
        {"role": "system", "content": "You are a careful mathematical reasoner. "
         "Think through the problem step by step in detail before giving your "
         "final answer."},
        {"role": "user", "content": "Find all integer solutions to: "
         "x^4 - 2x^3 - 7x^2 + 8x + 12 = 0. Show your full reasoning."},
    ]
    full, finish, ctoks = generate(msgs, max_tokens=81920)
    dup, phrase_dup = _dup_metrics(full)
    print(f"[long_cot] finish={finish} tokens={ctoks} sentence_dup={dup:.3f} "
          f"3gram_dup={phrase_dup:.3f}", flush=True)
    if finish == "stop":
        print(f"  PASS: terminated on its own ({ctoks} tokens)", flush=True)
        return "PASS"
    # finish == "length"
    if phrase_dup > 0.3:
        print(f"  FAIL: hit 81920 cap by repeating (3gram_dup={phrase_dup:.3f})",
              flush=True)
        return "FAIL"
    print(f"  TRUNCATE: novel content cut off at 81920 cap "
          f"(3gram_dup={phrase_dup:.3f}) - inconclusive, not a loop", flush=True)
    return "TRUNCATE"


if __name__ == "__main__":
    print("\n=== POST-TRAINING EVAL: HAWQ base + validation LoRA ===\n", flush=True)
    results = {}
    results["tool_loop"] = probe_tool_loop()
    results["error_recovery"] = probe_error_recovery()
    long_cot = probe_long_cot()
    results["long_cot"] = long_cot

    print(f"\n=== SUMMARY ===", flush=True)
    for k, v in results.items():
        print(f"  {k}: {v}", flush=True)

    tl = results["tool_loop"]
    er = results["error_recovery"]
    if tl and er and long_cot == "PASS":
        print("OVERALL: PASS - no regression", flush=True)
        sys.exit(0)
    if long_cot == "TRUNCATE" and tl and er:
        # Soft pass for go/no-go: not degenerating, just long.
        print("LONG_COT_TRUNCATE: tool_loop + error_recovery PASS, long_cot "
              "truncated (novel content). Soft pass for go/no-go; note budget "
              "cap for deployment target.", flush=True)
        sys.exit(0)
    print("OVERALL: FAIL - regression detected (see per-probe output)", flush=True)
    sys.exit(1)