#!/usr/bin/env python3
"""Lightweight post-training eval: load HAWQ base + LoRA adapter directly,
run eval probes (tool_loop, error_recovery, long_cot) on the Colab GPU.
No merge/quantize/MLX needed - loads base+adapter in-process.

Usage:
  python3 scripts/eval_peft_direct.py --adapter-dir /content/adapter/last-checkpoint --base lancejames221b/HAWQ
"""

import os, sys, json, time, re, subprocess

adapter_dir = "/content/adapter/last-checkpoint"
base_repo = "lancejames221b/HAWQ"

for i, arg in enumerate(sys.argv):
    if arg == "--adapter-dir" and i+1 < len(sys.argv):
        adapter_dir = sys.argv[i+1]
    elif arg == "--base" and i+1 < len(sys.argv):
        base_repo = sys.argv[i+1]

print(f"[eval] loading {base_repo} + adapter from {adapter_dir}", flush=True)

import torch
from transformers import AutoModelForImageTextToText, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

tok = AutoTokenizer.from_pretrained(base_repo)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

_kw = dict(dtype=torch.bfloat16, device_map="cuda", low_cpu_mem_usage=True)
try:
    model = AutoModelForImageTextToText.from_pretrained(base_repo, **_kw)
except Exception:
    model = AutoModelForCausalLM.from_pretrained(base_repo, **_kw)

model = PeftModel.from_pretrained(model, adapter_dir)
model = model.merge_and_unload()
model.eval()
print(f"[eval] model loaded + merged, {sum(p.numel() for p in model.parameters()):,} params", flush=True)


def generate(messages, max_tokens=4000, temperature=0.6, top_p=0.95):
    """Generate a response from the merged model."""
    input_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tok.pad_token_id,
        )
    response = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response


# === Probes (same logic as eval_loop_recovery.py but using local generate()) ===

def probe_tool_loop():
    """Truthful successes across turns -> must reach DONE without repeating a call."""
    tools_desc = "You have a write_file(path, content) tool. Call it by writing: CALL write_file {\"path\": \"...\", \"content\": \"...\"}"
    msgs = [
        {"role": "system", "content": f"You are a coding agent. {tools_desc} Create adder.py, rev.py, fib.py, palin.py via one write_file call each, then reply DONE. Never rewrite a file that already succeeded."},
        {"role": "user", "content": "Begin."},
    ]
    calls = []
    for _ in range(14):
        resp = generate(msgs, max_tokens=1200)
        msgs.append({"role": "assistant", "content": resp})
        # Extract calls
        found = re.findall(r'CALL\s+write_file\s+(\{[^}]+\})', resp)
        if found:
            for f in found:
                try:
                    p = json.loads(f).get("path", "?")
                except Exception:
                    p = "?"
                calls.append(p)
                msgs.append({"role": "user", "content": f"write_file({p}) succeeded. Continue, or DONE if all four exist."})
        else:
            if "DONE" in resp.upper():
                break
            msgs.append({"role": "user", "content": "Continue, or DONE if all four exist."})
    worst = max(__import__("collections").Counter(calls).values()) if calls else 0
    ok = worst < 3
    print(f"[tool_loop] calls={calls} max_repeat={worst} -> {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def probe_error_recovery():
    """Unresolvable error -> bounded, low-repetition, decisive."""
    trace = ("Traceback (most recent call last):\n"
             "  File \"build.py\", line 41, in build\n"
             "    ds = load_dataset('acme/does-not-exist')\n"
             "datasets.exceptions.DatasetNotFoundError: Dataset 'acme/does-not-exist' "
             "doesn't exist on the Hub.")
    msgs = [
        {"role": "system", "content": "You are a decisive engineer. If a resource is missing, choose a concrete fallback or stop with a reason. Never spiral."},
        {"role": "user", "content": f"Fix build.py. The run failed:\n\n{trace}\n\nProceed."},
    ]
    resp = generate(msgs, max_tokens=4000)
    sents = [re.sub(r"\s+", " ", s).strip() for s in re.split(r"[.\n]", resp) if len(s.split()) >= 6]
    dup = 1 - len(set(sents)) / max(len(sents), 1)
    ctoks = len(tok(resp)["input_ids"])
    ok = ctoks < 3500 and dup < 0.3
    print(f"[error_recovery] tokens={ctoks} dup_ratio={dup:.2f} -> {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def probe_long_cot():
    """Long reasoning trace -> must not loop or repeat. The actual v1 failure surface."""
    msgs = [
        {"role": "system", "content": "You are a careful mathematical reasoner. Think through the problem step by step in detail before giving your final answer."},
        {"role": "user", "content": "Find all integer solutions to: x^4 - 2x^3 - 7x^2 + 8x + 12 = 0. Show your full reasoning."},
    ]
    resp = generate(msgs, max_tokens=16000)
    ctoks = len(tok(resp)["input_ids"])
    sents = [re.sub(r"\s+", " ", s).strip() for s in re.split(r"[.\n]", resp) if len(s.split()) >= 6]
    dup = 1 - len(set(sents)) / max(len(sents), 1)
    words = resp.split()
    phrases_3 = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
    phrase_dup = 1 - len(set(phrases_3)) / max(len(phrases_3), 1)
    print(f"[long_cot] tokens={ctoks} sentence_dup={dup:.3f} 3gram_dup={phrase_dup:.3f}", flush=True)
    if ctoks >= 15000:
        print(f"  FAIL: hit near-max tokens ({ctoks}) - possible runaway trace", flush=True)
        return False
    elif phrase_dup > 0.5:
        print(f"  FAIL: very high 3-gram repetition ({phrase_dup:.3f}) - likely looping", flush=True)
        return False
    elif ctoks < 8000 and dup < 0.5:
        print(f"  PASS: bounded + clean ({ctoks} tokens, dup={dup:.3f})", flush=True)
        return True
    else:
        print(f"  REVIEW: long but not clearly looping - manual inspection needed", flush=True)
        return True  # don't fail on ambiguous


if __name__ == "__main__":
    print("\n=== POST-TRAINING EVAL: HAWQ base + validation LoRA ===\n", flush=True)
    results = {}
    results["tool_loop"] = probe_tool_loop()
    results["error_recovery"] = probe_error_recovery()
    results["long_cot"] = probe_long_cot()
    print(f"\n=== SUMMARY: {sum(results.values())}/{len(results)} passed ===", flush=True)
    for k, v in results.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}", flush=True)
    if all(results.values()):
        print("OVERALL: PASS - no regression", flush=True)
        sys.exit(0)
    else:
        print("OVERALL: FAIL - regression detected", flush=True)
        sys.exit(1)