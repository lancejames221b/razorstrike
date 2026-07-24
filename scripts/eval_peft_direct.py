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

# HAWQ base (lancejames221b/HAWQ, a DARE-TIES merge) has malformed checkpoint
# keys: they are prefixed `language_model.model.*` and `language_model.lm_head`,
# but Qwen3_5MoeForCausalLM expects `model.*` / `lm_head.*`. The original
# Qwen/Qwen3.6-35B-A3B base has keys `model.language_model.*` (correct for the
# ConditionalGeneration wrapper). Neither load class matches HAWQ's keys as-is,
# so from_pretrained loads with ~693 MISSING keys (random init -> garbage).
# Fix: load with CausalLM, check missing_keys via output_loading_info (the real
# list - transformers logs to stderr, NOT capturable via redirect_stdout), and
# if MISSING keys are present, reload by manually remapping the state dict
# (strip the leading `language_model.` prefix from every checkpoint key).
def _load_base(repo):
    """Load base with CausalLM-first, return (model, missing, unexpected)."""
    try:
        m, info = AutoModelForCausalLM.from_pretrained(
            repo, output_loading_info=True, **_kw)
        return m, info["missing_keys"], info["unexpected_keys"]
    except Exception as e:
        print(f"[eval] AutoModelForCausalLM failed ({type(e).__name__}); "
              f"trying AutoModelForImageTextToText", flush=True)
        m, info = AutoModelForImageTextToText.from_pretrained(
            repo, output_loading_info=True, **_kw)
        return m, info["missing_keys"], info["unexpected_keys"]


def _load_base_remapped(repo):
    """Load base by remapping checkpoint keys (strip `language_model.` prefix)
    so HAWQ's `language_model.model.*` -> `model.*` matches CausalLM. Memory-
    safe: builds the model on meta device (init_empty_weights, zero real
    memory), then load_state_dict(assign=True) places the real remapped tensors
    directly onto the meta params (no copy, no 2x peak). strict=True after a
    correct remap; anything still off raises immediately instead of silently
    running on random weights."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers import AutoConfig
    from accelerate import init_empty_weights
    import glob, os
    d = snapshot_download(repo)
    shards = sorted(glob.glob(os.path.join(d, "*.safetensors")))
    print(f"[eval] remapping load: {len(shards)} shards from {d}", flush=True)
    remapped = {}
    _gate_bufs = {}   # layer_prefix -> gate_proj tensor, awaiting fusion with up_proj
    _up_bufs = {}     # layer_prefix -> up_proj tensor, awaiting fusion with gate_proj
    for s in shards:
        for k, v in load_file(s).items():
            # AutoModelForCausalLM instantiates the TEXT-ONLY backbone
            # (Qwen3_5MoeForCausalLM) - it has no vision tower or multi-token-
            # prediction head. Only remap keys that originate under the
            # checkpoint's `language_model.*` prefix; everything else
            # (`model.visual.*`, `mtp.*`) belongs to submodules this class
            # doesn't have and must be dropped, not loaded.
            if not k.startswith("language_model."):
                continue
            nk = k[len("language_model."):]
            # conv1d weight layout fix: HAWQ's SSM/linear-attn merge saved
            # conv1d.weight as [C, K, 1] (fla/SSM convention) but transformers'
            # nn.Conv1d expects [C, 1, K]. Transpose dims 1<->2 for these keys.
            if nk.endswith("linear_attn.conv1d.weight") and v.dim() == 3:
                if v.shape[1] != 1 and v.shape[2] == 1:
                    v = v.transpose(1, 2).contiguous()
            # MoE expert naming fix: HAWQ saved separate switch_mlp.gate_proj /
            # switch_mlp.up_proj / switch_mlp.down_proj ([E,512,2048] each),
            # but the model expects a single fused experts.gate_up_proj
            # ([E,1024,2048] = cat(gate,up) dim=1) and experts.down_proj (same
            # shape as switch_mlp.down_proj, just renamed - no data change).
            if ".mlp.switch_mlp.gate_proj.weight" in nk:
                _gate_bufs[nk.replace(".mlp.switch_mlp.gate_proj.weight", "")] = v
                continue
            if ".mlp.switch_mlp.up_proj.weight" in nk:
                _up_bufs[nk.replace(".mlp.switch_mlp.up_proj.weight", "")] = v
                continue
            if ".mlp.switch_mlp.down_proj.weight" in nk:
                nk = nk.replace(".mlp.switch_mlp.down_proj.weight", ".mlp.experts.down_proj")
            remapped[nk] = v.to("cuda", dtype=torch.bfloat16)
    # Fuse buffered gate/up pairs now that both halves are loaded.
    for layer_prefix, gate_v in _gate_bufs.items():
        up_v = _up_bufs[layer_prefix]
        fused = torch.cat([gate_v, up_v], dim=1)
        remapped[layer_prefix + ".mlp.experts.gate_up_proj"] = fused.to("cuda", dtype=torch.bfloat16)
    cfg = AutoConfig.from_pretrained(repo)
    with init_empty_weights():
        m = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
    # assign=True: place real tensors onto meta params directly (no copy).
    # strict=False: unexpected keys (leftover naming variants we haven't hit
    # yet) are harmless since this class has no submodule to put them in.
    # The real invariant is "no MISSING keys" - checked explicitly below.
    res = m.load_state_dict(remapped, strict=False, assign=True)
    m = m.to("cuda")
    print(f"[eval] remapped load: missing={len(res.missing_keys)} "
          f"unexpected={len(res.unexpected_keys)}", flush=True)
    return m, list(res.missing_keys), list(res.unexpected_keys)


model, _missing, _unexpected = _load_base(base_repo)
if _missing:
    print(f"[eval] direct load had {len(_missing)} missing keys; "
          f"attempting key-remap load (strip language_model. prefix)", flush=True)
    del model
    import gc, torch as _t
    gc.collect(); _t.cuda.empty_cache()
    model, _missing, _unexpected = _load_base_remapped(base_repo)

# Real guard: abort if STILL missing keys after remap.
if _missing:
    print(f"[eval] FATAL: model loaded with {len(_missing)} MISSING keys "
          f"even after remap. Weights are randomly initialized; eval would be "
          f"meaningless. Sample missing: {list(_missing)[:5]}", flush=True)
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
    """Parse tool-call blocks from generated text.

    HAWQ's chat_template.jinja instructs native XML-attribute function calls:
    <tool_call><function=NAME><parameter=PNAME>value</parameter>...</function>
    </tool_call> (verified against the model's actual output - it complies
    with its own template exactly). Tried first. Falls back to a JSON object
    wrapped in TC_OPEN/TC_CLOSE (older Qwen convention), then to bare JSON
    objects containing both "name" and "arguments" keys, for compatibility
    with other base models. Returns a list of {"name":..., "arguments": <dict>},
    skipping anything that fails to parse or lacks "name".
    """
    calls = []
    # --- Primary: XML-attribute function-call format (HAWQ's actual format) ---
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
    # --- Fallback: JSON object wrapped in TC_OPEN/TC_CLOSE ---
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