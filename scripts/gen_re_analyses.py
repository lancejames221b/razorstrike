#!/usr/bin/env python3
"""Phase 5a - Generate base-vs-tuned RE analyses in-process (no HTTP serving
required). Companion to eval_re_analysis.py's judge step.

No "hawq"/"hawq-sec-re" model is being served over HTTP anywhere in this
infra (LM Studio on generic serves different models), so instead of hitting
an OpenAI-compatible endpoint, this script loads base HAWQ-v1 and the trained
adapter directly on the training GPU host (same pattern as
eval_peft_direct.py's loader) and generates both sets of analyses in one
pass. Output feeds eval_re_analysis.py --from-json for the frontier judge
step, which only needs network access to the judge endpoint, not to this
GPU host.

Usage (on the training VM, GPU available):
    python3 scripts/gen_re_analyses.py \
        --tasks /content/hawq_re/eval_re_tasks.jsonl \
        --base-repo lancejames221b/HAWQ-v1 \
        --adapter-dir /content/adapter \
        --out /content/hawq_re/gen_results.json
"""

import os
import sys
import json
import argparse

RE_ANALYST_SYSTEM = (
    "You are an elite reverse engineer. Given x86-64 assembly from a compiled "
    "binary, produce a precise technical analysis: purpose, inputs/outputs, "
    "algorithm, and security-relevant behavior."
)


def build_prompt(tok, asm):
    messages = [
        {"role": "system", "content": RE_ANALYST_SYSTEM},
        {"role": "user",
         "content": f"Analyze this x86-64 function and explain what it does.\n\n```asm\n{asm}\n```"},
    ]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--base-repo", default="lancejames221b/HAWQ-v1")
    ap.add_argument("--adapter-dir", default="/content/adapter")
    ap.add_argument("--out", default="gen_results.json")
    ap.add_argument("--max-new-tokens", type=int, default=1800)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--n-tasks", type=int, default=0, help="limit to first N tasks (0 = all)")
    args = ap.parse_args()

    tasks = []
    with open(args.tasks) as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    if args.n_tasks:
        tasks = tasks[:args.n_tasks]
    print(f"[gen] {len(tasks)} held-out tasks", flush=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained(args.base_repo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # required for correct batched generation

    _kw = dict(dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True)
    print("[gen] loading base model", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.base_repo, **_kw)
    model.eval()

    _im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
    eos_ids = {tok.eos_token_id}
    if isinstance(_im_end_id, int) and _im_end_id != tok.eos_token_id and _im_end_id >= 0:
        eos_ids.add(_im_end_id)

    def generate_batch(prompt_texts):
        inputs = tok(prompt_texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens,
                do_sample=True, temperature=0.6, top_p=0.95,
                pad_token_id=tok.pad_token_id, eos_token_id=list(eos_ids))
        gen_ids = out[:, inputs["input_ids"].shape[1]:]
        return [tok.decode(g, skip_special_tokens=True).strip() for g in gen_ids]

    BATCH = args.batch_size

    def generate_all(items, key_fn, partial_key, ckpt_path):
        """items: list of dicts with 'asm'; returns list of generated strings in order.
        Writes ckpt_path after every batch (durability against spot preemption)."""
        out_texts = [None] * len(items)
        for start in range(0, len(items), BATCH):
            chunk = items[start:start + BATCH]
            prompts = [build_prompt(tok, it["asm"]) for it in chunk]
            texts = generate_batch(prompts)
            for j, txt in enumerate(texts):
                out_texts[start + j] = txt
            done = min(start + BATCH, len(items))
            print(f"[gen] {key_fn} {done}/{len(items)}", flush=True)
            with open(ckpt_path, "w") as f:
                json.dump({"key": partial_key, "done": done, "texts": out_texts}, f)
        return out_texts

    results = [{"asm": t["asm"], "code": t["code"]} for t in tasks]
    print("[gen] generating BASE analyses", flush=True)
    base_texts = generate_all(tasks, "base", "base", args.out + ".base_ckpt.json")
    for r, base_text in zip(results, base_texts):
        r["base_analysis"] = base_text
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[gen] base pass complete, checkpoint written -> {args.out}", flush=True)

    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    print("[gen] loading base + adapter (tuned)", flush=True)
    base2 = AutoModelForCausalLM.from_pretrained(args.base_repo, **_kw)
    tuned = PeftModel.from_pretrained(base2, args.adapter_dir)
    tuned = tuned.merge_and_unload()
    tuned.eval()
    model = tuned

    print("[gen] generating TUNED analyses", flush=True)
    tuned_texts = generate_all(results, "tuned", "tuned", args.out + ".tuned_ckpt.json")
    for r, tuned_text in zip(results, tuned_texts):
        r["tuned_analysis"] = tuned_text

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[gen] wrote {len(results)} rows -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
