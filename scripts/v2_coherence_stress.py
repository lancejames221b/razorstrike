#!/usr/bin/env python3
"""Extra coherence stress-test for razorstrike-v2 bf16 - check if the
language-mixing glitch seen in the coding prompt is systemic or a one-off."""
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

MODEL = "/Volumes/Scratch/ml-workspace/merged/razorstrike_v2_mlx_bf16"
SYS = ("You are a senior expert. Think fast, not long - a few sentences of reasoning "
       "at most, then answer directly.")

PROMPTS = [
    ("coding-retry-1", "Write a Python function that reverses a linked list.", 500),
    ("coding-retry-2", "Write a Python function that reverses a linked list.", 500),
    ("coding-2",  "Write a Python function to check if a string is a palindrome.", 400),
    ("coding-3",  "Write a binary search function in Python.", 400),
    ("prose",     "Describe the water cycle in 3 sentences.", 300),
    ("agentic",   "You need to find all TODO comments in a codebase. What command would you run?", 300),
]

def run(model, processor, config, prompt, maxtok, temp):
    formatted = apply_chat_template(processor, config, [{"role": "system", "content": SYS},
                                                        {"role": "user", "content": prompt}])
    res = generate(model, processor, formatted, max_tokens=maxtok, temperature=temp,
                   top_p=0.95, verbose=False)
    return res.text if hasattr(res, "text") else str(res)

def has_cjk(text):
    return any('\u4e00' <= c <= '\u9fff' or '\u3040' <= c <= '\u30ff' for c in text)

def main():
    print(f"Loading bf16: {MODEL}", flush=True)
    model, processor = load(MODEL)
    config = model.config
    print("Loaded.", flush=True)
    for tag, p, mt in PROMPTS:
        for temp in [0.6, 0.0]:
            t = run(model, processor, config, p, mt, temp)
            cjk = has_cjk(t)
            print(f"\n### [{tag}] temp={temp} {'!!CJK-CONTAMINATION!!' if cjk else 'clean'} len={len(t)}", flush=True)
            print(t[:400], flush=True)

if __name__ == "__main__":
    main()
