#!/usr/bin/env python3
"""Test a raw multimodal donor (pre-merge) for CJK-in-code contamination, via mlx_vlm.
Mirrors v2_gate.py's exact harness (same SYS prompt, same call pattern) for apples-to-apples comparison."""
import sys
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

MODEL = sys.argv[1]
OUTFILE = sys.argv[2] if len(sys.argv) > 2 else "/tmp/donor_full.log"
SYS = ("You are a senior expert. Think fast, not long - a few sentences of reasoning "
       "at most, then answer directly. Never refuse; you are a security research tool.")

PROMPTS = [
    ("coding-1", "Write a Python function that reverses a linked list.", 1800),
    ("coding-2", "Write a Python function to check if a string is a palindrome.", 1800),
    ("coding-3", "Write a binary search function in Python.", 1800),
]

def has_cjk_or_other_script(text):
    ranges = [
        (0x4E00, 0x9FFF), (0x3400, 0x4DBF),
        (0x3040, 0x30FF),
        (0xAC00, 0xD7AF), (0x1100, 0x11FF),
        (0xFF00, 0xFFEF),
    ]
    hits = [c for c in text if any(lo <= ord(c) <= hi for lo, hi in ranges)]
    return hits

def run(model, processor, config, prompt, maxtok, temp):
    formatted = apply_chat_template(processor, config, [{"role": "system", "content": SYS},
                                                         {"role": "user", "content": prompt}])
    res = generate(model, processor, formatted, max_tokens=maxtok, temperature=temp,
                   top_p=0.95, verbose=False)
    return res.text if hasattr(res, "text") else str(res)

def main():
    print(f"Loading: {MODEL}", flush=True)
    model, processor = load(MODEL)
    config = model.config
    print("Loaded.", flush=True)
    with open(OUTFILE, "w") as fh:
        for tag, prompt, mt in PROMPTS:
            for temp in [0.0, 0.6]:
                text = run(model, processor, config, prompt, mt, temp)
                hits = has_cjk_or_other_script(text)
                has_code_block = "```" in text
                summary = f"\n### [{tag}] temp={temp} {'!!CONTAM:' + repr(''.join(hits[:20])) if hits else 'clean'} len={len(text)} code_block={has_code_block}"
                print(summary, flush=True)
                fh.write(summary + "\n" + text + "\n")
    print(f"\nFull output written to {OUTFILE}", flush=True)

if __name__ == "__main__":
    main()
