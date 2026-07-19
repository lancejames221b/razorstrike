#!/usr/bin/env python3
"""Test a raw donor (pre-merge) for CJK-in-code contamination, via mlx_lm (text-only)."""
import sys
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

MODEL = sys.argv[1] if len(sys.argv) > 1 else "/Volumes/Scratch/ml-workspace/diag/siq1_mlx_bf16"
OUTFILE = sys.argv[2] if len(sys.argv) > 2 else "/tmp/donor_full.log"
SYS = "You are a senior expert. Think fast, not long - a few sentences of reasoning at most, then answer directly."

PROMPTS = [
    ("coding-1", "Write a Python function that reverses a linked list.", 1800),
    ("coding-2", "Write a Python function to check if a string is a palindrome.", 1800),
    ("coding-3", "Write a binary search function in Python.", 1800),
]

def has_cjk_or_other_script(text):
    # CJK unified, CJK ext, Hiragana/Katakana, Hangul syllables+jamo, fullwidth forms
    ranges = [
        (0x4E00, 0x9FFF), (0x3400, 0x4DBF),   # CJK unified + ext A
        (0x3040, 0x30FF),                      # Hiragana/Katakana
        (0xAC00, 0xD7AF), (0x1100, 0x11FF),   # Hangul syllables + jamo
        (0xFF00, 0xFFEF),                      # fullwidth forms
    ]
    hits = [c for c in text if any(lo <= ord(c) <= hi for lo, hi in ranges)]
    return hits

def main():
    print(f"Loading: {MODEL}", flush=True)
    model, tokenizer = load(MODEL)
    print("Loaded.", flush=True)
    with open(OUTFILE, "w") as fh:
        for tag, prompt, mt in PROMPTS:
            for temp in [0.0, 0.6]:
                messages = [{"role": "system", "content": SYS}, {"role": "user", "content": prompt}]
                formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                sampler = make_sampler(temp, 0.95)
                text = generate(model, tokenizer, prompt=formatted, max_tokens=mt, verbose=False, sampler=sampler)
                hits = has_cjk_or_other_script(text)
                has_code_block = "```" in text
                summary = f"\n### [{tag}] temp={temp} {'!!CONTAM:' + repr(''.join(hits[:20])) if hits else 'clean'} len={len(text)} code_block={has_code_block}"
                print(summary, flush=True)
                fh.write(summary + "\n" + text + "\n")
    print(f"\nFull output written to {OUTFILE}", flush=True)

if __name__ == "__main__":
    main()
