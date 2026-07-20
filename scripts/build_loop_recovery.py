#!/usr/bin/env python3
"""Phase 1h - Build loop_recovery: anti-doom-loop traces.

Self-contained SYNTHETIC generator (no external HF dataset). Teaches the model
the single behavior that abliteration damaged and that we are re-instilling via
data: when an action repeats with the same result, STOP and change strategy
instead of retrying/ruminating.

Each row is a short agentic trace ending in the recovery move:
  detect repetition -> name it -> switch strategy or halt with a concrete reason.

Output: datasets.DatasetDict with train (~3,000 rows), family="loop_recovery".
"""

import os
import random
from datasets import Dataset, DatasetDict

SEED = 42
TARGET_ROWS = 3_000

SYSTEM = (
    "You are a disciplined engineering agent. You never repeat an action that has "
    "already produced the same result. When you notice you are repeating yourself, "
    "you stop, state what is actually happening, and either change strategy or halt "
    "with a concrete reason. You never spiral."
)

# (situation, repeated_action, observed_result, correct_recovery)
SCENARIOS = [
    ("rewriting a file whose content the tool reports unchanged",
     "write {path}",
     "the tool reports the file was written but a reviewer says it is still identical",
     "The write is persisting; 'unchanged' means my new content equals the old content, "
     "not that the write failed. Retrying the same write cannot help. I will stop, diff "
     "the current file against my intended content, and change only the specific lines that differ."),
    ("a shell command that keeps returning the same error",
     "run `{cmd}`",
     "exit code 1 with the identical traceback each time",
     "I have run this three times with the identical failure. Re-running will not change it. "
     "I will read the traceback's top frame, fix that root cause, and only then re-run once."),
    ("loading a dataset that does not exist on the Hub",
     "load_dataset('{path}')",
     "DatasetNotFoundError: the repo does not exist",
     "The dataset genuinely does not exist; retrying the load is futile. I will switch to a "
     "concrete fallback: generate the data synthetically or point at a real source, then continue."),
    ("a search that returns no matches",
     "grep '{pat}'",
     "zero results, repeatedly, for the same pattern",
     "The pattern is absent. Repeating the same search is pointless. I will broaden the pattern "
     "or search a different path once, then move on."),
    ("an edit that fails to apply with the same anchor error",
     "edit {path}",
     "'no preceding hunk header' on every attempt",
     "The edit format is wrong the same way each time; another identical attempt fails identically. "
     "I will re-read the file to get exact line numbers, or fall back to a full rewrite once."),
    ("waiting on a job that never changes state",
     "poll job {jid}",
     "status 'running' unchanged after many polls",
     "Polling faster changes nothing. I will set one bounded wait, and if it is still unchanged, "
     "report the stall and stop rather than polling forever."),
]

PATHS = ["scripts/build_x.py", "src/main.rs", "app/models.py", "lib/crypto.c", "notebooks/run.ipynb"]
CMDS = ["python build.py", "cargo test", "make all", "pytest -x", "npm run build"]
PATTERNS = ["def build_", "TODO", "unsafe", "api_key", "loop {"]
DATASETS = ["acme/private-set", "internal/traces-v3", "org/does-not-exist", "team/archived"]


def build_loop_recovery_dataset(cap=0):
    """Build the loop_recovery dataset (synthetic, self-contained)."""
    random.seed(SEED)
    n = TARGET_ROWS if cap <= 0 else min(cap, TARGET_ROWS)
    rows = []
    for i in range(n):
        situation, action, result, recovery = SCENARIOS[i % len(SCENARIOS)]
        fill = dict(
            path=random.choice(PATHS + DATASETS),
            cmd=random.choice(CMDS),
            pat=random.choice(PATTERNS),
            jid=f"job_{random.randint(1000,9999)}",
        )
        act = action.format(**fill)
        attempts = random.randint(2, 4)
        user = (
            f"You are {situation}. You have already tried `{act}` {attempts} times and each time "
            f"{result}. What do you do next?"
        )
        assistant = (
            f"I have repeated `{act}` {attempts} times with the same outcome, so another identical "
            f"attempt is guaranteed to fail the same way. {recovery}"
        )
        rows.append({
            "source": "synthetic/loop_recovery",
            "family": "loop_recovery",
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ],
        })
    ds_out = DatasetDict({"train": Dataset.from_list(rows)})
    print(f"Built loop_recovery dataset: {len(rows)} rows")
    return ds_out


if __name__ == "__main__":
    cap = int(os.environ.get("CAP", "0"))
    ds = build_loop_recovery_dataset(cap=cap)
    print(f"Train: {len(ds['train'])} rows")
    print("loop_recovery dataset ready (family=\"loop_recovery\")")
