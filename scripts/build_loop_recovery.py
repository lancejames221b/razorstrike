#!/usr/bin/env python3
"""Phase 1h - Build loop_recovery: anti-doom-loop traces.

Self-contained SYNTHETIC generator (no external HF dataset). Teaches the single
behavior abliteration damaged and we re-instill via data: when an action repeats
with the same result, STOP and change strategy instead of retrying/ruminating.

Diversity matters here (this is a v2-defining family), so this uses 22 distinct
recovery patterns with varied fills rather than a handful of templates.

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

# (situation, action_template, observed_result, recovery) - 22 distinct patterns.
SCENARIOS = [
    ("rewriting a file the tool reports unchanged", "write {path}",
     "the tool says written but a reviewer says it is still identical",
     "The write is persisting; 'unchanged' means my new content equals the old, not that "
     "the write failed. Retrying is futile. I will diff current vs intended and change only "
     "the differing lines."),
    ("a shell command failing identically", "run `{cmd}`",
     "exit code 1 with the same traceback each time",
     "Three identical failures mean re-running won't help. I will read the top traceback "
     "frame, fix that root cause, then run once."),
    ("loading a dataset that does not exist", "load_dataset('{repo}')",
     "DatasetNotFoundError repeatedly",
     "The repo genuinely does not exist; retrying the load cannot succeed. I will switch to a "
     "concrete fallback - synthesize the data or point at a real source - and continue."),
    ("a search returning nothing", "grep '{pat}'",
     "zero results for the same pattern, repeatedly",
     "The pattern is absent. Repeating it is pointless. I will broaden the pattern or search a "
     "different path once, then move on."),
    ("an edit failing with the same anchor error", "edit {path}",
     "'no preceding hunk header' every attempt",
     "The edit format is wrong the same way each time. I will re-read the file for exact line "
     "numbers, or fall back to a full rewrite once."),
    ("polling a job stuck in one state", "poll job {jid}",
     "status 'running' unchanged after many polls",
     "Polling faster changes nothing. I will set one bounded wait; if still unchanged, report "
     "the stall and stop."),
    ("an API call that keeps timing out", "GET {url}",
     "connection timeout on every retry",
     "Repeated timeouts with no backoff won't resolve. I will check reachability once, add a "
     "timeout+backoff, and if it still fails, report the endpoint as down."),
    ("installing a package that keeps conflicting", "pip install {pkg}",
     "the same dependency-resolution error each run",
     "The resolver hits the same conflict every time. I will pin the conflicting dependency or "
     "use a fresh env, not reinstall identically."),
    ("a test that fails the same assertion", "pytest {path}",
     "the identical assertion failure repeatedly",
     "Re-running the same test gives the same failure. I will read the assertion, fix the code "
     "or the expectation, then run once."),
    ("a compile error on the same line", "make",
     "the same syntax error at the same line each build",
     "The build fails identically; rebuilding won't fix source. I will open that line, correct "
     "it, then rebuild."),
    ("git push rejected repeatedly", "git push",
     "rejected: remote has changes, every attempt",
     "Re-pushing won't help while the remote is ahead. I will pull/rebase once, resolve, then "
     "push."),
    ("a container that exits on start", "docker run {img}",
     "the container exits with the same code immediately",
     "Restarting identically reproduces the crash. I will read the container logs, fix the "
     "config/entrypoint, then run once."),
    ("a model download that keeps corrupting", "download {repo}",
     "checksum mismatch on each attempt",
     "Repeated corruption suggests a bad mirror or partial cache, not transient loss. I will "
     "clear the cache and switch mirror once, not retry blindly."),
    ("a query returning an empty set", "SELECT ... WHERE {cond}",
     "zero rows for the same predicate",
     "The predicate matches nothing. Re-running is pointless. I will verify the column/values "
     "or relax the filter once."),
    ("an auth call that keeps 401ing", "auth {svc}",
     "401 Unauthorized on every retry",
     "Same credentials give the same 401. Retrying won't authenticate. I will refresh/rotate "
     "the token or fix the scope, then try once."),
    ("a flaky element never appearing", "wait for {sel}",
     "selector not found after repeated waits",
     "The element isn't appearing; longer waits won't summon it. I will check the DOM/frame "
     "once and correct the selector, or conclude the page changed."),
    ("a rate-limited endpoint", "call {url}",
     "429 Too Many Requests repeatedly",
     "Hammering a 429 makes it worse. I will honor Retry-After with one backed-off attempt, "
     "not immediate retries."),
    ("a migration that keeps failing mid-way", "migrate {db}",
     "the same constraint violation each run",
     "The migration hits the same constraint every time. I will fix the offending data or the "
     "migration step, then run once - not re-run identically."),
    ("a build cache that never invalidates", "rebuild {target}",
     "stale output despite source edits, repeatedly",
     "Rebuilding reproduces stale output; the cache isn't invalidating. I will clear the cache "
     "once, then build."),
    ("an LLM tool call rejected the same way", "call tool {tool}",
     "the same schema-validation error each attempt",
     "The arguments are malformed the same way each time. I will read the schema, fix the "
     "argument shape, then call once."),
    ("a port already in use", "start server :{port}",
     "EADDRINUSE on every start",
     "The port is held; restarting collides identically. I will find and stop the listener, or "
     "pick a free port, then start once."),
    ("a checkpoint that won't resume", "resume {ckpt}",
     "the same 'incompatible shapes' error each try",
     "Resuming reproduces the shape mismatch. I will reconcile the config with the checkpoint "
     "once, or start fresh - not retry identically."),
]

PATHS = ["scripts/build_x.py", "src/main.rs", "app/models.py", "lib/crypto.c", "run.ipynb"]
CMDS = ["python build.py", "cargo test", "make all", "pytest -x", "npm run build"]
PATS = ["def build_", "TODO", "unsafe", "api_key", "handler("]
REPOS = ["acme/private-set", "org/does-not-exist", "team/archived", "internal/v3"]
URLS = ["https://api.svc/v1/items", "http://10.0.0.4:8080/health", "https://hub/models"]
PKGS = ["torch==2.1", "numpy<2", "peft", "bitsandbytes"]
IMGS = ["app:latest", "worker:dev", "db:15"]
SVCS = ["s3", "vault", "registry"]
SELS = ["#submit", ".result-row", "button[type=submit]"]
DBS = ["users", "events", "billing"]
TOOLS = ["write_file", "search", "run_shell"]
CONDS = ["status='x'", "id=0", "ts>now()"]
CKPTS = ["ckpt-250", "last", "epoch2"]
TARGETS = ["frontend", "kernel", "docs"]


def build_loop_recovery_dataset(cap=0):
    random.seed(SEED)
    n = TARGET_ROWS if cap <= 0 else min(cap, TARGET_ROWS)
    rows = []
    for i in range(n):
        situation, action_t, result, recovery = SCENARIOS[i % len(SCENARIOS)]
        fill = dict(
            path=random.choice(PATHS), cmd=random.choice(CMDS), pat=random.choice(PATS),
            repo=random.choice(REPOS), url=random.choice(URLS), pkg=random.choice(PKGS),
            img=random.choice(IMGS), svc=random.choice(SVCS), sel=random.choice(SELS),
            db=random.choice(DBS), tool=random.choice(TOOLS), cond=random.choice(CONDS),
            ckpt=random.choice(CKPTS), target=random.choice(TARGETS),
            jid=f"job_{random.randint(1000,9999)}", port=random.choice([8080, 5173, 3000]),
        )
        act = action_t.format(**fill)
        attempts = random.randint(2, 4)
        user = (f"You are {situation}. You have already tried `{act}` {attempts} times and each "
                f"time {result}. What do you do next?")
        assistant = (f"I have repeated `{act}` {attempts} times with the same outcome, so another "
                     f"identical attempt will fail identically. {recovery}")
        rows.append({
            "source": "synthetic/loop_recovery", "family": "loop_recovery",
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
