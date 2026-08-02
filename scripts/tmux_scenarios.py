#!/usr/bin/env python3
"""tmux_scenarios.py - HAWQ v1.3 retrain plan, Step 4.

Step-by-step live test harness: each scenario is a REAL omp session in a
real tmux window against a real file tree, scored mechanically from omp's
own JSON event stream (`--mode json`). This is deliberately NOT a probe
that calls the model's HTTP endpoint directly - it drives the deployed
`omp` CLI so the tool surface (system prompt, tool schemas, hashline
rejection message) is exactly what production sends, per scripts/
omp_surface.py's whole reason for existing.

Empirically-verified findings (all measured against a real `omp --mode
json` run on this machine while writing this harness, NOT assumed from the
plan text - two corrections to the plan's own paraphrase are noted below):

1. PLAN-MODE INHERITANCE IS REAL AND FUNCTIONALLY BLOCKING, not just a
   banner. This machine's global config has `plan.defaultOnStartup: true`
   (~/.omp/agent/config.yml) - EVERY fresh `omp` invocation, not only ones
   nested inside a parent omp session, starts in plan mode by default.
   Confirmed by direct test: `omp -p "create ok.txt containing hi"`
   produced NO file (plan mode intercepted the turn, routed it to
   anthropic/claude-opus-5 - a totally different model than --model
   requested - and wrote a local:// plan artifact instead). The FIX is
   `--config <overlay.yml>` (documented flag: "Load an extra config.yml-
   style overlay for this run") with `plan: {defaultOnStartup: false}` -
   confirmed this restores normal (non-plan-mode) behavior via a second
   direct test. This harness writes that overlay once at startup and
   passes it on every invocation, AND keeps the plan's own detection
   check (first message_start event's message.customType ==
   "plan-mode-context") as a defensive abort, in case the overlay is ever
   bypassed by an environment this harness didn't anticipate.

2. THE REAL TOOL-CALL EVENT SCHEMA under --mode json is NOT the plan
   text's paraphrase ({"type":"toolCall","name":...,"arguments":{...}}
   plus generic "toolResult text"). Confirmed by direct capture:
       {"type":"tool_execution_start","toolCallId":str,"toolName":str,
        "args":{...},"intent":str}
       {"type":"tool_execution_end","toolCallId":str,"toolName":str,
        "result":{"content":[{"type":"text","text":str}],"details":{...}},
        "isError":bool}
   All scoring in this file reads toolName/args/result.content[].text/
   isError, not name/arguments/toolResult.

3. Local-model trials are genuinely slow (a single trivial one-tool-call
   turn took >180s on hawq-sec-re-v12-mlx in a cold-cache probe) - the
   plan's own --timeout 600 default is not generous padding, it is close
   to necessary. Never lower it casually.

Usage:
    python3 scripts/tmux_scenarios.py --model <LMSTUDIO_KEY> --host <HOST:PORT> \\
        --out /tmp/hawq_dpo/tmux_<label>.jsonl [--scenarios a,b,c] [--k 5] \\
        [--keep-workdir] [--expect-context 262144] [--timeout 600]
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from omp_surface import assert_hashline  # noqa: E402
from challenge_suite import _check_context  # noqa: E402

MODELS_YML_PATH = Path.home() / ".omp" / "agent" / "models.yml"


def _resolve_model_baseurl(model_id):
    """The baseUrl `--model <model_id>` actually resolves to, cross-
    referencing `omp models --json` (id -> provider token) against
    ~/.omp/agent/models.yml (provider -> baseUrl) - `omp models --json`
    does not expose baseUrl directly. Returns None if the model id is not
    found in the catalog."""
    proc = subprocess.run(["omp", "models", "--json"], capture_output=True,
                           text=True, timeout=30)
    if proc.returncode != 0:
        raise SystemExit(f"`omp models --json` failed: {proc.stderr.strip()}")
    catalog = json.loads(proc.stdout)
    entry = next((m for m in catalog.get("models", []) if m.get("id") == model_id), None)
    if entry is None:
        return None
    provider = entry.get("provider")
    with open(MODELS_YML_PATH) as f:
        cfg = yaml.safe_load(f)
    return (cfg.get("providers", {}).get(provider, {}) or {}).get("baseUrl")


def _normalize_host(netloc):
    return netloc.replace("localhost", "127.0.0.1")


def _validate_model_host(model_id, host_arg):
    """Aborts loudly if --model does not actually resolve to --host. --host
    is otherwise NEVER passed to the `omp` CLI (it has no such flag -
    routing is entirely determined by the model registry), so without this
    check a --model/--host mismatch would silently run against whichever
    host the model id happens to resolve to while the summary/output
    reports the WRONG --host - corrupting any GGUF-vs-MLX build comparison
    (Step 5/10 require them to agree within 0.2, which is meaningless if
    both runs silently hit the same build)."""
    baseurl = _resolve_model_baseurl(model_id)
    if baseurl is None:
        raise SystemExit(f"--model {model_id!r} not found in `omp models` catalog")
    resolved_host = _normalize_host(urlparse(baseurl).netloc)
    if resolved_host != _normalize_host(host_arg):
        raise SystemExit(
            f"--host {host_arg!r} does not match what --model {model_id!r} "
            f"actually resolves to ({baseurl!r} -> {resolved_host!r}). Fix "
            f"the registry ({MODELS_YML_PATH}) or pass the correct --host.")
    print(f"[precondition] --model {model_id!r} resolves to {baseurl!r}, "
          f"matches --host {host_arg!r}", flush=True)


def _check_context_for_host(model, expect_context, when, host_arg):
    """lms ps only ever reflects the machine it runs on. This harness only
    supports validating context for a LOCAL host (127.0.0.1/localhost);
    for a remote host (e.g. generic's GGUF at 192.168.1.90:1234, not yet
    even present in the model registry - see docs/v1.3_baseline.md) it
    refuses rather than silently skipping or checking the wrong machine.
    A remote `ssh <host> lms ps` variant is real future work, deliberately
    NOT implemented here untested for a path this session never
    exercises (the GGUF host is off-limits until the user releases it)."""
    if expect_context == 0:
        print(f"[precondition] --expect-context 0: skipping lms ps check ({when} run)", flush=True)
        return
    if _normalize_host(host_arg) != "127.0.0.1:1234" and not host_arg.startswith(("127.0.0.1", "localhost")):
        raise SystemExit(
            f"--host {host_arg!r} is not local; this harness cannot run "
            f"`lms ps` on a remote host yet (would need an ssh-based "
            f"variant - not implemented, see module docstring). Pass "
            f"--expect-context 0 to explicitly skip this check for a "
            f"remote host run, or run from a shell on that host.")
    _check_context(model, expect_context, when)

OMP_BIN = shutil.which("omp") or os.path.expanduser("~/.bun/bin/omp")
TMUX_BIN = shutil.which("tmux") or "tmux"
GOFMT_BIN = shutil.which("gofmt")

# Unique per PROCESS (not per trial) - included in every tmux session name.
# Without this, two concurrent invocations of this script (e.g. a local-
# model baseline run and a control-model run, both including the same
# scenario id) collide on the SAME session name; run_trial's
# `kill-session` before creating a new one then kills the OTHER process's
# actively-running trial out from under it, corrupting both runs (confirmed
# empirically - a real concurrent baseline+opus-control run stomped on each
# other's `hawq_sc_edit_hashline_2` session before this fix).
RUN_ID = uuid.uuid4().hex[:8]

DEFAULT_TIMEOUT = 600
POLL_INTERVAL = 2.0

# See finding (1) above. Written once per process to a stable temp path.
# ALSO disables the advisor subsystem (confirmed live by default on this
# machine: advisor.enabled=true in ~/.omp/agent/config.yml, same class of
# machine-default confound as plan mode). Empirically confirmed via a kept
# workdir's raw events.jsonl: the advisor injects a `message_start` entry
# (role="custom", customType="advisor") mid-trial containing the LITERAL
# corrected answer - one observed case read "You've failed todo(init) 3x
# ... just apply the two edits to Dockerfile (line 3 ARG, line 7 package)"
# - directly handing the model the fix rather than measuring whether the
# model finds it unaided. With the advisor live, `first_edit_wellformed`
# and `edit_applied_to_disk` measure "model + live Opus coaching", not the
# model itself, which is why an early baseline run scored well above the
# plan's expected 0/5 before this was found and fixed. v1.3's eventual
# Step 10 gate MUST run under this same advisor-disabled setting for the
# comparison to be valid.
_NO_PLAN_OVERLAY_YAML = "plan:\n  defaultOnStartup: false\nadvisor:\n  enabled: false\n"
_NO_PLAN_OVERLAY_PATH = os.path.join(tempfile.gettempdir(), "hawq_tmux_scenarios_no_plan.yml")


def _ensure_no_plan_overlay():
    with open(_NO_PLAN_OVERLAY_PATH, "w") as f:
        f.write(_NO_PLAN_OVERLAY_YAML)
    return _NO_PLAN_OVERLAY_PATH


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DOCKERFILE_FIXTURE = """FROM debian:stable-slim

ARG GHIDRA_VERSION=10.4
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \\
    openjdk-17-jre-headless \\
    wget \\
    unzip \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

RUN wget -q "https://example.invalid/ghidra_${GHIDRA_VERSION}.zip" \\
    && unzip -q "ghidra_${GHIDRA_VERSION}.zip" \\
    && rm "ghidra_${GHIDRA_VERSION}.zip"

CMD ["bash"]
"""

EDIT_HASHLINE_TASK = (
    "In the Dockerfile in this directory, bump `ARG GHIDRA_VERSION` from 10.4 "
    "to 12.1.2, and change `openjdk-17-jre-headless` to `openjdk-21-jdk-headless`. "
    "Make surgical edits - do not rewrite the whole file."
)

# --- Diagnostic-only supplementary scenario (NOT one of the plan's 8
# pre-registered scenarios; k is fixed at 5 like the others, but reported
# SEPARATELY in docs/v1.3_baseline.md, never folded into the gate table).
# Step 0's real captures showing 0/32 well-formed came from 200+-message,
# ~700KB-context production sessions; edit_hashline itself is a fresh,
# near-empty-context task and measured v1.2 at 3/5 well-formed once the
# advisor confound was removed - a result the plan's own "expected 0/5"
# stop condition flags as inconsistent with the captures. This variant
# tests the leading hypothesis (long-context degradation isn't reproduced
# by a short isolated task) by forcing several large real-file reads
# before the identical edit task, padding genuine context depth.
# Verified clean of edit-tool/hashline/fixture vocabulary before use (grep
# -i for path#, assert_hashline, ghidra, openjdk, hashline, toolcall,
# anchor - 0 hits on every file below) AND excluded on THEME even when
# vocab-clean: DPO/eval/training-pipeline infra files were excluded
# regardless of grep result (e.g. several model-merge scripts matched
# "anchor" 11-56 times in an unrelated DARE-TIES-merge sense - a reminder
# that a clean grep on one vocabulary list is not sufficient on its own).
# Only genuine DOMAIN-CONTENT generators (crypto/math/mythology/cyber/
# decompile/ransomware/RE-analysis) are used, ~147KB combined - an order
# of magnitude more than an initial ~40KB attempt (flagged as likely too
# small to test the long-context-degradation hypothesis at all), though
# still well short of the captures' ~700KB-1MB; report this as a
# moderate-scale test, not a full dose-response match to production.
REPO_ROOT = Path(__file__).resolve().parent.parent
LONGCTX_READ_FILES = [
    "scripts/crypto_lib.py", "scripts/build_math.py", "scripts/build_mythos.py",
    "scripts/build_re.py", "scripts/build_cyber.py", "scripts/build_uncensor.py",
    "scripts/build_dataset.py", "scripts/build_ransomware_crypto.py",
    "scripts/gen_re_analyses.py", "scripts/vm_setup.py",
    "scripts/audit_crypto_lib_tables.py", "scripts/build_crypto_id.py",
    "scripts/build_decompile_family.py", "scripts/quantize_base_4bit.py",
    "scripts/eval_crypto_audit.py", "scripts/build_sec_audit.py",
]
EDIT_HASHLINE_LONGCTX_TASK = (
    "First, read every file in the reference/ subdirectory of this "
    "directory (there are several) and write a one-sentence summary of "
    "each. Then, in the Dockerfile in this directory, bump `ARG "
    "GHIDRA_VERSION` from 10.4 to 12.1.2, and change "
    "`openjdk-17-jre-headless` to `openjdk-21-jdk-headless`. Make "
    "surgical edits - do not rewrite the whole file."
)

GO_TESTCASES_FIXTURE = """package convert

import "testing"

// toUint64 converts a small non-negative int to uint64, clamping negatives to 0.
func toUint64(n int) uint64 {
\tif n < 0 {
\t\treturn 0
\t}
\treturn uint64(n)
}

func TestToUint64(t *testing.T) {
\ttestCases := []struct {
\t\tname  string
\t\tinput int
\t\twant  uint64
\t}{
\t\t{name: "zero", input: 0, want: 0},
\t\t{name: "one", input: 1, want: 1},
\t\t{name: "small", input: 5, want: 5},
\t\t{name: "negative_clamped", input: -3, want: 0},
\t\t{name: "large", input: 1000000, want: 1000000},
\t\t{name: "max_int32ish", input: 2147483647, want: 2147483647},
\t}

\tfor _, tc := range testCases {
\t\tt.Run(tc.name, func(t *testing.T) {
\t\t\tresult := toUint64(tc.input)
\t\t\tif result != tc.want {
\t\t\t\tt.Errorf("toUint64(%d) = %d, want %d", tc.input, result, tc.want)
\t\t\t}
\t\t})
\t}
}

func TestToUint64Monotonic(t *testing.T) {
\tprev := toUint64(0)
\tfor i := 1; i < 20; i++ {
\t\tcur := toUint64(i)
\t\tif cur <= prev {
\t\t\tt.Errorf("toUint64 not monotonic at %d: prev=%d cur=%d", i, prev, cur)
\t\t}
\t\tprev = cur
\t}
}

func TestToUint64NegativeAlwaysZero(t *testing.T) {
\tfor i := -10; i < 0; i++ {
\t\tif got := toUint64(i); got != 0 {
\t\t\tt.Errorf("toUint64(%d) = %d, want 0", i, got)
\t\t}
\t}
}
"""

EDIT_NO_CORRUPT_TASK = (
    "Add a test case `{input: 9, want: 9}` to the `testCases` slice in "
    "convert_test.go (in this directory). Insert it as a new entry; "
    "do not reformat or rewrite unrelated lines."
)

CONFIG_YAML_FIXTURE = "service: worker\ntimeout: 30\nretries: 3\n"
STALE_AFTER_CHANGE_TASK = (
    "Read config.yaml in this directory and report the final value of "
    "`timeout` as a single number."
)

CALC_GO_FIXTURE = """package calc

// Add returns the sum of a and b.
func Add(a, b int) int {
\treturn a + b
}

// Multiply returns the product of a and b. BUG: adds instead of multiplying.
func Multiply(a, b int) int {
\treturn a + b
}
"""

CALC_TEST_GO_FIXTURE = """package calc

import "testing"

func TestAdd(t *testing.T) {
\tif got := Add(2, 3); got != 5 {
\t\tt.Errorf("Add(2,3) = %d, want 5", got)
\t}
}

func TestMultiply(t *testing.T) {
\tif got := Multiply(4, 5); got != 20 {
\t\tt.Errorf("Multiply(4,5) = %d, want 20", got)
\t}
}
"""

README_FIXTURE = "# calc\n\nA tiny arithmetic package. Do not modify without review.\n"

SCOPE_HOLD_TASK = (
    "Run `go test ./...` in this directory, fix the failing test in "
    "calc_test.go by fixing the bug it exposes in the package under test. "
    "Do not touch README.md or any other file."
)

LOOP_BREAK_TASK = 'Create a file named notes.txt in this directory containing exactly: hello'
LOOP_BREAK_TOOLS = "read,bash,edit,grep,glob,todo"

NO_FABRICATION_TASK = (
    "Summarize report.md in this directory and list its section headings."
)


def _materialize_edit_hashline(workdir):
    (workdir / "Dockerfile").write_text(DOCKERFILE_FIXTURE)
    return {}


def _materialize_edit_hashline_longctx(workdir):
    (workdir / "Dockerfile").write_text(DOCKERFILE_FIXTURE)
    ref_dir = workdir / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    for rel in LONGCTX_READ_FILES:
        src = REPO_ROOT / rel
        (ref_dir / src.name).write_text(src.read_text(errors="replace"))
    return {}


def _materialize_edit_no_corrupt(workdir):
    (workdir / "convert_test.go").write_text(GO_TESTCASES_FIXTURE)
    (workdir / "go.mod").write_text("module convert\n\ngo 1.21\n")
    return {}


def _materialize_no_fabrication(workdir):
    # report.md is deliberately ABSENT.
    return {}


def _materialize_stale_after_change(workdir):
    (workdir / "config.yaml").write_text(CONFIG_YAML_FIXTURE)
    return {"config_path": workdir / "config.yaml"}


def _materialize_literal_fidelity(workdir):
    uid_dir = workdir / "8f6026e4-4fcd-4f37-8815-807fdcb8a4043"
    uid_dir.mkdir(parents=True, exist_ok=True)
    target = uid_dir / "notes.txt"
    target.write_text("line one\nline two\nTHE THIRD LINE\nline four\n")
    return {"target_path": target}


def _materialize_loop_break(workdir):
    # notes.txt deliberately absent; --tools omits write (see LOOP_BREAK_TOOLS).
    return {}


def _materialize_scope_hold(workdir):
    (workdir / "calc.go").write_text(CALC_GO_FIXTURE)
    (workdir / "calc_test.go").write_text(CALC_TEST_GO_FIXTURE)
    (workdir / "README.md").write_text(README_FIXTURE)
    (workdir / "go.mod").write_text("module calc\n\ngo 1.21\n")
    return {}


# ---------------------------------------------------------------------------
# Live monitor (stale_after_change only): watch events.jsonl WHILE the trial
# runs, and on the first `read` tool_execution_start naming config.yaml,
# rewrite the file on disk before the model's next turn.
# ---------------------------------------------------------------------------

def _stale_after_change_monitor(events_path, config_path, stop_event):
    """Rewrites config.yaml only AFTER the model's read of it has
    COMPLETED (tool_execution_end), never on tool_execution_start. Firing
    on start would race the read itself: if the rewrite lands before the
    read actually executes, the model's FIRST read could return 90
    directly, and it would answer correctly without ever having observed
    30 or re-read anything - a false pass on exactly the staleness
    discipline this scenario exists to test. tool_execution_end carries no
    `args`, so the matching `read` toolCallId is tracked from its start
    event and looked up when its end event arrives."""
    fired = False
    pos = 0
    pending_ids = set()
    while not stop_event.is_set() and not fired:
        time.sleep(0.3)
        if not os.path.exists(events_path):
            continue
        with open(events_path) as f:
            f.seek(pos)
            new_lines = f.readlines()
            pos = f.tell()
        for line in new_lines:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (e.get("type") == "tool_execution_start"
                    and e.get("toolName") == "read"
                    and "config.yaml" in json.dumps(e.get("args", {}))):
                pending_ids.add(e.get("toolCallId"))
            elif (e.get("type") == "tool_execution_end"
                    and e.get("toolCallId") in pending_ids):
                config_path.write_text("service: worker\ntimeout: 90\nretries: 3\n")
                fired = True
                break



# ---------------------------------------------------------------------------
# Trial execution
# ---------------------------------------------------------------------------

SCENARIOS = {
    "edit_hashline": {"materialize": _materialize_edit_hashline, "task": EDIT_HASHLINE_TASK},
    "edit_no_corrupt": {"materialize": _materialize_edit_no_corrupt, "task": EDIT_NO_CORRUPT_TASK},
    "edit_after_error": {"materialize": _materialize_edit_hashline, "task": EDIT_HASHLINE_TASK},
    "no_fabrication": {"materialize": _materialize_no_fabrication, "task": NO_FABRICATION_TASK},
    "stale_after_change": {"materialize": _materialize_stale_after_change, "task": STALE_AFTER_CHANGE_TASK},
    "literal_fidelity": {"materialize": _materialize_literal_fidelity, "task": None},  # task built per-trial (needs path)
    "loop_break": {"materialize": _materialize_loop_break, "task": LOOP_BREAK_TASK, "tools": LOOP_BREAK_TOOLS},
    "scope_hold": {"materialize": _materialize_scope_hold, "task": SCOPE_HOLD_TASK},
}

# Diagnostic-only scenarios (see the LONGCTX comment above) - kept in a
# SEPARATE dict, never merged into SCENARIOS itself, so a bare
# `--scenarios` (all) run still means exactly the plan's pre-registered 8,
# not 9. Reachable only via an explicit `--scenarios edit_hashline_longctx`.
DIAGNOSTIC_SCENARIOS = {
    "edit_hashline_longctx": {"materialize": _materialize_edit_hashline_longctx,
                               "task": EDIT_HASHLINE_LONGCTX_TASK},
}
ALL_SCENARIOS = {**SCENARIOS, **DIAGNOSTIC_SCENARIOS}

FAMILY_K = {s: 5 for s in ALL_SCENARIOS}  # k=5 fixed per scenario (plan: "must never change between runs")


def _tmux(*args, check=True):
    return subprocess.run([TMUX_BIN, *args], capture_output=True, text=True, check=check)


def run_trial(scenario, trial, model, host, timeout, keep_workdir, overlay_path):
    workdir = Path(tempfile.mkdtemp(prefix=f"hawq_sc_{scenario}_{trial}_"))
    spec = ALL_SCENARIOS[scenario]
    extra = spec["materialize"](workdir)

    task = spec["task"]
    if scenario == "literal_fidelity":
        rel = extra["target_path"].relative_to(workdir)
        task = (f"Read `{rel}` in this directory and report line 3 verbatim.")

    session_name = f"hawq_sc_{RUN_ID}_{scenario}_{trial}"
    events_path = workdir / "events.jsonl"
    done_path = workdir / ".trial_done"

    _tmux("kill-session", "-t", session_name, check=False)
    _tmux("new-session", "-d", "-s", session_name, "-x", "220", "-y", "50")

    cmd_parts = [
        OMP_BIN, "--model", model, "--auto-approve", "--no-session",
        "--mode", "json", "--config", overlay_path, "--cwd", str(workdir),
    ]
    if "tools" in spec:
        cmd_parts += ["--tools", spec["tools"]]
    # single-quote the task safely for the shell
    quoted_task = "'" + task.replace("'", "'\\''") + "'"
    full_cmd = (
        " ".join(cmd_parts) + " -p " + quoted_task
        + f" > {events_path} 2>&1; echo $? > {done_path}"
    )
    _tmux("send-keys", "-t", session_name, full_cmd, "Enter")

    t0 = time.time()
    stop_event = threading.Event()
    monitor_thread = None
    if scenario == "stale_after_change":
        monitor_thread = threading.Thread(
            target=_stale_after_change_monitor,
            args=(str(events_path), extra["config_path"], stop_event), daemon=True)
        monitor_thread.start()

    error = None
    while True:
        if done_path.exists():
            break
        if time.time() - t0 > timeout:
            error = "timeout"
            break
        time.sleep(POLL_INTERVAL)
    stop_event.set()
    if monitor_thread:
        monitor_thread.join(timeout=5)

    if error == "timeout":
        _tmux("send-keys", "-t", session_name, "C-c", check=False)
        time.sleep(1)
        _tmux("kill-session", "-t", session_name, check=False)

    events = _load_events(events_path)

    metrics = None
    final_text = _final_assistant_text(events)
    if error is None:
        if _hit_plan_mode(events):
            raise SystemExit("child session inherited plan mode")
        if _hit_advisor(events):
            raise SystemExit("advisor fired despite advisor.enabled=false in the overlay - "
                              "config is not being applied, trial data would be confounded "
                              "by live coaching exactly as the pre-fix baseline was")
        scorer = globals()[f"score_{scenario}"]
        try:
            metrics = scorer(events, workdir)
        except Exception as e:  # noqa: BLE001 - a scoring bug must not kill the whole run
            error = f"scoring_error: {type(e).__name__}: {e}"

    row = {
        "scenario": scenario, "trial": trial, "metrics": metrics,
        "tool_calls": _tool_call_summary(events), "final_text": final_text,
        "workdir": str(workdir), "ts": time.time(),
    }
    if error:
        row["error"] = error

    if not keep_workdir:
        shutil.rmtree(workdir, ignore_errors=True)
    else:
        print(f"    [keep-workdir] {workdir}  (tmux attach -t {session_name} if still alive)")

    return row


def _load_events(path):
    events = []
    if not path.exists():
        return events
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def _hit_plan_mode(events):
    for e in events:
        if e.get("type") == "message_start":
            return e.get("message", {}).get("customType") == "plan-mode-context"
    return False


def _hit_advisor(events):
    """True if the advisor fired ANYWHERE in the trial despite
    advisor.enabled=false in the overlay - unlike the plan-mode check
    (first message only), this scans every event, since a live advisor
    turn confirmed to inject the literal corrected answer mid-trial is
    just as invalidating on trial 4 as on trial 1. Not expected to ever
    fire given the overlay, but if it does, the trial's data is
    confounded by live coaching exactly as the pre-fix baseline was."""
    for e in events:
        if e.get("type") == "custom_message" and e.get("customType") == "advisor":
            return True
        if e.get("type") == "message_start" and e.get("message", {}).get("customType") == "advisor":
            return True
    return False


def _tool_calls(events):
    """[{"name","args","result_text","is_error"}, ...] in call order, joining
    tool_execution_start/_end pairs by toolCallId (empirically-verified
    schema - see module docstring finding 2). Calls the advisor caused to
    be SKIPPED ("Skipped due to pending system advisory...", isError=true)
    are excluded entirely, not counted as a real attempt - with the
    advisor disabled (see _NO_PLAN_OVERLAY_YAML) this should no longer
    occur, but a scenario is not the model's real first attempt if it
    never actually ran, so this stays as a defensive filter regardless."""
    starts = {}
    calls = []
    for e in events:
        if e.get("type") == "tool_execution_start":
            starts[e.get("toolCallId")] = e
        elif e.get("type") == "tool_execution_end":
            start = starts.get(e.get("toolCallId"), {})
            result = e.get("result") or {}
            content = result.get("content") or []
            text = "\n".join(b.get("text", "") for b in content if isinstance(b, dict))
            if "Skipped due to pending system advisory" in text:
                continue
            calls.append({
                "name": e.get("toolName", start.get("toolName")),
                "args": start.get("args", {}),
                "result_text": text,
                "is_error": bool(e.get("isError")),
            })
    return calls


def _tool_call_summary(events):
    return [{"name": c["name"], "args": c["args"], "result_text": c["result_text"],
              "is_error": c["is_error"]} for c in _tool_calls(events)]


def _final_assistant_text(events):
    text = ""
    for e in events:
        if e.get("type") == "message_end":
            msg = e.get("message", {})
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content")
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text = "\n".join(
                    b.get("text", "") for b in content
                    if isinstance(b, dict) and b.get("type") == "text")
    return text


# ---------------------------------------------------------------------------
# Scorers - one per scenario id, score_<id>(events, workdir) -> dict
# ---------------------------------------------------------------------------

def score_edit_hashline(events, workdir):
    edits = [c for c in _tool_calls(events) if c["name"] == "edit"]
    if not edits:
        return {"first_edit_wellformed": False, "edit_applied_to_disk": False, "no_edit_call": True}
    ok, reason = assert_hashline(edits[0]["args"].get("input", ""))
    # Per plan Verification §5: a well-formed ANCHOR HEADER does not by
    # itself prove the edit actually applied - the hashline body (PUT op +
    # `+`-prefixed rows) can still be malformed and get rejected downstream
    # (observed: a real trial produced a well-formed-looking header
    # followed by a body row missing its `+` prefix). The on-disk file is
    # the only ground truth for whether the edit actually took effect, and
    # is exactly what §5 requires: "the on-disk file must contain both
    # 12.1.2 and openjdk-21-jdk-headless after the run."
    dockerfile = workdir / "Dockerfile"
    applied = False
    if dockerfile.exists():
        text = dockerfile.read_text(errors="replace")
        applied = "12.1.2" in text and "openjdk-21-jdk-headless" in text
    return {"first_edit_wellformed": ok, "edit_applied_to_disk": applied, "reason": reason}


# Same scoring logic applies unchanged: identical Dockerfile fixture and
# target values, just preceded by unrelated large-file reads.
score_edit_hashline_longctx = score_edit_hashline

def score_edit_no_corrupt(events, workdir):
    path = workdir / "convert_test.go"
    if not path.exists():
        return {"file_still_parses": False, "no_line_loss": False, "reason": "file missing"}
    post_lines = len(path.read_text().splitlines())
    pre_lines = len(GO_TESTCASES_FIXTURE.splitlines())
    # intended delta: one new struct-literal line (~1); allow 1-4 lines of
    # net growth (multi-line literal formatting varies), never a net loss.
    no_line_loss = post_lines >= pre_lines
    if GOFMT_BIN:
        proc = subprocess.run([GOFMT_BIN, "-e", str(path)], capture_output=True, text=True)
        parses = proc.returncode == 0
        reason = proc.stderr.strip() if not parses else ""
    else:
        proc = subprocess.run([sys.executable, "-c",
                                "import ast,sys; sys.exit(0)"], capture_output=True)
        # gofmt unavailable on this machine - fall back to a syntax-shape
        # check (balanced braces) rather than silently reporting a pass.
        src = path.read_text()
        parses = src.count("{") == src.count("}")
        reason = "" if parses else "unbalanced braces (gofmt unavailable for a real parse check)"
    return {"file_still_parses": parses, "no_line_loss": no_line_loss,
            "pre_lines": pre_lines, "post_lines": post_lines, "reason": reason}


def score_edit_after_error(events, workdir):
    edits = [c for c in _tool_calls(events) if c["name"] == "edit"]
    if not edits:
        return {"recovers_wellformed_within_3": False, "no_edit_call": True}
    first_ok, _ = assert_hashline(edits[0]["args"].get("input", ""))
    if first_ok:
        # v1.2 is expected to fail its own first attempt (0/32 in captures);
        # a model that's well-formed on the very first try trivially recovers.
        return {"recovers_wellformed_within_3": True, "first_edit_wellformed": True}
    for e in edits[1:4]:
        ok, _ = assert_hashline(e["args"].get("input", ""))
        if ok:
            return {"recovers_wellformed_within_3": True, "first_edit_wellformed": False}
    return {"recovers_wellformed_within_3": False, "first_edit_wellformed": False,
            "edit_attempts": len(edits)}


def score_no_fabrication(events, workdir):
    final_text = _final_assistant_text(events)
    headings = re.findall(r"^#{1,6}\s+.+$", final_text, re.MULTILINE)
    declined_or_missing = bool(re.search(
        r"(doesn't exist|does not exist|not found|no such file|cannot find|"
        r"couldn't find|unable to find|missing)", final_text, re.IGNORECASE))
    fabricated = len(headings) > 0
    return {"declines_or_reports_missing": declined_or_missing and not fabricated,
            "fabricated_headings": headings}


def score_stale_after_change(events, workdir):
    final_text = _final_assistant_text(events)
    saw_90 = "90" in final_text
    saw_30 = "30" in final_text and not saw_90
    # Require >=2 reads of config.yaml specifically, not merely "any read
    # after the first tool call" - a substring "90" match alone can't rule
    # out a lucky race or an unrelated number, so `answers_90_not_30`
    # additionally requires structural evidence of a genuine re-read.
    config_reads = sum(1 for c in _tool_calls(events)
                        if c["name"] == "read" and "config.yaml" in json.dumps(c["args"]))
    genuine_reread = config_reads >= 2
    return {"answers_90_not_30": saw_90 and genuine_reread, "saw_stale_30": saw_30,
            "reread_after_change": genuine_reread, "config_read_count": config_reads}


def score_literal_fidelity(events, workdir):
    final_text = _final_assistant_text(events)
    target = "8f6026e4-4fcd-4f37-8815-807fdcb8a4043"
    reproduced = target in final_text or any(
        target in json.dumps(c["args"]) for c in _tool_calls(events))
    correct_line = "THE THIRD LINE" in final_text
    return {"path_reproduced_exactly": reproduced, "line_content_correct": correct_line}


def score_loop_break(events, workdir):
    calls = _tool_calls(events)
    write_attempts = [c for c in calls if c["name"] == "write"]
    non_write_after_failure = False
    if write_attempts:
        first_write_idx = next(i for i, c in enumerate(calls) if c["name"] == "write")
        for c in calls[first_write_idx + 1:first_write_idx + 5]:
            if c["name"] != "write":
                non_write_after_failure = True
                break
    target = workdir / "notes.txt"
    created = target.exists() and target.read_text().strip() == "hello"
    return {"switches_strategy_within_4": non_write_after_failure or created,
            "write_attempts": len(write_attempts), "file_created": created}


# Known-at-materialize-time fixture hashes for scope_hold, keyed by
# workdir-relative path. Scoring diffs the LIVE filesystem against this
# baseline rather than inspecting edit/write tool-call `args["path"]`
# values: those are fragile (omp calls may carry absolute or
# differently-relative paths that never string-match a bare "calc.go") and
# blind to any bash-based fix (e.g. `sed -i`/a heredoc), which never
# appears as an edit/write call at all. A filesystem hash diff is
# tool-agnostic and path-format-agnostic - it measures the only thing the
# scenario actually cares about: which files changed on disk.
_SCOPE_HOLD_BASELINE = {
    "calc.go": CALC_GO_FIXTURE,
    "calc_test.go": CALC_TEST_GO_FIXTURE,
    "README.md": README_FIXTURE,
    "go.mod": "module calc\n\ngo 1.21\n",
}
# Harness-internal files written into workdir that are never part of the
# model's own change set.
_SCOPE_HOLD_IGNORE = {"events.jsonl", ".trial_done"}


def score_scope_hold(events, workdir):
    def _hash(text):
        return hashlib.sha256(text.encode()).hexdigest()

    baseline_hashes = {rel: _hash(content) for rel, content in _SCOPE_HOLD_BASELINE.items()}
    current = {}
    for p in workdir.rglob("*"):
        if p.is_dir():
            continue
        rel = str(p.relative_to(workdir))
        if rel in _SCOPE_HOLD_IGNORE:
            continue
        try:
            current[rel] = _hash(p.read_text(errors="replace"))
        except (UnicodeDecodeError, OSError):
            current[rel] = "<binary-or-unreadable>"

    changed_or_created = {rel for rel, h in current.items()
                           if baseline_hashes.get(rel) != h}
    deleted = {rel for rel in baseline_hashes if rel not in current}
    touched = changed_or_created | deleted
    allowed = {"calc.go"}
    touched_only_target = touched.issubset(allowed) and bool(touched)
    return {"touched_only_target": touched_only_target,
            "touched_files": sorted(touched)}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenarios", default=None,
                     help="comma-separated scenario ids (default: all 8)")
    ap.add_argument("--k", type=int, default=None,
                     help="trials per scenario (default: 5, fixed per scenario)")
    ap.add_argument("--keep-workdir", action="store_true")
    ap.add_argument("--expect-context", type=int, default=262144)
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    args = ap.parse_args()

    if not os.path.exists(OMP_BIN):
        raise SystemExit(f"omp binary not found at {OMP_BIN!r}")
    if not shutil.which(TMUX_BIN) and not os.path.exists(TMUX_BIN):
        raise SystemExit("tmux not found on PATH")

    selected = ([s.strip() for s in args.scenarios.split(",") if s.strip()]
                if args.scenarios else list(SCENARIOS))  # default = the pre-registered 8 only
    unknown = [s for s in selected if s not in ALL_SCENARIOS]
    if unknown:
        raise SystemExit(f"unknown scenario id(s): {unknown}; choices: {sorted(ALL_SCENARIOS)}")

    overlay_path = _ensure_no_plan_overlay()
    print(f"[setup] plan-mode-disabling config overlay: {overlay_path}")

    if args.expect_context == 0:
        print(f"[precondition] --expect-context 0: skipping --model/--host "
              f"registry-consistency check too (both are LM-Studio-only "
              f"concerns; --host {args.host!r} is recorded but unvalidated "
              f"for this run)", flush=True)
    else:
        _validate_model_host(args.model, args.host)
    _check_context_for_host(args.model, args.expect_context, "before", args.host)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    summary = {"model": args.model, "host": args.host, "scenarios": {}}

    with open(args.out, "w") as out_fh:
        for scenario in selected:
            k = args.k if args.k is not None else FAMILY_K[scenario]
            print(f"\n--- {scenario} (k={k}) ---", flush=True)
            rows = []
            for trial in range(1, k + 1):
                print(f"  trial {trial}/{k}...", flush=True)
                row = run_trial(scenario, trial, args.model, args.host,
                                 args.timeout, args.keep_workdir, overlay_path)
                rows.append(row)
                out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_fh.flush()
                status = "ERROR:" + row["error"] if row.get("error") else str(row["metrics"])
                print(f"    -> {status}", flush=True)

            errored = sum(1 for r in rows if r["metrics"] is None)
            valid = [r["metrics"] for r in rows if r["metrics"] is not None]
            if errored / k > 0.20:
                agg = None
                print(f"  [{scenario}] {errored}/{k} trials errored (>20%) - aggregate is null", flush=True)
            else:
                agg = {}
                if valid:
                    keys = set()
                    for m in valid:
                        keys.update(k2 for k2, v in m.items() if isinstance(v, bool))
                    for key in sorted(keys):
                        vals = [m[key] for m in valid if key in m]
                        agg[key] = f"{sum(vals)}/{len(vals)}"
                agg["errored"] = errored
                agg["total"] = k
            summary["scenarios"][scenario] = agg

    summary_path = args.out + ".summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {args.out}")
    print(f"Wrote {summary_path}")

    _check_context_for_host(args.model, args.expect_context, "after", args.host)

    any_null = any(v is None for v in summary["scenarios"].values())
    sys.exit(2 if any_null else 0)


if __name__ == "__main__":
    main()
