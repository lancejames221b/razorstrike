#!/usr/bin/env python3
"""Self-healing driver for the RazorStrike v2 full training run.

Loops:
  - session dead/missing -> reprovision (new G4, clone, vm_setup, HF login) + launch
  - session alive but training process dead (crash/OOM/manual kill) -> relaunch
    on the SAME session (no reprovision needed, much cheaper)
  - session alive and training running -> poll for errors/completion

Crash-loop detection: after every relaunch we check the Hub's global_step. If
it hasn't advanced (or is unreadable) across MAX_STUCK_CYCLES consecutive
relaunches, something is structurally broken and we halt for human review
instead of burning compute forever. Counters persist to a state file so a
driver-process crash+restart can't silently reset the guard.

IMPORTANT: this process is normally launched under `hub start` with
`restart: on-failure`, which means a plain `sys.exit(nonzero)` gets silently
RELAUNCHED, not halted - defeating any "exit for human review" intent. All
terminal-failure paths therefore go through `halt()`, which writes a sentinel
file checked at the top of every `main()` invocation (so even a hub-forced
restart just re-halts instead of doing real work) and fires an actual macOS
notification so a silent multi-day wedge surfaces to the user.
"""
import subprocess
import time
import sys
import json
import os

SESSION = "rs-g4"
ADAPTER_FULL = "lancejames221b/HAWQ-SEC-lora"
MAX_STUCK_CYCLES = 5
STATE_FILE = "/Volumes/Scratch/razorstrike-repo/.driver_state/state.json"
HALT_FILE = "/Volumes/Scratch/razorstrike-repo/.driver_state/HALT.txt"


def notify(title, message):
    """Soundless macOS banner - sound was disabled after the alert-storm
    incident, but a genuine halt should still surface visually rather than
    go completely silent until morning."""
    try:
        script = f'display notification "{message}" with title "{title}"'
        subprocess.run(["osascript", "-e", script], timeout=10)
    except Exception as e:
        print(f"[driver] WARNING: notify() failed: {e}", flush=True)


def halt(reason):
    """Terminal failure: write a sentinel file, notify, and exit. If hub
    restarts us anyway (restart: on-failure), main() sees the sentinel
    immediately on the next startup and re-halts instead of silently
    burning compute / provision cycles."""
    print(f"[driver] HALTING: {reason}", flush=True)
    try:
        with open(HALT_FILE, "w") as f:
            f.write(reason)
    except Exception as e:
        print(f"[driver] WARNING: could not write halt sentinel: {e}", flush=True)
    notify("RazorStrike training driver HALTED", reason[:200])
    sys.exit(1)


def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {"stuck_cycles": 0, "consecutive_provision_failures": 0, "last_known_step": None}


def save_state(state):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f)
    except Exception as e:
        print(f"[driver] WARNING: could not persist state: {e}", flush=True)


def get_hf_token(max_attempts=5):
    # Priority 1: explicitly-injected HF_TOKEN env var (set at `hub start` time).
    # Priority 2: a fixed /tmp path written by the launching shell - HOME-independent,
    #   so it works even if the hub-managed process gets a different/minimal HOME.
    # Priority 3: huggingface_hub's get_token() file resolution (depends on HOME,
    #   known to fail silently in a minimal-env hub-managed process - see commit history).
    env_tok = os.environ.get("HF_TOKEN", "").strip()
    if env_tok:
        return env_tok
    for token_file in ["/Volumes/Scratch/razorstrike-repo/.driver_state/hftoken.txt", "/tmp/_rs_hftoken.txt"]:
        if os.path.exists(token_file):
            with open(token_file) as f:
                file_tok = f.read().strip()
            if file_tok:
                print(f"[driver] HF_TOKEN loaded from {token_file}", flush=True)
                return file_tok
    print("[driver] HF_TOKEN not in env or token file, falling back to huggingface_hub.get_token() file resolution", flush=True)
    for attempt in range(max_attempts):
        r = subprocess.run(
            ["python3", "-c", "from huggingface_hub import get_token; print(get_token() or '')"],
            cwd="/Volumes/Scratch", capture_output=True, text=True
        )
        tok = r.stdout.strip()
        if tok:
            return tok
        print(f"[driver] HF_TOKEN read attempt {attempt+1}/{max_attempts} came back empty (stderr: {r.stderr[-200:]}), retrying in 5s", flush=True)
        time.sleep(5)
    halt("could not obtain a non-empty HF_TOKEN (env var, token file, AND get_token() resolution all failed after retries) - "
         "proceeding would launch training with no auth against private repos.")


if os.path.exists(HALT_FILE):
    with open(HALT_FILE) as f:
        _halt_reason = f.read()
    print(f"[driver] HALT sentinel present, refusing to run: {_halt_reason}", flush=True)
    notify("RazorStrike training driver still HALTED", _halt_reason[:200])
    sys.exit(0)

HF_TOKEN = get_hf_token()
print(f"[driver] HF_TOKEN acquired (len={len(HF_TOKEN)})", flush=True)

LAUNCH_PY = f'''
import subprocess, os, time
r0 = subprocess.run("HF_TOKEN='{HF_TOKEN}' python3 -c \\"from huggingface_hub import login; login('{HF_TOKEN}'); print('HF_LOGIN_OK')\\"", shell=True, capture_output=True, text=True)
print(r0.stdout, r0.stderr[-300:])
if "HF_LOGIN_OK" not in r0.stdout:
    print("HF_LOGIN_FAILED - aborting launch")
else:
    r1 = subprocess.run("test -d /content/razorstrike && cd /content/razorstrike && git fetch origin && git reset --hard origin/main 2>&1 || (rm -rf /content/razorstrike && git clone --depth 1 https://github.com/lancejames221b/razorstrike.git /content/razorstrike 2>&1)", shell=True, capture_output=True, text=True)
    print("SYNC:", r1.stdout[-300:])
    r2 = subprocess.run("cd /content/razorstrike && python3 scripts/vm_setup.py 2>&1 | tail -25", shell=True, capture_output=True, text=True, timeout=300)
    print("SETUP:", r2.stdout)
    subprocess.run("pkill -f train_lora 2>/dev/null", shell=True)
    time.sleep(2)
    env = os.environ.copy()
    env.update({{
        "BASE_REPO": "lancejames221b/HAWQ-v1",
        "DATA_REPO": "lancejames221b/hawq-sec-sft",
        "ADAPTER_REPO": "{ADAPTER_FULL}",
        "OUT_DIR": "/content/adapter",
        "MAXLEN": "3072",
        "TARGET_MLP": "0",
        "SAVE_STEPS": "250",
        "EVAL_STEPS": "250",
        "MAX_STEPS": "-1",
        "FORCE_CAUSAL_LM": "1",
        "HF_TOKEN": "{HF_TOKEN}",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }})
    # Only set RESUME=1 if the adapter repo already has a checkpoint-N/ dir to resume from.
    # The HubCheckpointPusher callback pushes checkpoints as checkpoint-N/ dirs at repo root,
    # not as last-checkpoint/ (which the old Trainer hub_strategy=checkpoint never created).
    try:
        from huggingface_hub import list_repo_files
        files = list_repo_files("{ADAPTER_FULL}", token="{HF_TOKEN}")
        ckpt_dirs = [f for f in files if f.startswith("checkpoint-")]
        if ckpt_dirs:
            env["RESUME"] = "1"
            print("RESUME=1 ({{}} checkpoints found on Hub)".format(len(ckpt_dirs)))
        else:
            print("RESUME not set (no checkpoint dirs on Hub - fresh run)")
    except Exception as e:
        print("RESUME not set (repo check failed: {{}})".format(type(e).__name__))
    cmd = "cd /content/razorstrike && nohup python3 -m scripts.train_lora > /content/train.log 2>&1 &"
    subprocess.Popen(cmd, shell=True, env=env)
    time.sleep(3)
    r3 = subprocess.run("pgrep -af train_lora || echo NOT_RUNNING", shell=True, capture_output=True, text=True)
    print("PROC:", r3.stdout.strip())
    print("TRAIN_LAUNCHED")
'''

POLL_PY = '''
import subprocess
r = subprocess.run("grep -E \\"OutOfMemory|RuntimeError|Traceback|TRAINING_COMPLETE\\" /content/train.log | tail -10", shell=True, capture_output=True, text=True)
print("SIGNALS:", r.stdout)
r2 = subprocess.run("tail -3 /content/train.log", shell=True, capture_output=True, text=True)
print("TAIL:", r2.stdout)
r3 = subprocess.run("pgrep -f train_lora >/dev/null && echo STILL_RUNNING || echo PROCESS_EXITED", shell=True, capture_output=True, text=True)
print(r3.stdout.strip())
'''


class _TimedOut:
    """Sentinel returned when a subprocess call hangs past its timeout, so
    callers can treat it as a normal (failed) result instead of an uncaught
    exception that would kill the whole driver process."""
    stdout = ""
    stderr = "TIMED_OUT"
    returncode = -1


def run(cmd, timeout=120):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"[driver] WARNING: command timed out after {timeout}s: {cmd[:100]}", flush=True)
        return _TimedOut()


def write(path, content):
    with open(path, "w") as f:
        f.write(content)


def session_alive():
    r = run(f"colab status -s {SESSION}", timeout=30)
    return ("IDLE" in r.stdout or "RUNNING" in r.stdout) and "not found" not in r.stdout.lower(), r.stdout


def get_local_step():
    """Ground-truth progress signal read directly from the live VM's
    train.log tqdm output (e.g. '2761/12538'), independent of Hub push
    lag. Returns None if the VM/session is unreachable or no step line
    is found yet (e.g. still in preprocessing)."""
    import re as _re
    write("/tmp/_rs_localstep.py", '''
import subprocess
r = subprocess.run("tail -c 4000 /content/train.log 2>&1", shell=True, capture_output=True, text=True)
print(r.stdout)
''')
    r = run(f"colab exec -s {SESSION} -f /tmp/_rs_localstep.py --timeout 30", timeout=45)
    matches = _re.findall(r"(\d+)/\d+", r.stdout)
    if matches:
        return int(matches[-1])
    return None


def get_hub_global_step():
    """Authoritative progress signal: query the Hub checkpoint directly,
    independent of anything happening on the (possibly dead) VM. Retries
    on transient failure; returns None only after all retries are exhausted,
    which callers treat as a stuck-signal (not a silent skip).

    Calls hf_hub_download IN-PROCESS rather than via a spawned `python3`
    subprocess: under a hub-managed process's minimal environment, a bare
    `python3` on PATH could resolve to an interpreter without
    huggingface_hub installed, silently producing empty stdout -> int('')
    failure on every attempt. Importing directly here removes that
    ambiguity entirely."""
    from huggingface_hub import list_repo_files, hf_hub_download
    import json as _json
    for attempt in range(3):
        try:
            files = list_repo_files(ADAPTER_FULL, token=HF_TOKEN)
            ckpt_dirs = sorted({f.split("/")[0] for f in files if f.startswith("checkpoint-")},
                               key=lambda s: int(s.split("-")[1]))
            if not ckpt_dirs:
                return None
            latest = ckpt_dirs[-1]
            p = hf_hub_download(ADAPTER_FULL, f"{latest}/trainer_state.json", token=HF_TOKEN)
            with open(p) as f:
                step = _json.load(f).get("global_step")
            return int(step)
        except Exception as e:
            print(f"[driver] hub global_step read attempt {attempt+1}/3 failed: {e}", flush=True)
            time.sleep(10)
    print("[driver] hub global_step unreadable after 3 attempts", flush=True)
    return None


def reprovision_and_launch():
    print(f"[driver] reprovisioning session {SESSION}...", flush=True)
    r = run(f"colab new -s {SESSION} --gpu G4", timeout=90)
    print("[driver] colab new:", r.stdout.strip(), flush=True)
    if "READY" not in r.stdout:
        print("[driver] provision may have failed, will retry next cycle:", r.stdout, r.stderr, flush=True)
        return False
    return relaunch_on_live_session()


def relaunch_on_live_session():
    write("/tmp/_rs_launch.py", LAUNCH_PY)
    r = run(f"colab exec -s {SESSION} -f /tmp/_rs_launch.py --timeout 300", timeout=330)
    print("[driver] launch output:", r.stdout[-1500:], flush=True)
    if "HF_LOGIN_FAILED" in r.stdout:
        print("[driver] HF login failed on the VM itself - treating as launch failure", flush=True)
        return False
    return "TRAIN_LAUNCHED" in r.stdout


def poll_once():
    write("/tmp/_rs_poll.py", POLL_PY)
    r = run(f"colab exec -s {SESSION} -f /tmp/_rs_poll.py --timeout 60", timeout=90)
    return r.stdout


def check_progress_or_die(state):
    """Mutates and persists state; halts for human review if stuck too long.

    Prefers the LOCAL training-log step (ground truth, read directly off the
    live VM) over the Hub checkpoint step, since the Hub push is async and can
    lag well behind actual progress (confirmed: 40-step lag observed). But the
    only thing that resets stuck_cycles is an ACTUAL step advance past
    last_known_step - local-reachable-but-stagnant, local-unreachable, and
    local-unknown all count identically as non-progress toward the same
    bounded MAX_STUCK_CYCLES threshold. The 480s wait before this check
    already covers legitimate preprocess+load+resume warm-up time, so a
    stagnant read at that point is a real signal, not noise - no unbounded
    grace period that could mask a genuine crash-loop-with-VM-alive."""
    local_step = get_local_step()
    hub_step = get_hub_global_step()
    last_known_step = state["last_known_step"]
    print(f"[driver] progress check: local={local_step} hub={hub_step} (was {last_known_step})", flush=True)

    best_step = max([s for s in (local_step, hub_step) if s is not None], default=None)
    made_progress = (best_step is not None and last_known_step is not None and best_step > last_known_step)

    if made_progress:
        state["stuck_cycles"] = 0
    else:
        state["stuck_cycles"] += 1
        print(f"[driver] no confirmed forward progress ({state['stuck_cycles']}/{MAX_STUCK_CYCLES})", flush=True)
        if state["stuck_cycles"] >= MAX_STUCK_CYCLES:
            save_state(state)
            halt(f"no step progress across {MAX_STUCK_CYCLES} consecutive checks (stuck at step {last_known_step}, "
                 f"last read local={local_step} hub={hub_step})")
    if best_step is not None:
        state["last_known_step"] = best_step
    save_state(state)


def main():
    print("[driver] starting autodrive loop", flush=True)
    state = load_state()
    if state["last_known_step"] is None:
        state["last_known_step"] = get_hub_global_step()
    print(f"[driver] resumed state: {state}", flush=True)

    while True:
        alive, status = session_alive()

        if not alive:
            print(f"[driver] session dead/missing ({status.strip()}), reprovisioning", flush=True)
            ok = reprovision_and_launch()
            if not ok:
                state["consecutive_provision_failures"] += 1
                save_state(state)
                print(f"[driver] provision/launch failed ({state['consecutive_provision_failures']} in a row), backing off 60s", flush=True)
                time.sleep(60)
                if state["consecutive_provision_failures"] >= 10:
                    halt(f"too many consecutive provision failures ({state['consecutive_provision_failures']})")
                continue
            state["consecutive_provision_failures"] = 0
            save_state(state)
            time.sleep(480)  # cover preprocess+model-load+resume+next save_steps boundary before judging progress
            check_progress_or_die(state)
            continue

        out = poll_once()
        print(f"[driver] poll: {out.strip()[-400:]}", flush=True)

        if "TRAINING_COMPLETE" in out:
            print("[driver] TRAINING COMPLETE - exiting driver loop", flush=True)
            notify("RazorStrike training COMPLETE", f"Reached target step count. Adapter pushed to {ADAPTER_FULL}.")
            sys.exit(0)

        if "OutOfMemoryError" in out or "Traceback" in out or "PROCESS_EXITED" in out:
            reason = "error" if ("OutOfMemoryError" in out or "Traceback" in out) else "process exited"
            print(f"[driver] {reason} on live session - relaunching training on SAME session (cheap, no reprovision)", flush=True)
            ok = relaunch_on_live_session()
            if not ok:
                print("[driver] same-session relaunch failed, will try reprovision next cycle", flush=True)
                time.sleep(15)
                continue
            time.sleep(480)  # same rationale as above
            check_progress_or_die(state)
            continue

        time.sleep(90)


if __name__ == "__main__":
    main()
