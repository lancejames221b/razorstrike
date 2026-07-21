#!/usr/bin/env python3
"""Self-healing driver for the RazorStrike v2 full training run.

Loops:
  - session dead/missing -> reprovision (new G4, clone, vm_setup, HF login) + launch
  - session alive but training process dead (crash/OOM/manual kill) -> relaunch
    on the SAME session (no reprovision needed, much cheaper)
  - session alive and training running -> poll for errors/completion

Crash-loop detection: after every relaunch we check the Hub's global_step. If
it hasn't advanced (or is unreadable) across MAX_STUCK_CYCLES consecutive
relaunches, something is structurally broken and we exit for human review
instead of burning compute forever. Counters persist to a state file so a
driver-process crash+restart (e.g. from an uncaught subprocess timeout)
can't silently reset the guard.
"""
import subprocess
import time
import sys
import json
import os

SESSION = "rs-g4"
ADAPTER_FULL = "lancejames221b/razorstrike-v2-offsec-lora"
MAX_STUCK_CYCLES = 5
STATE_FILE = "/tmp/rs_autodrive_state.json"


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
    # Prefer an explicitly-injected HF_TOKEN env var (set at `hub start` time,
    # captured from THIS interactive shell) over huggingface_hub's get_token(),
    # which resolves the cache path from HOME - a hub-managed process may get
    # a different/minimal HOME and silently fail to find the token file even
    # though it works fine in an interactive shell.
    env_tok = os.environ.get("HF_TOKEN", "").strip()
    if env_tok:
        return env_tok
    print("[driver] HF_TOKEN not in env, falling back to huggingface_hub.get_token() file resolution", flush=True)
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
    print("[driver] FATAL: could not obtain a non-empty HF_TOKEN (env var absent AND get_token() file resolution failed after retries). "
          "Exiting for human review - proceeding would launch training with no auth against private repos.", flush=True)
    sys.exit(3)


HF_TOKEN = get_hf_token()
print(f"[driver] HF_TOKEN acquired (len={len(HF_TOKEN)})", flush=True)

LAUNCH_PY = f'''
import subprocess, os, time
r0 = subprocess.run("HF_TOKEN='{HF_TOKEN}' python3 -c \\"from huggingface_hub import login; login('{HF_TOKEN}'); print('HF_LOGIN_OK')\\"", shell=True, capture_output=True, text=True)
print(r0.stdout, r0.stderr[-300:])
if "HF_LOGIN_OK" not in r0.stdout:
    print("HF_LOGIN_FAILED - aborting launch")
else:
    r1 = subprocess.run("test -d /content/razorstrike && cd /content/razorstrike && git pull --ff-only 2>&1 || (rm -rf /content/razorstrike && git clone --depth 1 https://github.com/lancejames221b/razorstrike.git /content/razorstrike 2>&1)", shell=True, capture_output=True, text=True)
    print("SYNC:", r1.stdout[-300:])
    r2 = subprocess.run("cd /content/razorstrike && python3 scripts/vm_setup.py 2>&1 | tail -25", shell=True, capture_output=True, text=True, timeout=300)
    print("SETUP:", r2.stdout)
    subprocess.run("pkill -f train_lora 2>/dev/null", shell=True)
    time.sleep(2)
    env = os.environ.copy()
    env.update({{
        "BASE_REPO": "Qwen/Qwen3.6-35B-A3B",
        "DATA_REPO": "lancejames221b/razorstrike-v2-sft",
        "ADAPTER_REPO": "{ADAPTER_FULL}",
        "OUT_DIR": "/content/adapter",
        "MAXLEN": "3072",
        "TARGET_MLP": "0",
        "SAVE_STEPS": "20",
        "EVAL_STEPS": "100",
        "RESUME": "1",
        "HF_TOKEN": "{HF_TOKEN}",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }})
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


def get_hub_global_step():
    """Authoritative progress signal: query the Hub checkpoint directly,
    independent of anything happening on the (possibly dead) VM. Retries
    on transient failure; returns None only after all retries are exhausted,
    which callers treat as a stuck-signal (not a silent skip)."""
    for attempt in range(3):
        try:
            r = subprocess.run(
                ["python3", "-c",
                 f"from huggingface_hub import hf_hub_download; import json; "
                 f"p = hf_hub_download('{ADAPTER_FULL}', 'last-checkpoint/trainer_state.json', token='{HF_TOKEN}'); "
                 f"print(json.load(open(p)).get('global_step'))"],
                cwd="/Volumes/Scratch", capture_output=True, text=True, timeout=30
            )
            return int(r.stdout.strip())
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
    """Mutates and persists state; exits process if stuck too long.
    A persistently-unreadable step counts as non-progress too, so an auth/network
    failure can't silently disable the crash-loop guard."""
    new_step = get_hub_global_step()
    last_known_step = state["last_known_step"]
    print(f"[driver] post-relaunch hub global_step={new_step} (was {last_known_step})", flush=True)
    made_progress = (new_step is not None and last_known_step is not None and new_step > last_known_step)
    if made_progress:
        state["stuck_cycles"] = 0
    else:
        state["stuck_cycles"] += 1
        print(f"[driver] no confirmed forward progress ({state['stuck_cycles']}/{MAX_STUCK_CYCLES})", flush=True)
        if state["stuck_cycles"] >= MAX_STUCK_CYCLES:
            print("[driver] STUCK: no step progress across multiple relaunches, exiting for human review", flush=True)
            save_state(state)
            sys.exit(2)
    if new_step is not None:
        state["last_known_step"] = new_step
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
                    print("[driver] too many consecutive provision failures, exiting for human review", flush=True)
                    sys.exit(1)
                continue
            state["consecutive_provision_failures"] = 0
            save_state(state)
            time.sleep(90)
            check_progress_or_die(state)
            continue

        out = poll_once()
        print(f"[driver] poll: {out.strip()[-400:]}", flush=True)

        if "TRAINING_COMPLETE" in out:
            print("[driver] TRAINING COMPLETE - exiting driver loop", flush=True)
            sys.exit(0)

        if "OutOfMemoryError" in out or "Traceback" in out or "PROCESS_EXITED" in out:
            reason = "error" if ("OutOfMemoryError" in out or "Traceback" in out) else "process exited"
            print(f"[driver] {reason} on live session - relaunching training on SAME session (cheap, no reprovision)", flush=True)
            ok = relaunch_on_live_session()
            if not ok:
                print("[driver] same-session relaunch failed, will try reprovision next cycle", flush=True)
                time.sleep(15)
                continue
            time.sleep(90)
            check_progress_or_die(state)
            continue

        time.sleep(90)


if __name__ == "__main__":
    main()
