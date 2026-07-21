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
instead of burning compute forever.
"""
import subprocess
import time
import sys

SESSION = "rs-g4"
ADAPTER_FULL = "lancejames221b/razorstrike-v2-offsec-lora"
MAX_STUCK_CYCLES = 5

HF_TOKEN = subprocess.run(
    ["python3", "-c", "from huggingface_hub import get_token; print(get_token())"],
    cwd="/Volumes/Scratch", capture_output=True, text=True
).stdout.strip()

LAUNCH_PY = f'''
import subprocess, os, time
r0 = subprocess.run("HF_TOKEN='{HF_TOKEN}' python3 -c \\"from huggingface_hub import login; login('{HF_TOKEN}'); print('HF_LOGIN_OK')\\"", shell=True, capture_output=True, text=True)
print(r0.stdout, r0.stderr[-300:])
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


def run(cmd, timeout=120):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)


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
                 f"p = hf_hub_download('{ADAPTER_FULL}', 'last-checkpoint/trainer_state.json'); "
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
    return "TRAIN_LAUNCHED" in r.stdout


def poll_once():
    write("/tmp/_rs_poll.py", POLL_PY)
    r = run(f"colab exec -s {SESSION} -f /tmp/_rs_poll.py --timeout 60", timeout=90)
    return r.stdout


def check_progress_or_die(last_known_step, stuck_cycles):
    """Returns (new_last_known_step, new_stuck_cycles); exits process if stuck too long.
    A persistently-unreadable step counts as non-progress too, so an auth/network
    failure can't silently disable the crash-loop guard."""
    new_step = get_hub_global_step()
    print(f"[driver] post-relaunch hub global_step={new_step} (was {last_known_step})", flush=True)
    made_progress = (new_step is not None and last_known_step is not None and new_step > last_known_step)
    if made_progress:
        stuck_cycles = 0
    else:
        stuck_cycles += 1
        print(f"[driver] no confirmed forward progress ({stuck_cycles}/{MAX_STUCK_CYCLES})", flush=True)
        if stuck_cycles >= MAX_STUCK_CYCLES:
            print("[driver] STUCK: no step progress across multiple relaunches, exiting for human review", flush=True)
            sys.exit(2)
    if new_step is not None:
        last_known_step = new_step
    return last_known_step, stuck_cycles


def main():
    print("[driver] starting autodrive loop", flush=True)
    consecutive_provision_failures = 0
    last_known_step = get_hub_global_step()
    stuck_cycles = 0
    print(f"[driver] initial hub global_step={last_known_step}", flush=True)

    while True:
        alive, status = session_alive()

        if not alive:
            print(f"[driver] session dead/missing ({status.strip()}), reprovisioning", flush=True)
            ok = reprovision_and_launch()
            if not ok:
                consecutive_provision_failures += 1
                print(f"[driver] provision/launch failed ({consecutive_provision_failures} in a row), backing off 60s", flush=True)
                time.sleep(60)
                if consecutive_provision_failures >= 10:
                    print("[driver] too many consecutive provision failures, exiting for human review", flush=True)
                    sys.exit(1)
                continue
            consecutive_provision_failures = 0
            time.sleep(90)
            last_known_step, stuck_cycles = check_progress_or_die(last_known_step, stuck_cycles)
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
            last_known_step, stuck_cycles = check_progress_or_die(last_known_step, stuck_cycles)
            continue

        time.sleep(90)


if __name__ == "__main__":
    main()
