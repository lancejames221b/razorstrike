#!/usr/bin/env python3
"""Phase 3 - External benchmark harness for HAWQ-SEC-RE-lora-v3's decompile
family: Decompile-Bench-Eval re-executability, scored against published
baselines (GPT-4.1-mini, LLM4Decompile-*, SK2Decompile) so v3 finally has a
number externally comparable, not just an internal judge win-rate.

Data: LLM4Binary/decompile-eval (cc0-1.0) is a save_to_disk DatasetDict with
THREE real splits (humaneval, mbpp, github) - plain load_dataset() returns a
flattened synthetic `train` split and silently loses this structure.
Schema (verified): index, func_name, func_dep, func, test, opt, language,
asm, ida_asm, ida_pseudo, ghidra_asm, ghidra_pseudo.

Generation: same system+user chat format as build_decompile_family.py's
format_decompile_row (Step 1) - train/eval prompt parity is the point, even
though it deviates from LLM4Decompile's own raw-completion harness (HAWQ is
a chat/reasoning model; raw-completion prompting degrades it). This is a
deliberate deviation, reported alongside the numbers wherever they're used.

Scoring: humaneval/mbpp use LLM4Decompile's own model-agnostic primitive,
ported line-for-line from decompile-bench/metrics/cal_execute_rate.py (MIT,
github.com/albertan017/LLM4Decompile) - same compile-then-run semantics,
same fixed "-O0" recompilation of the model's OUTPUT regardless of the
INPUT asm's original optimization level (matches the reference harness'
own run_exe_rate.py, which never threads per-row opt into execute_rate_main
either - opt only labels which O-level produced the *input* asm, used for
the per-level breakdown). Ported to compile+run inside a --network=none
Docker sandbox rather than directly on this host: `generic` also runs Plex
and the LM Studio server serving this same model - never compile/execute
untrusted model output as that user. Falls back to a ulimit/unprivileged
subprocess sandbox if Docker is unavailable.

`github` (the leakage-resistant 2025 split) has NO `test` field (verified:
always empty string) and its `func_dep` is a provenance file path, not
compilable header text - it cannot be scored by execute_rate at all.
Reported instead as an edit-similarity contamination check against ground
truth: unusually HIGH similarity on github (a split less likely to already
be memorized by the base model's pretraining) vs humaneval/mbpp would flag
memorization rather than decompilation skill.

Usage:
    python3 scripts/eval_decompile_bench.py --split humaneval --model hawq-sec-re-v1
    python3 scripts/eval_decompile_bench.py --split all --model hawq-sec-re-v1 \
        --host localhost:1235 --out /tmp/decompile_bench_results.json

Run this ON generic (gcc 13.3.0 + Docker confirmed present there).
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

DECOMPILE_SYSTEM = (
    "You are an expert decompiler. Given x86-64 assembly from a compiled binary, "
    "reconstruct the original C source function. Output only the C code in a single "
    "```c fenced block, with no commentary."
)

DOCKER_IMAGE = "gcc:13"


def format_prompt(asm):
    """Mirrors build_decompile_family.py's format_decompile_row exactly
    (train/eval prompt parity)."""
    return [
        {"role": "system", "content": DECOMPILE_SYSTEM},
        {"role": "user",
         "content": f"# This is the assembly code:\n\n```asm\n{asm}\n```\n\n# What is the source code?"},
    ]


def _post(host, model, messages, max_tokens=4000, temperature=0, timeout=180):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    req = urllib.request.Request(
        f"http://{host}/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode())
    return d, time.time() - t0


_FENCE_RE = re.compile(r"```c\s*\n(.*?)```", re.DOTALL)
_BARE_FENCE_RE = re.compile(r"```\s*\n(.*?)```", re.DOTALL)


def generate_c(host, model, asm):
    """Returns (c_source_or_None, elapsed). Ignores reasoning_content - this
    model emits a separate reasoning channel and burning the whole budget
    there yields empty content (observed on v2, eval_re_v2_http_probes.py
    convention)."""
    d, elapsed = _post(host, model, format_prompt(asm))
    msg = d["choices"][0]["message"]
    content = msg.get("content") or ""
    m = _FENCE_RE.search(content) or _BARE_FENCE_RE.search(content)
    if not m:
        return None, elapsed
    return m.group(1).strip(), elapsed


def _docker_available():
    return shutil.which("docker") is not None


def docker_execute_rate(func_dep, func, func_test, timeout=10, language="c", opt="-O0"):
    """Faithful port of LLM4Decompile's cal_execute_rate.execute_rate
    (decompile-bench/metrics/cal_execute_rate.py, MIT) - identical
    compile-then-run semantics, executed inside a network-isolated Docker
    container instead of directly on the host. Returns (flag_comp, flag_exe)."""
    func_exe = func_dep + "\n" + func + "\n" + func_test
    with tempfile.TemporaryDirectory() as d:
        src_name = "exe.cpp" if language == "cpp" else "exe.c"
        with open(os.path.join(d, src_name), "w") as f:
            f.write(func_exe)
        compiler = "g++" if language == "cpp" else "gcc"
        std = ["-std=c++17"] if language == "cpp" else []
        libs = ["-lm", "-lcrypto"] if language == "cpp" else ["-lm"]
        compile_cmd = " ".join([compiler, opt] + std + [src_name, "-o", "exe.out"] + libs)
        run_cmd = (f"timeout {timeout} {compile_cmd} 2>/tmp/build.err && echo __COMPILE_OK__ "
                   f"&& timeout {timeout} ./exe.out && echo __RUN_OK__")
        try:
            r = subprocess.run(
                ["docker", "run", "--rm", "--network=none", "--memory=256m",
                 "--pids-limit=64", "--cpus=1", "--user", "1000:1000",
                 "-v", f"{d}:/work", "-w", "/work", DOCKER_IMAGE,
                 "bash", "-c", run_cmd],
                capture_output=True, text=True, timeout=timeout * 2 + 20)
            comp = 1 if "__COMPILE_OK__" in r.stdout else 0
            exe = 1 if "__RUN_OK__" in r.stdout else 0
        except subprocess.TimeoutExpired:
            comp, exe = 0, 0
        except Exception:
            comp, exe = 0, 0
    return comp, exe


def sandboxed_execute_rate(func_dep, func, func_test, timeout=10, language="c", opt="-O0"):
    """Fallback if Docker is unavailable: unprivileged-subprocess sandbox
    with a virtual-memory ulimit and a hard timeout - the required minimum
    per plan for compiling/running arbitrary model-generated C on a host
    that also serves Plex and LM Studio."""
    func_exe = func_dep + "\n" + func + "\n" + func_test
    with tempfile.TemporaryDirectory() as d:
        src_name = "exe.cpp" if language == "cpp" else "exe.c"
        src_path = os.path.join(d, src_name)
        bin_path = os.path.join(d, "exe.out")
        with open(src_path, "w") as f:
            f.write(func_exe)
        compiler = "g++" if language == "cpp" else "gcc"
        std = ["-std=c++17"] if language == "cpp" else []
        libs = ["-lm", "-lcrypto"] if language == "cpp" else ["-lm"]

        def _preexec():
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))

        comp, exe = 0, 0
        try:
            r = subprocess.run([compiler, opt] + std + [src_path, "-o", bin_path] + libs,
                                timeout=timeout, capture_output=True, preexec_fn=_preexec)
            comp = 1 if r.returncode == 0 and os.path.exists(bin_path) else 0
        except Exception:
            return 0, 0
        if not comp:
            return comp, exe
        try:
            r = subprocess.run([bin_path], timeout=timeout, capture_output=True, preexec_fn=_preexec)
            exe = 1 if r.returncode == 0 else 0
        except Exception:
            exe = 0
    return comp, exe


def execute_rate(func_dep, func, func_test, timeout=10, language="c", opt="-O0"):
    if _docker_available():
        return docker_execute_rate(func_dep, func, func_test, timeout, language, opt)
    print("[sandbox] WARNING: docker not found, falling back to unprivileged-subprocess "
          "sandbox (ulimit -v + timeout)", file=sys.stderr)
    return sandboxed_execute_rate(func_dep, func, func_test, timeout, language, opt)


def _levenshtein(a, b):
    """Dependency-free Levenshtein distance (no `editdistance` package
    assumed present on generic)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def edit_similarity(target, prediction):
    """Port of LLM4Decompile's cal_edit_sim.compute_ES (MIT), whitespace-
    normalized line-based edit similarity in [0, 1]."""
    t = "\n".join(l.strip() for l in target.splitlines() if l.strip())
    p = "\n".join(l.strip() for l in prediction.splitlines() if l.strip())
    return 1 - (_levenshtein(t, p) / max(len(t), len(p), 1))


def load_split(split_name):
    from huggingface_hub import snapshot_download
    from datasets import load_from_disk
    p = snapshot_download(repo_id="LLM4Binary/decompile-eval", repo_type="dataset")
    dd = load_from_disk(p)
    return dd[split_name]


def score_scoreable_split(split_name, rows, host, model, workers, timeout):
    """humaneval/mbpp: full generate -> compile -> execute pipeline."""
    per_opt = defaultdict(lambda: {"n": 0, "compiled": 0, "ran": 0, "no_fence": 0})

    def one(row):
        c, elapsed = generate_c(host, model, row["asm"])
        if c is None:
            return row["opt"], 0, 0, 1
        comp, exe = execute_rate(row["func_dep"], c, row["test"], timeout=timeout,
                                  language=row["language"], opt="-O0")
        return row["opt"], comp, exe, 0

    n = len(rows)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (opt, comp, exe, no_fence) in enumerate(ex.map(one, rows)):
            per_opt[opt]["n"] += 1
            per_opt[opt]["compiled"] += comp
            per_opt[opt]["ran"] += exe
            per_opt[opt]["no_fence"] += no_fence
            if (i + 1) % 25 == 0 or (i + 1) == n:
                print(f"[{split_name}] {i + 1}/{n} processed", flush=True)

    result = {"per_opt": {}, "overall": {"n": 0, "compiled": 0, "ran": 0, "no_fence": 0}}
    for opt, d in per_opt.items():
        result["per_opt"][opt] = {
            **d,
            "compile_rate": d["compiled"] / max(d["n"], 1),
            "exe_rate": d["ran"] / max(d["n"], 1),
        }
        for k in ("n", "compiled", "ran", "no_fence"):
            result["overall"][k] += d[k]
    result["overall"]["compile_rate"] = result["overall"]["compiled"] / max(result["overall"]["n"], 1)
    result["overall"]["exe_rate"] = result["overall"]["ran"] / max(result["overall"]["n"], 1)
    return result


def score_github_split(rows, host, model, workers):
    """github: no test/usable func_dep - edit-similarity contamination
    check against ground truth instead of execute_rate."""
    sims = []

    def one(row):
        c, elapsed = generate_c(host, model, row["asm"])
        if c is None:
            return None
        return edit_similarity(row["func"], c)

    n = len(rows)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, sim in enumerate(ex.map(one, rows)):
            if sim is not None:
                sims.append(sim)
            if (i + 1) % 25 == 0 or (i + 1) == n:
                print(f"[github] {i + 1}/{n} processed", flush=True)
    avg = sum(sims) / max(len(sims), 1)
    return {"n": n, "n_scored": len(sims), "avg_edit_similarity": avg}


def stratified_sample(rows, n, seed=42):
    """Even sample across (opt, language) strata, preserving the dataset's
    natural balance rather than taking a biased prefix. Single-GPU serial
    inference makes the full humaneval (1312) + mbpp (7792) sets a
    multi-day run (measured ~5-25s/call against the deployed IQ4_XS
    endpoint) - subsample by default, --full opts back into the complete
    set matching the published-baseline methodology exactly."""
    import random
    strata = defaultdict(list)
    for i, row in enumerate(rows):
        strata[(row["opt"], row["language"])].append(i)
    rng = random.Random(seed)
    for idxs in strata.values():
        rng.shuffle(idxs)
    per_stratum = max(1, n // max(len(strata), 1))
    picked = []
    for idxs in strata.values():
        picked.extend(idxs[:per_stratum])
    picked.sort()
    return rows.select(picked[:n])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost:1235")
    ap.add_argument("--model", default="hawq-sec-re-v1")
    ap.add_argument("--split", choices=["humaneval", "mbpp", "github", "all"], default="all")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=10)
    ap.add_argument("--humaneval-limit", type=int, default=328,
                     help="stratified subsample size (opt x language balanced); "
                          "full split is 1312 rows")
    ap.add_argument("--mbpp-limit", type=int, default=800,
                     help="stratified subsample size (opt x language balanced); "
                          "full split is 7792 rows")
    ap.add_argument("--github-limit", type=int, default=300,
                     help="github split is 66.5k rows and only a qualitative honesty "
                          "check, not the scored metric - subsample by default")
    ap.add_argument("--full", action="store_true",
                     help="run the complete humaneval/mbpp sets (matches published-"
                          "baseline methodology exactly; multi-day on single-GPU serial "
                          "inference - see module docstring)")
    ap.add_argument("--limit", type=int, default=None,
                     help="cap rows per split (smoke-testing; overrides stratified limits)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    splits = ["humaneval", "mbpp", "github"] if args.split == "all" else [args.split]
    results = {}
    for split_name in splits:
        print(f"\n=== {split_name} ===", flush=True)
        rows = load_split(split_name)
        if split_name == "github":
            rows = rows.shuffle(seed=42).select(range(min(args.github_limit, len(rows))))
        elif split_name == "humaneval" and not args.full:
            rows = stratified_sample(rows, args.humaneval_limit)
        elif split_name == "mbpp" and not args.full:
            rows = stratified_sample(rows, args.mbpp_limit)
        if args.limit:
            rows = rows.select(range(min(args.limit, len(rows))))
        rows = list(rows)
        print(f"[{split_name}] scoring {len(rows)} rows against {args.model} @ {args.host} "
              f"(full_set={args.full})", flush=True)

        if split_name == "github":
            results[split_name] = score_github_split(rows, args.host, args.model, args.workers)
        else:
            results[split_name] = score_scoreable_split(
                split_name, rows, args.host, args.model, args.workers, args.timeout)

    print("\n=== SUMMARY ===", flush=True)
    for split_name, r in results.items():
        if split_name == "github":
            print(f"{split_name}: avg_edit_similarity={r['avg_edit_similarity']:.3f} "
                  f"(n_scored={r['n_scored']}/{r['n']}, contamination check only, not a "
                  f"re-executability score - dataset has no test harness)")
        else:
            ov = r["overall"]
            print(f"{split_name}: re-executability={ov['exe_rate']*100:.2f}% "
                  f"compile_rate={ov['compile_rate']*100:.2f}% "
                  f"(n={ov['n']}, no_fence={ov['no_fence']})")
            for opt in sorted(r["per_opt"]):
                d = r["per_opt"][opt]
                print(f"  {opt}: exe={d['exe_rate']*100:.2f}% compile={d['compile_rate']*100:.2f}% (n={d['n']})")

    print("\nNOTE: generation used chat-format prompting (system+user turns matching "
          "this model's training format), not LLM4Decompile's raw-completion harness. "
          "Numbers are metric-comparable to the published baselines but not produced by "
          "an identical harness.", flush=True)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[save] results -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
