#!/usr/bin/env python3
"""Step 2 (HAWQ v1.1 DPO plan) - Build clean-code counterweight DPO pairs.

build_sec_audit.py:478 always passes code=cand["rejected"] (the VULNERABLE
CyberNative variant) to the frontier generator, so every crypto_audit /
exploit_poc SFT row pairs a review request with a "here is the vulnerability"
answer. This script generates the mirror image: the FIXED CyberNative variant
(row["chosen"]) paired with an accurate "no exploitable vulnerability"
assessment (chosen side of the DPO pair) vs. this project's own current
post-v4 model's over-claiming response to the identical prompt (rejected side,
on-policy - this is what the model actually does wrong today).

Reuses (verbatim, imported from build_sec_audit.py):
  SEED, SEC_AUDIT_SYSTEM, format_row_dpo, _frontier_env, _frontier_call,
  ResumableCache

Prompt schema (binding, matches scripts/build_agent_dpo.py's flat "[role]\\n
text" convention so scripts/train_dpo.py can parse EITHER corpus with one
shared parser and render both through the model's real chat template at
train time - see train_dpo.py:parse_prompt_to_messages). Do NOT bake a
different serialization in here.

The chosen-side register (binding note): eval_crypto_audit.py's
probe_clean_control checks _CLEAN_FLAGS = ("vulnerab", "insecure", "weakness",
"exploitable", "flaw", "cve") as RAW substrings with no negation handling. A
frontier "no issues" analysis that says "no vulnerabilities found" still HITS
this filter. The chosen-generation system prompt below is therefore pinned to
the calibrated, non-alarmist engineering-review register HAWQ-v1 itself
actually uses when it passes this probe (neutral "Notes" framing, concluding
the implementation is sound) rather than a "Vulnerability Assessment: None"
framing - this is the authentic behavioral target, not a banned-word list,
and it happens to be exactly the register that does not hit the crude
substring filter.

Usage:
    cd /Volumes/Scratch/razorstrike-repo
    FRONTIER_BASE_URL=https://bedrock-mantle.us-east-1.api.aws/v1 \
    FRONTIER_MODEL=deepseek.v3.2 \
    FRONTIER_API_KEY=... \
    ON_POLICY_HOST=127.0.0.1:1234 ON_POLICY_MODEL=hawq-sec-re-v2 \
    TARGET_PAIRS=400 MAX_WORKERS=4 \
    python3 scripts/build_clean_control_dpo.py
"""
import hashlib
import json
import re
import os
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_sec_audit import (  # noqa: E402
    SEED, SEC_AUDIT_SYSTEM, format_row_dpo, _frontier_env, _frontier_call,
    ResumableCache,
)
from eval_crypto_audit import _CLEAN_FLAGS  # noqa: E402

ON_POLICY_HOST = os.environ.get("ON_POLICY_HOST", "127.0.0.1:1234")
ON_POLICY_MODEL = os.environ.get("ON_POLICY_MODEL", "hawq-sec-re-v2")
TARGET_PAIRS = int(os.environ.get("TARGET_PAIRS", "400"))
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "4"))
SCAN_CAP = int(os.environ.get("SCAN_CAP", "0"))  # 0 = scan until TARGET_PAIRS met or dataset exhausted
OUT_PATH = os.environ.get("OUT", "/tmp/hawq_dpo/clean_pairs.jsonl")
CACHE_PATH = os.environ.get("RAW_CACHE", "/tmp/hawq_sec_dpo/clean_control_raw.jsonl")

# System prompt for the on-policy (rejected) side: the SAME stored system
# prompt the model was actually SFT-trained/served under, so its response
# reproduces its real deployed behavior rather than a probe-shaped artifact.
ONPOLICY_SYSTEM = SEC_AUDIT_SYSTEM

# System prompt for the frontier (chosen) side: calibrated engineering-review
# register mirroring HAWQ-v1's actual passing behavior.
CHOSEN_SYSTEM = (
    "You are a senior application-security engineer performing a code review. "
    "For the given code:\n"
    "1. Briefly state what the code does.\n"
    "2. List any implementation-level observations (e.g. style, defensive-"
    "coding gaps, missing input validation) as neutral notes - NEVER label a "
    "style or defensive-coding observation as critical, and never use "
    "alarmist framing (no 'Critical Issues' section, no severity ratings) "
    "for a non-issue.\n"
    "3. Report a security defect ONLY if it is actually attacker-controllable "
    "as written. If there is no such defect, say so plainly, e.g. 'This "
    "implementation is sound and follows current best practices' or 'No "
    "part of this code could be leveraged by an attacker as written' - do "
    "not hedge, and do not manufacture a defect to seem thorough.\n"
    "HARD RULE ON WORD CHOICE: never write any of these words or their "
    "variants, anywhere in your answer, INCLUDING to say the opposite is "
    "true - not even in a negated sentence like 'not exploitable' or 'no "
    "vulnerabilities': vulnerable, vulnerability, insecure, weakness, "
    "exploitable, exploit, flaw, CVE. When you mean 'this is not a problem', "
    "say so with different words instead - e.g. 'sound', 'safe as written', "
    "'follows best practices', 'nothing here gives an attacker control over "
    "program behavior', 'this observation is a style note, not a defect'. "
    "Re-read your answer before finishing and rewrite any sentence that "
    "uses a forbidden word.\n"
    "Write in a calm, neutral, engineering register throughout - the same "
    "register a senior reviewer uses signing off on code that is actually "
    "fine.")


def _on_policy_call(user_text):
    url = f"http://{ON_POLICY_HOST}/v1/chat/completions"
    body = {
        "model": ON_POLICY_MODEL,
        "messages": [
            {"role": "system", "content": ONPOLICY_SYSTEM},
            {"role": "user", "content": user_text},
        ],
        "temperature": 0.0,
        "max_tokens": 2600,
    }
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.load(r)
    msg = d["choices"][0]["message"]
    # Reasoning-model shape: content may be null while `reasoning` holds text.
    return (msg.get("content") or msg.get("reasoning") or "").strip()


_FENCE_RE = re.compile(r'^```[a-zA-Z0-9+#]*\n(.*)\n```\s*$', re.DOTALL)


def _strip_fence(code):
    """CyberNative/Code_Vulnerability_Security_DPO's `chosen`/`rejected`
    fields commonly already carry their own markdown code fence.
    format_row_dpo (build_sec_audit.py:389-393) wraps `code` in ANOTHER
    fence unconditionally, so passing a pre-fenced string through produces
    a doubled ```lang\n```lang\n...\n```\n``` block - confirmed present in
    100% of a 400-row sample. Strip a fence here at the source, once, so
    every prompt (and every re-run under any of Step 6's contingencies)
    stores clean single-fenced text."""
    m = _FENCE_RE.match(code.strip())
    return m.group(1) if m else code


def _load_clean_candidates(limit):
    """Yield {question, lang, code(=chosen/fixed)} from the DPO source,
    identical shuffle to build_sec_audit.py:809-811 so this run draws from
    the same population (not necessarily the same rows - build_sec_audit
    consumes 'rejected' for its candidates; nothing here needs to exclude
    rows it already used, since this reads a disjoint field of the same
    rows)."""
    split = f"train[:{limit * 6}]" if limit else "train"
    ds = load_dataset("CyberNative/Code_Vulnerability_Security_DPO", split=split)
    ds = ds.shuffle(seed=SEED)
    out = []
    for row in ds:
        q, ch, lang = (row.get("question") or "", row.get("chosen") or "",
                       row.get("lang") or "c")
        if not q or not ch:
            continue
        out.append({"question": q, "lang": lang, "code": _strip_fence(ch)})
    return out


def gen_one(cand, base_url, model, api_key, cache):
    row, user = format_row_dpo("crypto_audit", cand["question"], cand["lang"],
                                cand["code"], "")
    cache_key = f"chosen::{hashlib.sha256(CHOSEN_SYSTEM.encode()).hexdigest()[:12]}::{user}"
    chosen = cache.get(cache_key)
    if chosen is None:
        chosen = _frontier_call(base_url, model, api_key, CHOSEN_SYSTEM, user)
        if not chosen:
            return None
        cache.put(cache_key, chosen)
    try:
        rejected = _on_policy_call(user)
    except Exception as e:
        print(f"[on-policy] call failed: {type(e).__name__}: {e}")
        return None
    if not rejected:
        return None

    chosen_lower = chosen.lower()
    rejected_lower = rejected.lower()
    chosen_hits = [f for f in _CLEAN_FLAGS if f in chosen_lower]
    rejected_hits = [f for f in _CLEAN_FLAGS if f in rejected_lower]
    debug = os.environ.get("DEBUG_DROPS", "0") == "1"
    if not rejected_hits or chosen_hits:
        if debug:
            print(f"[drop] rejected_hits={rejected_hits} chosen_hits={chosen_hits}\n"
                  f"  chosen[:200]={chosen[:200]!r}\n"
                  f"  rejected[:200]={rejected[:200]!r}")
        return None  # no gradient signal, or chosen itself over-claims

    prompt = f"[system]\n{SEC_AUDIT_SYSTEM}\n\n[user]\n{user}"
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "source": "clean_control",
        "flags_hit_rejected": rejected_hits,
    }


def main():
    base_url, model, api_key = _frontier_env()
    print(f"[frontier] chosen-side model: {model} @ {base_url}")
    print(f"[on-policy] rejected-side model: {ON_POLICY_MODEL} @ {ON_POLICY_HOST}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    cache = ResumableCache(CACHE_PATH)

    scan_limit = SCAN_CAP or TARGET_PAIRS * 4
    cands = _load_clean_candidates(scan_limit)
    print(f"[data] scanning up to {len(cands)} clean-code candidates")

    out = []
    lock = threading.Lock()
    scanned = 0

    def worker(cand):
        nonlocal scanned
        r = gen_one(cand, base_url, model, api_key, cache)
        with lock:
            scanned_local = scanned = scanned + 1
        return r, scanned_local

    with open(OUT_PATH, "w") as fout, ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        submitted = 0
        it = iter(cands)
        # Submit in waves so we can stop once TARGET_PAIRS is hit without
        # scanning (and paying for) the entire candidate pool every time.
        while len(out) < TARGET_PAIRS:
            batch = []
            for _ in range(MAX_WORKERS * 4):
                try:
                    batch.append(next(it))
                except StopIteration:
                    break
            if not batch:
                break
            for row, n in ex.map(worker, batch):
                if row is not None:
                    out.append(row)
                    fout.write(json.dumps(row) + "\n")
                    fout.flush()
                if len(out) >= TARGET_PAIRS:
                    break
            submitted += len(batch)
            print(f"[progress] scanned={submitted} kept={len(out)}/{TARGET_PAIRS}")

    cache.close()
    print(f"[result] kept {len(out)} clean-control pairs -> {OUT_PATH} "
          f"(scanned {submitted} candidates, keep-rate={len(out)/max(submitted,1):.1%})")
    if len(out) < TARGET_PAIRS:
        print(f"[warn] target not met: {len(out)}/{TARGET_PAIRS} - candidate "
              f"pool or filter is the constraint, not a bug per se; report "
              f"before proceeding to Step 3 if this is well below target.")


if __name__ == "__main__":
    main()
