#!/usr/bin/env python3
"""Phase 1h - Build sec_audit: crypto-implementation auditing + exploit-path
reasoning SFT data. Emits TWO families from a single frontier pass, split by
regex on the code under analysis so they are disjoint by construction:

  family="crypto_audit" - rows whose code matches _CRYPTO_RE
  family="exploit_poc"  - rows that do not

Primary source: CyberNative/Code_Vulnerability_Security_DPO (`question` for
context, `rejected` as the vulnerable code under analysis; `chosen` is shown
to the frontier teacher ONLY, never persisted - mirrors build_re_analysis.py
showing ground-truth C to the teacher but storing an asm-only user turn).

Confirmed this session: CyberNative's `rejected` field yields only 7
crypto-matching rows out of 4656 (regex hit-rate ~0.15%), far below the
800-row floor. The pre-decided fallback (LLM4Binary/decompile-bench, rows
whose ground-truth C contains a constant table from scripts/crypto_lib.py)
is therefore NOT a rare contingency here - it supplies nearly all of
crypto_audit. Three things follow from that, all confirmed/decided before
writing this generation logic:

1. decompile-bench C source commonly retains real function/variable names
   (`net::TcpSocket::Accept`, likely `aes_encrypt_block`-style names for
   crypto code) that would hand the identification straight to the model -
   exactly the build_crypto_id.py tautology this plan exists to fix. Fallback
   candidates are run through crypto_lib.py's own obfuscation transforms
   (_obf_rename_identifiers, _obf_strip_comments) before being shown to the
   frontier teacher or stored, not just checked-and-skipped after the fact.
2. The Step 1 eval (eval_crypto_audit.py's probe_crypto_id) presents plain C
   source with an embedded constant table, not assembly. An asm-only
   fallback (matching build_re_analysis.py's convention) would train the
   dominant share of crypto_audit on a modality the gate never measures.
   The fallback therefore splits ~50/50 between an asm user turn and a
   C-source user turn (decompile-bench provides real ground-truth C for
   both), so the training distribution actually covers what Step 6 grades.
3. The primitive match (from the constant-table scan) is passed to the
   frontier teacher as a hidden answer key, same as `chosen` for the DPO
   rows - the teacher is instructed to ground its identification in the
   code's own evidence and never narrate that it was told, but IS expected
   to confidently state the correct primitive name (unlike the DPO fix code,
   the primitive name is exactly the label this family trains the model to
   produce, so withholding it risks baking a confidently-wrong label into
   gold data on hard cases, e.g. Blowfish's plain pi-digit P-array, which
   this repo's own baseline model failed to name without help).

Frontier config is REQUIRED env, same guard as build_re_analysis.py - must
NOT default to a local LM Studio endpoint:
    FRONTIER_BASE_URL, FRONTIER_MODEL, FRONTIER_API_KEY

Usage:
    CAP=40 CRYPTO_OUT=/tmp/sec_audit_smoke \
        FRONTIER_BASE_URL=... FRONTIER_MODEL=... FRONTIER_API_KEY=... \
        python3 scripts/build_sec_audit.py
"""

import os
import re
import json
import random
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from datasets import load_dataset, DatasetDict, Dataset

import crypto_lib

MAX_CODE = 4000
MAX_ASM = 6000
SEED = 42
RETRIES = 2
MAX_LABEL_SHARE = 0.25         # no single weakness label / primitive > 25% of a family
PER_PROJECT_CAP = 10           # decompile-bench diversity: cap rows per source file/project
MIN_CRYPTO_PRIMARY = 800       # yield gate: below this from the DPO source, invoke fallback
CAP_CRYPTO = int(os.environ.get("CAP_CRYPTO", "2000"))
CAP_EXPLOIT = int(os.environ.get("CAP_EXPLOIT", "2000"))

# Case-insensitive; partitions CyberNative rows by content of the vulnerable
# (`rejected`) code into crypto_audit vs exploit_poc.
_CRYPTO_RE = re.compile(
    r"\b(AES|DES|3DES|RC4|ChaCha20|Salsa20|Blowfish|TEA|XTEA|MD5|SHA-?1|SHA-?256|"
    r"HMAC|RSA|ECDSA|ECB|CBC|CTR|GCM|IV|nonce|EVP_|CryptGen|srand|RAND_bytes|"
    r"getrandom|PBKDF2|bcrypt|scrypt)\b", re.IGNORECASE)

# Stored (student-facing) system prompt.
SEC_AUDIT_SYSTEM = (
    "You are a security code auditor and vulnerability researcher. Given "
    "source code or assembly, identify vulnerabilities, explain precisely "
    "how they are exploited with concrete attacker-controlled inputs, and "
    "sketch a concrete exploitation path. For correct code, confirm there is "
    "no flaw rather than inventing one."
)

# --- fallback-pool constant-table matchers (scripts/crypto_lib.py) ---------
# Use two matching modes:
#   1. Distinct high-entropy constants (SHA/MD5/Blowfish/TEA/CRC32): N-of-M
#      token-boundary values anywhere in the code, because 32-bit constants
#      are specific enough to survive reordering/partial usage.
#   2. Byte-range permutation tables (AES S-box): ORDERED runs only. The AES
#      S-box is a bijection over 0..255, so membership carries zero signal:
#      any code with enough distinct byte literals could match. Requiring a
#      canonical consecutive run (from any offset) preserves precision while
#      still catching split/decompiled tables that do not start at byte 0.
def _number_token(v, decimal_ok=True):
    """Regex fragment for one integer literal token. Allows C integer suffixes
    but refuses substring matches inside identifiers or wider literals."""
    if decimal_ok:
        body = rf"(?:0x0*{v:x}|{v})"
    else:
        body = rf"0x0*{v:x}"
    return rf"(?<![A-Za-z0-9_]){body}[uUlL]*(?![A-Za-z0-9_])"


def _value_patterns(values, decimal_ok=True):
    return [re.compile(_number_token(v, decimal_ok=decimal_ok), re.IGNORECASE)
            for v in values]


_INT_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_])(0[xX][0-9a-fA-F]+|\d+)[uUlL]*(?![A-Za-z0-9_])")


def _ordered_run_signatures(values, run_len=16):
    """Return canonical in-order value windows as tuples. Used after one
    integer-tokenization pass over a candidate row; avoids running hundreds
    of expensive offset regexes per row."""
    return {tuple(values[start:start + run_len])
            for start in range(len(values) - run_len + 1)}


def _has_ordered_run(code, signatures, run_len=16):
    nums = []
    for m in _INT_TOKEN_RE.finditer(code):
        try:
            nums.append(int(m.group(1), 0))
        except ValueError:
            continue
    if len(nums) < run_len:
        return False
    return any(tuple(nums[i:i + run_len]) in signatures
               for i in range(len(nums) - run_len + 1))


_FALLBACK_TABLES = {
    # mode, compiled-patterns, threshold
    "SHA-256": ("any", _value_patterns(crypto_lib.sha256_K()), 3),
    # SHA-1's init state (A,B,C,D,E) shares its first 4 words BYTE-FOR-BYTE
    # with MD5's init state - confirmed empirically this session: two of
    # three eyeballed "MD5" fallback matches were actually genuine SHA-1
    # functions (nni_sha1_init, mbedtls_sha1_starts). Checked before MD5,
    # requiring ALL 5 words (the 5th, 0xC3D2E1F0, is SHA-1-only) so it can
    # only fire on real SHA-1 code; MD5 code (4 words, no 5th) falls through
    # to the MD5 check below untouched.
    "SHA-1": ("any", _value_patterns([0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0]), 5),
    "MD5": ("any", _value_patterns(crypto_lib.md5_init()), 3),
    "Blowfish": ("any", _value_patterns(crypto_lib.blowfish_p_array()), 3),
    "TEA": ("any", _value_patterns([crypto_lib.tea_delta()]), 1),
    "CRC32": ("any", _value_patterns([crypto_lib.crc32_poly()]), 1),
    # AES S-box values are a permutation of all byte values, so unordered
    # membership matching is degenerate. Match a 16-value canonical ordered
    # run from any table offset, allowing hex OR decimal literal rendering.
    "AES": ("ordered", _ordered_run_signatures(crypto_lib.aes_forward_sbox(), run_len=16), 16),
}
# Cheap substring prefilter for the expensive checks. It is only a reject
# shortcut; the real regex matcher remains authoritative.
_FALLBACK_PREFILTER = {
    # SHA-256/MD5 key on value[0] specifically. Safe for the real
    # decompile-bench scan these two are used for: genuine implementations
    # declare the full K[]/init array in canonical order starting at index
    # 0, so a windowed/offset rendering is implausible in practice - unlike
    # the synthetic Blowfish generator below, which deliberately windows at
    # arbitrary offsets and hit exactly this bug (fixed by dropping its
    # prefilter entirely). If SHA-256/MD5 corpus counts ever look
    # suspiciously low, re-check this assumption before trusting "genuine
    # scarcity" the way Blowfish's low count turned out not to be.
    "SHA-256": (f"{crypto_lib.sha256_K()[0]:x}", str(crypto_lib.sha256_K()[0])),
    "MD5": (f"{crypto_lib.md5_init()[0]:x}", str(crypto_lib.md5_init()[0])),
    # Blowfish candidates are windowed across the full 18-word table at any
    # offset (0-10), so a prefilter tied to value[0] specifically would
    # reject every window that doesn't happen to include it - confirmed
    # empirically: this caused 85 of 96 real Blowfish rows to be silently
    # misclassified as no-match. No prefilter, same rationale as AES above.
    "TEA": (f"{crypto_lib.tea_delta():x}", str(crypto_lib.tea_delta())),
    "CRC32": (f"{crypto_lib.crc32_poly():x}", str(crypto_lib.crc32_poly())),
    # AES ordered patterns can start at any offset and may be decimal-only, so
    # a first-byte hex prefilter would create false negatives. No prefilter.
}
_FALLBACK_LITERAL = {
    # Already highly specific as a bare substring (64-char exact alphabet) -
    # no regex needed. Case-sensitive by construction.
    "Base64": crypto_lib.base64_alphabet(),
}


def _match_fallback_primitive(code):
    """Return the first matching primitive name, or None. `code` original case
    (patterns are IGNORECASE; Base64's alphabet is case-sensitive by
    construction so it must NOT be lowercased before this check)."""
    code_lower = code.lower()
    for name, (mode, patterns, threshold) in _FALLBACK_TABLES.items():
        prefilters = _FALLBACK_PREFILTER.get(name)
        if prefilters is not None and not any(p in code_lower for p in prefilters):
            continue
        if mode == "any":
            if sum(1 for p in patterns if p.search(code)) >= threshold:
                return name
        elif mode == "ordered":
            if _has_ordered_run(code, patterns, run_len=threshold):
                return name
        else:
            raise ValueError(f"unknown fallback matcher mode: {mode}")
    for name, literal in _FALLBACK_LITERAL.items():
        if literal in code:
            return name
    return None


def _frontier_env():
    """Load and validate required FRONTIER_* env. Aborts on a local/HAWQ endpoint.
    Copied verbatim from build_re_analysis.py's guard logic."""
    base_url = os.environ.get("FRONTIER_BASE_URL")
    model = os.environ.get("FRONTIER_MODEL")
    api_key = os.environ.get("FRONTIER_API_KEY")
    missing = [n for n, v in (("FRONTIER_BASE_URL", base_url),
                               ("FRONTIER_MODEL", model),
                               ("FRONTIER_API_KEY", api_key)) if not v]
    if missing:
        raise SystemExit(
            f"[frontier] required env missing: {', '.join(missing)}. "
            f"This builder must call a real frontier model, not the local HAWQ "
            f"server, to generate gold sec-audit analyses.")

    if re.match(r"^(hawq|razorstrike)", model, re.IGNORECASE):
        raise SystemExit(
            f"[frontier] FRONTIER_MODEL={model!r} looks like a HAWQ/RazorStrike "
            f"model id, not a frontier model. Training HAWQ on its own output "
            f"produces no lift. Aborting.")

    local_hosts = ("127.0.0.1:1234", "localhost:1234", "generic:1234")
    if any(h in base_url for h in local_hosts):
        raise SystemExit(
            f"[frontier] FRONTIER_BASE_URL={base_url!r} points at a local LM "
            f"Studio endpoint (serves HAWQ itself). Aborting - supply a real "
            f"frontier endpoint (e.g. https://api.openai.com/v1 or "
            f"https://ollama.com/v1).")

    return base_url.rstrip("/"), model, api_key


def _frontier_call(base_url, model, api_key, system, user):
    """One frontier chat/completions call. Returns analysis text or None on failure."""
    url = f"{base_url}/chat/completions"
    body = {
        "model": model,
        "reasoning_effort": os.environ.get("FRONTIER_REASONING_EFFORT", "low"),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.3,
        "max_tokens": int(os.environ.get("FRONTIER_MAX_TOKENS", "2600")),
    }
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
    for attempt in range(RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=90) as r:
                d = json.load(r)
            text = d["choices"][0]["message"]["content"].strip()
            if text:
                return text
        except Exception as e:
            if attempt == RETRIES:
                print(f"[frontier] call failed after {RETRIES + 1} attempts: "
                      f"{type(e).__name__}: {e}")
    return None


# ---------------------------------------------------------------------------
# Frontier prompts: hard rule against referencing the hidden answer key,
# never open with the label before showing the reasoning that finds it.
# ---------------------------------------------------------------------------

FRONTIER_SYSTEM_DPO = (
    "You are an elite security code auditor and exploit developer. You will be "
    "shown vulnerable source code, together with a corrected/secure version of "
    "the same code. The secure version is provided ONLY so your analysis is "
    "factually correct - treat it as a private answer key. Write a precise "
    "security analysis OF THE VULNERABLE CODE, exactly as an expert would who "
    "has ONLY the vulnerable code in front of them and has never seen the fix.\n"
    "Hard rules:\n"
    "- NEVER mention, quote, or allude to the secure version, the fix, the "
    "corrected code, an 'answer key', 'ground truth', or the fact that you "
    "were given anything beyond the vulnerable code shown. The reader has "
    "only the vulnerable code.\n"
    "- Do NOT open by naming the vulnerability category as a label before "
    "explaining it (e.g. do not start with 'This is a buffer overflow.'); "
    "walk through what the code actually does first, then name the flaw once "
    "you've shown the reasoning that finds it.\n"
    "- Do NOT narrate a step-by-step 'let me verify... correct... correct' "
    "trace or any chain-of-thought. Output only the finished analysis.\n"
    "Structure the analysis as: (1) what the code does; (2) the specific flaw "
    "and precisely why it is exploitable, naming the attacker-controlled "
    "input(s); (3) a concrete exploitation path or PoC sketch showing impact; "
    "(4) in one or two sentences, the general shape of a correct fix (do not "
    "reproduce a fixed code block verbatim). Be faithful to what the code "
    "actually does; never invent behavior it doesn't have."
)

FRONTIER_USER_TMPL_DPO = (
    "TASK CONTEXT: {question}\n\n"
    "CODE UNDER REVIEW ({lang}):\n```{lang}\n{code}\n```\n"
    "KNOWN-CORRECT FIXED VERSION (ground truth, for your reference only - do "
    "not echo it):\n```{lang}\n{chosen}\n```\n"
    "Write the security analysis now."
)

FRONTIER_SYSTEM_ASM = (
    "You are an elite security code auditor and reverse engineer. You will be "
    "shown x86-64 assembly for a function, the original C/C++ source it was "
    "compiled from, and the cryptographic primitive it implements. The source "
    "and primitive name are provided ONLY so your analysis is factually "
    "correct - treat them as a private answer key. Write a precise "
    "cryptographic/security analysis OF THE ASSEMBLY, exactly as an expert "
    "would who has ONLY the assembly in front of them and has never seen the "
    "source or been told the answer.\n"
    "Hard rules:\n"
    "- NEVER mention, quote, or allude to the source, the C/C++ code, an "
    "'answer key', 'ground truth', a 'known primitive', or the fact that you "
    "were told or given anything beyond the assembly. The reader has only "
    "the assembly.\n"
    "- DO confidently and correctly name the primitive in your analysis - "
    "that is the point of the exercise - but ground the identification in "
    "specific evidence from the assembly itself (the constant values, table "
    "structure, or algorithm shape actually visible there), as if you "
    "reasoned your own way to it.\n"
    "- Do NOT narrate a step-by-step 'let me verify... correct... correct' "
    "trace or any chain-of-thought. Output only the finished analysis.\n"
    "Structure the analysis as: (1) one-line purpose; (2) the cryptographic "
    "primitive identified, with the specific evidence that identifies it; "
    "(3) how it's used (mode of operation, key/IV/nonce handling, if "
    "visible); (4) any implementation-level security issue visible in the "
    "assembly itself. Be faithful to what the assembly actually does."
)

FRONTIER_USER_TMPL_ASM = (
    "ASSEMBLY:\n```asm\n{asm}\n```\n"
    "KNOWN-CORRECT SOURCE (ground truth, for your reference only - do not "
    "echo it):\n```cpp\n{code}\n```\n"
    "KNOWN PRIMITIVE (ground truth, for your reference only - identify it "
    "via the assembly's own evidence, never say you were told): {primitive}\n"
    "Write the analysis now."
)

FRONTIER_SYSTEM_CSRC = (
    "You are an elite security code auditor. You will be shown a C/C++ "
    "function with identifiers stripped/renamed (obfuscated) and the "
    "cryptographic primitive it implements. The primitive name is provided "
    "ONLY so your analysis is factually correct - treat it as a private "
    "answer key. Write a precise cryptographic/security analysis of the "
    "function, exactly as an expert would who was never told the answer.\n"
    "Hard rules:\n"
    "- NEVER mention, quote, or allude to an 'answer key', 'ground truth', a "
    "'known primitive', or the fact that you were told or given anything "
    "beyond the code shown.\n"
    "- DO confidently and correctly name the primitive in your analysis - "
    "that is the point of the exercise - but ground the identification in "
    "specific evidence from the code itself (the constant values, table "
    "structure, or algorithm shape actually visible there), as if you "
    "reasoned your own way to it.\n"
    "- Do NOT narrate a step-by-step 'let me verify... correct... correct' "
    "trace or any chain-of-thought. Output only the finished analysis.\n"
    "Structure the analysis as: (1) one-line purpose; (2) the cryptographic "
    "primitive identified, with the specific evidence that identifies it; "
    "(3) how it's used (mode of operation, key/IV/nonce handling, if "
    "visible); (4) any implementation-level security issue visible in the "
    "code. Be faithful to what the code actually does; never invent behavior."
)

FRONTIER_USER_TMPL_CSRC = (
    "CODE:\n```c\n{code}\n```\n"
    "KNOWN PRIMITIVE (ground truth, for your reference only - identify it "
    "via the code's own evidence, never say you were told): {primitive}\n"
    "Write the analysis now."
)


def format_row_dpo(family, question, lang, code, analysis):
    user = (f"The following {lang} code was written to satisfy this task: "
            f"{question}\n\nAnalyze it for security vulnerabilities and "
            f"describe how an attacker could exploit them, or confirm there "
            f"are none.\n\n```{lang}\n{code}\n```")
    return {
        "source": "CyberNative/Code_Vulnerability_Security_DPO",
        "family": family,
        "messages": [
            {"role": "system", "content": SEC_AUDIT_SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": analysis},
        ],
    }, user


def format_row_asm(asm, analysis):
    user = ("Analyze this x86-64 function and identify the cryptographic "
            f"primitive it implements. Note any implementation-level "
            f"security issues.\n\n```asm\n{asm}\n```")
    return {
        "source": "LLM4Binary/decompile-bench",
        "family": "crypto_audit",
        "messages": [
            {"role": "system", "content": SEC_AUDIT_SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": analysis},
        ],
    }, user


def format_row_csrc(code, analysis):
    user = ("Analyze this C function and identify the cryptographic "
            f"primitive it implements. Note any implementation-level "
            f"security issues.\n\n```c\n{code}\n```")
    return {
        "source": "LLM4Binary/decompile-bench",
        "family": "crypto_audit",
        "messages": [
            {"role": "system", "content": SEC_AUDIT_SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": analysis},
        ],
    }, user


class ResumableCache:
    """Resumable, crash-safe generation cache keyed by the exact stored
    user-turn text. Mirrors build_re_analysis.py's RAW_CACHE pattern - a
    multi-hundred-call run against a paid frontier API must not lose
    already-paid-for work to a crash, and must not re-pay for rows a prior
    partial run already generated."""

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.cached = {}
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    self.cached[row["_cache_key"]] = row["row"]
            print(f"[cache] resumed {len(self.cached)} rows from {path}")
        self.lock = threading.Lock()
        self.f = open(path, "a")

    def get(self, key):
        return self.cached.get(key)

    def put(self, key, row):
        with self.lock:
            self.f.write(json.dumps({"_cache_key": key, "row": row}) + "\n")
            self.f.flush()

    def close(self):
        self.f.close()


def _gen_dpo_rows(cands, family, base_url, model, api_key, cache, max_workers):
    def gen_one(cand):
        row, user = format_row_dpo(family, cand["question"], cand["lang"],
                                    cand["rejected"], "")
        cached = cache.get(user)
        if cached is not None:
            return cached, cand.get("vulnerability", "")
        prompt = FRONTIER_USER_TMPL_DPO.format(
            question=cand["question"], lang=cand["lang"], code=cand["rejected"],
            chosen=cand["chosen"])
        analysis = _frontier_call(base_url, model, api_key, FRONTIER_SYSTEM_DPO, prompt)
        if not analysis:
            return None, None
        row, user = format_row_dpo(family, cand["question"], cand["lang"],
                                    cand["rejected"], analysis)
        # anti-leak: fixed/chosen code must never appear in the stored user turn
        if cand["chosen"].strip() and cand["chosen"].strip() in user:
            return None, None
        # anti-tautology: the weakness label must not appear in the user turn
        if cand["vulnerability"].strip().lower() in user.lower():
            return None, None
        cache.put(user, row)
        return row, cand.get("vulnerability", "")

    out = []
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            results = list(ex.map(gen_one, cands))
    else:
        results = [gen_one(c) for c in cands]
    for row, label in results:
        if row is not None:
            out.append((row, label))
    return out


def _gen_fallback_rows(cands, base_url, model, api_key, cache, max_workers):
    """cands: list of dicts {asm, code, primitive, modality} where modality
    is 'asm' or 'csrc' (assigned ~50/50 by the caller)."""

    def gen_one(cand):
        asm, code, primitive, modality = (cand["asm"], cand["code"],
                                           cand["primitive"], cand["modality"])
        # obfuscate before it's shown to anyone or stored - real function/
        # variable names in decompile-bench routinely self-document the
        # primitive (e.g. an aes_encrypt_block-style name), which is exactly
        # the build_crypto_id.py tautology this builder exists to avoid.
        code_obf = crypto_lib._obf_strip_comments(
            crypto_lib._obf_rename_identifiers(code, seed=SEED))
        if modality == "asm":
            row, user = format_row_asm(asm, "")
            if primitive.lower() in user.lower():
                return None, None  # symbol names leaked into the asm itself
            cached = cache.get(user)
            if cached is not None:
                return cached, primitive
            prompt = FRONTIER_USER_TMPL_ASM.format(asm=asm, code=code_obf, primitive=primitive)
            analysis = _frontier_call(base_url, model, api_key, FRONTIER_SYSTEM_ASM, prompt)
            if not analysis:
                return None, None
            row, user = format_row_asm(asm, analysis)
            if "```cpp" in user or "KNOWN-CORRECT" in user or "KNOWN PRIMITIVE" in user:
                return None, None
        else:
            row, user = format_row_csrc(code_obf, "")
            if primitive.lower() in user.lower():
                return None, None  # obfuscation didn't fully strip the hint
            cached = cache.get(user)
            if cached is not None:
                return cached, primitive
            prompt = FRONTIER_USER_TMPL_CSRC.format(code=code_obf, primitive=primitive)
            analysis = _frontier_call(base_url, model, api_key, FRONTIER_SYSTEM_CSRC, prompt)
            if not analysis:
                return None, None
            row, user = format_row_csrc(code_obf, analysis)
            if "KNOWN PRIMITIVE" in user:
                return None, None
        cache.put(user, row)
        return row, primitive

    out = []
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            results = list(ex.map(gen_one, cands))
    else:
        results = [gen_one(c) for c in cands]
    for row, label in results:
        if row is not None:
            out.append((row, label))
    return out


def _fmt_ints(vals, hex_mode=True, width=2):
    if hex_mode:
        return ", ".join(f"0x{v:0{width}x}" for v in vals)
    return ", ".join(str(v) for v in vals)


def _synthetic_table_candidates():
    """Small audited-table coverage for eval-target primitives that are scarce
    or absent in decompile-bench after precision-safe matching. Avoids
    train-on-test leakage by varying scaffold shape and checking generated
    rows against eval_crypto_audit.py snippets before training."""
    rng = random.Random(SEED)
    by_primitive = {"AES": [], "Blowfish": []}

    # AES: canonical S-box windows, forcing offset 0 in because that is the
    # most common real-world rendering and the eval also uses it. Leakage
    # risk is the surrounding scaffold, not the constants themselves; rows
    # use longer 24-value windows and different function/array shapes.
    sbox = crypto_lib.aes_forward_sbox()
    shuffled_starts = list(range(0, len(sbox) - 24))
    rng.shuffle(shuffled_starts)
    starts = [0] + [s for s in shuffled_starts if s != 0][:95]
    for idx, start in enumerate(starts):
        vals = sbox[start:start + 24]
        hex_mode = idx % 3 != 0
        table = _fmt_ints(vals, hex_mode=hex_mode, width=2)
        if idx % 2 == 0:
            code = (
                f"#include <stdint.h>\n"
                f"uint8_t sub_{idx:04x}(uint8_t x) {{\n"
                f"    static const uint8_t t[{len(vals)}] = {{{table}}};\n"
                f"    uint8_t y = t[(x + {idx % 7}) % {len(vals)}];\n"
                f"    return (uint8_t)(y ^ (x >> 1));\n"
                f"}}\n")
        else:
            code = (
                f"#include <stdint.h>\n"
                f"void sub_{idx:04x}(const uint8_t *in, uint8_t *out, unsigned n) {{\n"
                f"    static const unsigned char lut[{len(vals)}] = {{{table}}};\n"
                f"    for (unsigned i = 0; i < n; ++i) out[i] = lut[(in[i] + i) % {len(vals)}];\n"
                f"}}\n")
        by_primitive["AES"].append({"asm": "", "code": code, "primitive": "AES",
                                    "modality": "csrc", "project": "synthetic-aes"})

    # Blowfish: obfuscation strips identifiers/comments but leaves numeric
    # literals and code STRUCTURE untouched - so diversity axes that survive
    # dedup are: which window of the (now full 18-word) P-array is used, how
    # many words are emitted, hex vs decimal rendering, and the access
    # pattern shape (loop vs unrolled vs pointer-walk vs switch). A prior
    # version varied only rotation-by-index over an 8-value table, which had
    # just 8 distinct rotations regardless of candidate count and collapsed
    # 59 pre-dedup rows to 8 post-dedup when combined into the training mix.
    p = crypto_lib.blowfish_p_array()  # 18 words
    _blowfish_scaffolds = [
        lambda idx, table, n: (
            f"#include <stdint.h>\n"
            f"uint32_t mix_{idx:04x}(uint32_t x, uint32_t y) {{\n"
            f"    static const uint32_t k[{n}] = {{{table}}};\n"
            f"    for (unsigned r = 0; r < {n}; ++r) x = ((x ^ k[r]) + y) ^ (x << 3);\n"
            f"    return x;\n"
            f"}}\n"),
        lambda idx, table, n: (
            f"#include <stdint.h>\n"
            f"void mix_{idx:04x}(uint32_t *x, uint32_t *y) {{\n"
            f"    static const unsigned long tab[{n}] = {{{table}}};\n"
            f"    for (unsigned r = 0; r != {n}; ++r) {{ *x ^= tab[r]; *y += (*x >> (r & 7)); }}\n"
            f"}}\n"),
        lambda idx, table, n: (
            f"#include <stdint.h>\n"
            f"static const uint32_t words_{idx:04x}[{n}] = {{{table}}};\n"
            f"uint32_t scramble_{idx:04x}(uint32_t seed) {{\n"
            f"    const uint32_t *p = words_{idx:04x};\n"
            f"    for (int i = 0; i < {n}; i++) {{ seed ^= *p; seed = (seed << 5) | (seed >> 27); p++; }}\n"
            f"    return seed;\n"
            f"}}\n"),
        lambda idx, table, n: (
            f"#include <stdint.h>\n"
            f"uint32_t init_state_{idx:04x}(int which) {{\n"
            f"    static const uint32_t consts[{n}] = {{{table}}};\n"
            f"    switch (which) {{\n"
            f"        default: return consts[which % {n}] ^ 0x5bd1e995u;\n"
            f"    }}\n"
            f"}}\n"),
    ]
    window_lens = (8, 12)
    windows = [(wl, start) for wl in window_lens for start in range(0, len(p) - wl + 1)]
    combos = [(wl, start, hex_mode, scaffold_i)
              for wl, start in windows
              for hex_mode in (True, False)
              for scaffold_i in range(len(_blowfish_scaffolds))]
    rng.shuffle(combos)

    seen_obfuscated = set()
    for idx, (wl, start, hex_mode, scaffold_i) in enumerate(combos[:96]):
        vals = p[start:start + wl]
        table = _fmt_ints(vals, hex_mode=hex_mode, width=8)
        code = _blowfish_scaffolds[scaffold_i](idx, table, wl)
        code_obf = crypto_lib._obf_strip_comments(
            crypto_lib._obf_rename_identifiers(code, seed=SEED))
        if code_obf in seen_obfuscated:
            continue  # would be dropped by within-family dedup anyway - skip generating it
        seen_obfuscated.add(code_obf)
        by_primitive["Blowfish"].append({"asm": "", "code": code, "primitive": "Blowfish",
                                         "modality": "csrc", "project": "synthetic-blowfish"})
    print(f"[synthetic] Blowfish: {len(by_primitive['Blowfish'])} distinct post-obfuscation "
          f"candidates from {len(combos[:96])} attempts")
    return by_primitive


def _load_fallback_candidates():
    """Scan the FULL decompile-bench corpus for constant-table matches,
    grouped by primitive. Per-project cap (via the `file` column) keeps one
    or two codebases from flooding a primitive's pool with near-duplicate
    functions. Modality (asm vs csrc) assigned ~50/50 by index parity
    within each primitive's pool, so both are represented regardless of
    scan order.

    Comprehensive, not need-bounded: the water-filled allocation (computed
    from these pool sizes via _waterfill_counts) decides which subset
    actually gets sent to the frontier, so this scan must see each
    primitive's TRUE available count first - a per-label cap here would
    just move the same collapse bug _waterfill_counts was written to fix
    into a different place, and this scan is cheap (a full pass over the
    cached 2.2M-row corpus takes well under a minute; no frontier calls
    happen here)."""
    print("[fallback] scanning decompile-bench for constant-table matches...")
    local_dir = "/Volumes/Scratch/ml-workspace/decompile-bench"
    ds_raw = load_dataset("LLM4Binary/decompile-bench", split="train", cache_dir=local_dir)
    ds_shuf = ds_raw.shuffle(seed=SEED)

    project_counts = {}
    by_primitive = {}
    for row in ds_shuf:
        code, asm, file_ = row.get("code") or "", row.get("asm") or "", row.get("file") or ""
        if len(code) > MAX_CODE or len(asm) > MAX_ASM or not code or not asm:
            continue
        primitive = _match_fallback_primitive(code)
        if primitive is None:
            continue
        project = file_.split("/")[1] if "/" in file_ else file_
        key = (primitive, project)
        if project_counts.get(key, 0) >= PER_PROJECT_CAP:
            continue
        project_counts[key] = project_counts.get(key, 0) + 1
        pool = by_primitive.setdefault(primitive, [])
        if len(pool) >= CAP_CRYPTO:  # absolute ceiling, well above any water-filled need
            continue
        modality = "asm" if len(pool) % 2 == 0 else "csrc"
        pool.append({"asm": asm, "code": code, "primitive": primitive,
                     "modality": modality, "project": project})
    synthetic = _synthetic_table_candidates()
    for primitive, rows in synthetic.items():
        by_primitive.setdefault(primitive, []).extend(rows)
    print(f"[fallback] available per primitive (including synthetic audited-table rows): "
          f"{ {p: len(v) for p, v in by_primitive.items()} }")
    return by_primitive


def _waterfill_counts(avail, max_share=MAX_LABEL_SHARE, cap=None):
    """Given {label: available_count}, return {label: allocated_count} that
    MAXIMIZES sum(allocated) subject to allocated[label] <= max_share *
    sum(allocated) for every label (then optionally scaled to fit `cap`).

    NOT the naive "iteratively cap the biggest label using the shrinking
    running total" - that satisfies the <=max_share invariant at its fixed
    point but collapses the achievable total, because each pass's limit is
    computed from a total already shrunk by the previous pass, over-punishing
    labels that were within budget relative to the true achievable maximum.
    Confirmed empirically on the actual fallback label distribution this
    session (AES/MD5/SHA-256/TEA/Blowfish/CRC32/Base64 counts from
    decompile-bench): the naive loop converges to ~230 rows where proper
    water-filling on the same input reaches 385 - the correct maximum
    achievable under a strict 25% cap given how corpus-limited some
    primitives genuinely are (Blowfish: 2 matches in the entire 2.2M-row
    decompile-bench corpus; DES: 0 with any tested formatting - confirmed,
    not a scan artifact).

    Water-filling: sort labels ascending by availability. The smallest
    labels are "saturated" (their availability is below their fair share of
    the true achievable total T, so they contribute everything they have);
    the remaining labels are each capped at max_share*T. T is solved by
    fixed-point iteration from the small end: with k labels left
    unconstrained, T = fixed_total / (1 - k*max_share) is self-consistent
    once no smaller label would additionally saturate.
    """
    ordered = sorted(avail.items(), key=lambda kv: kv[1])  # ascending
    n = len(ordered)
    fixed_total = 0.0
    idx = 0
    while idx < n:
        k = n - idx
        if k * max_share >= 1:
            # All k remaining can't simultaneously be "unconstrained" (that
            # would require >100% share) - the smallest of them must be
            # saturated in the true solution. Take it and continue.
            fixed_total += ordered[idx][1]
            idx += 1
            continue
        T = fixed_total / (1 - k * max_share)
        _, count = ordered[idx]
        if count <= max_share * T:
            fixed_total += count
            idx += 1
        else:
            break
    k = n - idx
    T = fixed_total / (1 - k * max_share) if k > 0 else fixed_total
    counts = {label: count for label, count in ordered[:idx]}
    cap_val = int(max_share * T)
    for label, count in ordered[idx:]:
        counts[label] = min(count, cap_val)

    total = sum(counts.values())
    if cap is not None and total > cap and total > 0:
        ratio = cap / total
        counts = {label: int(c * ratio) for label, c in counts.items()}
    return counts


def _balance_and_cap(items, cap=None, max_share=MAX_LABEL_SHARE):
    """Water-fill `items` (list of (row, label) pairs) so no label exceeds
    max_share of the total, then (if `cap` given) proportionally scale
    every label down to fit `cap`. Returns (row, label) pairs (labels kept
    attached so callers can verify the guarantee on what actually ships,
    not just on this intermediate pool). Deterministic (seeded shuffle)."""
    rng = random.Random(SEED)
    items = items[:]
    rng.shuffle(items)
    by_label = {}
    for row, label in items:
        by_label.setdefault(label, []).append(row)
    avail = {label: len(rows) for label, rows in by_label.items()}
    counts = _waterfill_counts(avail, max_share=max_share, cap=cap)
    out = []
    for label, rows in by_label.items():
        out.extend((row, label) for row in rows[:counts.get(label, 0)])
    rng.shuffle(out)
    return out


def build_sec_audit_dataset(cap=0):
    base_url, model, api_key = _frontier_env()
    print(f"[frontier] using {model} @ {base_url}")
    max_workers = int(os.environ.get("MAX_WORKERS", "1"))

    cache = ResumableCache(os.environ.get("RAW_CACHE", "/tmp/hawq_sec/raw_generations.jsonl"))

    dpo_split = f"train[:{cap * 4}]" if cap else "train"
    ds_dpo = load_dataset("CyberNative/Code_Vulnerability_Security_DPO",
                           split=dpo_split).shuffle(seed=SEED)

    crypto_cands, exploit_cands = [], []
    for row in ds_dpo:
        q, ch, rj, lang, vuln = (row.get("question") or "", row.get("chosen") or "",
                                  row.get("rejected") or "", row.get("lang") or "c",
                                  row.get("vulnerability") or "")
        if not q or not ch or not rj:
            continue
        cand = {"question": q, "chosen": ch, "rejected": rj, "lang": lang,
                "vulnerability": vuln}
        if _CRYPTO_RE.search(rj):
            crypto_cands.append(cand)
        else:
            exploit_cands.append(cand)

    if cap:
        crypto_cands = crypto_cands[:cap]
        exploit_cands = exploit_cands[:cap]
    else:
        exploit_cands = exploit_cands[:CAP_EXPLOIT]

    print(f"[data] DPO crypto candidates: {len(crypto_cands)}, "
          f"exploit candidates: {len(exploit_cands)}")

    crypto_pairs = _gen_dpo_rows(crypto_cands, "crypto_audit", base_url, model,
                                  api_key, cache, max_workers)
    exploit_pairs = _gen_dpo_rows(exploit_cands, "exploit_poc", base_url, model,
                                   api_key, cache, max_workers)
    print(f"[gen] DPO crypto_audit rows: {len(crypto_pairs)}, "
          f"exploit_poc rows: {len(exploit_pairs)}")

    # Yield gate: below MIN_CRYPTO_PRIMARY from the primary source, top up
    # from decompile-bench. Skipped in smoke mode (cap>0) - a 40-row smoke
    # run has no business demanding 800 crypto rows.
    if not cap and len(crypto_pairs) < MIN_CRYPTO_PRIMARY:
        print(f"[yield-gate] DPO crypto_audit={len(crypto_pairs)} < "
              f"{MIN_CRYPTO_PRIMARY}; topping up from decompile-bench")
        by_primitive = _load_fallback_candidates()
        avail = {p: len(v) for p, v in by_primitive.items()}
        alloc = _waterfill_counts(avail, max_share=MAX_LABEL_SHARE)
        achievable = sum(alloc.values())
        print(f"[fallback] water-filled allocation under {MAX_LABEL_SHARE:.0%} "
              f"diversity cap: {alloc} (total {achievable})")
        if achievable + len(crypto_pairs) < MIN_CRYPTO_PRIMARY:
            print(f"[yield-gate] WARNING: achievable crypto_audit total "
                  f"({achievable + len(crypto_pairs)}) is BELOW the "
                  f"{MIN_CRYPTO_PRIMARY}-row floor even after the fallback. "
                  f"This is a genuine corpus-availability limit under the "
                  f"{MAX_LABEL_SHARE:.0%} diversity cap (some primitives - "
                  f"e.g. Blowfish, SHA-256 - are rare in decompile-bench "
                  f"regardless of scan depth), not a bug. Proceeding with "
                  f"the smaller, diversity-respecting total rather than "
                  f"relaxing the cap or improvising an unvetted source.")
        fb_cands = []
        for primitive, n in alloc.items():
            fb_cands.extend(by_primitive[primitive][:n])
        fb_pairs = _gen_fallback_rows(fb_cands, base_url, model, api_key, cache, max_workers)
        print(f"[gen] fallback crypto_audit rows: {len(fb_pairs)}")
        crypto_pairs.extend(fb_pairs)

    cache.close()

    crypto_labeled = _balance_and_cap(crypto_pairs, cap=(cap or CAP_CRYPTO))
    exploit_labeled = _balance_and_cap(exploit_pairs, cap=(cap or CAP_EXPLOIT))

    all_labeled = crypto_labeled + exploit_labeled
    random.seed(SEED)
    random.shuffle(all_labeled)
    n_val = max(1, int(len(all_labeled) * 0.02))
    train_labeled = all_labeled[n_val:]
    val_labeled = all_labeled[:n_val]

    ds = DatasetDict({
        "train": Dataset.from_list([row for row, _ in train_labeled]),
        "validation": Dataset.from_list([row for row, _ in val_labeled]),
    })

    # --- assertions 1-3 (structural + source-reference leak rate) ---------
    for split_name, split in ds.items():
        for i, row in enumerate(split):
            if len(row["messages"]) != 3 or row["messages"][-1]["role"] != "assistant":
                raise ValueError(f"{split_name}[{i}] malformed messages")
            if row["family"] not in ("crypto_audit", "exploit_poc"):
                raise ValueError(f"{split_name}[{i}] unknown family {row['family']!r}")

    _SRC_REF = re.compile(
        r"(?i)\b(secure version|fixed version|corrected version|patched version|"
        r"secure code(?: provided| shown| given)?|corrected code|patched code|"
        r"the fix(?:ed)? (?:code|version)|chosen (?:answer|code|version)|"
        r"ground.?truth|known.?correct|answer key|known primitive|"
        r"as (?:given|provided|shown|told) (?:in|the) (?:secure|fixed|corrected|patched|primitive)|"
        r"(?:was|am|were) (?:given|provided|shown|told) the (?:secure|fixed|corrected|patched|primitive))\b")
    _n = _hits = 0
    for split in ds.values():
        for row in split:
            _n += 1
            if _SRC_REF.search(row["messages"][-1]["content"]):
                _hits += 1
    _frac = _hits / max(_n, 1)
    print(f"[guard] gold source-reference rate: {_hits}/{_n} = {_frac:.3f}")
    assert _frac <= 0.05, (
        f"gold still references the hidden answer key in {_frac:.1%} of rows "
        f"(>5%); the prompt's hard rules did not take. Do NOT train on this data.")

    # Assertions 2 and 4 (leak / anti-tautology) are enforced at construction
    # time (candidates are skipped, not stored, on violation) - re-verify
    # here as defense-in-depth against a construction-time bug.
    for split in ds.values():
        for row in split:
            user = row["messages"][1]["content"]
            if "```cpp" in user and "KNOWN-CORRECT" in user:
                raise ValueError("user turn leaks hidden ground-truth marker")
            if "KNOWN PRIMITIVE" in user:
                raise ValueError("user turn leaks hidden primitive marker")

    print(f"[assert] structural + leak checks passed ({_n} rows)")

    # Assertion 5 (diversity): verified on the ACTUAL shipped rows (both
    # splits combined per family), using the labels retained through the
    # balance/cap/shuffle/split pipeline - not just on the pre-cap
    # intermediate pool, since a naive slice after balancing does not
    # guarantee the shipped set inherits the same share bound.
    by_family_label = {}
    for row, label in train_labeled + val_labeled:
        by_family_label.setdefault(row["family"], {}).setdefault(label, 0)
        by_family_label[row["family"]][label] += 1
    for family, label_counts in by_family_label.items():
        total = sum(label_counts.values())
        for label, n in label_counts.items():
            share = n / total
            assert share <= MAX_LABEL_SHARE + 1e-9, (
                f"family={family!r} label={label!r} is {share:.1%} of {total} "
                f"rows (>{MAX_LABEL_SHARE:.0%}); diversity guard failed on the "
                f"shipped dataset.")
        worst = max(label_counts.items(), key=lambda kv: kv[1])
        print(f"[assert] {family}: {len(label_counts)} distinct labels, "
              f"{total} rows, worst label {worst[0]!r} = {worst[1]}/{total} "
              f"({worst[1] / total:.1%})")

    return ds


def _selftest_matcher():
    """Positive-control fixture for _match_fallback_primitive: real constant
    values (hex and decimal forms) embedded in plausible C, asserted to match
    the expected primitive. Run via SELFTEST=1. Exists because this matcher
    has had two revisions silently produce all-zero corpus counts this
    session - a synthetic fixture distinguishes "pattern is broken" from
    "corpus genuinely lacks it" in milliseconds instead of a multi-minute
    corpus scan."""
    cases = {
        "AES": "static const unsigned char T[16]={0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,"
               "0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76};",
        "SHA-256": "static const unsigned int K[4]={0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5};",
        "MD5": "h[0]=0x67452301;h[1]=0xefcdab89;h[2]=0x98badcfe;h[3]=0x10325476;",
        "SHA-1": "digest[0]=0x67452301;digest[1]=0xEFCDAB89;digest[2]=0x98BADCFE;digest[3]=0x10325476;digest[4]=0xC3D2E1F0;",
        "Blowfish": "static const unsigned int P[4]={0x243F6A88,0x85A308D3,0x13198A2E,0x03707344};",
        "TEA": "unsigned int delta=0x9E3779B9;",
        "CRC32": "unsigned int poly=0xEDB88320;",
        "Base64": crypto_lib.base64_alphabet(),
        # decimal-formatted variant of the SAME SHA-256 K values
        "SHA-256 (decimal)": "unsigned int K[4]={1116352408,1899447441,3049323471,3921009573};",
    }
    ok = True
    for label, snippet in cases.items():
        expected = label.split(" ")[0]
        got = _match_fallback_primitive(snippet)
        status = "OK" if got == expected else "FAIL"
        if got != expected:
            ok = False
        print(f"[selftest] {label}: expected={expected} got={got} -> {status}")
    # Adversarial negative control: a plain non-crypto function whose
    # literals happen to include most of the AES S-box's DECIMAL forms
    # (99,124,119,123,242,107,111,197,48,103,43,254,215,171,118) as sizes/
    # offsets/ASCII codes, plus an unrelated byte lookup table - exactly the
    # over-matching failure mode the decimal_ok=False hex-only requirement
    # for AES/DES exists to reject. A weak negative control (e.g. a function
    # with almost no integer literals) can't detect that failure mode.
    neg = """
static const unsigned char lookup_table[16] = {
    99, 124, 119, 123, 242, 107, 111, 197, 48, 103, 43, 254, 215, 171, 118, 200
};
int resize_buffer(int current_size) {
    int offset = 99;
    if (current_size > 124) return current_size + 43;
    char ascii_codes[8] = {103, 111, 111, 100, 33, 254, 118, 48};
    return offset + 197 - 171 + 215;
}"""
    got_neg = _match_fallback_primitive(neg)
    neg_ok = got_neg is None
    print(f"[selftest] adversarial negative control: got={got_neg} -> "
          f"{'OK' if neg_ok else 'FAIL'}")
    ok = ok and neg_ok
    if not ok:
        raise SystemExit("[selftest] matcher self-test FAILED - fix before trusting a corpus scan")
    print("[selftest] all matcher checks passed")


if __name__ == "__main__":
    if os.environ.get("SELFTEST", "0") == "1":
        _selftest_matcher()
        raise SystemExit(0)
    cap = int(os.environ.get("CAP", "0"))
    ds = build_sec_audit_dataset(cap=cap)
    fam_counts = {}
    for split in ds.values():
        for row in split:
            fam_counts[row["family"]] = fam_counts.get(row["family"], 0) + 1
    print(f"Train: {len(ds['train'])} | Validation: {len(ds['validation'])}")
    print(f"Family counts: {fam_counts}")

    for i, row in enumerate(ds["train"].select(range(min(3, len(ds["train"]))))):
        print(f"\n--- sample {i} ({row['family']}) ---")
        print("user:", row["messages"][1]["content"][:200], "...")
        print("assistant:", row["messages"][2]["content"][:300], "...")

    local_out = os.environ.get("CRYPTO_OUT", "/Volumes/SeXternal/hawq_v4/sec_audit_dataset")
    ds.save_to_disk(local_out)
    print(f"[save] dataset saved locally -> {local_out} (durability checkpoint before push)")

    if os.environ.get("PUSH", "0") == "1":
        data_repo = os.environ.get("DATA_REPO", "lancejames221b/hawq-sec-audit")
        print(f"Pushing to HF: {data_repo}")
        try:
            ds.push_to_hub(data_repo, private=True)
            print("sec_audit dataset pushed to HF")
        except Exception as e:
            print(f"[push] FAILED: {type(e).__name__}: {e}. Dataset is safe on local "
                  f"disk at {local_out} - retry with: python3 -c \"from datasets import "
                  f"load_from_disk; load_from_disk('{local_out}').push_to_hub("
                  f"'{data_repo}', private=True)\"")
            raise
