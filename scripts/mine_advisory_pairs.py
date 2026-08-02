#!/usr/bin/env python3
"""mine_advisory_pairs.py - HAWQ v1.3 retrain plan, Step 3.

Mine the omp/pi advisory-correction corpus into DPO pairs using a BOUNDED
LOCAL WINDOW, not build_agent_dpo.py's whole-session prompt (mean 774,020
chars, max 2,695,001 - at MAX_PROMPT_LEN=1024 only a tail survives, so the
advisory and the turn it corrects are usually truncated apart; see Step 0
of the v1.3 retrain plan). Do NOT extend build_agent_dpo.py.

Schema (confirmed against real session files on generic - see managed skill
`omp-advisor-jsonl-schema-for-dpo-mining`, and empirically re-verified while
writing this script):
  - Advisories are `type: "custom_message", customType: "advisor"` entries,
    content = '<advisory severity="..." guidance="...">BODY</advisory>'.
  - `parentId` overwhelmingly resolves to a `toolResult`-role message entry
    (the advisor reviews a tool call's OUTCOME), whose own ancestry chains
    through non-message `custom`/`tool_execution_*` bookkeeping entries back
    to the assistant turn that issued the call. Rather than following that
    multi-hop parentId chain, this miner resolves to the NEAREST PRECEDING
    assistant-role message entry in the same file as the first-hop
    resolution target - the exact approach in scripts/build_agent_dpo.py's
    `_resolve_rejected_idx`, confirmed correct against a real capture.
  - ~88% of advisor entries resolve CROSS-FILE in general (advisor lives in
    a nested subagent/__advisor.jsonl transcript, parentId points into the
    PARENT session's own file) - this miner builds ONE GLOBAL
    id -> (file, index) index across every *.jsonl under --root before
    resolving any parentId, rather than a single-file resolution (which the
    schema skill measured as recovering only ~12% of pairs on its own).
  - Message roles observed in practice: assistant, user, developer (a
    system-level reminder/instruction role - NOT literally "system"),
    toolResult (a tool call's return, a DISTINCT top-level message role,
    not a nested content block as scripts/build_agent_dpo.py's block_text
    assumes). Mapped onto the stored prompt's 4-role vocabulary
    (system|user|assistant|tool - scripts/dpo_common.py's _ROLE_SPLIT_RE)
    as: developer->system, toolResult->tool, user->user, assistant->assistant.
  - Content blocks: "text", "thinking" (excluded - a separate channel from
    the served `content` string, never concatenated into it by
    chat_template.jinja), "toolCall" (fields: name, arguments - NOT
    "input"; rendered via omp_surface.render_tool_calls to reproduce the
    model's actual deployed <tool_call><function=...> wire format
    character-for-character - cross-verified against a real jinja2 render
    of chat_template.jinja, including the parallel-tool-call separator
    rule - rather than build_agent_dpo.py's simplified "[tool_use:name]
    {json}" form, because Step 3 requires the real wire format).

`chosen` is FRONTIER-GENERATED (unlike build_agent_dpo.py's "next assistant
turn" - the advisory states what was wrong but rarely emits the literal
corrected turn, and "next turn" is not reliably the fix: an advisory can be
surfaced and not acted on, or acted on several turns later past an
unrelated turn). ONE frontier call per candidate does both (a) emit the
corrected assistant turn under a strict grounding constraint (retry once on
an ungrounded literal, drop on second failure) and (b) classify the pair
into one of the 8 v1.3 families (drop if none fit - not given a 9th label).

completion(model="slow") in the plan text is the `eval`-tool primitive, not
callable from this standalone script. Reuses the FRONTIER_* env-var
convention and hard local-endpoint abort already established in
scripts/build_re_analysis.py / scripts/build_sec_audit.py (self-labeling
with HAWQ would silently contaminate `chosen` with the very defect this
retrain is trying to fix).

Usage:
    FRONTIER_BASE_URL=... FRONTIER_MODEL=... FRONTIER_API_KEY=... \\
    python3 scripts/mine_advisory_pairs.py \\
        --root /tmp/hawq_dpo/generic_sessions_mirror \\
        --out /tmp/hawq_dpo/advisory_pairs_v13.jsonl \\
        --window 6 --cap-per-family 200 --max-workers 16 [--dry-run]

--dry-run scans and resolves candidates (including dedup and the
write-fallback ban) WITHOUT calling the frontier, and prints the same
stage-count breakdown scripts/build_agent_dpo.py's --dry-run does. Always
run this first (per the schema skill: a resolution bug silently produces a
near-zero corpus that looks exactly like "the corpus is just thin").
"""
import argparse
import hashlib
import json
import os
import re
import ssl
import sys
import threading
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from omp_surface import render_tool_calls  # noqa: E402

# Explicit certifi CA bundle rather than relying on ssl.get_default_verify_paths():
# confirmed in this environment that a bash-tool interactive shell and a
# hub-start-launched process can resolve different default trust stores (a
# known class of gotcha - hub-managed process env differs from interactive
# shell), and the default path failed with CERTIFICATE_VERIFY_FAILED under
# hub start despite working interactively. certifi's bundle is
# environment-independent.
try:
    import certifi
    _SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CONTEXT = ssl.create_default_context()

DEFAULT_ROOT = Path.home() / ".omp" / "agent" / "sessions"
RETRIES = 2

VALID_FAMILIES = {
    "adv_destructive_edit", "adv_stale_state", "adv_scope_drift",
    "adv_literal_fidelity", "adv_unverified_claim", "adv_hard_loop",
    "adv_premature_done", "adv_malformed_call",
}
_SEV_ORDER = {"blocker": 0, "concern": 1, "nit": 2}

# Per Assumptions & contingencies: advisories are themselves model-generated
# and occasionally wrong - this specific instruction (observed twice in the
# Step 0 captures) contradicts the hashline objective and would train the
# wrong behavior if mined.
_WRITE_FALLBACK_RE = re.compile(r"just .{0,20}(write|rewrite) the (whole|full|entire)", re.IGNORECASE)

_SEVERITY_RE = re.compile(r'severity="([^"]*)"')
_BODY_RE = re.compile(r"<advisory[^>]*>\s*(.*?)\s*</advisory>", re.DOTALL)
_TOKEN_RE = re.compile(r"[A-Za-z0-9_/.\-]{8,}")

# developer (system-level reminders in practice, not literally "system") and
# toolResult (a distinct message role for a tool call's return, not a nested
# content block) map onto the stored prompt's 4-role vocabulary.
_ROLE_MAP = {
    "assistant": "assistant", "user": "user",
    "system": "system", "developer": "system",
    "toolResult": "tool", "tool": "tool",
}


# ---------------------------------------------------------------------------
# JSONL loading / entry-type predicates
# ---------------------------------------------------------------------------

def load_jsonl(path):
    entries = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def is_message(e):
    return e.get("type") == "message"


def is_assistant(e):
    return is_message(e) and e.get("message", {}).get("role") == "assistant"


def is_advisor(e):
    return e.get("type") == "custom_message" and e.get("customType") == "advisor"


def find_session_files(root):
    root = Path(root)
    if not root.exists():
        return []
    return sorted(root.rglob("*.jsonl"))


# ---------------------------------------------------------------------------
# Cross-file id index + parentId resolution
# ---------------------------------------------------------------------------

def build_global_index(files):
    """(file_entries: {Path: [entry,...]}, id_index: {id: (Path, idx)})
    across EVERY file, so a parentId minted in a nested subagent/advisor
    transcript resolves into its parent session's own file."""
    file_entries = {}
    id_index = {}
    for f in files:
        entries = load_jsonl(f)
        file_entries[f] = entries
        for i, e in enumerate(entries):
            eid = e.get("id")
            if eid and eid not in id_index:
                id_index[eid] = (f, i)
    return file_entries, id_index


def resolve_rejected(id_index, file_entries, parent_id):
    """(file, idx) of the nearest preceding (or exact) assistant-role
    message entry, starting from parent_id's first-hop resolution.
    Mirrors build_agent_dpo.py's _resolve_rejected_idx, extended to
    cross-file via the global index above."""
    loc = id_index.get(parent_id)
    if loc is None:
        return None
    f, idx = loc
    entries = file_entries[f]
    if is_assistant(entries[idx]):
        return f, idx
    for k in range(idx - 1, -1, -1):
        if is_assistant(entries[k]):
            return f, k
    return None


# ---------------------------------------------------------------------------
# Content rendering (text + toolCall blocks -> the served `content` string)
# ---------------------------------------------------------------------------

def render_message_content(entry):
    """type:'message' entry -> the flat string stored in prompt/chosen/
    rejected. ALL text blocks are concatenated first (thinking excluded -
    a separate channel, never part of the served content string), then ALL
    toolCall blocks are rendered in order via omp_surface.render_tool_calls
    - matching how chat_template.jinja treats `content` (already-flattened
    text) and `tool_calls` (a separate structured list) as independent
    fields, regardless of how session JSONL interleaves the blocks."""
    msg = entry.get("message", {})
    content = msg.get("content")
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    text_parts = []
    calls = []
    for block in content:
        if not isinstance(block, dict):
            continue
        t = block.get("type")
        if t == "text":
            text_parts.append(block.get("text", ""))
        elif t in ("toolCall", "tool_use"):
            name = block.get("name", "")
            args = block.get("arguments", block.get("input", {})) or {}
            calls.append((name, args))
        # "thinking" and any other block type intentionally ignored.
    text = "\n".join(p for p in text_parts if p is not None)
    return render_tool_calls(text, calls)


def _cap_block(text, cap=350):
    """Hard-cap one window block at `cap` chars, truncating from the LEFT
    with a leading ellipsis marker (Step 3: "truncating from the left").
    Default lowered from an earlier 4000 to 350: at up to 6 blocks that
    bounds the whole window near ~2100 chars (~550-600 tokens at ~3.5-4
    chars/token for mixed English/code text), comfortably inside the
    training pipeline's max_prompt_tokens=1024 budget - a 4000-char cap
    left most windows 2-6x over budget, and dpo_common.
    truncate_messages_to_budget's whole-turn-drop truncation then produced
    a stored (post-truncation) prompt materially SMALLER than what
    `chosen` was grounded against at generation time, up to and including
    dropping the row entirely (see build_window's docstring)."""
    if len(text) <= cap:
        return text
    return "\u2026" + text[-(cap - 1):]


def build_window(entries, before_idx, window_n, block_cap=350):
    """Last `window_n` type:'message' entries preceding entries[before_idx],
    walked backward then reversed to chronological order, each rendered
    "[role]\\ntext" (role-mapped, see _ROLE_MAP) and hard-capped at
    `block_cap` chars. Entries with an unmapped role or empty rendered
    text are skipped without consuming a window slot.

    GUARANTEES the window contains at least one 'user' turn if one exists
    anywhere earlier in `entries`: dpo_common.truncate_messages_to_budget
    (and the underlying chat template) DROP THE ENTIRE ROW if no user turn
    is present anywhere in the prompt (confirmed empirically: Qwen's
    template raises "No user query found in messages" otherwise) - a
    window built from "last N messages" alone routinely has none, since
    agentic sessions commonly run many consecutive assistant/tool turns
    after a single user instruction, and mining measured >85% of advisory
    windows this way as having no user turn (preflight survival on the
    adv_* sources came in at 3-13% before this fix, vs 100% for the
    sources whose prompts always include one). If the nearest user turn
    falls outside the last `window_n` messages, it is walked back to and
    PREPENDED (still capped), counting as an EXTRA slot beyond window_n
    rather than displacing more-recent context."""
    collected = []
    earliest_idx = before_idx
    for k in range(before_idx - 1, -1, -1):
        e = entries[k]
        if e.get("type") != "message":
            continue
        role = _ROLE_MAP.get(e.get("message", {}).get("role"))
        if role is None:
            continue
        text = render_message_content(e).strip()
        if not text:
            continue
        collected.append((role, text))
        earliest_idx = k
        if len(collected) >= window_n:
            break
    if not any(role == "user" for role, _ in collected):
        for k in range(earliest_idx - 1, -1, -1):
            e = entries[k]
            if e.get("type") != "message":
                continue
            if _ROLE_MAP.get(e.get("message", {}).get("role")) != "user":
                continue
            text = render_message_content(e).strip()
            if text:
                collected.append(("user", text))
            break  # nearest preceding user turn only - not a full rescan
    collected.reverse()
    return "\n\n".join(f"[{role}]\n{_cap_block(text, block_cap)}" for role, text in collected)



# ---------------------------------------------------------------------------
# Advisory parsing
# ---------------------------------------------------------------------------

def parse_advisory(entry):
    """(severity, body) from a custom_message/advisor entry's `content`."""
    content = entry.get("content", "") or ""
    sev_m = _SEVERITY_RE.search(content)
    body_m = _BODY_RE.search(content)
    severity = sev_m.group(1) if sev_m else ""
    body = body_m.group(1).strip() if body_m else content.strip()
    return severity, body


# ---------------------------------------------------------------------------
# Candidate collection (scan + resolve + dedup + write-fallback filter)
# ---------------------------------------------------------------------------

def collect_candidates(root, window_n):
    files = find_session_files(root)
    file_entries, id_index = build_global_index(files)
    stats = Counter()
    stats["session_files"] = len(files)
    candidates = []
    seen_dedup = set()
    for f, entries in file_entries.items():
        for e in entries:
            if not is_advisor(e):
                continue
            stats["advisor_entries"] += 1
            loc = resolve_rejected(id_index, file_entries, e.get("parentId"))
            if loc is None:
                stats["unresolved_parent"] += 1
                continue
            rf, ridx = loc
            rentries = file_entries[rf]
            severity, body = parse_advisory(e)
            if not body:
                stats["empty_advisory_body"] += 1
                continue
            if _WRITE_FALLBACK_RE.search(body):
                stats["dropped_write_fallback"] += 1
                continue
            rejected = render_message_content(rentries[ridx])
            if not rejected.strip():
                stats["empty_rejected"] += 1
                continue
            window = build_window(rentries, ridx, window_n)
            dedup_key = (severity, body, rejected)
            if dedup_key in seen_dedup:
                stats["dedup_dropped"] += 1
                continue
            seen_dedup.add(dedup_key)
            candidates.append({
                "severity": severity,
                "advisory_body": body,
                "rejected": rejected,
                "window": window,
                "source_file": str(rf),
            })
    stats["unique_candidates"] = len(candidates)
    candidates.sort(key=lambda c: _SEV_ORDER.get(c["severity"], 3))
    return candidates, stats


# ---------------------------------------------------------------------------
# Frontier call (FRONTIER_* convention copied from scripts/build_re_analysis.py
# / scripts/build_sec_audit.py - hard-aborts on a local/HAWQ endpoint so this
# script cannot silently self-label with the model it is trying to fix).
# ---------------------------------------------------------------------------

def _frontier_env():
    base_url = os.environ.get("FRONTIER_BASE_URL")
    model = os.environ.get("FRONTIER_MODEL")
    api_key = os.environ.get("FRONTIER_API_KEY")
    missing = [n for n, v in (("FRONTIER_BASE_URL", base_url),
                               ("FRONTIER_MODEL", model),
                               ("FRONTIER_API_KEY", api_key)) if not v]
    if missing:
        raise SystemExit(
            f"[frontier] required env missing: {', '.join(missing)}. This "
            f"miner must call a real frontier model, not the local HAWQ "
            f"server, to generate `chosen`. Use --dry-run to scan/resolve "
            f"candidates without a frontier call, or see the plan's "
            f"Assumptions & contingencies for the fenced-code-block fallback.")
    if re.match(r"^(hawq|razorstrike)", model, re.IGNORECASE):
        raise SystemExit(
            f"[frontier] FRONTIER_MODEL={model!r} looks like a HAWQ/RazorStrike "
            f"model id, not a frontier model. Self-labeling `chosen` with "
            f"HAWQ's own output would train toward the defect being fixed. Aborting.")
    local_hosts = ("127.0.0.1:1234", "localhost:1234", "generic:1234")
    if any(h in base_url for h in local_hosts):
        raise SystemExit(
            f"[frontier] FRONTIER_BASE_URL={base_url!r} points at a local LM "
            f"Studio endpoint (serves HAWQ itself). Aborting - supply a real "
            f"frontier endpoint (e.g. https://api.openai.com/v1).")
    return base_url.rstrip("/"), model, api_key


def _frontier_call(base_url, model, api_key, system, user):
    url = f"{base_url}/chat/completions"
    body = {
        "model": model,
        "reasoning_effort": os.environ.get("FRONTIER_REASONING_EFFORT", "low"),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
        "max_tokens": int(os.environ.get("FRONTIER_MAX_TOKENS", "1200")),
    }
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
    for attempt in range(RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=90, context=_SSL_CONTEXT) as r:
                d = json.load(r)
            text = d["choices"][0]["message"]["content"].strip()
            if text:
                return text
        except Exception as e:
            if attempt == RETRIES:
                print(f"[frontier] call failed after {RETRIES + 1} attempts: "
                      f"{type(e).__name__}: {e}")
            else:
                time.sleep(2 ** attempt)  # backoff: 1s, 2s before the 2nd/3rd try
    return None


FRONTIER_SYSTEM = (
    "You are labeling AI coding-agent training data for a preference-tuning "
    "(DPO) dataset. You will be shown a bounded conversation window, an "
    "assistant turn that was flagged as a mistake, and an advisory note "
    "explaining what was wrong. Respond with STRICT JSON only (no markdown "
    "fences, no prose outside the JSON object):\n"
    '{"chosen": "<the assistant turn that should have happened instead>", '
    '"family": "<one of the family ids below, or null if none fit>"}\n\n'
    "Emit only the assistant turn that the advisory says should have "
    "happened. Copy every literal - paths, tags, line numbers, identifiers, "
    "commands - VERBATIM from the window, the mistake shown below, or the "
    "advisory (e.g. if the mistake already targeted the right path/tag and "
    "only the tool-call SHAPE was wrong, reuse that same path/tag - do not "
    "invent a different one). Invent nothing: if a path, tag, or number is "
    "not present in the window, the mistake, or the advisory, do not use it. "
    "If the corrected turn is a tool call, render "
    "it exactly as: <tool_call>\\n<function=NAME>\\n<parameter=KEY>\\nVALUE"
    "\\n</parameter>\\n...\\n</function>\\n</tool_call>\n\n"
    "Family ids (choose exactly one, or null if the advisory does not "
    "describe a real behavioral mistake in one of these categories):\n"
    "  adv_destructive_edit  - a destructive/corrupting edit: deleted or "
    "altered code/content beyond what was intended\n"
    "  adv_stale_state       - acted on stale/remembered/invented state "
    "instead of re-reading current state after a change\n"
    "  adv_scope_drift       - touched files or did work beyond what was asked\n"
    "  adv_literal_fidelity  - mis-typed or altered a literal (path, UUID, "
    "tag, identifier, line number) instead of copying it verbatim\n"
    "  adv_unverified_claim  - stated a result/summary without having "
    "verified it, or fabricated content\n"
    "  adv_hard_loop         - repeated an identical failing action\n"
    "  adv_premature_done    - claimed completion before the work was done\n"
    "  adv_malformed_call    - a malformed tool-call argument or shape\n"
)

FRONTIER_USER_TMPL = (
    "=== CONVERSATION WINDOW (context leading up to the mistake) ===\n"
    "{window}\n\n"
    "=== THE MISTAKE (rejected assistant turn) ===\n"
    "{rejected}\n\n"
    "=== ADVISORY (severity={severity}) ===\n"
    "{advisory_body}\n\n"
    "Respond with the JSON object described in the system prompt."
)


def _parse_frontier_json(text):
    text = text.strip()
    m = re.match(r"^```(?:json)?\s*\n(.*)\n```\s*$", text, re.DOTALL)
    if m:
        text = m.group(1)
    try:
        d = json.loads(text)
    except json.JSONDecodeError:
        m2 = re.search(r"\{.*\}", text, re.DOTALL)
        if not m2:
            return None
        try:
            d = json.loads(m2.group(0))
        except json.JSONDecodeError:
            return None
    return d if isinstance(d, dict) else None


def _ungrounded_tokens(chosen_text, grounding_text):
    """Tokens in chosen_text matching [A-Za-z0-9_/.-]{8,} absent verbatim
    from grounding_text (window + rejected turn + advisory concatenated)."""
    out, seen = [], set()
    for m in _TOKEN_RE.finditer(chosen_text):
        tok = m.group(0)
        if tok in seen:
            continue
        seen.add(tok)
        if tok not in grounding_text:
            out.append(tok)
    return out


# ---------------------------------------------------------------------------
# Resumable cache (mirrors scripts/build_sec_audit.py's ResumableCache - a
# multi-hundred-call paid frontier run must not lose already-paid-for work
# to a crash, and must not re-pay for a row a prior partial run finished).
# ---------------------------------------------------------------------------

class ResumableCache:
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


def _cache_key(cand):
    # Includes the WINDOW content, not just window_n/block_cap as separate
    # params: `chosen` is grounded against cand["window"] specifically, so
    # any change to window construction (cap, guaranteed-user-turn logic,
    # or anything else) must invalidate the cache and force regeneration -
    # reusing a `chosen` generated/grounded against a DIFFERENT window
    # would silently store literals absent from the new stored prompt.
    raw = "|".join([cand["severity"], cand["advisory_body"], cand["rejected"], cand["window"]])
    return hashlib.sha256(raw.encode()).hexdigest()


def gen_one(cand, base_url, model, api_key, cache, stats, lock):
    key = _cache_key(cand)
    cached = cache.get(key)
    if cached is not None:
        status, row = cached
    else:
        grounding_text = cand["window"] + "\n" + cand["rejected"] + "\n" + cand["advisory_body"]
        user = FRONTIER_USER_TMPL.format(
            window=cand["window"] or "(no preceding context)",
            rejected=cand["rejected"], severity=cand["severity"] or "unknown",
            advisory_body=cand["advisory_body"])
        status, row = "drop_call_failed", None
        retry_note = ""
        for attempt in range(2):
            text = _frontier_call(base_url, model, api_key, FRONTIER_SYSTEM, user + retry_note)
            if not text:
                break
            d = _parse_frontier_json(text)
            if not d or not isinstance(d.get("chosen"), str) or not d["chosen"].strip():
                status = "drop_unparseable"
                break
            family = d.get("family")
            if family not in VALID_FAMILIES:
                status = "drop_family"
                break
            chosen = d["chosen"].strip()
            bad = _ungrounded_tokens(chosen, grounding_text)
            if not bad:
                status = "ok"
                row = {
                    "prompt": cand["window"],
                    "chosen": chosen,
                    "rejected": cand["rejected"],
                    "source": family,
                    "severity": cand["severity"],
                    "advisory_text": cand["advisory_body"],
                    "source_session": cand["source_file"],
                }
                break
            if attempt == 0:
                retry_note = (
                    f"\n\nYour previous answer used token(s) {bad!r} that do "
                    f"NOT appear verbatim in the window or advisory above. Do "
                    f"not invent literals - copy only what is present. Try again.")
                continue
            status = "drop_ungrounded"
        if status != "drop_call_failed":  # transient/infra failure - retry on next run, don't poison the cache
            cache.put(key, (status, row))
    with lock:
        stats[status] += 1
    return row


# ---------------------------------------------------------------------------
# Post-hoc per-family cap (Step 3: "Cap at 200 pairs per family ... prefer
# blocker over concern over nit when capping" - applied AFTER labeling, so
# the cap operates on real (frontier-assigned) families, not a guess).
# ---------------------------------------------------------------------------

def apply_family_cap(pairs, cap):
    by_family = defaultdict(list)
    for p in pairs:
        by_family[p["source"]].append(p)
    out = []
    family_counts = {}
    for fam, plist in by_family.items():
        plist.sort(key=lambda p: _SEV_ORDER.get(p["severity"], 3))
        kept = plist[:cap]
        family_counts[fam] = (len(kept), len(plist))
        out.extend(kept)
    return out, family_counts


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--out", default="/tmp/hawq_dpo/advisory_pairs_v13.jsonl")
    ap.add_argument("--window", type=int, default=6)
    ap.add_argument("--cap-per-family", type=int, default=200)
    ap.add_argument("--target", type=int, default=1200,
                     help="informational only - logged against the post-cap "
                          "total, not enforced directly (the per-family cap "
                          "is what's enforced)")
    ap.add_argument("--max-workers", type=int, default=16)
    ap.add_argument("--max-candidates", type=int, default=0,
                     help="0 = process every deduped candidate")
    ap.add_argument("--cache", default="/tmp/hawq_dpo/advisory_mine_cache.jsonl")
    ap.add_argument("--dry-run", action="store_true",
                     help="scan/resolve/dedup only, no frontier calls")
    args = ap.parse_args()

    print(f"[scan] root={args.root}")
    candidates, scan_stats = collect_candidates(args.root, args.window)
    print(f"[scan] session_files={scan_stats['session_files']} "
          f"advisor_entries={scan_stats['advisor_entries']} "
          f"unresolved_parent={scan_stats['unresolved_parent']} "
          f"empty_advisory_body={scan_stats['empty_advisory_body']} "
          f"dropped_write_fallback={scan_stats['dropped_write_fallback']} "
          f"empty_rejected={scan_stats['empty_rejected']} "
          f"dedup_dropped={scan_stats['dedup_dropped']}")
    print(f"[scan] {scan_stats['unique_candidates']} unique candidates "
          f"after resolution + dedup")
    sev_hist = Counter(c["severity"] or "(none)" for c in candidates)
    print(f"[scan] severity histogram: {dict(sev_hist)}")

    if args.dry_run:
        print("[dry-run] not calling frontier. Re-run without --dry-run to "
              "generate `chosen` + label families.")
        if candidates:
            print("\n[sample candidate]")
            c = candidates[0]
            print(f"  severity: {c['severity']}")
            print(f"  source_file: {c['source_file']}")
            print(f"  advisory_body[:200]: {c['advisory_body'][:200]!r}")
            print(f"  rejected[:200]: {c['rejected'][:200]!r}")
            print(f"  window chars: {len(c['window'])}")
        return

    base_url, model, api_key = _frontier_env()
    if args.max_candidates:
        candidates = candidates[: args.max_candidates]
    print(f"[frontier] {len(candidates)} candidates -> {base_url} model={model} "
          f"max_workers={args.max_workers}")

    cache = ResumableCache(args.cache)
    gen_stats = Counter()
    lock = threading.Lock()
    pairs = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = [ex.submit(gen_one, c, base_url, model, api_key, cache, gen_stats, lock)
                for c in candidates]
        for i, fut in enumerate(futs, 1):
            row = fut.result()
            if row is not None:
                pairs.append(row)
            if i % 100 == 0 or i == len(futs):
                print(f"[frontier] {i}/{len(futs)} processed, "
                      f"{len(pairs)} kept so far, stats={dict(gen_stats)}")
    cache.close()

    print(f"[frontier] done. generation stats: {dict(gen_stats)}")
    print(f"[frontier] {len(pairs)} pairs before per-family cap")

    kept, family_counts = apply_family_cap(pairs, args.cap_per_family)
    print(f"\n[family] cap={args.cap_per_family} (kept/available per family):")
    for fam in sorted(VALID_FAMILIES):
        k, avail = family_counts.get(fam, (0, 0))
        print(f"  {fam:24s} {k:4d} / {avail:4d}")
    sev_final = Counter(p["severity"] or "(none)" for p in kept)
    print(f"\n[result] {len(kept)} pairs total (target ~{args.target})")
    print(f"[result] severity histogram: {dict(sev_final)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for p in kept:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"[write] {len(kept)} pairs -> {out_path}")


if __name__ == "__main__":
    main()
