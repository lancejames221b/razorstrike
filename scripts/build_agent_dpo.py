#!/usr/bin/env python3
"""Phase 6 - Mine omp/pi agent-session JSONL for real advisory-correction
trajectories and build a DPO preference dataset (for a swappable
agent-behavior LoRA on HAWQ-v1 base, kept SEPARATE from HAWQ-SEC-RE-v2 so the
two specializations don't fight for the same weights).

Confirmed on-disk schema (verified against real session files, not inferred):
  Advisories are a DISTINCT top-level JSONL entry, not nested inside a
  "message" entry's content blocks:
      {"type": "custom_message", "customType": "advisor",
       "content": "<advisory severity=\"...\" guidance=\"...\">...</advisory>",
       "id": "<hex>", "parentId": "<hex-of-the-message-entry-it-critiques>", ...}
  `parentId` resolves EXACTLY to the `id` of the `type:"message"` (assistant)
  entry the advisory was attached to -- this replaces proximity/nearest-
  neighbor guessing with an exact link.

Design (see managed skill `omp-session-traces` for background; this
supersedes its illustrative proximity-based sketch with the exact parentId
linkage confirmed here):
  - A DPO pair is a TRAJECTORY DIFF, not the advisory text itself:
      prompt   = conversation state right before the flawed action (the
                 advisory content is NOT included -- production has no
                 reviewer injecting help, so the prompt must match what the
                 policy actually sees at inference time).
      rejected = the assistant turn the advisory's parentId points at (the
                 mistake it reacted to).
      chosen   = the next assistant-role message entry after that turn (what
                 was actually done afterward -- not the advisory's suggested
                 text, which may have been followed loosely or not at all).
  - Pairs where chosen and rejected are near-identical (the advisory was
    surfaced but not actually acted on) are DISCARDED, not inverted.
  - Covers BOTH the standing background-advisor injections and any
    Opus-model plan-review/replan pass, since both use this same
    custom_message/advisor injection mechanism -- one extractor, one corpus.

Usage:
    python3 scripts/build_agent_dpo.py --out /tmp/agent_dpo_pairs.jsonl [--dry-run]

--dry-run prints corpus size (sessions found, advisor entries found, pairs
kept after the acted-on filter, WITH per-stage counts) without writing the
dataset -- use this FIRST to decide whether the real pair count justifies a
training run at all.
"""
import argparse
import json
from difflib import SequenceMatcher
from pathlib import Path

SESSION_ROOTS = [
    Path.home() / ".omp" / "agent" / "sessions",
    Path.home() / ".pi" / "agent" / "sessions",
]
SIMILARITY_DISCARD_THRESHOLD = 0.92  # chosen vs rejected text similarity above
                                      # this = advisory was not really acted on


def load_jsonl(p: Path):
    out = []
    with open(p, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def block_text(block) -> str:
    if isinstance(block, str):
        return block
    if not isinstance(block, dict):
        return ""
    t = block.get("type")
    if t == "text":
        return block.get("text", "")
    if t == "thinking":
        return block.get("thinking", "")
    if t in ("tool_use", "toolCall"):
        args = block.get("input", block.get("arguments", {}))
        return f"[tool_use:{block.get('name')}] {json.dumps(args, sort_keys=True)}"
    if t == "tool_result":
        c = block.get("content")
        if isinstance(c, list):
            return " ".join(block_text(x) for x in c)
        return str(c) if c else ""
    return json.dumps(block, sort_keys=True)


def message_text(entry) -> str:
    """Flattened text of a type:'message' entry's message.content."""
    msg = entry.get("message", {})
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(block_text(b) for b in content)
    return ""


def is_message(entry) -> bool:
    return entry.get("type") == "message"


def is_assistant(entry) -> bool:
    return is_message(entry) and entry.get("message", {}).get("role") == "assistant"


def is_advisor(entry) -> bool:
    return entry.get("type") == "custom_message" and entry.get("customType") == "advisor"


def serialize_prompt(entries, upto_idx: int) -> str:
    """Serialize conversation state (role: text) up to (not including)
    entries[upto_idx] -- this becomes the DPO prompt. Advisory entries are
    never type:'message' so they're naturally excluded by the is_message filter."""
    lines = []
    for e in entries[:upto_idx]:
        if not is_message(e):
            continue
        role = e.get("message", {}).get("role")
        if role not in ("user", "assistant"):
            continue
        text = message_text(e).strip()
        if not text:
            continue
        lines.append(f"[{role}]\n{text}")
    return "\n\n".join(lines)


def _session_model(entries):
    """The acting model for this session, from its `model_change`/`session`
    entries. Returns None if not found."""
    for e in entries:
        if e.get("type") in ("model_change", "session") and e.get("model"):
            return e.get("model")
    return None


def extract_pairs_from_session(session_file: Path, stats: dict, model_filter=None):
    entries = load_jsonl(session_file)
    if model_filter is not None:
        model = _session_model(entries)
        if model is None or not any(m in model for m in model_filter):
            return []
    id_to_idx = {e["id"]: i for i, e in enumerate(entries) if "id" in e}
    message_idxs = [i for i, e in enumerate(entries) if is_message(e)]

    advisors = [e for e in entries if is_advisor(e)]
    stats["advisor_entries"] += len(advisors)

    def _resolve_rejected_idx(adv):
        parent_id = adv.get("parentId")
        idx = id_to_idx.get(parent_id)
        if idx is None:
            return None
        if is_assistant(entries[idx]):
            return idx
        # Confirmed empirically: parentId overwhelmingly resolves to a
        # message/toolResult entry (the advisor reviews a tool call's
        # OUTCOME), not the assistant turn that issued the call. Walk
        # backward to the nearest preceding assistant-role message - that
        # turn is the actual flawed action being critiqued.
        for k in range(idx - 1, -1, -1):
            if is_assistant(entries[k]):
                return k
        return None

    # Resolve every advisor's real target FIRST, so the chain-skip set below
    # is built from actual assistant-turn ids (not raw toolResult parentIds,
    # which would never match a candidate assistant turn's own id).
    resolved = [(adv, _resolve_rejected_idx(adv)) for adv in advisors]
    advisor_target_ids = {entries[idx]["id"] for _adv, idx in resolved
                          if idx is not None and "id" in entries[idx]}
    pairs = []

    for adv, rejected_idx in resolved:
        if rejected_idx is None:
            stats["no_resolvable_parent"] += 1
            continue

        # Position of the parent within message_idxs, to find the NEXT
        # assistant turn that is itself clean (not chained: skip forward
        # past any assistant turn that is ITSELF the parentId of another
        # advisor entry - a still-broken fix attempt, not the real
        # correction. Confirmed empirically: a SyntaxError-inducing edit
        # took two advisor-flagged attempts before the actual fix landed;
        # naively taking "next assistant turn" would train toward the
        # still-broken intermediate attempt.
        try:
            pos = message_idxs.index(rejected_idx)
        except ValueError:
            stats["no_resolvable_parent"] += 1
            continue

        chosen_idx = None
        for j in range(pos + 1, len(message_idxs)):
            cand_idx = message_idxs[j]
            cand = entries[cand_idx]
            if cand.get("message", {}).get("role") != "assistant":
                continue
            cand_id = cand.get("id")
            if cand_id is not None and cand_id in advisor_target_ids:
                stats["skipped_chained_correction"] += 1
                continue  # still-flawed intermediate attempt, keep looking
            chosen_idx = cand_idx
            break
        if chosen_idx is None:
            stats["no_following_assistant_turn"] += 1
            continue

        rejected = message_text(entries[rejected_idx]).strip()
        chosen = message_text(entries[chosen_idx]).strip()
        if not rejected or not chosen:
            stats["empty_text"] += 1
            continue

        sim = SequenceMatcher(None, rejected, chosen).ratio()
        if sim >= SIMILARITY_DISCARD_THRESHOLD:
            stats["not_acted_on"] += 1
            continue

        prompt = serialize_prompt(entries, rejected_idx)
        if not prompt:
            stats["empty_prompt"] += 1
            continue

        pairs.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "similarity": round(sim, 3),
            "advisory_text": adv.get("content", ""),
            "advisory_severity": adv.get("content", "").split('severity="')[1].split('"')[0]
                if 'severity="' in adv.get("content", "") else "",
            "source_session": str(session_file),
        })

    return pairs


def find_sessions():
    """Recursive: primary sessions AND nested subagent/advisor transcripts
    can both carry custom_message/advisor entries with useful parentId links
    into the SAME file's own message entries."""
    files = []
    for root in SESSION_ROOTS:
        if not root.exists():
            continue
        files.extend(sorted(root.rglob("*.jsonl")))
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/agent_dpo_pairs.jsonl")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--model-filter", default="",
                    help="comma-separated substrings; only sessions whose "
                         "acting model contains one of these are mined "
                         "(e.g. 'hawq-sec-re-v1,razorstrike-v1'). Empty = "
                         "no filter (mines every session, including "
                         "frontier-driven ones - not recommended for a "
                         "local-model behavior fix).")
    args = ap.parse_args()
    model_filter = [m.strip() for m in args.model_filter.split(",") if m.strip()] or None
    sessions = find_sessions()
    print(f"[scan] {len(sessions)} session jsonl files (recursive) under "
          f"{[str(r) for r in SESSION_ROOTS]}")

    stats = {
        "advisor_entries": 0,
        "no_resolvable_parent": 0,
        "no_following_assistant_turn": 0,
        "empty_text": 0,
        "not_acted_on": 0,
        "empty_prompt": 0,
        "skipped_chained_correction": 0,
    }
    all_pairs = []
    sessions_with_advisors = 0
    sessions_matched_filter = 0
    for sf in sessions:
        pairs = extract_pairs_from_session(sf, stats, model_filter=model_filter)
        all_pairs.extend(pairs)

    # Recompute per-session advisor/filter presence for reporting (cheap
    # second pass, avoids threading extra state through the extractor).
    for sf in sessions:
        entries = load_jsonl(sf)
        if model_filter is not None:
            model = _session_model(entries)
            if model is None or not any(m in model for m in model_filter):
                continue
        sessions_matched_filter += 1
        if any(is_advisor(e) for e in entries):
            sessions_with_advisors += 1

    if model_filter is not None:
        print(f"[scan] model_filter={model_filter}: {sessions_matched_filter} "
              f"sessions matched")

    print(f"[scan] {sessions_with_advisors} sessions contain advisor entries")
    print(f"[scan] {stats['advisor_entries']} total advisor entries found")
    print(f"[filter] no_resolvable_parent={stats['no_resolvable_parent']} "
          f"no_following_assistant_turn={stats['no_following_assistant_turn']} "
          f"empty_text={stats['empty_text']} "
          f"not_acted_on(sim>={SIMILARITY_DISCARD_THRESHOLD})={stats['not_acted_on']} "
          f"empty_prompt={stats['empty_prompt']} "
          f"skipped_chained_correction={stats['skipped_chained_correction']}")
    print(f"[result] {len(all_pairs)} usable DPO pairs")

    if args.dry_run:
        print("[dry-run] not writing dataset. Re-run without --dry-run to write.")
        if all_pairs:
            print("\n[sample pair]")
            s = all_pairs[0]
            print(f"  source: {s['source_session']}")
            print(f"  similarity: {s['similarity']}")
            print(f"  advisory_text: {s['advisory_text'][:200]!r}")
            print(f"  rejected[:200]: {s['rejected'][:200]!r}")
            print(f"  chosen[:200]:   {s['chosen'][:200]!r}")
        return

    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        for p in all_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"[write] {len(all_pairs)} pairs -> {out_path}")


if __name__ == "__main__":
    main()
