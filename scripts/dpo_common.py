#!/usr/bin/env python3
"""Shared parsing/tokenization helpers for the HAWQ v1.1 DPO pipeline
(scripts/train_dpo.py, scripts/preflight_dpo_maxlen.py). Deliberately
import-light (no torch/transformers model loading, no top-level side
effects) so preflight can import it without pulling in train_dpo.py's heavy
model-load block - the same reason preflight_v4_maxlen.py reimplements
train_lora.py's to_features logic locally rather than importing it.

Prompt schema: both scripts/build_agent_dpo.py (Step 1, omp-session mining)
and scripts/build_clean_control_dpo.py (Step 2, clean-code counterweight)
store `prompt` as flat "[role]\ntext" blocks joined by blank lines (roles:
system, user, assistant - Step 1 never emits a system block). This module
parses that back into a standard messages list and renders it through the
model's REAL chat template at train/eval time, so the stored DPO pairs match
the tokens the model actually sees at inference - a plain-text/flat-string
training format would teach a preference the deployed chat-template-driven
serving path never exercises.
"""
import re


_ROLE_SPLIT_RE = re.compile(r'(?:^|\n\n)\[(system|user|assistant|tool)\]\n')
_WARNED = {"once": False}


def dpo_loss(pol_chosen_lp, pol_rejected_lp, ref_chosen_lp, ref_rejected_lp, beta=0.1):
    """Standard DPO loss (Rafailov et al.). Each *_lp is the SUM of
    per-token log-probs over the response span. Shared by
    scripts/train_dpo.py (real training) and scripts/test_dpo_loss.py (unit
    tests) so the tests validate the exact function that runs on the A100,
    not a duplicate copy. Imports torch lazily so modules that only need
    parse_prompt_to_messages/truncate_messages_to_budget (e.g.
    preflight_dpo_maxlen.py) can run without torch installed."""
    import torch
    logits = (pol_chosen_lp - ref_chosen_lp) - (pol_rejected_lp - ref_rejected_lp)
    return -torch.nn.functional.logsigmoid(beta * logits).mean()


def parse_prompt_to_messages(prompt_text):
    """Flat "[role]\ntext" string -> [{"role":..., "content":...}, ...]."""
    parts = _ROLE_SPLIT_RE.split(prompt_text)
    messages = []
    it = iter(parts[1:])
    for role, content in zip(it, it):
        content = content.strip("\n")
        if content:
            messages.append({"role": role, "content": content})
    return messages


def _merge_consecutive_same_role(msgs):
    """Agentic sessions routinely have multiple consecutive assistant
    'message' entries with no intervening user turn (tool-call/tool-result
    entries are filtered out upstream by build_agent_dpo.py's is_message
    check, but a multi-step assistant turn can still emit >1 message-type
    entry in a row). Collapsing these preserves the content - which is
    exactly the text nearest the response, the highest-priority context -
    instead of a naive "must start with user" rule discarding it outright."""
    merged = []
    for m in msgs:
        if merged and merged[-1]["role"] == m["role"]:
            merged[-1] = {"role": m["role"],
                          "content": merged[-1]["content"] + "\n\n" + m["content"]}
        else:
            merged.append(dict(m))
    return merged


def _content_truncate_search(tok, system_msgs, turn_list, target_idx, max_prompt_tokens,
                              base_ids):
    """Binary-search the character offset into turn_list[target_idx]'s
    content that keeps the tail (nearest the response) within budget,
    trimming that turn's HEAD. Returns (candidate_messages, ids) - falls
    back to base_ids/the untrimmed candidate if no offset helps (e.g. the
    OTHER turns alone already exceed budget)."""
    def render(msgs):
        if not msgs:
            return []
        return tok.apply_chat_template(msgs, add_generation_prompt=True,
                                        tokenize=True)["input_ids"]

    target = turn_list[target_idx]
    content = target["content"]
    lo, hi = 0, len(content)
    best_ids = base_ids
    best_candidate = system_msgs + turn_list
    # ~2.7M-char messages seen in practice; cap generously above log2(that)
    # (~22) so the search always converges instead of exiting mid-range.
    for _ in range(40):
        mid = (lo + hi) // 2
        trial_list = list(turn_list)
        trial_list[target_idx] = {"role": target["role"], "content": content[mid:]}
        trial_candidate = system_msgs + trial_list
        trial_ids = render(trial_candidate)
        if len(trial_ids) <= max_prompt_tokens:
            best_candidate, best_ids = trial_candidate, trial_ids
            hi = mid
        else:
            lo = mid + 1
        if hi - lo <= 1:
            break
    return best_candidate, best_ids


def truncate_messages_to_budget(messages, tok, max_prompt_tokens):
    """Drop WHOLE LEADING turns (never a raw token/char slice mid-message)
    until the rendered prompt fits max_prompt_tokens. System message(s) are
    always kept. Consecutive same-role turns are merged first so an
    agentic multi-step assistant turn survives as one unit instead of being
    discarded for "not starting with user".

    Performance note: omp session prompts run up to ~2.7M raw chars across
    dozens of turns. Re-rendering the FULL chat template on every candidate
    while dropping one turn at a time (an earlier version of this function)
    means up to O(num_turns) expensive jinja-template tokenizations of a
    multi-hundred-KB string each - measured too slow at corpus scale. This
    version tokenizes each turn's raw content ONCE with plain (non-template)
    tokenization to pick the turn-count boundary via a single O(num_turns)
    scan, then calls the real apply_chat_template exactly once (occasionally
    twice, if the plain-token estimate + fixed per-turn overhead undershoots
    the true template cost) on the already-small final candidate.

    The model's chat template requires at least one 'user' turn to render
    at all (confirmed empirically: Qwen's template raises "No user query
    found in messages" otherwise) - so the drop-whole-turns pass never
    drops past the LAST 'user' turn, guaranteeing a renderable minimum
    window of [..., last_user, everything_after_it_merged].

    If that minimum window still overflows the budget on its own, the
    OLDEST turn within it (closest to the truncation boundary, furthest
    from the response) is content-truncated at character level (head
    trimmed, tail kept) via binary search on token length - consistent
    with "prompts truncate from the left, keeping tokens nearest the
    response" applied at sub-message granularity. If trimming that turn to
    nothing still overflows, the newest turn is trimmed next as a last
    resort.

    Returns (kept_messages, prompt_token_ids, dropped_turn_count), or
    (None, [], dropped) if the row has no 'user' turn anywhere and the
    template cannot render it at all (unusable - caller should drop it).
    """
    # Per-turn wrapper overhead (role header/footer tokens the chat
    # template adds around each turn) a plain tokenize() call doesn't
    # capture. Small, fixed overestimate - the exact render at the end
    # corrects for any remaining gap.
    TEMPLATE_OVERHEAD_PER_TURN = 12

    def exact_render(msgs):
        if not msgs:
            return []
        return tok.apply_chat_template(msgs, add_generation_prompt=True,
                                        tokenize=True)["input_ids"]

    def turn_len(content):
        return len(tok(content, add_special_tokens=False)["input_ids"]) + \
            TEMPLATE_OVERHEAD_PER_TURN

    # Coalesced, not just collected: the chat template only permits a
    # system/developer message at position 0 (chat_template.jinja raises
    # "System message must be at the beginning." for any later one) - a
    # window with 2+ system-mapped entries (omp's mid-session `developer`
    # reminders map to "system"; long agentic runs commonly have several)
    # would crash apply_chat_template otherwise. Merging preserves every
    # system message's content instead of silently dropping the extras.
    _system_msgs_raw = [m for m in messages if m["role"] == "system"]
    if len(_system_msgs_raw) > 1:
        system_msgs = [{"role": "system",
                         "content": "\n\n".join(m["content"] for m in _system_msgs_raw)}]
    else:
        system_msgs = _system_msgs_raw
    turn_msgs = _merge_consecutive_same_role(
        [m for m in messages if m["role"] != "system"])

    user_idxs = [i for i, m in enumerate(turn_msgs) if m["role"] == "user"]
    if not user_idxs:
        return None, [], 0
    min_start = user_idxs[-1]

    system_est = sum(turn_len(m["content"]) for m in system_msgs)

    # Required floor: min_start..end (small - the tail nearest the
    # response). Tokenize it unconditionally; this is the ONLY guaranteed
    # cost regardless of how huge the earlier history is.
    running = system_est + sum(turn_len(m["content"]) for m in turn_msgs[min_start:])
    start = min_start

    # Try to extend further back for more context, turn-by-turn, stopping
    # the INSTANT one more turn would exceed budget - so a multi-MB early
    # history costs at most one extra tokenize() call, not a full scan.
    for i in range(min_start - 1, -1, -1):
        tlen = turn_len(turn_msgs[i]["content"])
        if running + tlen > max_prompt_tokens:
            break
        running += tlen
        start = i

    dropped = start
    candidate = system_msgs + turn_msgs[start:]
    ids = exact_render(candidate)
    # The plain-token estimate can undershoot the true template cost;
    # retry a bounded number of times by dropping one more turn at a time
    # (never past min_start) before falling back to content truncation.
    retries = 0
    while len(ids) > max_prompt_tokens and start < min_start and retries < 8:
        start += 1
        dropped += 1
        retries += 1
        candidate = system_msgs + turn_msgs[start:]
        ids = exact_render(candidate)
    if len(ids) <= max_prompt_tokens:
        return candidate, ids, dropped

    turn_list = turn_msgs[min_start:]
    candidate = system_msgs + turn_list
    ids = exact_render(candidate)
    dropped = min_start
    if len(ids) <= max_prompt_tokens:
        return candidate, ids, dropped

    # Minimum renderable window still overflows: trim the OLDEST turn's
    # content first (furthest from the response), then the newest as a
    # last resort if that alone isn't enough.
    best_candidate, best_ids = _content_truncate_search(
        tok, system_msgs, turn_list, 0, max_prompt_tokens, ids)
    if len(best_ids) > max_prompt_tokens and len(turn_list) > 1:
        trimmed_oldest = [dict(turn_list[0], content="")] + turn_list[1:]
        best_candidate, best_ids = _content_truncate_search(
            tok, system_msgs, trimmed_oldest, len(trimmed_oldest) - 1,
            max_prompt_tokens, ids)
    if len(best_ids) > max_prompt_tokens:
        print(f"[dpo_common] WARNING: pair still exceeds max_prompt_tokens="
              f"{max_prompt_tokens} after full truncation (len={len(best_ids)}) "
              f"- both turns in the minimum window are individually too "
              f"large; row will overflow downstream.")
    return best_candidate, best_ids, dropped


def _response_features_from_prompt(kept_messages, prompt_ids, response_text, tok, maxlen):
    """Build masked (input_ids, labels) for ONE response against an
    ALREADY-TRUNCATED prompt (kept_messages/prompt_ids from a single shared
    truncate_messages_to_budget call - see build_dpo_pair_features, the
    intended entry point). Mirrors train_lora.py's to_features prompt-
    prefix masking.

    Returns features_dict_or_None. None if the response contributes zero
    tokens, or prompt+response still exceeds maxlen - per plan policy this
    is a DROP, never a right-truncate: (1) DPO sums per-token log-probs
    over the response span, so truncating chosen/rejected to different
    lengths injects a raw length artifact straight into the preference
    signal; (2) for the Step 2 clean-control pairs, the calibrated "this is
    sound" conclusion sits at the END of the response - right-truncating
    would remove exactly the text that pass is training toward.
    """
    if not kept_messages:
        return None
    full_messages = kept_messages + [{"role": "assistant", "content": response_text}]
    full_ids = tok.apply_chat_template(full_messages, add_generation_prompt=False,
                                        tokenize=True)["input_ids"]
    if len(prompt_ids) >= len(full_ids):
        return None
    boundary = len(prompt_ids)
    if full_ids[:boundary] != prompt_ids:
        # Chat-template rendering divergence between the two render calls
        # (add_generation_prompt=True vs False) can shift the boundary by a
        # token or two (e.g. a header/newline token). Recompute the actual
        # boundary from the real common prefix rather than silently masking
        # the wrong span - training on a misaligned boundary would corrupt
        # every row the same way.
        boundary = 0
        for a, b in zip(prompt_ids, full_ids):
            if a != b:
                break
            boundary += 1
        delta = len(prompt_ids) - boundary
        if not _WARNED["once"]:
            print(f"[dpo_common] WARNING: prompt/full chat-template prefix "
                  f"mismatch (expected boundary {len(prompt_ids)}, actual "
                  f"common prefix {boundary}, delta={delta} tokens) - using "
                  f"actual common prefix. This message prints once per "
                  f"process; a large delta means real prompt tokens are "
                  f"landing inside the loss span and the render path needs "
                  f"fixing, not just the fallback.")
            _WARNED["once"] = True
    if boundary >= len(full_ids):
        return None
    if len(full_ids) > maxlen:
        return None  # overflow: DROP, never right-truncate the response
    labels = [-100] * boundary + full_ids[boundary:]
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids),
            "labels": labels}


def build_dpo_pair_features(messages, chosen_text, rejected_text, tok, maxlen,
                             max_prompt_tokens):
    """Truncate the (shared) prompt EXACTLY ONCE, then build masked features
    for chosen and rejected against that SAME truncated prompt. Both sides
    of a DPO pair share one prompt by construction - truncating twice would
    be pure waste and, worse, could silently diverge (the content-
    truncation fallback is length-sensitive), conditioning chosen vs
    rejected on different prompts and making the DPO logit difference
    meaningless.

    Returns (chosen_features_or_None, rejected_features_or_None,
    dropped_turn_count). Caller drops the pair if either side is None.
    """
    kept_messages, prompt_ids, dropped = truncate_messages_to_budget(
        messages, tok, max_prompt_tokens)
    if not kept_messages:
        return None, None, dropped
    chosen_feat = _response_features_from_prompt(
        kept_messages, prompt_ids, chosen_text, tok, maxlen)
    rejected_feat = _response_features_from_prompt(
        kept_messages, prompt_ids, rejected_text, tok, maxlen)
    return chosen_feat, rejected_feat, dropped
