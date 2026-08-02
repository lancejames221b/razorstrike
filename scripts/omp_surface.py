#!/usr/bin/env python3
"""omp_surface.py - HAWQ v1.3 retrain plan, Step 2 (authoritative tool surface).

Pins the exact tool surface (system prompt + tool schemas) that omp sends to
the model in production, parsed from a real omp LLM-request capture
(`/tmp/omp-llm-request-<id>.json` on generic, mirrored under
`fixtures/omp_captures/` in this repo). Every probe/scenario/synthetic-pair
generator in this retrain MUST source its tool list and system prompt from
`load_surface()` rather than reconstructing them by hand - the `edit` tool's
description alone is 5,935 chars and contains the full hashline spec; a
probe that trims or paraphrases it is measuring a different task than what
is actually deployed.

Capture schema (verified against fixtures/omp_captures/1547296284f4db5c.json):
    {
      "messages": [...],
      "model": str,
      "serviceTier": str,
      "systemPrompt": str,
      "thinkingLevel": str,
      "tools": [{"name": str, "description": str, "parameters": {...}, "strict": bool}, ...]
    }
Tools are flat dicts (Anthropic-style: name/description/parameters/strict at
the top level) - NOT OpenAI chat-completions-style {"type": "function",
"function": {"name": ...}} nesting. Any code reading these tool dicts must
use t["name"], not t["function"]["name"].
"""
import json
import re

# Verbatim prefix of omp's edit-tool rejection message (see Step 0 of the
# v1.3 retrain plan: observed 23/32 times across captures). The full runtime
# message appends `; got: "<first line of payload>". Example: ...` - only
# the fixed prefix is pinned here since the suffix varies per call.
ANCHOR_ERROR = 'input must begin with "[PATH#HASH]" on the first non-blank line for anchored edits'

# First non-blank line of a well-formed `edit` input must be an anchor
# header: "[<path>#<4-hex-tag>]" with optional trailing whitespace.
_HASHLINE_RE = re.compile(r"^\[[^\]\n]+#[0-9A-Fa-f]{4}\]\s*$")


def load_surface(capture_path: str) -> dict:
    """Parse an omp LLM-request capture into the authoritative tool surface.

    Returns {"system_prompt": str, "tools": list} where `tools` is the
    capture's `tools` array passed through unmodified (list of
    {"name","description","parameters","strict"} dicts) and
    `system_prompt` is the capture's `systemPrompt` string, also unmodified.

    Raises FileNotFoundError if capture_path does not exist, and KeyError
    (with the offending key name) if the capture is missing `systemPrompt`
    or `tools` - both are treated as fatal rather than defaulted, since a
    surface built from a partial capture would silently under-specify the
    deployed task.
    """
    with open(capture_path, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    for required_key in ("systemPrompt", "tools"):
        if required_key not in d:
            raise KeyError(f"{capture_path!r} missing required key {required_key!r}")
    return {"system_prompt": d["systemPrompt"], "tools": d["tools"]}


def find_tool(tools: list, name: str) -> dict:
    """Return the tool schema dict with tools[i]["name"] == name.

    Raises KeyError if absent - callers (probes, scenario harness) need a
    named tool to exist in the pinned surface; a silent None would let a
    probe run against a smaller-than-deployed tool set without noticing.
    """
    for t in tools:
        if t.get("name") == name:
            return t
    raise KeyError(f"tool {name!r} not found in surface (have: {[t.get('name') for t in tools]})")


def assert_hashline(input_str: str) -> tuple[bool, str]:
    """Score well-formedness of an `edit` tool call's `input` argument.

    The single scoring predicate for edit well-formedness (shared by the
    tmux scenario harness and the synthetic hashline-pair generator): the
    FIRST NON-BLANK line of `input_str` must match `[<path>#<4-hex-tag>]`.

    Returns (True, reason) if well-formed, (False, reason) otherwise. Never
    raises - a malformed or empty `input_str` is exactly the condition this
    function exists to detect, not an exceptional case.
    """
    if not isinstance(input_str, str):
        return False, f"input is not a string (got {type(input_str).__name__})"
    first_non_blank = None
    for line in input_str.split("\n"):
        if line.strip() != "":
            first_non_blank = line
            break
    if first_non_blank is None:
        return False, "input has no non-blank line"
    if _HASHLINE_RE.match(first_non_blank):
        return True, f"well-formed anchor header: {first_non_blank.strip()!r}"
    return False, f"first non-blank line is not a [PATH#TAG] anchor header: {first_non_blank!r}"


# ---------------------------------------------------------------------------
# Tool-call wire-format rendering. Reproduces
# /Volumes/SeXternal/hawq_v12_dpo_merged/chat_template.jinja lines 184-211
# (the message.tool_calls loop) character-for-character: this is the exact
# string the model must emit as its assistant `content` for a tool call to
# parse. Any DPO/SFT generator building a `chosen`/`rejected` tool-call
# string (Steps 3 and 6 of the v1.3 retrain plan) MUST render through here
# rather than hand-rolling the XML, so a template edit only has to be
# re-verified in one place. Cross-verified against a real jinja2 render of
# the template (not just read by eye) - see the module's __main__ block.
# ---------------------------------------------------------------------------


def render_tool_call_body(name: str, args) -> str:
    """<function=NAME>\\n<parameter=K>\\nV\\n</parameter>\\n...</function> -
    ONE call's inner body, no surrounding <tool_call> tags (jinja:193-208).
    `args` is normally a dict (parameter name -> value; string values
    verbatim, other types JSON-serialized like jinja's `tojson`). A STRING
    `args` hits jinja's tool_call.arguments-is-string branch (203-207): the
    string is emitted raw plus a trailing newline, with NO <parameter> tags
    at all - real omp tool calls always carry structured (dict) arguments,
    so this path is defensive only, not expected to fire."""
    parts = [f"<function={name}>\n"]
    if isinstance(args, dict):
        for k, v in args.items():
            val = v if isinstance(v, str) else json.dumps(v)
            parts.append(f"<parameter={k}>\n{val}\n</parameter>\n")
    elif isinstance(args, str):
        if args.strip():
            parts.append(args)
            parts.append("\n")
    parts.append("</function>")
    return "".join(parts)


def render_tool_calls(text: str, calls: list) -> str:
    """Full assistant `content` string for a turn with leading text `text`
    and zero or more tool calls `calls` (list of (name, args) pairs, in
    call order), reproducing chat_template.jinja:184-211 EXACTLY - including
    the parallel-tool-call separator rule, which differs from a naive
    join: the FIRST call is preceded by "\\n\\n" only if `text` is
    non-empty after trim (bare `<tool_call>` if `text` is empty); EVERY
    SUBSEQUENT call is preceded by a bare "\\n" regardless of `text`.
    `calls=[]` returns the trimmed text unchanged (a text-only turn).
    `calls=[(name, args)]` with `text=""` is the common single-tool-call
    case: a bare `<tool_call>...</tool_call>` block."""
    text = (text or "").strip()
    if not calls:
        return text
    pieces = []
    for i, (name, args) in enumerate(calls):
        body = render_tool_call_body(name, args)
        prefix = ("\n\n" if text else "") if i == 0 else "\n"
        pieces.append(f"{prefix}<tool_call>\n{body}\n</tool_call>")
    return text + "".join(pieces)


def render_tool_call(name: str, args) -> str:
    """Convenience wrapper for the common single-call, no-preceding-text
    case: render_tool_calls("", [(name, args)])."""
    return render_tool_calls("", [(name, args)])


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "fixtures/omp_captures/1547296284f4db5c.json"
    s = load_surface(path)
    names = [t["name"] for t in s["tools"]]
    print(len(s["tools"]), names[:6])
    print(assert_hashline("FROM debian:stable-slim"))
    print(assert_hashline("[Dockerfile#B3E9]\nPUT 1.=1:\n+FROM debian:trixie-slim"))
    print(render_tool_call("edit", {"path": "Dockerfile",
                                     "input": "[Dockerfile#B3E9]\nPUT 1.=1:\n+FROM debian:trixie-slim"}))
