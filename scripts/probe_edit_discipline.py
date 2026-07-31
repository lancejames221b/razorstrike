"""probe_edit_discipline - grades on RESULTING FILE STATE, not tool error,
per the confirmed gap: a stale line-range edit can apply "successfully"
and still corrupt the file (the exact failure mode found in real HAWQ
sessions - a SWAP against stale line numbers spliced code into the wrong
place and deleted a needed assignment, with the tool itself reporting no
error at all).

Scenario: the model is told it read a file "earlier"; the file has since
changed (one line inserted upstream of the target function, shifting all
line numbers below it - a realistic background-edit scenario). It's given
one concrete task (add a service fee to calculate_total) and two tools:
  - read_file(path): returns the file's ACTUAL current content.
  - edit_file(path, old_string, new_string) OR
    edit_file(path, start_line, end_line, new_content): applies against
    the ACTUAL current content. Line-range edits are applied POSITIONALLY
    (mirroring the omp SWAP semantics that silently splice/delete on a
    stale range) - no content-match safety net. String-match edits are
    applied only on an exact substring match (mirroring Claude Code's
    Edit tool) and error loudly otherwise.

Grading is entirely on the FINAL file content after the interaction:
PASS requires the fee logic present, syntactically intact (compiles),
and everything else byte-identical to the real current file. A
line-range edit that "succeeds" but lands on the wrong lines FAILS here
even though no tool ever reported an error - closing the gap a
tool-error-only grader would miss.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_re_v2_http_probes import generate, normalize_tool_calls  # noqa: E402

STALE_VIEW = '''def apply_discount(total, pct):
    return total * (1 - pct / 100)


def calculate_total(items):
    subtotal = sum(i["price"] * i["qty"] for i in items)
    return round(subtotal, 2)


def format_receipt(total):
    return f"Total: ${total:.2f}"
'''

# LIVE content: one blank-then-comment line inserted before apply_discount
# (shifting every subsequent line number down by 2 from the stale view -
# tests line-range discipline), AND calculate_total's own subtotal line
# gained an inline comment (tests string-match discipline too - a stale
# exact-string anchor copied from STALE_VIEW no longer matches here
# either, so BOTH edit_file paths require a fresh read to succeed).
LIVE_CONTENT = '''# NOTE: pricing module - keep in sync with docs/pricing.md

def apply_discount(total, pct):
    return total * (1 - pct / 100)


def calculate_total(items):
    subtotal = sum(i["price"] * i["qty"] for i in items)  # raw subtotal
    return round(subtotal, 2)


def format_receipt(total):
    return f"Total: ${total:.2f}"
'''

TASK = (
    "Update calculate_total in pricing.py to add a 5% service fee on top "
    "of the subtotal before rounding."
)

EXPECTED_LOGIC_RE = re.compile(
    r'subtotal\s*\*\s*1\.05|subtotal\s*\+\s*subtotal\s*\*\s*0\.05|'
    r'subtotal\s*\*\s*\(\s*1\s*\+\s*0\.05\s*\)',
)

TOOLS = [
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read the current content of a file.",
        "parameters": {"type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "edit_file",
        "description": (
            "Apply an edit to a file. Either provide old_string+new_string "
            "(exact substring match required), or start_line+end_line+"
            "new_content (positional line replacement, 1-indexed inclusive, "
            "against the file's CURRENT line numbering - no content "
            "verification)."),
        "parameters": {"type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "old_string": {"type": "string"},
                            "new_string": {"type": "string"},
                            "start_line": {"type": "integer"},
                            "end_line": {"type": "integer"},
                            "new_content": {"type": "string"},
                        },
                        "required": ["path"]}}},
]


def apply_edit(current_text, args):
    """Returns (new_text_or_None, result_message)."""
    if args.get('old_string'):
        old, new = args['old_string'], args.get('new_string', '')
        if old not in current_text:
            return None, "String to replace not found in file."
        return current_text.replace(old, new, 1), "Edit applied successfully."
    if 'start_line' in args and 'end_line' in args:
        lines = current_text.split('\n')
        s, e = int(args['start_line']), int(args['end_line'])
        if s < 1 or e > len(lines) or s > e:
            return None, f"Line range {s}-{e} out of bounds (file has {len(lines)} lines)."
        new_content = args.get('new_content', '')
        new_lines = lines[:s - 1] + new_content.split('\n') + lines[e:]
        return '\n'.join(new_lines), "Edit applied successfully."
    return None, "edit_file requires either old_string/new_string or start_line/end_line/new_content."


def _func_source(tree, text, name):
    """Return the exact source text of function `name`'s body via AST
    line boundaries, or None if not found / not a top-level function."""
    import ast
    lines = text.split('\n')
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            start = node.body[0].lineno
            end = max(getattr(n, 'end_lineno', n.lineno) for n in node.body)
            return '\n'.join(lines[start - 1:end])
    return None


def grade(final_text):
    import ast
    if final_text is None:
        return False, "no edit ever applied"
    try:
        tree = ast.parse(final_text)
    except SyntaxError as e:
        return False, f"result does not parse: {e}"
    # Structural check, not substring presence: apply_discount and
    # format_receipt bodies must be EXACTLY unchanged from LIVE_CONTENT -
    # this is what a marker-substring check misses (dead code spliced
    # into an unrelated function's body still contains the marker
    # strings and "parses", but is genuinely corrupted).
    live_tree = ast.parse(LIVE_CONTENT)
    for untouched_fn in ('apply_discount', 'format_receipt'):
        live_body = _func_source(live_tree, LIVE_CONTENT, untouched_fn)
        final_body = _func_source(tree, final_text, untouched_fn)
        if final_body is None:
            return False, f"{untouched_fn} is missing or no longer a top-level function"
        if final_body != live_body:
            return False, f"{untouched_fn} body was modified (should be untouched): {final_body!r}"
    # The fee logic must be specifically INSIDE calculate_total's body,
    # not merely present somewhere in the file.
    ct_body = _func_source(tree, final_text, 'calculate_total')
    if ct_body is None:
        return False, "calculate_total is missing or no longer a top-level function"
    if not EXPECTED_LOGIC_RE.search(ct_body):
        return False, "service fee logic not present inside calculate_total's body"
    # calculate_total must still be the ONLY thing between apply_discount
    # and format_receipt (catches dead code spliced between functions).
    names_in_order = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    if names_in_order != ['apply_discount', 'calculate_total', 'format_receipt']:
        return False, f"unexpected top-level function set/order: {names_in_order}"
    return True, "ok"


def probe_edit_discipline(host, model):
    msgs = [
        {"role": "system", "content": (
            "You are a coding agent with read_file and edit_file tools. "
            "You read pricing.py earlier in this session; its content was:\n\n"
            f"{STALE_VIEW}\n\nNote: other processes in this environment may "
            "modify files, so content you read earlier is not guaranteed to "
            "still be current.")},
        {"role": "user", "content": TASK},
    ]
    current_text = LIVE_CONTENT
    final_text = None
    read_before_first_edit = False
    saw_any_read = False
    saw_any_edit = False
    for turn in range(6):
        print(f"  [edit_discipline] turn {turn+1}/6", flush=True)
        reasoning, content, finish, ctoks, api_tcs = generate(
            host, model, msgs, max_tokens=3000, tools=TOOLS)
        tcs = normalize_tool_calls(api_tcs, content)
        assistant_msg = {"role": "assistant", "content": content}
        if api_tcs:
            assistant_msg["tool_calls"] = api_tcs
        msgs.append(assistant_msg)
        if not tcs:
            break
        for i, tc in enumerate(tcs):
            name = tc.get("name", "")
            args = tc.get("arguments", {}) or {}
            call_id = (api_tcs[i].get("id") if i < len(api_tcs) and api_tcs[i].get("id")
                       else f"call_{turn}_{i}")
            if name == "read_file":
                saw_any_read = True
                if not saw_any_edit:
                    read_before_first_edit = True
                result = current_text
            elif name == "edit_file":
                saw_any_edit = True
                new_text, msg = apply_edit(current_text, args)
                if new_text is not None:
                    current_text = new_text
                    final_text = current_text
                result = msg if new_text is None else f"{msg}\n\n{new_text}"
            else:
                result = "unknown tool"
            msgs.append({"role": "tool", "tool_call_id": call_id, "content": json.dumps(result)})
        if final_text is not None:
            break

    ok, why = grade(final_text)
    print(f"[edit_discipline] read_before_first_edit={read_before_first_edit} "
          f"saw_any_read={saw_any_read} saw_any_edit={saw_any_edit} "
          f"result={'PASS' if ok else 'FAIL'} ({why})", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost:1234")
    ap.add_argument("--model", default="hawq-sec-re-v12")
    args = ap.parse_args()
    print(f"\n=== EDIT DISCIPLINE PROBE: {args.model} @ {args.host} ===\n", flush=True)
    ok = probe_edit_discipline(args.host, args.model)
    print(f"OVERALL: {'PASS' if ok else 'FAIL'}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
