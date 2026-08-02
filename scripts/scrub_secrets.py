#!/usr/bin/env python3
"""scrub_secrets.py - HAWQ v1.3 retrain plan, Step 6 (secret-leak gate).

The Step 3 advisory corpus is mined from REAL omp session history across
this machine's projects, which routinely includes real credentials
encountered during real work (server passwords, API tokens, hashes).
v1.3's LoRA is published to a PUBLIC Hugging Face repo
(lancejames221b/HAWQ-SEC-RE-lora-v13, verified publicly fetchable by Step
10's unauthenticated curl check) - training on unscrubbed session data
would memorize live credentials into a public model. Confirmed present in
/tmp/hawq_dpo/advisory_pairs_v13.jsonl by direct grep before writing this
script: 76 password-assignment matches, 3 bcrypt hashes, 3 generic
api_key/token/secret assignments, 1 PEM private-key block, 0 AWS keys.

Policy: DROP the whole pair (not redact in place) on any high-confidence
secret-pattern hit anywhere in prompt+chosen+rejected. Redaction risks a
false sense of safety (a pattern that almost-but-not-quite matches survives
untouched) and can silently corrupt JSON/training text; dropping is
unambiguous and this corpus has enough volume to absorb losing flagged
rows. Every one of the three v1.3 corpus sources (advisory-mined,
hashline-synthetic, and the pre-existing clean-control pairs) is scanned
independently - synthetic/pre-existing sources are not exempted just
because they're less likely to contain real secrets.

Usage:
    python3 scripts/scrub_secrets.py --inputs /tmp/hawq_dpo/advisory_pairs_v13.jsonl \\
        /tmp/hawq_dpo/hashline_pairs_v13.jsonl /tmp/hawq_dpo/clean_pairs.jsonl \\
        --out-suffix _scrubbed
"""
import argparse
import json
import os
import re
import sys

# High-confidence patterns only (per-pattern false-positive rate matters
# less than recall here - the cost of a missed real secret in a PUBLIC
# model vastly exceeds the cost of dropping a borderline training pair).
# KEY name matching is NOT \b-anchored on the sensitive word itself:
# real identifiers routinely compound it via underscores (DB_PASSWORD,
# SLACK_TOKEN, STRIPE_API_KEY) where "_" is a \w character, so \btoken\b
# would never match inside "SLACK_TOKEN" (no boundary before "TOKEN").
# Anchor on a leading identifier-start instead and allow the sensitive
# word to appear anywhere within a longer compound identifier.
# Left edge is a lookbehind, NOT a mandatory prefix char class: a bare
# `password=` or `token=` (no compound prefix at all) is the single most
# common real form and MUST still match, while `(?<![A-Za-z0-9_])` still
# prevents matching mid-identifier (e.g. inside an unrelated longer word).
_ASSIGN_KEY = (r"(?<![A-Za-z0-9_])[A-Za-z0-9_]*"
               r"(?:password|passwd|pwd|api[_-]?key|apikey|secret|token)"
               r"[A-Za-z0-9_]*")

# A value that is itself an env-var reference or template placeholder
# (`TOKEN=$GITHUB_TOKEN`, `api_key=os.environ.get(...)`, `token={{secret}}`)
# names/reads a secret without EMBEDDING one - excluded so the assignment
# rules don't drop pairs over code that correctly avoids hardcoding.
_PLACEHOLDER_VALUE = r"(?:\$|os\.environ|process\.env|getenv|<|\{\{)"

SECRET_PATTERNS = {
    "assignment_quoted": re.compile(
        rf"""(?i){_ASSIGN_KEY}\s*[:=]\s*['"](?!{_PLACEHOLDER_VALUE})([^'"\s]{{6,}})['"]"""),
    "assignment_unquoted": re.compile(
        # KEY=VALUE / KEY: VALUE with no quotes - `export FOO_TOKEN=abc123`,
        # `PASSWORD=hunter2`, common in shell commands and .env-style
        # output throughout real bash-tool session transcripts.
        rf"""(?i){_ASSIGN_KEY}\s*[:=]\s*(?!{_PLACEHOLDER_VALUE})[^\s'"]{{8,}}"""),
    "bcrypt_hash": re.compile(r"\$2[aby]\$\d{2}\$[A-Za-z0-9./]{53}"),
    "aws_access_key_id": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "github_token": re.compile(
        r"\b(ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36}\b|\bgithub_pat_[A-Za-z0-9_]{22,}\b"),
    "slack_token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    "pem_private_key": re.compile(r"-----BEGIN[ A-Z]*PRIVATE KEY-----"),
    # `sk-` API-key style prefix (OpenAI, Anthropic, many others). NOT
    # `[A-Za-z0-9]{20,}` right after `sk-` - Anthropic keys are
    # `sk-ant-api03-...` and contain hyphens, which that character class
    # would reject, silently missing the dominant real-world case here.
    "sk_style_api_key": re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b"),
    # The dominant shape in this corpus's curl/mcporter command transcripts.
    "bearer_token": re.compile(r"(?i)Authorization:\s*Bearer\s+\S{16,}"),
    "url_basic_auth": re.compile(r"https?://[^/\s:@]+:[^/\s@]+@"),
}


def load_pairs(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _all_string_values(row):
    """Concatenate EVERY string value in the row, not just prompt/chosen/
    rejected - advisory_text (which can itself quote a secret back from a
    reviewer's note) and any other string field ride along into the
    GCS-staged file just as much as the three headline fields, and a
    schema that gains a field later must not silently bypass this scan."""
    parts = []
    for v in row.values():
        if isinstance(v, str):
            parts.append(v)
        elif isinstance(v, list):
            parts.extend(item for item in v if isinstance(item, str))
    return "\n".join(parts)


def scan_row(row):
    """Returns the FIRST matching rule name, or None. Never returns or logs
    the matched substring itself - callers must only report the rule name
    and a byte length, not the secret value, even to local stdout."""
    blob = _all_string_values(row)
    for rule, pattern in SECRET_PATTERNS.items():
        if pattern.search(blob):
            return rule
    return None


def scrub_file(path, out_path):
    rows = load_pairs(path)
    kept, dropped_by_rule = [], {}
    for row in rows:
        rule = scan_row(row)
        if rule is None:
            kept.append(row)
        else:
            dropped_by_rule[rule] = dropped_by_rule.get(rule, 0) + 1
    n_dropped = sum(dropped_by_rule.values())
    print(f"[scrub] {path}: {len(rows)} rows, {n_dropped} dropped, "
          f"{len(kept)} survive. by-rule: {dropped_by_rule}")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows), n_dropped, len(kept)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out-suffix", default="_scrubbed",
                     help="output path = <input stem>_scrubbed.jsonl next to the input")
    args = ap.parse_args()

    total_rows = total_dropped = total_kept = 0
    out_paths = []
    for path in args.inputs:
        stem, ext = os.path.splitext(path)
        out_path = f"{stem}{args.out_suffix}{ext}"
        n_rows, n_dropped, n_kept = scrub_file(path, out_path)
        total_rows += n_rows
        total_dropped += n_dropped
        total_kept += n_kept
        out_paths.append(out_path)

    print(f"\n[scrub] TOTAL: {total_rows} rows scanned, {total_dropped} dropped "
          f"({total_dropped / max(total_rows, 1):.2%}), {total_kept} survive")
    print(f"[scrub] scrubbed outputs: {out_paths}")

    # Re-scan every survivor as a hard self-check - a scrub pass that
    # silently missed a hit on its own output would be worse than no scrub
    # at all (false confidence).
    residual = 0
    for out_path in out_paths:
        for row in load_pairs(out_path):
            if scan_row(row) is not None:
                residual += 1
    if residual:
        print(f"[scrub] FATAL: {residual} residual hit(s) survived scrubbing "
              f"- refusing to report success")
        sys.exit(1)
    print("[scrub] verified: 0 residual hits across all scrubbed outputs")


if __name__ == "__main__":
    main()
