#!/usr/bin/env python3
"""build_hashline_pairs.py - HAWQ v1.3 retrain plan, Step 6.

Synthetic 300-pair corpus covering the exact `[PATH#TAG]` edit-anchor
token sequence directly, since the advisory corpus alone may under-cover
it. Every pair's `prompt` contains a `[tool]` turn holding a realistic
`read` result (header `[<relpath>#<TAG>]` followed by `N:<text>` numbered
lines - the real deployed `read` output shape); `chosen` is an `edit` tool
call whose `input` begins with that SAME header, one `PUT N.=M:` op, and
`+`-prefixed body rows. Rendering goes through scripts/omp_surface.py's
render_tool_call (cross-verified against the real chat_template.jinja - see
that module) so the wire format matches character-for-character.

Three `rejected` variants, 100 pairs each (900 total rows before the
prompt/chosen are shared per pair - i.e. 300 pairs, one rejected variant
each, evenly split):
  (a) raw_content   - `input` is the raw file content, no anchor header at
                       all (Step 0: "the model instead passes raw file
                       content or prose as input" - the exact observed
                       failure, 23/32 non-header cases' sibling).
  (b) tag_in_path    - the tag is appended to the `path` ARGUMENT instead
                       of the input header (`path="<relpath>#<TAG>"`),
                       `input` starts directly at raw content - the other
                       23/32 observed confusion from Step 0.
  (c) fabricated_tag - correct SHAPE (`[<relpath>#<TAG>]` + PUT op) but a
                       tag that does NOT appear anywhere in the prompt.
                       MANDATORY per the plan: a corpus teaching only the
                       shape (variants a/b alone) would train tag
                       hallucination - fails identically to today's
                       behavior, just with confident-looking syntax.

Usage:
    python3 scripts/build_hashline_pairs.py --out /tmp/hawq_dpo/hashline_pairs_v13.jsonl [--seed 42]
"""
import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from omp_surface import render_tool_call  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SELF_FILE = Path(__file__).resolve()

PAIRS_PER_VARIANT = 100
TOTAL_PAIRS = PAIRS_PER_VARIANT * 3  # 300
MAX_USES_PER_FILE = 8
MAX_PREFIX_SHARE = 0.05  # 15/300
MAX_TAG_REUSE = 2
MIN_LANGUAGES = 4


def _fail(msg):
    print(f"[fatal] {msg}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Contamination ban list: the two probe fixture constants (verbatim, as
# whole blocks) plus their three most distinctive identifiers, loaded
# directly from probe_edit_discipline.py rather than duplicated by hand so
# a future edit to that probe can't silently desync the ban list.
# ---------------------------------------------------------------------------

def _load_banned_substrings():
    src = (REPO_ROOT / "scripts" / "probe_edit_discipline.py").read_text()
    banned = ["pricing.py", "apply_discount", "calculate_total"]
    for const in ("STALE_VIEW", "LIVE_CONTENT"):
        m = re.search(rf"{const} = '''(.*?)'''", src, re.DOTALL)
        if not m:
            _fail(f"could not locate {const} in probe_edit_discipline.py - "
                  f"contamination ban list would be incomplete")
        banned.append(m.group(1))
    return banned


# ---------------------------------------------------------------------------
# Source pool: real scripts/*.py plus synthesized Go/C/TypeScript/Shell
# snippets (multiple parameterized instances each, so the pool comfortably
# exceeds the 300/8 = 38 distinct-file floor the usage cap implies).
# ---------------------------------------------------------------------------

class SourceFile:
    __slots__ = ("path", "language", "lines", "uses")

    def __init__(self, path, language, lines):
        self.path = path
        self.language = language
        self.lines = lines
        self.uses = 0


GO_TEMPLATE = """package @PKG@

import (
\t"context"
\t"sync"
)

// @IDENT@Pool manages a bounded worker pool for @PKG@ tasks.
type @IDENT@Pool struct {
\tmu      sync.Mutex
\tworkers int
\tqueue   chan func(context.Context) error
}

func New@IDENT@Pool(workers int) *@IDENT@Pool {
\treturn &@IDENT@Pool{
\t\tworkers: workers,
\t\tqueue:   make(chan func(context.Context) error, @THRESH@),
\t}
}

func (p *@IDENT@Pool) Submit(fn func(context.Context) error) {
\tp.mu.Lock()
\tdefer p.mu.Unlock()
\tp.queue <- fn
}

func (p *@IDENT@Pool) Run(ctx context.Context) error {
\tfor i := 0; i < p.workers; i++ {
\t\tgo p.worker(ctx)
\t}
\treturn nil
}

func (p *@IDENT@Pool) worker(ctx context.Context) {
\tfor fn := range p.queue {
\t\tif err := fn(ctx); err != nil {
\t\t\tcontinue
\t\t}
\t}
}
"""

C_TEMPLATE = """#include <stddef.h>
#include <stdint.h>

#define @IDENT@_CAP @THRESH@

typedef struct {
\tuint8_t *data;
\tsize_t len;
\tsize_t cap;
} @IDENT@_buf_t;

int @IDENT@_buf_init(@IDENT@_buf_t *buf, uint8_t *storage, size_t cap) {
\tif (buf == NULL || storage == NULL) {
\t\treturn -1;
\t}
\tbuf->data = storage;
\tbuf->len = 0;
\tbuf->cap = cap;
\treturn 0;
}

int @IDENT@_buf_push(@IDENT@_buf_t *buf, uint8_t byte) {
\tif (buf->len >= buf->cap) {
\t\treturn -1;
\t}
\tbuf->data[buf->len++] = byte;
\treturn 0;
}

void @IDENT@_buf_reset(@IDENT@_buf_t *buf) {
\tbuf->len = 0;
}
"""

TS_TEMPLATE = """type @IDENT@Handler<T> = (arg: T) => Promise<void>;

interface @IDENT@Options {
\tmaxRetries: number;
\ttimeoutMs: number;
}

const DEFAULT_@IDENT@_OPTIONS: @IDENT@Options = {
\tmaxRetries: @THRESH@,
\ttimeoutMs: 30000,
};

export class @IDENT@Queue<T> {
\tprivate items: T[] = [];
\tprivate handler: @IDENT@Handler<T> | null = null;

\tconstructor(private options: @IDENT@Options = DEFAULT_@IDENT@_OPTIONS) {}

\tregister(handler: @IDENT@Handler<T>): void {
\t\tthis.handler = handler;
\t}

\tasync enqueue(item: T): Promise<void> {
\t\tthis.items.push(item);
\t\tif (this.handler) {
\t\t\tawait this.handler(item);
\t\t}
\t}

\tsize(): number {
\t\treturn this.items.length;
\t}
}
"""

SH_TEMPLATE = """#!/usr/bin/env bash
set -euo pipefail

@IDENT@_THRESHOLD=@THRESH@
@IDENT@_LOG_DIR="/tmp/@IDENT@_logs"

mkdir -p "${@IDENT@_LOG_DIR}"

@IDENT@_retry() {
\tlocal attempts=0
\tlocal max_attempts="${1:-3}"
\tshift
\tuntil "$@"; do
\t\tattempts=$((attempts + 1))
\t\tif [ "$attempts" -ge "$max_attempts" ]; then
\t\t\techo "giving up after ${attempts} attempts" >&2
\t\t\treturn 1
\t\tfi
\t\tsleep $((attempts * 2))
\tdone
}

@IDENT@_main() {
\tlocal count=0
\twhile read -r line; do
\t\tcount=$((count + 1))
\t\tif [ "$count" -gt "${@IDENT@_THRESHOLD}" ]; then
\t\t\tbreak
\t\tfi
\t\techo "${line}"
\tdone
}

@IDENT@_main "$@"
"""

_IDENT_POOL = [
    "Sentinel", "Beacon", "Harbor", "Anchor", "Cascade", "Lattice", "Rampart",
    "Kestrel", "Marrow", "Thicket", "Vellum", "Umbral", "Pinion", "Gauntlet",
    "Ferrous", "Coalfire", "Driftwood", "Longbow",
]
_PKG_POOL = [
    "worker", "queue", "dispatch", "ingest", "resolve", "router", "cache",
    "session", "audit", "replay", "shard", "gate",
]
_THRESH_POOL = [3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 40, 50, 64]

LANG_TEMPLATES = {
    "go": (GO_TEMPLATE, "go"),
    "c": (C_TEMPLATE, "c"),
    "typescript": (TS_TEMPLATE, "ts"),
    "shell": (SH_TEMPLATE, "sh"),
}


def _fill_template(template, ident, pkg, thresh):
    return (template
            .replace("@IDENT@", ident)
            .replace("@PKG@", pkg)
            .replace("@THRESH@", str(thresh)))


def _synthesize_files(banned, n_per_lang=10):
    out = []
    rng = random.Random(1337)  # fixed seed: pool composition should be
    # reproducible across runs independent of --seed (which only controls
    # pair SELECTION), so a re-run's diagnostics are directly comparable.
    idents = rng.sample(_IDENT_POOL, k=min(n_per_lang, len(_IDENT_POOL)))
    for lang, (template, ext) in LANG_TEMPLATES.items():
        for i in range(n_per_lang):
            ident = idents[i % len(idents)] + str(i)
            pkg = rng.choice(_PKG_POOL)
            thresh = rng.choice(_THRESH_POOL)
            content = _fill_template(template, ident, pkg, thresh)
            if any(b in content for b in banned):
                continue  # extremely unlikely given the fixed templates, but never trust silently
            path = f"internal/{pkg}/{ident.lower()}.{ext}"
            lines = content.splitlines()
            if len(lines) >= 4:
                out.append(SourceFile(path, lang, lines))
    return out


def _collect_real_python_files(banned):
    out = []
    for py in sorted((REPO_ROOT / "scripts").glob("*.py")):
        if py.resolve() == SELF_FILE:
            continue  # never sample this generator's own source
        text = py.read_text(errors="replace")
        if any(b in text for b in banned):
            continue
        lines = text.splitlines()
        if len(lines) >= 10:
            out.append(SourceFile(f"scripts/{py.name}", "python", lines))
    return out


def build_pool(banned):
    pool = _collect_real_python_files(banned)
    pool += _synthesize_files(banned)
    langs = {sf.language for sf in pool}
    if len(langs) < MIN_LANGUAGES:
        _fail(f"source pool has only {len(langs)} language(s) ({sorted(langs)}), "
              f"need >= {MIN_LANGUAGES}")
    print(f"[pool] {len(pool)} source files across {len(langs)} languages: {sorted(langs)}")
    return pool


def pick_file(rng, pool):
    candidates = [sf for sf in pool if sf.uses < MAX_USES_PER_FILE]
    if not candidates:
        _fail("source pool exhausted (every file hit the per-file use cap)")
    return rng.choice(candidates)


def pick_anchor(rng, sf, min_len=2, max_len=8):
    n = len(sf.lines)
    length = rng.randint(min_len, min(max_len, max(min_len, n - 1)))
    start = rng.randint(1, max(1, n - length + 1))  # 1-indexed, matching `read`'s N: numbering
    snippet = sf.lines[start - 1:start - 1 + length]
    return start, snippet


COMMENT_PREFIX = {"python": "#", "go": "//", "c": "//", "typescript": "//", "shell": "#"}


def _mutate_line(line, language):
    marker = COMMENT_PREFIX[language]
    return f"{line}  {marker} reviewed"


# ---------------------------------------------------------------------------
# Task-instruction variety (only cosmetic - the language/path diversity is
# what actually drives the prefix-collision cap, since `chosen` always
# opens with the fixed <tool_call>/<function=edit>/<parameter=path> preamble
# before the varying relpath).
# ---------------------------------------------------------------------------

INSTRUCTIONS = [
    "Add a trailing review comment to the line you just read.",
    "Append a short review note to the highlighted line.",
    "Mark the line you read as reviewed with an inline comment.",
    "Tag the snippet's last line as reviewed.",
    "Annotate the line you just viewed to show it was reviewed.",
]


def _new_tag(rng, used_tags):
    for _ in range(1000):
        tag = f"{rng.randint(0, 0xFFFF):04X}"
        if used_tags[tag] < MAX_TAG_REUSE:
            return tag
    _fail("could not find a tag under the reuse cap after 1000 attempts "
          "(tag space unexpectedly exhausted)")


def build_pair(rng, pool, used_tags, variant):
    sf = pick_file(rng, pool)
    start, snippet = pick_anchor(rng, sf)
    sf.uses += 1
    tag = _new_tag(rng, used_tags)
    used_tags[tag] += 1

    read_result = f"[{sf.path}#{tag}]\n" + "\n".join(
        f"{start + i}:{line}" for i, line in enumerate(snippet))
    instruction = rng.choice(INSTRUCTIONS)
    read_call = render_tool_call("read", {"path": sf.path})
    prompt = (f"[user]\n{instruction}\n\n"
              f"[assistant]\n{read_call}\n\n"
              f"[tool]\n{read_result}")

    target_line = snippet[-1]
    mutated = _mutate_line(target_line, sf.language)
    end_line = start + len(snippet) - 1
    chosen = render_tool_call("edit", {
        "path": sf.path,
        "input": f"[{sf.path}#{tag}]\nPUT {end_line}.={end_line}:\n+{mutated}",
    })

    raw_content = "\n".join(snippet)
    if variant == "raw_content":
        rejected = render_tool_call("edit", {"path": sf.path, "input": raw_content})
    elif variant == "tag_in_path":
        rejected = render_tool_call("edit", {"path": f"{sf.path}#{tag}", "input": raw_content})
    elif variant == "fabricated_tag":
        fab_tag = _new_tag(rng, used_tags)
        while fab_tag == tag:
            fab_tag = _new_tag(rng, used_tags)
        used_tags[fab_tag] += 1
        rejected = render_tool_call("edit", {
            "path": sf.path,
            "input": f"[{sf.path}#{fab_tag}]\nPUT {end_line}.={end_line}:\n+{mutated}",
        })
    else:
        raise ValueError(f"unknown variant {variant!r}")

    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "source": "edit_hashline_format",
        "variant": variant,
        "tag": tag,
    }, sf


# ---------------------------------------------------------------------------
# Post-hoc verification suite over the FULL committed corpus. Any failure
# aborts nonzero with a diagnostic; nothing partial is ever written.
# ---------------------------------------------------------------------------

def _assert_grounding(rows):
    ok = 0
    for r in rows:
        tag_marker = f"#{r['tag']}]"
        if tag_marker in r["prompt"] and r["tag"] in r["chosen"]:
            ok += 1
    if ok != len(rows):
        _fail(f"grounding check: only {ok}/{len(rows)} chosen tags are "
              f"literally sourced from their own pair's prompt")
    print(f"[grounding] {ok}/{len(rows)} tags sourced from prompt")


def _assert_tag_reuse(used_tags):
    worst = max(used_tags.values(), default=0)
    if worst > MAX_TAG_REUSE:
        offenders = [t for t, c in used_tags.items() if c > MAX_TAG_REUSE]
        _fail(f"tag {offenders[0]!r} used {used_tags[offenders[0]]} times "
              f"(cap {MAX_TAG_REUSE})")
    print(f"[tags] max reuse {worst} (cap {MAX_TAG_REUSE}), "
          f"{len(used_tags)} distinct tags across {sum(used_tags.values())} uses")


def _assert_prefix_uniqueness(rows):
    from collections import Counter
    counts = Counter(r["chosen"][:120] for r in rows)
    cap = max(1, round(len(rows) * MAX_PREFIX_SHARE))
    worst_prefix, worst_n = counts.most_common(1)[0]
    if worst_n > cap:
        _fail(f"chosen[:120] prefix {worst_prefix!r} repeated {worst_n} times "
              f"(cap {cap}/{len(rows)})")
    print(f"[prefix] worst chosen[:120] repeat: {worst_n}/{len(rows)} (cap {cap})")


def _assert_file_usage_cap(pool):
    worst = max((sf.uses for sf in pool), default=0)
    if worst > MAX_USES_PER_FILE:
        offender = next(sf for sf in pool if sf.uses == worst)
        _fail(f"source file {offender.path!r} used {worst} times (cap {MAX_USES_PER_FILE})")
    used_files = sum(1 for sf in pool if sf.uses > 0)
    print(f"[file-cap] max uses {worst} (cap {MAX_USES_PER_FILE}), "
          f"{used_files} distinct files used")


def _assert_language_diversity(rows, pool):
    path_to_lang = {sf.path: sf.language for sf in pool}
    langs = {path_to_lang.get(r["chosen"].split("<parameter=path>\n", 1)[1].split("\n", 1)[0])
             for r in rows}
    langs.discard(None)
    if len(langs) < MIN_LANGUAGES:
        _fail(f"only {len(langs)} language(s) represented in the final corpus "
              f"({sorted(langs)}), need >= {MIN_LANGUAGES}")
    print(f"[languages] {len(langs)} represented: {sorted(langs)}")


def _assert_no_contamination(rows, banned):
    for i, r in enumerate(rows):
        blob = r["prompt"] + r["chosen"] + r["rejected"]
        for b in banned:
            if b in blob:
                _fail(f"row {i} (variant={r['variant']}) contains a banned "
                      f"probe-fixture substring (len={len(b)})")
    print(f"[contamination] 0 probe-fixture hits")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="/tmp/hawq_dpo/hashline_pairs_v13.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    banned = _load_banned_substrings()
    pool = build_pool(banned)

    rng = random.Random(args.seed)
    used_tags = {}
    from collections import defaultdict
    used_tags = defaultdict(int)

    rows = []
    variants = (["raw_content"] * PAIRS_PER_VARIANT
                + ["tag_in_path"] * PAIRS_PER_VARIANT
                + ["fabricated_tag"] * PAIRS_PER_VARIANT)
    rng.shuffle(variants)  # interleave variants so file/tag pressure is even, not phase-separated
    for variant in variants:
        row, sf = build_pair(rng, pool, used_tags, variant)
        rows.append(row)

    if len(rows) != TOTAL_PAIRS:
        _fail(f"generated {len(rows)} rows, expected {TOTAL_PAIRS}")

    variant_counts = {}
    for r in rows:
        variant_counts[r["variant"]] = variant_counts.get(r["variant"], 0) + 1
    print(f"[variants] {variant_counts}")

    _assert_grounding(rows)
    _assert_tag_reuse(used_tags)
    _assert_prefix_uniqueness(rows)
    _assert_file_usage_cap(pool)
    _assert_language_diversity(rows, pool)
    _assert_no_contamination(rows, banned)

    for r in rows:
        del r["tag"]  # internal bookkeeping only, not part of the stored DPO row

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[result] wrote {len(rows)} pairs -> {out_path}")


if __name__ == "__main__":
    main()
