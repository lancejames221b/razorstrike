# HAWQ-SEC-RE v4 ship/no-ship gate criteria (pre-registered 2026-07-30, before any v4 eval)

Written BEFORE running eval_crypto_audit.py or eval_re_v2_http_probes.py against the
merged v4 model, to prevent post-hoc threshold-shopping. Baseline itself has proven
noisy across runs (documented below) — criteria account for that noise explicitly
rather than pretending a single number is exact.

## Baseline evidence (all against hawq-sec-re-v1, pre-training)

| Run | k | AES | SHA-256 | MD5 | TEA | Blowfish | misuse_enum | clean_control | exploit_path |
|---|---|---|---|---|---|---|---|---|---|
| 1 (original, corrupted SHA constant) | 3 | PASS | PASS(artifact) | FAIL | PASS | FAIL | PASS | PASS | PASS |
| 2 (SHA-256 fixed) | 3 | PASS | PASS | PASS | PASS | FAIL | PASS | PASS | PASS |
| 3 (k9, discarded as anomalous) | 9 | PASS | PASS | PASS | PASS | FAIL | PASS | FAIL (9/9 "flaw", ~1e-7 vs spot-check rate) | PASS |
| 4 (k9b, in progress) | 9 | PASS | PASS | FAIL (1/9) | TBD | TBD | TBD | TBD | TBD |

**Stable signal**: AES and SHA-256 consistently PASS across every run. Blowfish
consistently FAILS across every valid run (this is the primitive with the weakest
real-corpus coverage pre-fix — 0 usable rows before this session's synthetic-table
addition). MD5 is a genuine boundary case that flips (documented serving-stack
nondeterminism, not a scoring bug — see session notes). `probe_clean_control` has one
discarded anomalous reading under investigation (possible caching artifact); spot-check
of 6 fresh live samples showed ~1/6 "flaw"-mention rate, consistent with the original
k=3 PASS.

## Gate criteria (fixed now, applied mechanically once v4 numbers exist)

1. **Primary claim (crypto capability)**: `probe_crypto_id` Blowfish case flips from the
   consistently-observed baseline FAIL to a majority PASS at k=9. This is the one
   primitive whose training coverage this session's work specifically and newly
   created (96 synthetic audited-table rows, 0 before). A Blowfish flip is the
   cleanest, most attributable signal available.
2. **No-regression on stable positives**: AES and SHA-256 must remain PASS at k=9.
   Any flip to FAIL on either is an automatic NO-SHIP regardless of other results.
3. **MD5/TEA are informational, not gating**: given demonstrated baseline
   nondeterminism (MD5 flipped FAIL/PASS/FAIL across 3 valid runs already), a
   flip in either direction on these two cases alone does not gate the decision.
   Report the k=9 numbers plainly.
4. **`probe_misuse_enum` and `probe_exploit_path` must remain PASS.** Both have been
   stable PASS across every baseline run; any regression here is a hard NO-SHIP.
5. **`probe_clean_control` is excluded from the hard gate** pending resolution of the
   run-3 anomaly (tracked as a follow-up, not a blocker). Report the k=9 result and
   flag explicitly if it recurs.
6. **No-regression suite (`eval_re_v2_http_probes.py`)**: all three probes
   (`tool_loop`, `error_recovery`, `long_cot`) must remain PASS, matching the
   pre-training baseline recorded in `/tmp/re_v2_baseline_pretrain_v4.txt`
   (OVERALL: PASS - no regression, tool_loop=True, error_recovery=True,
   long_cot=PASS).

## Ship decision rule

- **SHIP** iff: (2) holds AND (4) holds AND (6) holds AND (1) holds (Blowfish flips
  to PASS).
- **SHIP WITH CAVEAT** iff (2), (4), (6) hold but (1) does not flip (Blowfish stays
  FAIL) — the exploit_poc/broader crypto_audit capability gain may still justify
  shipping, but the specific headline claim ("moved Blowfish identification") would
  not be substantiated; say so plainly rather than declaring victory.
- **NO-SHIP** iff any of (2), (4), (6) fail — a real regression outweighs any crypto
  gain.

This file is the pre-committed reference for the final report; the actual Step 6 run
either satisfies it or explicitly deviates with justification, not the reverse.
Official k=9 baseline (crypto-baseline-k9d, 2026-07-30 04:09-04:39 UTC): overall PASS, crypto_id 4/5 (AES/SHA-256/MD5/TEA pass, Blowfish fail), misuse_enum PASS, clean_control PASS, exploit_path PASS
