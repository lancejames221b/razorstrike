# Continue.md — Living Plan

What's built/deployed, in-progress, decisions, and next steps. Update at every meaningful state change. See `REQ.md` for frozen requirements (base model, naming, scope, publish policy) — don't duplicate those here.

## Status as of 2026-08-01 (v1.3 challenge-loop cycle, in progress)

**Executing `hawq-v13-challenge-loop-plan.md`.** Deployed baseline is v1.2 (`hawq-sec-re-v12` GGUF on generic's 4090, `hawq-sec-re-v12-mlx` MLX on the Mac). v1.2's edit-discipline DPO training had **zero** measured effect — root cause: the training corpus taught prose commentary about anchors, but the eval metric (`probe_edit_discipline.py`) measures whether a `read_file` **tool call** precedes the first `edit_file` **tool call**. v1.3 retrains that behavior in the model's real tool-call surface form (see `REQ.md` for full frozen decision record once written; this file tracks live progress).

**Constraint from user (2026-08-01): do NOT launch the 4x A100 GCE training run until the local pipeline (generator → corpus → preflight) is validated. User is actively using the generic 4090 (GGUF host) for other work — baseline/gate runs target `hawq-sec-re-v12-mlx` / `hawq-sec-re-v13-mlx` on the Mac (localhost:1234) only until told otherwise.**

### Done this cycle
- `scripts/eval_crypto_audit.py` — fixed the 2 invalid crypto-id fixtures (MD5 vs SHA-1 IV ambiguity; Blowfish-vs-hedge scoring) so the metric is real before baselining.
- `scripts/probe_edit_discipline.py` — `probe_edit_discipline` now returns `(ok, read_before_first_edit, saw_any_read, saw_any_edit)`; added `probe_edit_reread` scenario (rejects first `edit_file`, checks the model re-reads rather than resubmitting identically).
- `scripts/challenge_suite.py` (new) — battery runner across all 9 challenge families with per-trial `raw_text`/`tool_calls` capture, `lms ps` context-guard precondition, per-family error thresholds. Live-smoke-tested against `hawq-sec-re-v12-mlx` @ localhost:1234 (real HTTP calls, 2 trials, `read_before_first_edit_rate=0.5` observed).
- `scripts/dpo_common.py` — `_ROLE_SPLIT_RE` now accepts a `tool` role (needed for tool-call-shaped DPO prompts); `torch` import made lazy (moved into `dpo_loss()`) so `preflight_dpo_maxlen.py` and other tokenizer-only consumers run without torch installed locally (this Mac has no local `torch`/`transformers` in system Python; `mlx_venv` has `transformers` only — verified working after the fix).
- `scripts/build_edit_discipline_toolcall.py` (new) — generates 600 tool-call-shaped DPO pairs (300 `edit_toolcall_read_first` + 300 `edit_toolcall_reread_after_failure`), enforces anti-memorization assertions, zero probe-fixture contamination. Verified independently: 600 rows, correct source split, 600/600 `read_file` in chosen, 600/600 `edit_file` in rejected, 0 contamination hits.
- `scripts/gce_cluster_train.sh` — threaded `DPO_BETA` through both launch branches (FSDP heredoc + non-FSDP `launch_cmd`), matching `MAX_PROMPT_LEN`'s existing mechanism. Prep-only — **no VM launched**.

### In progress / next
- Assemble+contaminate-check+preflight the 1000-row v1.3 corpus (600 new + 400 existing `clean_control`), stage to `gs://hawq-training-us-central1/datasets/hawq-dpo-v13/`.
- Pre-register `docs/v1.3_gate_criteria.md` (must be committed before any training run).
- Baseline the full 9-family battery against `hawq-sec-re-v12-mlx` (MLX only for now), write `docs/v1.3_baseline.md`.
- **BLOCKED pending explicit user go-ahead:** GCE DPO training run (Step 9, `a2-highgpu-4g`, 4x A100) — held per the constraint above.

### Prior cycle status (superseded, kept for history)
## Status as of 2026-07-24 09:14

**Plan `hawq-coherent-eval-plan.md` is complete (21/21 tasks). Validation PASSED.**

### What's built/deployed

- `lancejames221b/HAWQ-v1` (private, HF) — first-ever HF-`transformers`-loadable, text-only, bf16 `Qwen3_5MoeForCausalLM` checkpoint of the HAWQ base. 693/693 keys match canonical `AutoModelForCausalLM` target exactly. 69.4GB, 24 shards, byte-for-byte verified against local extraction.
- `razorstrike-repo/scripts/eval_peft_direct.py` — 3 real bugs fixed this session, pushed to `origin/main` (commits `7ef7b2e`, `40e8778`):
  1. `parse_tool_calls()` now handles HAWQ's native XML `<function=name><parameter=...>` tool-call format (was JSON-only, silently saw zero calls).
  2. Adapter LoRA keys use a `.language_model.` prefix segment; our text-only extraction strips that segment from the base. Old code would silently apply **0% of the LoRA** on merge (PEFT just warns, doesn't raise). New `_patch_adapter_dir()` strips the prefix from the adapter's saved keys before load, and hard-fails (`sys.exit(3)`) if any missing-key warning still fires. **Any future eval run MUST go through this path — do not bypass.**
- (Earlier, prior session) The extraction pipeline itself (`.driver_state` / local scripts, not committed to this repo) fixed a `conv1d.weight` MLX `[C,K,1]` → PyTorch `[C,1,K]` layout bug across all 29 affected tensors in `HAWQ-v1`'s shards.

### Decision reached

Step 7 decision matrix (in `hawq-validation-eval-fix-plan.md`), applied to a run where the adapter was verifiably merged correctly (log line `[eval] adapter loaded clean: 0 missing-key warnings after patch`):

| Probe | Result |
|---|---|
| `tool_loop` | PASS — 4 distinct calls, 0 repeats, DONE |
| `error_recovery` | PASS — dup_ratio 0.03 |
| `long_cot` | PASS — terminated on its own at 6,724 tokens |

**All three PASS → validation PASSED.** Matrix's mechanical next step: full HAWQ-SEC production run.

### Next steps (NOT yet started — holding for explicit user go-ahead, since it's a new multi-hour compute commitment)

1. `build_dataset.py PUSH=1` — build the full 8-family SFT mix (decompile/RE, crypto_id, ransomware-crypto, math, cyber, loop_recovery, mythos, uncensor). This is the full mix, **not** the narrow RE+loop_recovery validation subset checkpoint-300 was trained on.
2. `autodrive.py` with that dataset, `MAX_STEPS=-1` (full run, not the 500-step validation), adapter repo `lancejames221b/HAWQ-RE-lora`.
3. `merge_push.py` to merge + publish (private by default per `REQ.md`; public only on explicit `PUBLISH_PUBLIC=1`).

### Operational gotchas hit this session (for next time)

- `colab new -s <name> --tpu G4` is WRONG — `--tpu` only accepts `v5e1`/`v6e1`. G4 is a GPU tier: use `--gpu G4`. Using `--tpu G4` silently creates a CPU-only runtime (no error) and wastes a full model-load cycle before the mistake surfaces as a CUDA `AssertionError`.
- Colab sessions go stale/404 easily — always `colab status -s <name>` before assuming a session is alive; re-provision (`colab new` → clone/reset → `vm_setup.py` → re-download adapter checkpoint) is a full ~1min sequence, not just a reconnect.
- Repo clones to `/content/razorstrike` (not `/content/razorstrike-repo`) on Colab — mismatched `cwd` in a launch script fails with a silent-looking `FileNotFoundError` that's easy to misread as a deeper bug.
- Bash heredocs need to be UNQUOTED (`<< PYEOF`, not `<< 'PYEOF'`) when a shell variable like `$HF_TOKEN` needs to substitute into the generated script — quoting the delimiter suppresses all expansion and writes the literal `$HF_TOKEN` string into the file.
- Don't trust a HF `merge_and_unload()`/`PeftModel.from_pretrained()` "success" at face value — it warns instead of raising on missing adapter keys. Always check for `UserWarning` text (or now: rely on the `_patch_adapter_dir` gate in `eval_peft_direct.py`) before trusting an "adapter + base" eval result.
