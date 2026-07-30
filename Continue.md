# Continue.md — Living Plan

What's built/deployed, in-progress, decisions, and next steps. Update at every meaningful state change. See `REQ.md` for frozen requirements (base model, naming, scope, publish policy) — don't duplicate those here.

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
