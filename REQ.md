# REQ.md — Frozen Requirements

Authoritative record of the user's actual asks, base-model decisions, naming, and scope.
Update only when the user changes direction — this is not a scratchpad for implementation detail (see `Continue.md` for that).

## Base model lineage (PIVOTED — do not use the old lineage for new work)

- **Old lineage (superseded, DO NOT use as a training/eval base going forward):**
  `razorstrike-v1` = DARE-TIES merge: `Qwen/Qwen3.6-35B-A3B` anchor + `huihui-ai/...Claude-4.7-Opus-abliterated` (reasoning+uncensored donor) + `AlexWortega/SIQ-1-35B` (agentic/coding donor). Documented in the original `MANIFEST.json` at repo root (now describes a build later replaced on HF).
- **Current lineage (HAWQ):** `nightmedia/Qwen3.6-35B-A3B-Holo3-Qwopus-AgentWorld-qx64-hi-mlx` — a third-party MLX merge, dequantized to bf16. As of 2026-07-21 this is also what `lancejames221b/razorstrike-v1` serves on HF (weights replaced in place; old build preserved as `PRIOR_BUILD_MANIFEST.json`/`PRIOR_BUILD_README.md` in that same repo).
- **User's explicit decision:** the adapter under validation was never trained on the old razorstrike lineage — HAWQ is the correct/only base for this work. Do not substitute razorstrike-v1-bf16 (old lineage) for HAWQ in any eval or training step.

## Naming convention (per `hawq-naming-convention` skill — binding)

- Base: always `HAWQ` (no suffix, no version until public release).
- Domain specialists: `HAWQ-<DOMAIN>` (e.g. `HAWQ-SEC` = security/RE/crypto/offensive multi-domain).
- Version suffix (`-v1`, `-v1.1`, etc.) added ONLY after a public release.
- Never reuse "razorstrike" naming for new HAWQ-lineage work — that name is the old, superseded lineage.
- User explicitly renamed the coherent-base repo `lancejames221b/HAWQ-hf` → `lancejames221b/HAWQ-v1` (2026-07-23, this session's parent).

## Scope of the current effort

Validate that the already-trained `HAWQ-SEC-re-validation-lora` (checkpoint-300, best `eval_loss` 2.57, run destabilized ~step 385 — see `Continue.md`) actually works when loaded against a **correctly HF-loadable, coherent** HAWQ base. This is **validation, not training** — no new training happens until Step 7's decision matrix says PASS.

Out of scope for this effort: multimodal `HAWQ-hf` (vision-preserving upload), rebuilding the MLX→HF conversion from scratch, retraining unless the decision matrix's iterate/escalate branch is hit.

## Publish policy (binding)

- Private by default for all new repos (`create_repo(..., private=True)`).
- Public only on the user's **explicit** go — the plan encodes this as `PUBLISH_PUBLIC=1`, never automatic.
- `razorstrike-v1-bf16` was made public 2026-07-23 by explicit user choice, specifically to free HF private-storage quota headroom for `HAWQ-v1`'s upload — not a precedent for making anything else public.

## Downstream product (only after PASS)

Per the plan's Step 7 "all three PASS" branch: build the full 8-family SFT mix (`build_dataset.py PUSH=1`), train the full run (`autodrive.py`, `MAX_STEPS=-1`) under adapter repo `lancejames221b/HAWQ-RE-lora`, merge/publish via `merge_push.py` (private by default).
