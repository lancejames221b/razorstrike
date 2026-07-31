# Project Memory

Local-first memory fallback. Authoritative copies in Obsidian + hAIveMind.

- 2026-07-24 09:18 | ml | HAWQ coherent base + eval validation PASSED — published HAWQ-v1, fixed 3 eval harness bugs (conv1d layout, XML tool-call parser, adapter key-prefix zeroing 100% of LoRA), all 3 probes PASS with checkpoint-300 genuinely merged. Next: full HAWQ-SEC production run, holding for go-ahead. [obsidian:projects/hawq-coherent-eval-2026-07-24.md] [hv:e6f8a0bd-0263-48dc-a4f6-7ef9aa380ff3]

- 2026-07-31 17:30 | project | HAWQ-SEC-RE v1.2 DPO fix pass checkpoint — clean_control fix confirmed working (0/9 FP), edit-discipline fix confirmed NOT working (2/15, same as baseline, needs >=8/15) — shipping v1.2 with clean_control fix only, edit-discipline deferred to v1.3. GGUF+vision-merge done, MLX convert + Step 7 republish + Step 8 serving still open. Full detail in docs/v1.2_gate_criteria.md and /tmp/hawq_dpo/checkpoint_content.md. [obsidian:PENDING] [hv:PENDING]
