# RazorStrike-v1

Merge scripts, diagnostic tooling, and build manifest for **RazorStrike-v1** — a DARE-TIES task-arithmetic merge on the Qwen3.6-35B-A3B MoE base, published on HuggingFace:

- **MLX (4-bit, Apple Silicon)**: [lancejames221b/razorstrike-v1](https://huggingface.co/lancejames221b/razorstrike-v1)
- **GGUF (Q4_K_M, llama.cpp/Ollama)**: [lancejames221b/razorstrike-v1-GGUF](https://huggingface.co/lancejames221b/razorstrike-v1-GGUF)

This repo does **not** contain model weights — see the HuggingFace links above for those. It contains the build pipeline: the merge scripts, the diagnostic scripts used to root-cause and fix a coherence defect during development, and the full build manifest.

## Composition

DARE-TIES merge of two donors on a `Qwen/Qwen3.6-35B-A3B` anchor:

| Role | Model | Weight | DARE density |
|---|---|---|---|
| Anchor | `Qwen/Qwen3.6-35B-A3B` | — | — |
| Donor 1 (dominant, router source) | huihui-ai's Claude-4.7-Opus-abliterated build | 0.45 | 0.55 |
| Donor 2 | `AlexWortega/SIQ-1-35B` | 0.40 | 0.50 |

Full details, verification results, and the investigation writeup are in [`MANIFEST.json`](./MANIFEST.json).

## What was tried and dropped

A third donor, `UnipatAI/UniMath-35B-A3B`, was evaluated for math specialization and dropped after a systematic pairwise bisect found it caused deterministic CJK-language contamination in coding-context generation whenever combined with *any* other donor via DARE-TIES — an emergent property of the merge, not a defect in UniMath alone. See `MANIFEST.json`'s `investigation` block for the full bisect matrix and root-cause writeup.

## Scripts

- **`scripts/merge_pair_A.py`** — the merge script that produced the shipped build (huihui-ai + SIQ-1, no UniMath). This is the canonical, reproducible recipe.
- **`scripts/merge_qwen36_uncensored.py`** — the original 3-donor recipe (huihui-ai + SIQ-1 + UniMath) that exhibited the CJK-contamination defect; kept for reference/reproducibility of the diagnostic.
- **`scripts/merge_pair_B.py`**, **`scripts/merge_pair_C.py`** — pairwise-bisect variants used to isolate the root cause (huihui+UniMath and SIQ-1+UniMath respectively — both reproduce the defect, confirming UniMath as the common denominator).
- **`scripts/merge_salvage_D.py`** — a salvage attempt (heavily sparsified/de-weighted UniMath in the full 3-donor merge) that did not resolve the contamination; kept to document a negative result.
- **`scripts/patch_router_to_anchor.py`**, **`scripts/patch_router_pair_C.py`** — targeted router-tensor patch utility (swaps the router source into an already-merged output without a full re-merge, since the router is a direct-copy family never touched by DARE-TIES).
- **`scripts/convert_cpu.py`**, **`scripts/quantize_cpu.py`** — CPU-pinned MLX conversion/quantization, working around a Metal GPU command-buffer watchdog timeout that intermittently crashes `mlx_vlm convert` on this architecture.
- **`scripts/v2_gate.py`**, **`scripts/v2_coherence_stress.py`**, **`scripts/v2_final_gate.py`** — coherence and refusal-battery test scripts (the evolution of the test harness used to find, isolate, and confirm the fix for the CJK-contamination defect).
- **`scripts/test_raw_donor.py`**, **`scripts/test_raw_donor_vlm.py`** — raw single-donor test harnesses (`mlx_lm` for text-only donors, `mlx_vlm` for multimodal donors) used in the bisect.

## Reproducing the build

The scripts reference local absolute paths (`/Volumes/Scratch/ml-workspace/models/...`) matching the build environment they were written in — adjust the path constants at the top of each script (`_DEFAULT_ANCHOR_PATH`, `_DEFAULT_DONOR_PATHS`, `_DEFAULT_OUTPUT_PATH`) to your own local donor/anchor download locations. Requires `torch`, `safetensors`, and `mlx-vlm` (for conversion/quantization/testing).

## License

Apache 2.0, matching the Qwen3.6-35B-A3B base. This is a derivative merge — review the constituent donor models' own license terms before use.
