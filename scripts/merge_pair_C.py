#!/usr/bin/env python3
"""Pairwise bisect C: SIQ-1 + UniMath (huihui dropped). Router defaults to donor[0]=SIQ-1;
patched to anchor post-merge via patch_router_to_anchor.py (huihui was always the router
source in every other variant, so with huihui absent the anchor is the natural fallback -
already empirically shown inert for this defect via byte-identical greedy-output swap test
on the full 3-donor merge)."""
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from safetensors import safe_open
from safetensors.torch import save_file

_DEFAULT_ANCHOR_PATH = "/Volumes/Scratch/ml-workspace/models/Qwen3.6-35B-A3B"
_DEFAULT_DONOR_PATHS = [
    "/Volumes/Scratch/ml-workspace/models/SIQ-1-35B",
    "/Volumes/Scratch/ml-workspace/models/UniMath-35B-A3B",
]
_DEFAULT_OUTPUT_PATH = "/Volumes/Scratch/ml-workspace/merged/pair_C_siq1_unimath_bf16"
WEIGHTS = [0.40, 0.25]
DENSITIES = [0.50, 0.45]
DENSITY = 0.5


def categorize_tensors(all_keys: set) -> Tuple[set, set, set, dict, set]:
    router_tensors: set = set()
    anchor_keep_tensors: set = set()
    mergeable_tensors: set = set()
    verbatim_copy_tensors: set = set()
    family_map: dict = {}

    for key in all_keys:
        if key.endswith(".mlp.gate.weight") and "layers." in key:
            router_tensors.add(key); family_map[key] = "router"; continue
        if "embed_tokens.weight" in key or "lm_head.weight" in key:
            anchor_keep_tensors.add(key); family_map[key] = "anchor"; continue
        if (".input_layernorm.weight" in key or ".post_attention_layernorm.weight" in key or
            ".norm.weight" in key or ".q_norm.weight" in key or ".k_norm.weight" in key):
            anchor_keep_tensors.add(key); family_map[key] = "anchor"; continue
        if key.startswith("model.visual."):
            verbatim_copy_tensors.add(key); family_map[key] = "vision"; continue
        if key.startswith("mtp."):
            verbatim_copy_tensors.add(key); family_map[key] = "mtp"; continue
        if ".self_attn." in key and any(x in key for x in [".q_proj.weight", ".k_proj.weight", ".v_proj.weight", ".o_proj.weight"]):
            mergeable_tensors.add(key); family_map[key] = "full_attn"; continue
        if ".linear_attn." in key and any(x in key for x in [".A_log", ".dt_bias", ".conv1d.weight", ".in_proj_a.weight", ".in_proj_b.weight", ".norm.weight"]):
            anchor_keep_tensors.add(key); family_map[key] = "anchor"; continue
        if ".linear_attn." in key and any(x in key for x in [".in_proj_qkv.weight", ".in_proj_z.weight", ".out_proj.weight"]):
            mergeable_tensors.add(key); family_map[key] = "linear_attn"; continue
        if ".mlp.experts." in key and (".gate_up_proj" in key or ".down_proj" in key):
            mergeable_tensors.add(key); family_map[key] = "routed_expert"; continue
        if ".mlp.shared_expert." in key and (".gate_proj.weight" in key or ".up_proj.weight" in key or ".down_proj.weight" in key):
            mergeable_tensors.add(key); family_map[key] = "shared_expert"; continue
        if ".mlp.shared_expert_gate.weight" in key:
            mergeable_tensors.add(key); family_map[key] = "shared_expert"; continue
        family_map[key] = "unknown"

    return router_tensors, anchor_keep_tensors, mergeable_tensors, verbatim_copy_tensors, family_map


def load_safetensors_index(model_path: Path) -> dict:
    idx_file = model_path / "model.safetensors.index.json"
    if not idx_file.exists():
        raise FileNotFoundError(f"No safetensors index at {idx_file}")
    with open(idx_file) as f:
        idx = json.load(f)
    return idx["weight_map"]


def verify_model_complete(model_path: Path, weight_map: dict) -> Tuple[bool, list]:
    shard_files = sorted(set(weight_map.values()))
    missing = [f for f in shard_files if not (model_path / f).exists()]
    return (len(missing) == 0, missing)


def get_num_experts(anchor_path: Path) -> int:
    with open(anchor_path / "config.json") as f:
        cfg = json.load(f)
    for c in (cfg, cfg.get("text_config", {})):
        if "num_experts" in c:
            return c["num_experts"]
    raise ValueError("num_experts not found in anchor config.json")


def _load_tensors_grouped_by_file(model_path: Path, weight_map: dict, keys: list) -> Dict[str, torch.Tensor]:
    by_file: Dict[str, list] = {}
    for k in keys:
        by_file.setdefault(weight_map[k], []).append(k)
    result: Dict[str, torch.Tensor] = {}
    for file_name, ks in by_file.items():
        file_path = model_path / file_name
        if not file_path.exists():
            continue
        with safe_open(str(file_path), framework="pt") as sf:
            for k in ks:
                if k in sf.keys():
                    result[k] = sf.get_tensor(k)
    return result


def load_expert_tensor(donor_path: Path, donor_weight_map: dict, tensor_name: str,
                        num_experts: int) -> Optional[torch.Tensor]:
    if tensor_name in donor_weight_map:
        file_path = donor_path / donor_weight_map[tensor_name]
        if file_path.exists():
            with safe_open(str(file_path), framework="pt") as sf:
                if tensor_name in sf.keys():
                    return sf.get_tensor(tensor_name)
        return None

    if tensor_name.endswith(".mlp.experts.gate_up_proj"):
        prefix = tensor_name[: -len("gate_up_proj")]
        gate_keys = [f"{prefix}{i}.gate_proj.weight" for i in range(num_experts)]
        up_keys = [f"{prefix}{i}.up_proj.weight" for i in range(num_experts)]
        if not all(k in donor_weight_map for k in gate_keys):
            return None
        if not all(k in donor_weight_map for k in up_keys):
            return None
        gate_tensors = _load_tensors_grouped_by_file(donor_path, donor_weight_map, gate_keys)
        up_tensors = _load_tensors_grouped_by_file(donor_path, donor_weight_map, up_keys)
        if len(gate_tensors) != num_experts or len(up_tensors) != num_experts:
            return None
        per_expert = [torch.cat([gate_tensors[gk], up_tensors[uk]], dim=0)
                      for gk, uk in zip(gate_keys, up_keys)]
        return torch.stack(per_expert, dim=0)

    if tensor_name.endswith(".mlp.experts.down_proj"):
        prefix = tensor_name[: -len("down_proj")]
        down_keys = [f"{prefix}{i}.down_proj.weight" for i in range(num_experts)]
        if not all(k in donor_weight_map for k in down_keys):
            return None
        down_tensors = _load_tensors_grouped_by_file(donor_path, donor_weight_map, down_keys)
        if len(down_tensors) != num_experts:
            return None
        return torch.stack([down_tensors[k] for k in down_keys], dim=0)

    return None


def dare_ties_merge(tensor_name: str, indexed_task_vectors: list, density: float, weights: list,
                     merge_counter: int = 0, densities: list = None) -> Optional[torch.Tensor]:
    if not indexed_task_vectors:
        return None

    donor_indices = [idx for idx, _ in indexed_task_vectors]
    stacked = torch.stack([tv for _, tv in indexed_task_vectors])

    donor_weights = [weights[i] for i in donor_indices]
    weight_sum = sum(donor_weights)
    donor_weights = [w / weight_sum for w in donor_weights]
    weights_tensor = torch.tensor(donor_weights, dtype=stacked.dtype).view(-1, *([1] * (stacked.ndim - 1)))

    torch.random.manual_seed(42 + merge_counter)
    per_donor_density = [(densities[i] if densities and i < len(densities) else density) for i in donor_indices]
    dens_t = torch.tensor(per_donor_density, dtype=stacked.dtype).view(-1, *([1] * (stacked.ndim - 1)))
    dare_masks = torch.rand_like(stacked) < dens_t
    dared_vectors = torch.where(dare_masks, stacked / dens_t, torch.zeros_like(stacked))

    weighted_sum = (dared_vectors * weights_tensor).sum(dim=0)
    elected_sign = torch.sign(weighted_sum)

    agreement_mask = (torch.sign(dared_vectors) == elected_sign.unsqueeze(0)).to(stacked.dtype)
    numerator = (dared_vectors * weights_tensor * agreement_mask).sum(dim=0)
    denominator = (weights_tensor.expand_as(dared_vectors) * agreement_mask).sum(dim=0)
    merged_delta = torch.where(denominator != 0, numerator / denominator.clamp(min=1e-12),
                                torch.zeros_like(numerator))
    return merged_delta


def apply_dare_ties(tensor_name: str, anchor_tensor: torch.Tensor,
                     donor_tensors: dict, indexed_task_vectors: list, weights: list,
                     density: float = DENSITY, merge_counter: int = 0, densities: list = None) -> torch.Tensor:
    merged_vector = dare_ties_merge(tensor_name, indexed_task_vectors, density, weights, merge_counter, densities)
    if merged_vector is None:
        return anchor_tensor.clone()
    return (anchor_tensor + merged_vector).to(anchor_tensor.dtype)


def merge_qwen_moe(anchor_path: str = _DEFAULT_ANCHOR_PATH,
                    donor_paths: list = _DEFAULT_DONOR_PATHS,
                    output_path: str = _DEFAULT_OUTPUT_PATH):
    anchor = Path(anchor_path)
    donors = [Path(d) for d in donor_paths]
    output = Path(output_path)

    print("=== Pairwise Bisect C: SIQ-1 + UniMath (huihui dropped, router->anchor patch pending) ===")
    print(f"Anchor: {anchor.name}")
    print(f"Donors: {[d.name for d in donors]}  weights={WEIGHTS} densities={DENSITIES}")
    print(f"Output: {output}")

    print("\nLoading anchor index...")
    anchor_weight_map = load_safetensors_index(anchor)
    num_experts = get_num_experts(anchor)
    print(f"  num_experts (from config): {num_experts}")

    anchor_complete, anchor_missing = verify_model_complete(anchor, anchor_weight_map)
    if not anchor_complete:
        print(f"\nERROR: Anchor is INCOMPLETE - missing {len(anchor_missing)} shard files:")
        for f in anchor_missing:
            print(f"  {f}")
        exit(1)
    print(f"  Anchor completeness: OK ({len(set(anchor_weight_map.values()))} shards present)")

    all_keys = set(anchor_weight_map.keys())
    router_tensors, anchor_keep_tensors, mergeable_tensors, verbatim_copy_tensors, family_map = categorize_tensors(all_keys)

    print(f"\nTensor categorization:")
    print(f"  Router: {len(router_tensors)} tensors")
    print(f"  Anchor-keep: {len(anchor_keep_tensors)} tensors")
    print(f"  Mergeable: {len(mergeable_tensors)} tensors")
    print(f"  Verbatim-copy (vision/MTP): {len(verbatim_copy_tensors)} tensors")
    unknown_count = sum(1 for f in family_map.values() if f == "unknown")
    print(f"  Unknown: {unknown_count} tensors")

    if unknown_count > 0:
        print("\nERROR: Found unrecognized tensor patterns!")
        for key, family in family_map.items():
            if family == "unknown":
                print(f"  {key}")
        exit(1)

    output.mkdir(parents=True, exist_ok=True)

    print("\nProcessing tensors shard-by-shard...")
    output_tensor_map = {t: f for t, f in anchor_weight_map.items()}
    shard_files = set(output_tensor_map.values())

    total_processed = 0
    written_tensors = set()
    global_merge_counter = 0

    print("\nPre-loading donor indices...")
    donor_weight_maps = {}
    for donor_idx, donor_path in enumerate(donors):
        try:
            wm = load_safetensors_index(donor_path)
            is_complete, missing = verify_model_complete(donor_path, wm)
            if not is_complete:
                print(f"  WARNING: Donor {donor_idx + 1} ({donor_path.name}) is INCOMPLETE - "
                      f"missing {len(missing)} shard files. EXCLUDING this donor.")
                for f in missing[:10]:
                    print(f"    missing: {f}")
                continue
            donor_weight_maps[donor_idx] = wm
            print(f"  Loaded index for donor {donor_idx + 1}: {donor_path.name} "
                  f"({len(set(wm.values()))} shards, complete)")
        except Exception as e:
            print(f"  WARNING: Could not load donor {donor_idx + 1} index: {e} - EXCLUDING")

    if not donor_weight_maps:
        print("\nWARNING: No complete donors available - merge will be anchor-only.")

    expert_layout_stats = {"fused_matches": 0, "unfused_remapped": 0, "donor_missing": 0}

    for shard_file in sorted(shard_files):
        shard_tensor_names = [t for t in all_keys if output_tensor_map[t] == shard_file]
        shard_buffer = {}
        for tensor_name in shard_tensor_names:
            if tensor_name in router_tensors and len(donors) > 0:
                donor1 = donors[0]
                tensor = None
                if 0 in donor_weight_maps and tensor_name in donor_weight_maps[0]:
                    file_path = donor1 / donor_weight_maps[0][tensor_name]
                    if file_path.exists():
                        with safe_open(str(file_path), framework="pt") as sf:
                            if tensor_name in sf.keys():
                                tensor = sf.get_tensor(tensor_name)
                if tensor is None:
                    if tensor_name in anchor_weight_map:
                        file_path = anchor / anchor_weight_map[tensor_name]
                        if file_path.exists():
                            with safe_open(str(file_path), framework="pt") as sf:
                                if tensor_name in sf.keys():
                                    tensor = sf.get_tensor(tensor_name)
                if tensor is not None:
                    shard_buffer[tensor_name] = tensor
                    del tensor

            elif tensor_name in anchor_keep_tensors:
                if tensor_name in anchor_weight_map:
                    file_path = anchor / anchor_weight_map[tensor_name]
                    if file_path.exists():
                        with safe_open(str(file_path), framework="pt") as sf:
                            if tensor_name in sf.keys():
                                tensor = sf.get_tensor(tensor_name)
                                shard_buffer[tensor_name] = tensor
                                del tensor

            elif tensor_name in verbatim_copy_tensors:
                if tensor_name in anchor_weight_map:
                    file_path = anchor / anchor_weight_map[tensor_name]
                    if file_path.exists():
                        with safe_open(str(file_path), framework="pt") as sf:
                            if tensor_name in sf.keys():
                                tensor = sf.get_tensor(tensor_name)
                                shard_buffer[tensor_name] = tensor
                                del tensor

            elif tensor_name in mergeable_tensors:
                if tensor_name in anchor_weight_map:
                    anchor_file = anchor / anchor_weight_map[tensor_name]
                    if not anchor_file.exists():
                        continue
                    with safe_open(str(anchor_file), framework="pt") as sf:
                        if tensor_name not in sf.keys():
                            continue
                        anchor_tensor = sf.get_tensor(tensor_name)

                    is_routed_expert = family_map.get(tensor_name) == "routed_expert"
                    task_vectors = []
                    for donor_idx, donor_path in enumerate(donors):
                        if donor_idx not in donor_weight_maps:
                            continue
                        donor_tensor = None
                        if is_routed_expert:
                            was_fused = tensor_name in donor_weight_maps[donor_idx]
                            donor_tensor = load_expert_tensor(
                                donor_path, donor_weight_maps[donor_idx], tensor_name, num_experts
                            )
                            if donor_tensor is not None:
                                if was_fused:
                                    expert_layout_stats["fused_matches"] += 1
                                else:
                                    expert_layout_stats["unfused_remapped"] += 1
                            else:
                                expert_layout_stats["donor_missing"] += 1
                        else:
                            if tensor_name in donor_weight_maps[donor_idx]:
                                donor_file = donor_path / donor_weight_maps[donor_idx][tensor_name]
                                if donor_file.exists():
                                    with safe_open(str(donor_file), framework="pt") as sf:
                                        if tensor_name in sf.keys():
                                            donor_tensor = sf.get_tensor(tensor_name)

                        if donor_tensor is not None:
                            if donor_tensor.shape != anchor_tensor.shape:
                                print(f"  WARNING: shape mismatch {tensor_name} donor {donor_idx + 1}: "
                                      f"{tuple(donor_tensor.shape)} vs anchor {tuple(anchor_tensor.shape)} - skipping donor")
                            else:
                                task_vector = donor_tensor - anchor_tensor
                                task_vectors.append((donor_idx, task_vector))
                            del donor_tensor

                    merged_tensor = None
                    if task_vectors:
                        merged_tensor = apply_dare_ties(tensor_name, anchor_tensor, {}, task_vectors, WEIGHTS,
                                                         density=DENSITY, merge_counter=global_merge_counter, densities=DENSITIES)
                        shard_buffer[tensor_name] = merged_tensor
                    else:
                        shard_buffer[tensor_name] = anchor_tensor

                    del anchor_tensor
                    if merged_tensor is not None:
                        del merged_tensor
                    global_merge_counter += 1

        if shard_buffer:
            out_path = output / shard_file
            out_path.parent.mkdir(parents=True, exist_ok=True)
            save_file(shard_buffer, str(out_path))
            print(f"  Saved {shard_file} ({len(shard_buffer)} tensors)")
            total_processed += len(shard_buffer)
            written_tensors.update(shard_buffer.keys())
            del shard_buffer

    missing_tensors = all_keys - written_tensors
    if missing_tensors:
        print(f"\nERROR: Missing {len(missing_tensors)} tensors in output!")
        for key in sorted(missing_tensors):
            print(f"  {key}")
        exit(1)

    print(f"\n\u2713 Processed {total_processed} tensors in single streaming pass")
    print(f"  Expert layout: {expert_layout_stats['fused_matches']} fused-direct, "
          f"{expert_layout_stats['unfused_remapped']} unfused-remapped, "
          f"{expert_layout_stats['donor_missing']} donor-missing (anchor fallback)")

    index_output = {"metadata": {}, "weight_map": {}}
    for tensor_name, file_name in output_tensor_map.items():
        if tensor_name in written_tensors:
            index_output["weight_map"][tensor_name] = file_name
    total_size_bytes = sum(
        (output / f).stat().st_size for f in sorted(set(index_output["weight_map"].values()))
    )
    index_output["metadata"]["total_size"] = total_size_bytes
    with open(output / "model.safetensors.index.json", "w") as f:
        json.dump(index_output, f)

    import shutil
    print("\nCopying auxiliary files from anchor (tokenizer, config, etc.)...")
    skip_names = {"model.safetensors.index.json"}
    copied = []
    for item in anchor.iterdir():
        if item.is_file() and not item.name.endswith(".safetensors") and item.name not in skip_names:
            shutil.copy2(item, output / item.name)
            copied.append(item.name)
    print(f"  Copied {len(copied)} auxiliary files: {sorted(copied)}")

    print(f"\n\u2713 Merge complete! Output: {output}")
    print(f"  Total tensors processed: {total_processed}")


if __name__ == "__main__":
    merge_qwen_moe()
