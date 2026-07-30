"""Step 5a (HAWQ v1.1 DPO plan) - Hand-rolled LoRA-DPO training entrypoint.

Stacks a fresh LoRA adapter on top of the ALREADY-MERGED v4 crypto/exploit
weights (BASE_REPO points at that merge, not raw HAWQ-v1) to fold in the
omp-advisory correction (Step 1) and the clean-code counterweight (Step 2).

No `trl`: it is absent from every Python env on this Mac and from
vm_setup.py's deps, and the pinned transformers==5.14.1 + peft==0.18.0 stack
has documented breakage on either side of the peft pin - adding trl risks
pip re-resolving transformers on a paid VM. The DPO loss is a closed-form
expression over four log-probabilities (scripts/dpo_common.py:dpo_loss);
this script implements it directly against the SAME proven model-loading
path scripts/train_lora.py uses (copied verbatim below - only the data path
and the loss differ).

Reference model: NONE loaded separately. Reference log-probs come from the
SAME model with the LoRA adapter disabled (`with model.disable_adapter():`
under torch.no_grad()) - this is what keeps DPO in the SFT run's memory
envelope (no second 35B copy). If this misbehaves under FSDP-wrapped
modules (a real risk - flagged in the plan's own contingency), set
REF_LOGPROBS_PREPASS=1 to precompute reference log-probs ONCE up front
(base model has zero LoRA delta at PEFT-init and the reference model must
stay frozen throughout DPO training regardless, so a one-time precompute is
exactly equivalent to calling disable_adapter() at every step - not an
approximation) and cache them, rather than reaching for trl mid-run.

Reads env: BASE_REPO (the merged v4 checkpoint, local path or HF repo),
  DPO_DATA_GCS (gs:// path to combined_pairs.jsonl - REQUIRED; DATA_REPO is
  NOT read here, it exists only to satisfy gce_cluster_train.sh's launcher
  gate), OUT_DIR, ADAPTER_REPO, HF_TOKEN, MAXLEN(2048), MAX_PROMPT_LEN(1024),
  LORA_R(64), LORA_ALPHA(128), DPO_BETA(0.1), SAVE_STEPS(25), EVAL_STEPS(25),
  RESUME, REF_LOGPROBS_PREPASS(0).
"""

import gc
import json
import os
import subprocess
import sys

import torch
from transformers import (AutoModelForImageTextToText, AutoModelForCausalLM,
                          AutoTokenizer,
                          TrainingArguments, TrainerCallback,
                          EarlyStoppingCallback, Trainer)
from peft import LoraConfig, get_peft_model
from huggingface_hub import upload_folder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dpo_common import (parse_prompt_to_messages, build_dpo_pair_features,
                        dpo_loss)  # noqa: E402

BASE   = os.environ.get("BASE_REPO", "Qwen/Qwen3.6-35B-A3B")
OUT    = os.environ.get("OUT_DIR", "/content/adapter")
MAXLEN = int(os.environ.get("MAXLEN") or 2048)
MAX_PROMPT_LEN = int(os.environ.get("MAX_PROMPT_LEN") or 1024)
DPO_BETA = float(os.environ.get("DPO_BETA") or 0.1)
REF_LOGPROBS_PREPASS = os.environ.get("REF_LOGPROBS_PREPASS", "0") == "1"

# GCS staging: activate the service-account key (written to disk by the
# launcher, NEVER committed to this repo) before any `gcloud storage` call.
# GOOGLE_APPLICATION_CREDENTIALS is a client-library convention the gcloud
# CLI itself does not honor - explicit activate-service-account is required.
GCS_KEY_FILE = os.environ.get("GCS_KEY_FILE", "/content/gcs-key.json")
GCS_PROJECT = os.environ.get("GCS_PROJECT", "ewitness-dev")
_gcs_activated = False
# DDP: `gcloud auth activate-service-account` mutates a shared config dir
# (~/.config/gcloud by default). Give each rank its own isolated config dir.
os.environ.setdefault("CLOUDSDK_CONFIG", f"/tmp/gcloud-config-rank{os.environ.get('LOCAL_RANK', '0')}")


def _gcs_activate():
    global _gcs_activated
    if _gcs_activated:
        return
    subprocess.run(["gcloud", "auth", "activate-service-account",
                     f"--key-file={GCS_KEY_FILE}"], check=True)
    subprocess.run(["gcloud", "config", "set", "project", GCS_PROJECT], check=True)
    _gcs_activated = True


tok = AutoTokenizer.from_pretrained(BASE)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


_local_rank = int(os.environ.get("LOCAL_RANK", "0"))

# FSDP mode: accelerate sets this env var for every process under
# `accelerate launch --use_fsdp`. Detected once here and threaded through
# the model-load kwargs, TrainingArguments, and the final-save path below -
# FSDP owns parameter placement/sharding and is incompatible with the
# device_map="auto" pipeline-parallel path used otherwise.
_FSDP = os.environ.get("ACCELERATE_USE_FSDP", "").lower() in ("1", "true", "yes")

# --- DPO dataset: read the combined pairs JSONL (rank 0 downloads from GCS,
# other ranks wait on a sentinel - mirrors train_lora.py's gs:// dataset
# staging pattern exactly). ---------------------------------------------
DPO_DATA_GCS = os.environ["DPO_DATA_GCS"]
_dpo_local = "/content/dpo_pairs.jsonl"
_ready = "/content/.dpo_data_ready"
if _local_rank == 0:
    _gcs_activate()
    if os.path.exists(_ready):
        os.remove(_ready)
    subprocess.run(["gcloud", "storage", "cp", DPO_DATA_GCS, _dpo_local], check=True)
    with open(_ready, "w") as _f:
        _f.write(DPO_DATA_GCS)
else:
    import time as _time
    _waited = 0
    while not os.path.exists(_ready):
        _time.sleep(5)
        _waited += 5
        if _waited > 1800:
            raise RuntimeError(f"rank {_local_rank}: timed out waiting 30min for "
                                f"rank 0's DPO data download ({_ready} never appeared)")

with open(_dpo_local) as _f:
    _raw_pairs = [json.loads(line) for line in _f if line.strip()]
print(f"[data] loaded {len(_raw_pairs)} raw DPO pairs from {DPO_DATA_GCS}", flush=True)


def _build_examples(raw_pairs, tok, maxlen, max_prompt_len):
    """Apply Step 4's truncation policy (shared prompt truncation once per
    pair, response NEVER truncated - overflow pairs dropped) and return the
    list of usable {chosen_*, rejected_*} feature dicts."""
    examples = []
    n_dropped_overflow = 0
    for row in raw_pairs:
        messages = parse_prompt_to_messages(row["prompt"])
        chosen_feat, rejected_feat, _dropped_turns = build_dpo_pair_features(
            messages, row["chosen"], row["rejected"], tok, maxlen, max_prompt_len)
        if chosen_feat is None or rejected_feat is None:
            n_dropped_overflow += 1
            continue
        examples.append({
            "source": row.get("source", "agent"),
            "chosen_input_ids": chosen_feat["input_ids"],
            "chosen_attention_mask": chosen_feat["attention_mask"],
            "chosen_labels": chosen_feat["labels"],
            "rejected_input_ids": rejected_feat["input_ids"],
            "rejected_attention_mask": rejected_feat["attention_mask"],
            "rejected_labels": rejected_feat["labels"],
        })
    print(f"[data] built {len(examples)} usable examples "
          f"({n_dropped_overflow} dropped for MAXLEN/MAX_PROMPT_LEN overflow)",
          flush=True)
    return examples


_all_examples = _build_examples(_raw_pairs, tok, MAXLEN, MAX_PROMPT_LEN)
assert len(_all_examples) > 0, "no usable DPO examples survived truncation - abort"

# Held-out validation slice (10%, STRATIFIED by source - a naive positional
# slice of the concatenated corpus (agent_pairs.jsonl then clean_pairs.jsonl)
# would put clean_control pairs almost entirely in one split, so eval_loss
# would say nothing about the omp-agent half, and training would silently
# lose a chunk of the 400 clean-control counterweight pairs Step 6's gate
# depends on. Fixed literal seed (never rank/time-derived): all 4 FSDP
# ranks build this dataset independently at startup and MUST agree on the
# exact same split or the ranks desync on step counts.
import random as _random
_by_source = {}
for ex in _all_examples:
    _by_source.setdefault(ex["source"], []).append(ex)
_train_examples, _val_examples = [], []
for _src, _exs in _by_source.items():
    _rng = _random.Random(42)
    _shuffled = list(_exs)
    _rng.shuffle(_shuffled)
    _n_val_src = max(1, int(len(_shuffled) * 0.10))
    _val_examples.extend(_shuffled[:_n_val_src])
    _train_examples.extend(_shuffled[_n_val_src:])
_train_source_mix = {s: sum(1 for e in _train_examples if e["source"] == s) for s in _by_source}
_val_source_mix = {s: sum(1 for e in _val_examples if e["source"] == s) for s in _by_source}
print(f"[data] train={len(_train_examples)} {_train_source_mix} "
      f"val={len(_val_examples)} {_val_source_mix}", flush=True)


class DpoDataset(torch.utils.data.Dataset):
    def __init__(self, examples, split_name):
        self.examples = examples
        self.split_name = split_name

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = dict(self.examples[idx])
        ex["_dpo_split"] = self.split_name
        ex["_dpo_idx"] = idx
        return ex


def dpo_collate(features):
    """per_device_train_batch_size=1 (plan-fixed hyperparameter) - each
    batch is exactly one example, so no cross-example padding is needed."""
    assert len(features) == 1, (
        f"DPO training requires per_device_train_batch_size=1, got batch of "
        f"{len(features)}")
    f = features[0]
    batch = {
        "chosen_input_ids": torch.tensor([f["chosen_input_ids"]], dtype=torch.long),
        "chosen_attention_mask": torch.tensor([f["chosen_attention_mask"]], dtype=torch.long),
        "chosen_labels": torch.tensor([f["chosen_labels"]], dtype=torch.long),
        "rejected_input_ids": torch.tensor([f["rejected_input_ids"]], dtype=torch.long),
        "rejected_attention_mask": torch.tensor([f["rejected_attention_mask"]], dtype=torch.long),
        "rejected_labels": torch.tensor([f["rejected_labels"]], dtype=torch.long),
    }
    if "_dpo_split" in f:
        batch["_dpo_index"] = (f["_dpo_split"], f["_dpo_idx"])
    return batch


# --- Model load block, copied verbatim from scripts/train_lora.py (lines
# ~145-252 there) - _FSDP detection above, _load_kw construction,
# FORCE_CAUSAL_LM handling, gradient checkpointing, target_modules,
# LoraConfig, and the trainable-params guard are all unchanged. Only the
# data path and the loss differ for DPO. ---------------------------------
_QLORA_4BIT = os.environ.get("QLORA_4BIT", "0") == "1"
_device_map = os.environ.get("DEVICE_MAP", "").strip() or {"": _local_rank}
_max_mem_gib = os.environ.get("MAX_MEMORY_GIB", "").strip()
_max_memory = None
if _max_mem_gib and _device_map == "auto":
    _max_memory = {i: f"{_max_mem_gib}GiB" for i in range(torch.cuda.device_count())}
_already_quantized = False
if _QLORA_4BIT:
    try:
        from transformers import AutoConfig
        _already_quantized = getattr(AutoConfig.from_pretrained(BASE), "quantization_config", None) is not None
    except Exception:
        _already_quantized = False
    if _already_quantized:
        print(f"[load] {BASE} is already 4-bit quantized - loading directly, no re-quantization", flush=True)
if _QLORA_4BIT and not _already_quantized:
    from transformers import BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training
    _bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    _load_kw = dict(device_map=_device_map, quantization_config=_bnb_cfg, low_cpu_mem_usage=True)
    if _max_memory:
        _load_kw["max_memory"] = _max_memory
elif _QLORA_4BIT and _already_quantized:
    from peft import prepare_model_for_kbit_training
    _load_kw = dict(device_map=_device_map, low_cpu_mem_usage=True)
    if _max_memory:
        _load_kw["max_memory"] = _max_memory
else:
    if _FSDP:
        _load_kw = dict(torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    else:
        _load_kw = dict(device_map=_device_map, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        if _max_memory:
            _load_kw["max_memory"] = _max_memory
if os.environ.get("FORCE_CAUSAL_LM", "0") == "1":
    model = AutoModelForCausalLM.from_pretrained(BASE, **_load_kw)
else:
    try:
        model = AutoModelForImageTextToText.from_pretrained(BASE, **_load_kw)
    except Exception as e:
        print(f"[load] ImageTextToText failed ({type(e).__name__}); trying CausalLM")
        model = AutoModelForCausalLM.from_pretrained(BASE, **_load_kw)
if _QLORA_4BIT:
    model = prepare_model_for_kbit_training(model)
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()

targets = ["q_proj", "k_proj", "v_proj", "o_proj",
           "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"]
if os.environ.get("TARGET_MLP", "0") == "1":
    targets += ["gate_proj", "up_proj", "down_proj"]

lora = LoraConfig(
    r=int(os.environ.get("LORA_R", "64")),
    lora_alpha=int(os.environ.get("LORA_ALPHA", "128")),
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=targets)
model = get_peft_model(model, lora)
model.print_trainable_parameters()

_tp = sum(p.numel() for p in model.parameters() if p.requires_grad)
_tot = sum(p.numel() for p in model.parameters())
_pct = 100.0 * _tp / max(_tot, 1)
print(f"[guard] trainable params: {_tp:,} ({_pct:.4f}% of {_tot:,})")
assert _pct > 0.01, (f"LoRA matched almost nothing ({_pct:.4f}%). target_modules "
                     f"do not match {BASE}; inspect model.named_modules().")

ADAPTER_REPO = os.environ["ADAPTER_REPO"]
HF_TOKEN = os.environ.get("HF_TOKEN")


# --- Reference log-probs: precompute-and-cache fallback (REF_LOGPROBS_PREPASS=1).
# Runs BEFORE get_peft_model() would matter... but LoRA is already attached
# above (needed for the primary disable_adapter() path to exist at all), so
# this precompute call goes through `with model.disable_adapter():` too -
# mathematically identical to calling it fresh every step, since the
# reference model never changes. The env flag only controls WHEN the
# lookup happens (cached upfront vs recomputed live) and gives an escape
# hatch if repeated disable_adapter() calls misbehave mid-training under
# FSDP - a one-time prepass is far less exposed to that than thousands of
# repeated calls across the whole run. ------------------------------------
_ref_cache = {}


def _logprob_sum(fwd_model, input_ids, attention_mask, labels):
    out = fwd_model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[:, :-1, :]  # keep bf16 - cross_entropy's CUDA kernel
    # accumulates in fp32 internally (accscalar_t) for half/bf16 inputs, so
    # explicitly casting here would only materialize a redundant full-size
    # fp32 copy (~1.5GB per forward at MAXLEN=1536, retained in BOTH the
    # chosen and rejected policy graphs simultaneously until backward).
    targets_ = labels[:, 1:]
    # F.cross_entropy(ignore_index=-100) is fused (never materializes the
    # full [B,T,V] log_softmax distribution the way log_softmax+gather did)
    # and treats -100 positions as contributing exactly 0 to the per-token
    # loss, so summing directly is correct with no separate mask/gather.
    # Saves ~2.4GB of activation per forward at MAXLEN=2048 on this vocab
    # size - real headroom on a 4-forward-per-step DPO run.
    per_token_nll = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets_.reshape(-1),
        reduction="none", ignore_index=-100).view(targets_.shape)
    # Cast the SMALL [B,T] per-token result to fp32 before summing, not the
    # huge [B,T,V] logits before cross_entropy - summing hundreds-to-
    # thousands of bf16 terms directly loses real precision (measured
    # ~0.25% relative on a 128-token synthetic case, ~10x worse than
    # casting post-hoc), and DPO's signal is a difference of these sums so
    # that bias doesn't just wash out.
    return (-per_token_nll.float()).sum(dim=-1)


def _unwrap_to_adapter_toggle(m):
    """Find the nearest object in the wrap chain exposing disable_adapter()
    (handles a possible FSDP/accelerate wrapper around the PeftModel)."""
    seen = m
    for _ in range(5):
        if hasattr(seen, "disable_adapter"):
            return seen
        if hasattr(seen, "module"):
            seen = seen.module
        else:
            break
    return m  # fall through - will raise naturally if truly absent


if REF_LOGPROBS_PREPASS:
    print("[ref] REF_LOGPROBS_PREPASS=1: precomputing reference log-probs "
          "once for the full dataset before training begins", flush=True)
    _toggle = _unwrap_to_adapter_toggle(model)
    model.eval()
    with torch.no_grad(), _toggle.disable_adapter():
        for split_name, split in (("train", _train_examples), ("val", _val_examples)):
            for i, ex in enumerate(split):
                dev = next(model.parameters()).device
                c_ids = torch.tensor([ex["chosen_input_ids"]], device=dev)
                c_am = torch.tensor([ex["chosen_attention_mask"]], device=dev)
                c_lb = torch.tensor([ex["chosen_labels"]], device=dev)
                r_ids = torch.tensor([ex["rejected_input_ids"]], device=dev)
                r_am = torch.tensor([ex["rejected_attention_mask"]], device=dev)
                r_lb = torch.tensor([ex["rejected_labels"]], device=dev)
                ref_chosen_lp = _logprob_sum(model, c_ids, c_am, c_lb).item()
                ref_rejected_lp = _logprob_sum(model, r_ids, r_am, r_lb).item()
                _ref_cache[(split_name, i)] = (ref_chosen_lp, ref_rejected_lp)
                if (i + 1) % 100 == 0:
                    print(f"[ref] {split_name} {i + 1}/{len(split)}", flush=True)
    model.train()
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[ref] cached {len(_ref_cache)} reference log-prob pairs", flush=True)


class DpoTrainer(Trainer):
    """Overrides compute_loss for the DPO objective. train_dataset/
    eval_dataset carry a `_split_name` list attribute so REF_LOGPROBS_PREPASS
    can look each example up by (split, index) instead of recomputing."""

    _step0_checked = False

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        c_ids, c_am, c_lb = (inputs["chosen_input_ids"], inputs["chosen_attention_mask"],
                              inputs["chosen_labels"])
        r_ids, r_am, r_lb = (inputs["rejected_input_ids"], inputs["rejected_attention_mask"],
                              inputs["rejected_labels"])

        pol_chosen_lp = _logprob_sum(model, c_ids, c_am, c_lb)
        pol_rejected_lp = _logprob_sum(model, r_ids, r_am, r_lb)

        idx = inputs.get("_dpo_index")
        if REF_LOGPROBS_PREPASS and idx is not None:
            split_name, i = idx
            ref_chosen_lp_v, ref_rejected_lp_v = _ref_cache[(split_name, int(i))]
            ref_chosen_lp = torch.tensor([ref_chosen_lp_v], device=pol_chosen_lp.device)
            ref_rejected_lp = torch.tensor([ref_rejected_lp_v], device=pol_rejected_lp.device)
        else:
            _toggle = _unwrap_to_adapter_toggle(model)
            with torch.no_grad(), _toggle.disable_adapter():
                ref_chosen_lp = _logprob_sum(model, c_ids, c_am, c_lb)
                ref_rejected_lp = _logprob_sum(model, r_ids, r_am, r_lb)

        if not DpoTrainer._step0_checked:
            # LoRA's B matrix is zero-initialized, so the adapter
            # contributes EXACTLY zero at init: policy and reference
            # log-probs SHOULD be identical on this very first call. In
            # practice they won't be bit-identical: pol_*_lp runs through
            # train-mode + gradient-checkpointing + autograd, ref_*_lp
            # through no_grad - different kernel paths in bf16, and each
            # is a SUM over up to ~1024-2048 per-token log-probs, so
            # rounding noise accumulates. A loose bound (order 1.0, not
            # 1e-2) still cleanly separates that noise floor from a
            # genuinely broken reference - if disable_adapter() failed to
            # take effect under FSDP wrapping (the plan's named risk), the
            # divergence would be structural (wrong/garbage submodule
            # state), not a small rounding artifact, and would show up as
            # a difference orders of magnitude larger than noise.
            DpoTrainer._step0_checked = True
            c_diff = (pol_chosen_lp - ref_chosen_lp).abs().max().item()
            r_diff = (pol_rejected_lp - ref_rejected_lp).abs().max().item()
            print(f"[step0-check] |pol_chosen_lp - ref_chosen_lp|={c_diff:.6f} "
                  f"|pol_rejected_lp - ref_rejected_lp|={r_diff:.6f} "
                  f"(expect near-0, bf16 sum-of-~1000-tokens noise floor "
                  f"~O(0.1-1) is normal)", flush=True)
            assert c_diff < 2.0 and r_diff < 2.0, (
                f"STEP-0 SANITY CHECK FAILED: policy and reference log-probs "
                f"differ by up to {max(c_diff, r_diff):.4f} at initialization, "
                f"when the LoRA adapter is a mathematical no-op - far beyond "
                f"bf16 rounding noise. disable_adapter() did not take effect "
                f"(likely FSDP-wrapping issue) - the reference model is "
                f"wrong. Set REF_LOGPROBS_PREPASS=1 and relaunch rather than "
                f"continuing.")

        loss = dpo_loss(pol_chosen_lp, pol_rejected_lp, ref_chosen_lp, ref_rejected_lp,
                         beta=DPO_BETA)

        with torch.no_grad():
            hit = ((pol_chosen_lp - ref_chosen_lp) >
                   (pol_rejected_lp - ref_rejected_lp)).float().mean().item()
        # Instantaneous reward_accuracy is meaningless with batch size 1
        # (always exactly 0.0 or 1.0 per call) - the plan's own health
        # check ("climbs from ~0.5 toward 0.7+", "stop if it sits at ~0.5
        # through step 65") requires an aggregate. Track both a full-run
        # cumulative mean and a short recent window so a stall is visible
        # quickly rather than washed out by early-run noise.
        if not hasattr(self, "_rw_hits"):
            self._rw_hits = 0
            self._rw_total = 0
            self._rw_recent = []
        self._rw_hits += hit
        self._rw_total += 1
        self._rw_recent.append(hit)
        if len(self._rw_recent) > 50:
            self._rw_recent.pop(0)
        cumulative_acc = self._rw_hits / self._rw_total
        recent_acc = sum(self._rw_recent) / len(self._rw_recent)

        if self.state.global_step % max(self.args.logging_steps, 1) == 0:
            print(f"[train] step={self.state.global_step} loss={loss.item():.4f} "
                  f"reward_accuracy_instant={hit:.1f} "
                  f"reward_accuracy_cumulative={cumulative_acc:.4f} "
                  f"reward_accuracy_last50={recent_acc:.4f} "
                  f"(n={self._rw_total})", flush=True)

        # Always return a plain scalar loss - prediction_loss_only=True
        # means Trainer never requests return_outputs=True, but returning
        # a dict here would also risk Trainer treating {"reward_accuracy":
        # <0-dim tensor>} as logits to gather/pad across FSDP ranks at
        # eval time, which is not what that value is.
        return loss


args = TrainingArguments(
    output_dir=OUT, num_train_epochs=1,
    max_steps=int(os.environ.get("MAX_STEPS", "-1")),
    per_device_train_batch_size=1, per_device_eval_batch_size=1,
    gradient_accumulation_steps=int(os.environ.get("GRAD_ACCUM", "4")),
    learning_rate=float(os.environ.get("LEARNING_RATE", "1e-5")),
    lr_scheduler_type="cosine", warmup_ratio=0.1,
    bf16=True, gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    max_grad_norm=1.0, logging_steps=5,
    save_steps=int(os.environ.get("SAVE_STEPS", "25")), save_total_limit=3,
    eval_strategy="steps", eval_steps=int(os.environ.get("EVAL_STEPS", "25")),
    optim="adamw_torch",
    report_to="none", dataloader_num_workers=2,
    push_to_hub=False, prediction_loss_only=True,
    load_best_model_at_end=False,  # no single "eval_loss" minimum target for DPO the way SFT has
    remove_unused_columns=False)


class HubCheckpointPusher(TrainerCallback):
    """Blocking push of each checkpoint to the Hub after every save.
    Unchanged from scripts/train_lora.py."""
    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        from huggingface_hub import create_repo
        import glob
        ckpts = sorted(glob.glob(os.path.join(args.output_dir, "checkpoint-*")),
                       key=lambda p: int(p.split("-")[-1]))
        if not ckpts:
            return
        latest = ckpts[-1]
        name = os.path.basename(latest)
        try:
            create_repo(ADAPTER_REPO, token=HF_TOKEN, exist_ok=True, private=True)
            upload_folder(
                repo_id=ADAPTER_REPO,
                folder_path=latest,
                path_in_repo=name,
                token=HF_TOKEN,
                run_as_future=False,
            )
            print(f"[push] {name} -> Hub OK", flush=True)
        except Exception as e:
            print(f"[push] {name} FAIL {type(e).__name__}: {e}", flush=True)


ADAPTER_NAME = ADAPTER_REPO.rsplit("/", 1)[-1]
CKPT_GCS = os.environ.get("CKPT_GCS", "").strip()


class GcsCheckpointPusher(TrainerCallback):
    """Blocking rsync of each checkpoint to GCS after every save. Unchanged
    from scripts/train_lora.py."""
    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        import glob
        ckpts = sorted(glob.glob(os.path.join(args.output_dir, "checkpoint-*")),
                       key=lambda p: int(p.split("-")[-1]))
        if not ckpts:
            return
        latest = ckpts[-1]
        name = os.path.basename(latest)
        dest = f"{CKPT_GCS.rstrip('/')}/{ADAPTER_NAME}/{name}"
        try:
            _gcs_activate()
            subprocess.run(["gcloud", "storage", "rsync", "-r", latest, dest], check=True)
            print(f"[push] {name} -> {dest} OK", flush=True)
        except Exception as e:
            print(f"[push] {name} FAIL {type(e).__name__}: {e}", flush=True)


class FsdpMemoryCleanupCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        if _FSDP:
            gc.collect()
            torch.cuda.empty_cache()


_callbacks = [GcsCheckpointPusher() if CKPT_GCS else HubCheckpointPusher(),
              FsdpMemoryCleanupCallback()]
if not _FSDP:
    _callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))

train_dataset = DpoDataset(_train_examples, "train")
eval_dataset = DpoDataset(_val_examples, "val")

trainer = DpoTrainer(model=model, args=args,
    train_dataset=train_dataset, eval_dataset=eval_dataset,
    data_collator=dpo_collate,
    callbacks=_callbacks)

resume_path = None
if os.environ.get("RESUME"):
    if CKPT_GCS:
        try:
            _gcs_activate()
            prefix = f"{CKPT_GCS.rstrip('/')}/{ADAPTER_NAME}/"
            r = subprocess.run(["gcloud", "storage", "ls", prefix],
                               check=True, capture_output=True, text=True)
            ckpt_dirs = sorted(
                {line.strip().rstrip("/") for line in r.stdout.splitlines()
                 if "/checkpoint-" in line},
                key=lambda s: int(s.rsplit("-", 1)[-1]))
            if ckpt_dirs:
                latest = ckpt_dirs[-1]
                name = latest.rsplit("/", 1)[-1]
                resume_path = os.path.join(OUT, name)
                subprocess.run(["gcloud", "storage", "rsync", "-r", latest, resume_path], check=True)
                print(f"[resume] pulled {name} from GCS -> {resume_path}")
            else:
                print("[resume] no GCS checkpoint dirs found; starting fresh")
        except Exception as e:
            print(f"[resume] no GCS checkpoint found ({type(e).__name__}: {e}); starting fresh")
    else:
        from huggingface_hub import list_repo_files, snapshot_download
        try:
            files = list_repo_files(ADAPTER_REPO, token=HF_TOKEN)
            ckpt_dirs = sorted({f.split("/")[0] for f in files if f.startswith("checkpoint-")},
                                key=lambda s: int(s.split("-")[1]))
            if ckpt_dirs:
                latest = ckpt_dirs[-1]
                snapshot_download(ADAPTER_REPO, allow_patterns=[f"{latest}/*"],
                                   token=HF_TOKEN, local_dir=OUT)
                resume_path = os.path.join(OUT, latest)
                print(f"[resume] pulled {latest} from hub -> {resume_path}")
            else:
                print("[resume] no checkpoint dirs on hub; starting fresh")
        except Exception as e:
            print(f"[resume] no hub checkpoint found ({type(e).__name__}: {e}); starting fresh")

trainer.train(resume_from_checkpoint=resume_path)
print(f"[steps] completed global_step={trainer.state.global_step} of expected "
      f"max_steps={trainer.state.max_steps} - a mismatch means training stopped "
      f"early (e.g. EarlyStoppingCallback), not that it ran to completion.",
      flush=True)

if torch.cuda.is_available():
    _peak_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"[mem] rank {_local_rank} peak CUDA memory allocated: {_peak_gib:.2f} GiB", flush=True)

# FSDP final save: collective call on ALL ranks (see train_lora.py's
# extensive comment on this - accelerator.save_model all-gathers sharded
# params internally; guarding with is_world_process_zero() would deadlock
# the other ranks on a collective they'd never join).
if _FSDP:
    trainer.model.zero_grad(set_to_none=True)
    if hasattr(trainer, "optimizer") and trainer.optimizer is not None:
        del trainer.optimizer
    gc.collect()
    torch.cuda.empty_cache()
    trainer.save_model(OUT)

if trainer.is_world_process_zero():
    if not _FSDP:
        trainer.save_model(OUT)
    tok.save_pretrained(OUT)
    tok.push_to_hub(ADAPTER_REPO, private=True, token=HF_TOKEN)

print("TRAINING_COMPLETE")
