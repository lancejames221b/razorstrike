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
    lens = sorted(max(len(e["chosen_input_ids"]), len(e["rejected_input_ids"])) for e in examples)
    if lens:
        p98 = lens[int(len(lens) * 0.98)]
        print(f"[data] built {len(examples)} usable examples "
              f"({n_dropped_overflow} dropped for MAXLEN/MAX_PROMPT_LEN overflow) "
              f"seq_len p50={lens[len(lens)//2]} p98={p98} max={lens[-1]} "
              f"(memory-tuning visibility: if OOM recurs, the real ceiling to "
              f"target is p98/max, not a guessed MAXLEN)",
              flush=True)
    else:
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
    """Overrides training_step (train, single-graph-at-a-time split
    backward) and compute_loss (eval-only, combined 4-forward form - safe
    since eval never retains a graph across a backward call)."""

    _step0_checked = False

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Overrides the whole step (not just compute_loss) to do a SPLIT
        backward: chosen and rejected are each forwarded-with-grad and
        backpropagated SEPARATELY, one at a time, instead of both being
        forwarded-with-grad and held alive simultaneously until one
        combined loss.backward(). Confirmed OOM site was inside backward's
        gradient-checkpoint recompute while BOTH policy graphs were
        retained; this removes that by construction - only ONE retained
        graph ever exists at a time - at the cost of two extra forward
        passes per step (6 total instead of 4).

        Mathematically exact, not an approximation: for
        L = -logsigmoid(beta*h), h = (pc-rc)-(pr-rr) with rc/rr treated as
        constants, dL/dpc = -beta*sigmoid(-beta*h) and dL/dpr =
        +beta*sigmoid(-beta*h). Computing that scalar under no_grad (cheap,
        no retained graph) and feeding it to `tensor.backward(gradient=...)`
        on a SECOND, separately-retained forward of just that one side
        reproduces the identical gradient a combined-loss single backward
        would produce - verified locally against dpo_loss(...).backward()
        on a toy model to 1e-6 before deploying.
        """
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()
        inputs = self._prepare_inputs(inputs)

        c_ids, c_am, c_lb = (inputs["chosen_input_ids"], inputs["chosen_attention_mask"],
                              inputs["chosen_labels"])
        r_ids, r_am, r_lb = (inputs["rejected_input_ids"], inputs["rejected_attention_mask"],
                              inputs["rejected_labels"])

        # Reference log-probs: cached (REF_LOGPROBS_PREPASS) or live
        # disable_adapter(), always no_grad - never retained either way.
        idx = inputs.get("_dpo_index")
        if REF_LOGPROBS_PREPASS and idx is not None:
            split_name, i = idx
            ref_chosen_lp_v, ref_rejected_lp_v = _ref_cache[(split_name, int(i))]
            device = next(model.parameters()).device
            ref_chosen_lp = torch.tensor([ref_chosen_lp_v], device=device)
            ref_rejected_lp = torch.tensor([ref_rejected_lp_v], device=device)
        else:
            _toggle = _unwrap_to_adapter_toggle(model)
            with torch.no_grad(), _toggle.disable_adapter():
                ref_chosen_lp = _logprob_sum(model, c_ids, c_am, c_lb)
                ref_rejected_lp = _logprob_sum(model, r_ids, r_am, r_lb)

        # Pass 1 (no_grad, both sides): just to get the scalar gradient
        # coefficient g and the reportable loss value. No graph retained.
        with torch.no_grad():
            pc_ng = _logprob_sum(model, c_ids, c_am, c_lb)
            pr_ng = _logprob_sum(model, r_ids, r_am, r_lb)

        if not DpoTrainer._step0_checked:
            # LoRA's B matrix is zero-initialized, so the adapter
            # contributes EXACTLY zero at init: policy and reference
            # log-probs SHOULD be identical on this very first call. A
            # loose bound (order 1.0, not 1e-2) separates bf16 rounding
            # noise (different kernel paths, sums over ~1000+ tokens) from
            # a genuinely broken reference (disable_adapter() not taking
            # effect under FSDP wrapping - the plan's named risk).
            DpoTrainer._step0_checked = True
            c_diff = (pc_ng - ref_chosen_lp).abs().max().item()
            r_diff = (pr_ng - ref_rejected_lp).abs().max().item()
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

        h = (pc_ng - ref_chosen_lp) - (pr_ng - ref_rejected_lp)
        loss = -torch.nn.functional.logsigmoid(DPO_BETA * h).mean()
        n = pc_ng.shape[0]
        g = DPO_BETA * torch.sigmoid(-DPO_BETA * h)  # dL/dpc = -g, dL/dpr = +g

        # Pass 2 (WITH grad, chosen only): single retained graph, backward
        # immediately, then free before pass 3 ever allocates.
        pc_grad = _logprob_sum(model, c_ids, c_am, c_lb)
        self.accelerator.backward(pc_grad, gradient=(-g / n))
        del pc_grad
        torch.cuda.empty_cache()

        # Pass 3 (WITH grad, rejected only): same, after pass 2's graph is gone.
        pr_grad = _logprob_sum(model, r_ids, r_am, r_lb)
        self.accelerator.backward(pr_grad, gradient=(g / n))
        del pr_grad
        torch.cuda.empty_cache()

        with torch.no_grad():
            hit = ((pc_ng - ref_chosen_lp) > (pr_ng - ref_rejected_lp)).float().mean().item()
        # Instantaneous reward_accuracy is meaningless with batch size 1
        # (always exactly 0.0 or 1.0 per call) - track a cumulative mean
        # and a short recent window so a stall is visible quickly.
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

        return loss.detach()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """EVAL-ONLY path. Combined 4-forward-pass form is safe here: eval
        always runs under torch.no_grad() (this method's own context
        manager, and prediction_step below wraps it again), so no graph
        is ever retained during eval regardless - exactly why the OOM
        never happened at an eval boundary in earlier runs, only inside
        training's backward. Reached via prediction_step's explicit call
        below, NOT via Trainer's own has_labels routing (confirmed on-GPU
        in the Step 2 smoke test: this dataset's batches carry no "labels"
        key - only chosen_labels/rejected_labels - so Trainer.
        prediction_step's has_labels/loss_without_labels branch always
        picks the bare `model(**inputs)` path and ValueErrors ("must
        specify exactly one of input_ids or inputs_embeds") the moment
        EVAL_STEPS is hit; the prediction_step override is what actually
        routes eval to this method). Do NOT remove this method: eval will
        TypeError against the base Trainer's compute_loss (it expects
        model(**inputs) to accept chosen_input_ids/rejected_input_ids
        directly, not this dataset's key names) if this is deleted."""
        with torch.no_grad():
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
                with _toggle.disable_adapter():
                    ref_chosen_lp = _logprob_sum(model, c_ids, c_am, c_lb)
                    ref_rejected_lp = _logprob_sum(model, r_ids, r_am, r_lb)
            loss = dpo_loss(pol_chosen_lp, pol_rejected_lp, ref_chosen_lp, ref_rejected_lp,
                             beta=DPO_BETA)
        if return_outputs:
            return loss, {}
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Overridden: base Trainer.prediction_step's has_labels /
        loss_without_labels branching never routes to compute_loss for
        this dataset's chosen_*/rejected_* batch shape (confirmed on-GPU:
        it falls through to a bare model(**inputs) call, which ValueErrors
        since **inputs unpacks chosen_input_ids/rejected_input_ids, never
        input_ids). Call compute_loss directly instead - args sets
        prediction_loss_only=True, so no logits/labels are needed."""
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return (loss.detach(), None, None)


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

    def on_substep_end(self, args, state, control, **kwargs):
        # Fires after EACH grad-accumulation micro-step's backward (not
        # just the full accumulated step) - the OOM observed in practice
        # happened mid-backward on a later micro-step within one
        # accumulation cycle, after several micro-steps had already run.
        # compute_loss's own empty_cache() only covers the pre-backward
        # forward-pass memory; this covers what backward itself retains
        # (gradient buffers, checkpointing recomputation scratch). No
        # gc.collect() here - this runs every micro-step (hundreds per
        # run) and a full GC pass over a 35B-param object graph is real
        # wall-clock for no extra memory return beyond empty_cache().
        if _FSDP:
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

def _gcs_checkpoint_complete(gcs_dir, world_size):
    """A checkpoint-N dir in GCS is resumable iff its save-format-specific
    artifacts are all present at non-truncated size. Under FSDP
    (SHARDED_STATE_DICT) that means the DCP model/optimizer dirs each have
    .metadata + exactly world_size __i_0.distcp shards of near-uniform
    size; under the non-FSDP single-process pipeline path (CKPT_GCS is
    independent of _FSDP - see gce_cluster_train.sh's device_map=auto
    branch) checkpoints are plain single-file PEFT/optimizer dumps with no
    DCP sharding at all, so the DCP check would reject every checkpoint
    (missing pytorch_model_fsdp_0/.metadata) and silently restart training
    from step 0. Either way, plus trainer_state.json in both modes. A
    crash mid-push (GcsCheckpointPusher is a plain rsync with no
    manifest/commit step) leaves a subset of these objects - resuming
    from that silently corrupts the run. Returns (ok: bool, why: str)."""
    r = subprocess.run(["gcloud", "storage", "ls", "-l", "-r", gcs_dir.rstrip("/") + "/**"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return False, f"ls failed: {r.stderr.strip()[:200]}"
    sizes = {}
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0].isdigit():
            sizes[parts[-1]] = int(parts[0])
    if not any(k.endswith("/trainer_state.json") for k in sizes):
        return False, "missing trainer_state.json"
    if not _FSDP:
        for name in ("adapter_model.safetensors", "optimizer.pt"):
            match = [v for k, v in sizes.items() if k.endswith(f"/{name}")]
            if not match:
                return False, f"missing {name}"
            if match[0] == 0:
                return False, f"{name} is 0 bytes (truncated upload)"
        return True, "ok"
    def shards(sub):
        return sorted(v for k, v in sizes.items()
                      if f"/{sub}/" in k and k.endswith(".distcp"))
    for sub in ("pytorch_model_fsdp_0", "optimizer_0"):
        if not any(k.endswith(f"/{sub}/.metadata") for k in sizes):
            return False, f"missing {sub}/.metadata"
        s = shards(sub)
        if len(s) != world_size:
            return False, f"{sub}: {len(s)} distcp shards, expected {world_size}"
        if s and s[0] < 0.9 * s[-1]:
            return False, f"{sub}: shard size spread {s[0]}..{s[-1]} exceeds 10% (truncated upload)"
    return True, "ok"


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
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            latest = None
            for cand in reversed(ckpt_dirs):
                ok, why = _gcs_checkpoint_complete(cand, world_size)
                if ok:
                    latest = cand
                    break
                cand_name = cand.rsplit("/", 1)[-1]
                print(f"[resume] SKIPPING {cand_name}: {why} (partial push - delete it in GCS or it will be re-checked next resume)")
            if latest:
                name = latest.rsplit("/", 1)[-1]
                resume_path = os.path.join(OUT, name)
                subprocess.run(["gcloud", "storage", "rsync", "-r", latest, resume_path], check=True)
                print(f"[resume] pulled {name} from GCS -> {resume_path}")
            else:
                if ckpt_dirs:
                    print(f"[resume] all {len(ckpt_dirs)} GCS checkpoint(s) failed verification; starting fresh")
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
    # Informational only - Step 4 of the DPO plan merges on-VM from
    # /content/adapter (already saved above) using BASE_REPO's tokenizer,
    # never this pushed copy. Confirmed on-GPU: a private-storage-quota
    # 403 here crashed rank 0 (and, via torchrun's elastic launch, every
    # other rank with it) AFTER the real adapter save completed, so
    # TRAINING_COMPLETE never printed despite training having actually
    # finished. Don't let a non-critical upload fail a multi-hour run.
    try:
        tok.push_to_hub(ADAPTER_REPO, private=True, token=HF_TOKEN)
    except Exception as e:
        print(f"[push] tokenizer push to {ADAPTER_REPO} failed ({type(e).__name__}: {e}) - "
              f"non-fatal, adapter itself already saved to {OUT}", flush=True)

print("TRAINING_COMPLETE")
