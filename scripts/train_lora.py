"""Phase 2 - Training script (the core artifact) for RazorStrike v2.

RazorStrike v2 = clean Qwen/Qwen3.6-35B-A3B (35B MoE, 3B activated) base +
this multi-domain LoRA. Uncensoring is done via the `uncensor` data family,
not weight abliteration.

Loads in PLAIN bf16 (~70GB) - no quantization at all. On a 40GB GPU this
doesn't fit and forces on-the-fly 4-bit quantization, which hits a confirmed,
still-open transformers v5 bug class (huggingface/transformers#43032: MoE
expert weight-merge conversion materializes at full precision on GPU before
quantizing, OOMs even after patching two separate call sites, and the patched
load still bloated to ~39.7GB - a broken/non-functional quantization). On an
80-96GB GPU (Colab's G4 = RTX PRO 6000 Blackwell, ~96GB), plain bf16 LoRA
sidesteps that entire bug class - no BitsAndBytesConfig, no monkeypatches,
no on-the-fly quantize step.

Reads env: BASE_REPO (default Qwen/Qwen3.6-35B-A3B), DATA_REPO, OUT_DIR,
  ADAPTER_REPO, HF_TOKEN, MAXLEN(4096), TARGET_MLP(0), LORA_R(32),
  LORA_ALPHA(64), SAVE_STEPS(50), EVAL_STEPS(250), RESUME.
"""

import os, gc, torch, glob, subprocess
from transformers import (AutoModelForImageTextToText, AutoModelForCausalLM,
                          AutoTokenizer,
                          Trainer, TrainingArguments, DataCollatorForSeq2Seq,
                          TrainerCallback, EarlyStoppingCallback)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from huggingface_hub import upload_folder

BASE   = os.environ.get("BASE_REPO", "Qwen/Qwen3.6-35B-A3B")
DATA   = os.environ["DATA_REPO"]           # lancejames221b/razorstrike-v2-sft OR gs://bucket/path
OUT    = os.environ.get("OUT_DIR", "/content/adapter")
MAXLEN = int(os.environ.get("MAXLEN", "3072"))  # 4096 tail (0.2% of rows >4096, 1.1% >3072) OOMs on 96GB G4; verified via row-length sampling

# GCS staging: activate the service-account key (written to disk by the
# launcher, NEVER committed to this repo) before any `gcloud storage` call.
# GOOGLE_APPLICATION_CREDENTIALS is a client-library convention the gcloud
# CLI itself does not honor - explicit activate-service-account is required.
GCS_KEY_FILE = os.environ.get("GCS_KEY_FILE", "/content/gcs-key.json")
GCS_PROJECT = os.environ.get("GCS_PROJECT", "ewitness-dev")
_gcs_activated = False
# DDP: `gcloud auth activate-service-account` mutates a shared config dir
# (~/.config/gcloud by default). 8 ranks calling it concurrently race on
# the same credentials file. Give each rank its own isolated config dir -
# no cross-process coordination needed, and each rank's own token is fully
# independent (never shared secret state to corrupt).
os.environ.setdefault("CLOUDSDK_CONFIG", f"/tmp/gcloud-config-rank{os.environ.get('LOCAL_RANK', '0')}")


def _gcs_activate():
    global _gcs_activated
    if _gcs_activated:
        return
    if not os.path.exists(GCS_KEY_FILE):
        raise RuntimeError(f"GCS mode requested but key file missing: {GCS_KEY_FILE}")
    subprocess.run(["gcloud", "auth", "activate-service-account",
                     f"--key-file={GCS_KEY_FILE}", f"--project={GCS_PROJECT}"], check=True)
    _gcs_activated = True


tok = AutoTokenizer.from_pretrained(BASE)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


def to_features(ex):
    msgs = ex["messages"]
    prompt = tok.apply_chat_template(msgs[:-1], add_generation_prompt=True, tokenize=True)["input_ids"]
    full   = tok.apply_chat_template(msgs,       add_generation_prompt=False, tokenize=True)["input_ids"]
    if len(full) > MAXLEN or len(prompt) >= len(full):
        return {"input_ids": None, "attention_mask": None, "labels": None}
    labels = [-100] * len(prompt) + full[len(prompt):]
    return {"input_ids": full, "attention_mask": [1] * len(full), "labels": labels}


_local_rank = int(os.environ.get("LOCAL_RANK", "0"))  # torchrun sets this per-process; absent -> single-process (rank 0)

# FSDP mode: accelerate sets this env var for every process under
# `accelerate launch --use_fsdp`. Detected once here and threaded through
# the model-load kwargs, TrainingArguments, and the final-save path below -
# FSDP owns parameter placement/sharding and is incompatible with the
# device_map="auto" pipeline-parallel path used otherwise.
_FSDP = os.environ.get("ACCELERATE_USE_FSDP", "").lower() in ("1", "true", "yes")

if DATA.startswith("gs://"):
    _local = "/content/dataset"
    _ready = os.path.join(_local, ".rsync_complete")
    # DDP: all ranks import this module and reach here independently. Only
    # rank 0 downloads; concurrent rsyncs into the same dir from 8 ranks
    # would corrupt/partially-read the Arrow files. Other ranks poll a
    # sentinel written after rank 0's rsync finishes rather than assuming
    # any particular startup ordering.
    if _local_rank == 0:
        _gcs_activate()
        # Clear any stale sentinel from a prior run/relaunch on this VM
        # BEFORE starting the rsync - otherwise a waiting non-zero rank
        # could see a leftover sentinel from a previous (possibly
        # different DATA_REPO) run and load_from_disk a stale/wrong
        # dataset while rank 0 is still overwriting it underneath.
        if os.path.exists(_ready):
            os.remove(_ready)
        subprocess.run(["gcloud", "storage", "rsync", "-r", DATA, _local], check=True)
        with open(_ready, "w") as _f:
            _f.write(DATA)  # record which DATA_REPO this sentinel corresponds to
    else:
        import time as _time
        _waited = 0
        while not os.path.exists(_ready):
            _time.sleep(5)
            _waited += 5
            if _waited > 1800:
                raise RuntimeError(f"rank {_local_rank}: timed out waiting 30min for "
                                    f"rank 0's dataset download ({_ready} never appeared)")
        with open(_ready) as _f:
            _sentinel_data = _f.read().strip()
        if _sentinel_data != DATA:
            raise RuntimeError(f"rank {_local_rank}: sentinel records DATA_REPO="
                                f"{_sentinel_data!r} but this process expects {DATA!r} - "
                                f"stale sentinel from a different run, refusing to load")
    from datasets import load_from_disk
    ds = load_from_disk(_local)
else:
    ds = load_dataset(DATA)
ds = ds.map(to_features, remove_columns=ds["train"].column_names)
ds = ds.filter(lambda r: r["input_ids"] is not None)

_smoke_longest_n = int(os.environ.get("SMOKE_LONGEST_N", "0"))
if _smoke_longest_n > 0:
    # Worst-case memory validation, opt-in only. A short smoke run (a
    # handful of optimizer steps) sampling randomly can miss the long tail
    # of sequence lengths near MAXLEN that the full run will eventually
    # hit - passing clean on short sequences and then OOMing hours into the
    # $29.39/hr full run is the expensive failure mode this guards against.
    # Order the train split longest-first and keep only the top N so a
    # short smoke run is guaranteed to exercise near-MAXLEN activation
    # memory. The full run must NOT set this env var.
    _train = ds["train"]
    _lens = [len(x) for x in _train["input_ids"]]
    _order = sorted(range(len(_lens)), key=lambda i: -_lens[i])[:_smoke_longest_n]
    ds["train"] = _train.select(_order)
    print(f"[smoke] SMOKE_LONGEST_N={_smoke_longest_n}: train rows reordered longest-first, "
          f"max len={_lens[_order[0]] if _order else 0}", flush=True)

# Plain bf16 load (~70GB) - no quantization, no bnb, no monkeypatch. Needs an
# 80GB+ GPU (Colab G4 = RTX PRO 6000 Blackwell, ~96GB). ImageTextToText is
# the correct class for this *ForConditionalGeneration multimodal MoE arch
# (confirmed empirically earlier); CausalLM as fallback only if that class
# genuinely can't resolve the checkpoint.
_QLORA_4BIT = os.environ.get("QLORA_4BIT", "0") == "1"
# DEVICE_MAP override: fallback path when on-the-fly bnb 4-bit quantization
# OOMs during load (confirmed failure mode on this MoE arch - the expert
# weight-merge conversion materializes at full bf16 precision on GPU before
# quantizing, transformers#43032-class bug). Set DEVICE_MAP=auto with
# QLORA_4BIT=0 on a multi-GPU host (e.g. 2x A100-40GB) to shard plain bf16
# load across GPUs instead of quantizing on a single card.
# DDP: torchrun sets LOCAL_RANK per-process. Each rank MUST load its own
# quantized replica onto its OWN GPU ({"": local_rank}) - device_map="auto"
# under DDP causes ranks to independently naive-shard the model across ALL
# visible GPUs, silently colliding/deadlocking instead of each rank owning
# one full replica. LOCAL_RANK is absent for a single-process launch, so
# this is a no-op there (falls back to GPU 0 exactly as before).
_device_map = os.environ.get("DEVICE_MAP", "").strip() or {"": _local_rank}
# MAX_MEMORY_GIB caps how much of each GPU's weights budget device_map="auto"
# uses, leaving headroom for activations/gradients during backward (naive
# model-parallel splits by weight size only, ignoring activation memory -
# confirmed OOM in backward without this on 2x 40GB A100s).
_max_mem_gib = os.environ.get("MAX_MEMORY_GIB", "").strip()
_max_memory = None
if _max_mem_gib and _device_map == "auto":
    _max_memory = {i: f"{_max_mem_gib}GiB" for i in range(torch.cuda.device_count())}
# A checkpoint already saved in 4-bit (via bitsandbytes' own save/load
# support) carries its own quantization_config in config.json - passing a
# FRESH BitsAndBytesConfig on top of that is what triggers the OOM bug
# (transformers' loader materializes the source tensor at full precision
# before quantizing it AGAIN). Detect an already-quantized checkpoint and
# skip re-quantization entirely - it loads its saved 4-bit tensors
# directly, no full-precision materialization step at all.
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
        # FSDP owns placement: accelerate rank0-loads and broadcasts under
        # cpu_ram_efficient_loading, other ranks init on meta. Passing
        # device_map here is explicitly warned against by transformers
        # (modeling_utils.py:4270) and breaks that contract.
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

# Grounded targets: BOTH attention types (full + linear/SSM), verified against
# the real weight-map (40/40 text layers, zero vision collisions). MoE expert
# MLPs (256 experts) are opt-in via TARGET_MLP - large adapter, only needed if
# you want to adapt the expert weights directly.
targets = ["q_proj", "k_proj", "v_proj", "o_proj",
           "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"]
if os.environ.get("TARGET_MLP", "0") == "1":
    targets += ["gate_proj", "up_proj", "down_proj"]   # 256 experts -> large adapter

lora = LoraConfig(
    r=int(os.environ.get("LORA_R", "32")),
    lora_alpha=int(os.environ.get("LORA_ALPHA", "64")),
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=targets)
model = get_peft_model(model, lora)
model.print_trainable_parameters()

# Fail fast: PEFT silently drops non-matching target names. If almost nothing
# matched, the adapter would "train" for hours and learn nothing.
_tp = sum(p.numel() for p in model.parameters() if p.requires_grad)
_tot = sum(p.numel() for p in model.parameters())
_pct = 100.0 * _tp / max(_tot, 1)
print(f"[guard] trainable params: {_tp:,} ({_pct:.4f}% of {_tot:,})")
assert _pct > 0.01, (f"LoRA matched almost nothing ({_pct:.4f}%). target_modules "
                     f"do not match {BASE}; inspect model.named_modules().")

# ADAPTER_REPO is REQUIRED: durability depends on pushing checkpoints to the
# Hub during training, not just at the very end. A VM reclamation mid-run
# would otherwise wipe /content and lose everything.
ADAPTER_REPO = os.environ["ADAPTER_REPO"]
HF_TOKEN = os.environ.get("HF_TOKEN")

args = TrainingArguments(
    output_dir=OUT, num_train_epochs=2,
    max_steps=int(os.environ.get("MAX_STEPS", "-1")),  # -1 = full 2 epochs; positive caps for validation runs
    per_device_train_batch_size=1, per_device_eval_batch_size=1,
    # DDP: global effective batch = per_device(1) * grad_accum * WORLD_SIZE.
    # Divide GRAD_ACCUM by the GPU count when launching under torchrun so
    # the effective batch (and thus the tuned LR/schedule) stays identical
    # to the single-GPU config - more GPUs should cut wall-clock, not
    # silently change what's being trained.
    gradient_accumulation_steps=int(os.environ.get("GRAD_ACCUM", "16")),
    learning_rate=2e-4, lr_scheduler_type="cosine", warmup_ratio=0.03,
    bf16=True, gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    max_grad_norm=1.0, logging_steps=5,
    save_steps=int(os.environ.get("SAVE_STEPS", "50")), save_total_limit=3,
    eval_strategy="steps", eval_steps=int(os.environ.get("EVAL_STEPS", "250")), optim="adamw_torch",
    report_to="none", dataloader_num_workers=2,
    # Durable checkpointing via custom blocking-push callback (see below).
    # The built-in push_to_hub=True uses async Futures that silently swallow
    # errors (huggingface/transformers#29399 + Future exception swallowing).
    # We disable it and use our own callback that does blocking uploads.
    push_to_hub=False, prediction_loss_only=True,
    load_best_model_at_end=(not _FSDP), metric_for_best_model="eval_loss", greater_is_better=False)


class HubCheckpointPusher(TrainerCallback):
    """Blocking push of each checkpoint to the Hub after every save.
    Unlike push_to_hub=True, this uses run_as_future=False so upload errors
    are raised synchronously and logged, not silently swallowed."""
    def on_save(self, args, state, control, **kwargs):
        from huggingface_hub import create_repo
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
CKPT_GCS = os.environ.get("CKPT_GCS", "").strip()  # e.g. gs://hawq-training-us-central1/checkpoints


class GcsCheckpointPusher(TrainerCallback):
    """Blocking rsync of each checkpoint to GCS after every save. Sibling to
    HubCheckpointPusher, selected via CKPT_GCS instead of the Hub push so a
    failed GCS upload never kills training (caught, printed, not re-raised -
    mirrors HubCheckpointPusher's behavior exactly)."""
    def on_save(self, args, state, control, **kwargs):
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


trainer = Trainer(model=model, args=args,
    train_dataset=ds["train"], eval_dataset=ds["validation"],
    data_collator=DataCollatorForSeq2Seq(tok, label_pad_token_id=-100, padding=True),
    callbacks=[GcsCheckpointPusher() if CKPT_GCS else HubCheckpointPusher(),
               EarlyStoppingCallback(early_stopping_patience=3)])

resume_path = None
if os.environ.get("RESUME"):
    if CKPT_GCS:
        # List gs://.../checkpoints/<ADAPTER_NAME>/checkpoint-*, pick the
        # highest integer suffix, rsync it down, resume from it.
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
        # With hub_strategy="all", checkpoints push to the Hub repo root as
        # checkpoint-N/ dirs. Pull the highest-numbered one and resume from it.
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

# Peak-memory diagnostic, every rank: cheap, always-on visibility into how
# close a run came to OOM. Critical for the smoke test (3-15 steps of random
# sampling can miss the long tail of sequence lengths that a long training
# run will eventually hit) and useful headroom tracking on the full run too.
if torch.cuda.is_available():
    _peak_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"[mem] rank {_local_rank} peak CUDA memory allocated: {_peak_gib:.2f} GiB", flush=True)

# DDP: every rank reaches this point after trainer.train() returns.
# trainer.save_model() under FSDP is a COLLECTIVE call - it all-gathers
# sharded params across every rank internally (accelerator.save_model),
# rank-gating the actual disk write itself. Guarding it with
# is_world_process_zero() would make rank 0 block on a collective the other
# ranks never join (they'd fall through to TRAINING_COMPLETE and exit) -
# a deadlock discovered via transformers#24208 / accelerate collective-save
# reports, confirmed against this exact plan. So it runs on ALL ranks here,
# unconditionally. SHARDED_STATE_DICT (used during training for cheap
# periodic checkpoints) only loads back into FSDP; the final adapter must
# be gathered as FULL_STATE_DICT so it's a normal loadable PEFT directory.
# NOTE: accelerator.get_state_dict() gathers the WHOLE wrapped model (all
# 34.66B base+adapter params get FSDP-unsharded onto rank 0), not just the
# 65M trainable LoRA params - PEFT only filters down to adapter weights
# after the gather when Trainer._save() writes the file. This is expensive
# in both time (~12 saves if done every checkpoint - hence SHARDED_STATE_DICT
# for periodic saves, FULL only here at the very end) and peak memory, so
# free everything gather-adjacent first: optimizer states, autograd graph
# residue, and the CUDA caching allocator's fragmented pool. Verified
# necessary - without this, the final gather hit an NCCL "unhandled cuda
# error" under the tight 4-GPU fallback shape at ~40/41GB already in use.
if _FSDP:
    # zero_grad(set_to_none=True) actually frees gradient tensors (the
    # live-memory term); del'ing trainer.optimizer barely helps since
    # accelerate keeps its own reference to the wrapped optimizer -
    # empty_cache() is what matters, returning the caching allocator's
    # fragmented reserve so NCCL's all-gather buffer has room to allocate.
    trainer.model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model(OUT)      # FSDP-aware state-dict gather, all ranks

# Only rank 0 should write the tokenizer / push the final adapter -
# push_to_hub() called directly (not via Trainer's own internal methods) is
# NOT auto-gated by rank, unlike Trainer's own checkpoint saving. Every
# process racing to write the same OUT dir / push concurrently would
# corrupt the upload. Non-zero ranks skip straight to TRAINING_COMPLETE.
if trainer.is_world_process_zero():
    if not _FSDP:
        model.save_pretrained(OUT)
    tok.save_pretrained(OUT)

    if CKPT_GCS:
        # Durability first: land the final adapter in GCS before attempting the
        # HF push, so a private-storage quota failure (the exact failure mode
        # that has bitten this project before) doesn't strand a completed run.
        try:
            _gcs_activate()
            final_dest = f"{CKPT_GCS.rstrip('/')}/{ADAPTER_NAME}/final"
            subprocess.run(["gcloud", "storage", "rsync", "-r", OUT, final_dest], check=True)
            print(f"[push] final adapter -> {final_dest} OK", flush=True)
        except Exception as e:
            print(f"[push] final adapter GCS push FAILED {type(e).__name__}: {e}", flush=True)
        try:
            model.push_to_hub(ADAPTER_REPO, private=True, token=HF_TOKEN)
            tok.push_to_hub(ADAPTER_REPO, private=True, token=HF_TOKEN)
        except Exception as e:
            print(f"[push] final adapter HF push FAILED (non-fatal, GCS copy is durable) "
                  f"{type(e).__name__}: {e}", flush=True)
    else:
        model.push_to_hub(ADAPTER_REPO, private=True, token=HF_TOKEN)
        tok.push_to_hub(ADAPTER_REPO, private=True, token=HF_TOKEN)

print("TRAINING_COMPLETE")
