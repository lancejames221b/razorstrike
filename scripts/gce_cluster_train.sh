#!/usr/bin/env bash
# HAWQ-SEC-RE v3 training launch on a GCE on-demand VM. Two modes:
#
# FSDP=1: real data-parallel via `accelerate launch --use_fsdp`, sharding
# raw parameters - including the fused MoE expert nn.Parameter tensors that
# make 92.9% of this model unquantizable by bitsandbytes. Requires >=4 GPUs
# (per-GPU sharded weight budget). The committed approach once smoke-tested.
#
# FSDP unset/0 (default): bf16 device_map=auto pipeline-parallel, single
# process, no DDP. Kept as fallback. Chosen originally after confirming:
# (1) QLoRA-4bit OOMs on a single 40GB A100 for this MoE arch (transformers
#     materializes full-precision before quantizing); (2) an offline
#     pre-quantized checkpoint didn't help either - the 256 expert-MLP
#     layers aren't standard nn.Linear so bitsandbytes' quantizer skips
#     them, checkpoint stayed ~63GB, barely smaller than bf16; (3) naive
#     pipeline-parallel gives the SAME throughput on 2 vs 8 GPUs (layers
#     execute sequentially either way) - capacity-only, no real speedup.
#
# Usage (FSDP):
#   VM_NAME=hawq-v3-train ZONE=us-central1-f GPU_COUNT=4 MACHINE_TYPE=a2-highgpu-4g \
#   FSDP=1 GRAD_ACCUM=4 \
#   ADAPTER_FULL=lancejames221b/HAWQ-SEC-RE-lora-v3 \
#   DATA_REPO=gs://hawq-training-us-central1/datasets/hawq-re-v3 \
#   CKPT_GCS=gs://hawq-training-us-central1/checkpoints \
#   MAXLEN=4096 SAVE_STEPS=250 EVAL_STEPS=250 \
#   GCS_KEY_FILE_LOCAL=/Volumes/SeXternal/hawq_v3/hawq-training-vm-key.json \
#       ./scripts/gce_cluster_train.sh create      # provision + launch
#   ./scripts/gce_cluster_train.sh status           # check training log
#   ./scripts/gce_cluster_train.sh teardown         # delete the VM
#
# GRAD_ACCUM * GPU_COUNT must equal 16 (the tuned global effective batch)
# under FSDP=1 - enforced below, not left to silently default wrong.
set -eo pipefail  # NOT -u: macOS's bash 3.2 (default /bin/bash) errors on
                  # "${empty_array[@]}" under nounset - required vars below
                  # already use ${VAR:?msg} for explicit enforcement instead.

VM_NAME="${VM_NAME:-hawq-v3-train}"
ZONE="${ZONE:-us-central1-a}"
PROJECT="${PROJECT:-ewitness-dev}"
MACHINE_TYPE="${MACHINE_TYPE:-a2-highgpu-2g}"
GPU_TYPE="${GPU_TYPE:-nvidia-tesla-a100}"
GPU_COUNT="${GPU_COUNT:-2}"
IMAGE_FAMILY="${IMAGE_FAMILY:-common-cu129-ubuntu-2204-nvidia-580}"
IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"
BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-300GB}"
PREEMPTIBLE="${PREEMPTIBLE:-0}"  # 0 = on-demand (single-process pipeline run has no DDP resume-across-ranks story)

ADAPTER_FULL="${ADAPTER_FULL:?set ADAPTER_FULL}"
DATA_REPO="${DATA_REPO:?set DATA_REPO}"
CKPT_GCS="${CKPT_GCS:-}"
GCS_KEY_FILE_LOCAL="${GCS_KEY_FILE_LOCAL:-}"
LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
MAXLEN="${MAXLEN:-3072}"
# Single process (no DDP here) - full effective batch stays 16 regardless
# of GPU_COUNT since pipeline-parallel splits ONE replica across GPUs,
# it does not create multiple replicas to divide grad_accum across. FSDP
# mode is real data-parallel (one replica shard per rank), so its effective
# batch = per_device(1) * GRAD_ACCUM * GPU_COUNT - GRAD_ACCUM must scale
# inversely with GPU_COUNT to hold the tuned LR/schedule constant. No
# implicit default for FSDP: forgetting to recompute it when GPU_COUNT
# changes was flagged as a known trap, so it's enforced below instead of
# silently defaulting wrong.
GRAD_ACCUM="${GRAD_ACCUM:-16}"
# Per-GPU memory budget for device_map=auto's placement planner. Left
# generous (35GiB of 40GiB) since with only 1 replica (not N as in DDP)
# there's ample headroom versus the ~65GB model across 2x40GB=80GB.
MAX_MEMORY_GIB="${MAX_MEMORY_GIB:-35}"

# FSDP mode: real data-parallel via accelerate launch --use_fsdp instead of
# the single-process device_map=auto pipeline above. Chosen because FSDP
# shards raw parameters - including the fused MoE expert nn.Parameter
# tensors that make 92.9% of this model unquantizable by bitsandbytes and
# that pipeline-parallel just accepts (capacity-only, same throughput on
# 2 vs 8 GPUs since layers execute sequentially either way).
FSDP="${FSDP:-0}"
TARGET_MLP="${TARGET_MLP:-0}"
SAVE_STEPS="${SAVE_STEPS:-250}"
EVAL_STEPS="${EVAL_STEPS:-250}"
MAX_STEPS="${MAX_STEPS:--1}"
SMOKE_LONGEST_N="${SMOKE_LONGEST_N:-0}"
# Selects the training entrypoint module: scripts.train_lora (SFT, default)
# or scripts.train_dpo (HAWQ v1.1 DPO pass). Substituted at both FSDP and
# non-FSDP launch sites below - never fork this script per-entrypoint.
TRAIN_MODULE="${TRAIN_MODULE:-scripts.train_lora}"
# GCS path to the combined DPO pairs JSONL (scripts/train_dpo.py reads this
# directly; DATA_REPO is still required below by the pre-existing gate but
# train_dpo.py never reads it - only train_lora.py's SFT path does).
DPO_DATA_GCS="${DPO_DATA_GCS:-}"
# DPO-only tunables, threaded through to both launch branches below.
# MAX_PROMPT_LEN defaults to 1024 (matching train_dpo.py's own Python-side
# default) rather than empty string - an exported empty string is NOT the
# same as "unset" to os.environ.get() on the VM, and int("") raises
# ValueError at import, killing every rank instantly.
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-1024}"
DPO_BETA="${DPO_BETA:-0.1}"
REF_LOGPROBS_PREPASS="${REF_LOGPROBS_PREPASS:-0}"
# Empty-string-safe (no int() parsing on the Python side) - threaded so a
# preemptible run's resume-from-GCS-checkpoint path actually fires; it was
# silently never wired to either launch branch before this.
RESUME="${RESUME:-}"
# FSDP backward-prefetch strategy: BACKWARD_PRE (default, faster - keeps
# the NEXT layer's unsharded params resident during current backward) vs
# BACKWARD_POST (holds one fewer layer's full params at peak, real memory
# margin at some speed cost). Exposed here since DPO's two-policy-graph
# retention pattern runs measurably closer to the memory ceiling than the
# single-graph SFT workload this default was tuned for.
FSDP_BACKWARD_PREFETCH="${FSDP_BACKWARD_PREFETCH:-BACKWARD_PRE}"
if [ "$FSDP" = "1" ] && [ $((GRAD_ACCUM * GPU_COUNT)) -ne 16 ]; then
  echo "[gce] ERROR: FSDP=1 requires GRAD_ACCUM * GPU_COUNT == 16 (global effective batch, matches the tuned LR/schedule). Got GRAD_ACCUM=$GRAD_ACCUM * GPU_COUNT=$GPU_COUNT = $((GRAD_ACCUM * GPU_COUNT))." >&2
  exit 1
fi

# The account's own gcloud reauth is stale in this environment and requires
# an interactive browser login; Application Default Credentials remain
# valid, so mint a fresh access token from ADC before every gcloud call
# instead (refreshed per-call since VM boot + bootstrap can span many
# minutes, longer than a single token's safety margin).
_tok() { gcloud auth application-default print-access-token; }

# gcloud compute ssh --tunnel-through-iap was confirmed unreliable here
# (repeated ERROR: [/usr/bin/ssh] exited with return code [255] while VM
# status/connectivity checks were all clean) - direct SSH via the VM's
# external IP worked reliably in the same session. Default to direct SSH;
# set USE_IAP=1 to force the tunnel (e.g. if the VM has no external IP).
_gcloud() {
  local i
  for i in 1 2 3 4 5; do
    if CLOUDSDK_AUTH_ACCESS_TOKEN="$(_tok)" gcloud "$@"; then return 0; fi
    echo "[gce] gcloud call failed (attempt $i/5), retrying in $((i*10))s..." >&2
    sleep $((i*10))
  done
  return 1
}
USE_IAP="${USE_IAP:-0}"
_ssh() {
  local i iap_flag=()
  if [ "$USE_IAP" = "1" ]; then iap_flag=(--tunnel-through-iap); fi
  for i in 1 2 3 4 5; do
    if CLOUDSDK_AUTH_ACCESS_TOKEN="$(_tok)" gcloud compute ssh "$VM_NAME" --zone="$ZONE" --project="$PROJECT" "${iap_flag[@]}" --command="$1"; then return 0; fi
    echo "[gce] ssh call failed (attempt $i/5), retrying in $((i*10))s..." >&2
    sleep $((i*10))
  done
  return 1
}
_bootstrap_and_launch() {
    echo "[gce] waiting for SSH..."
    for i in $(seq 1 30); do
      if _ssh "echo ssh_ready" 2>/dev/null | grep -q ssh_ready; then break; fi
      sleep 10
    done

    echo "[gce] bootstrap: python3-pip, /content, repo clone, vm_setup.py"
    _ssh "sudo apt-get update -qq && sudo apt-get install -y -qq python3-pip >/dev/null 2>&1; \
      sudo mkdir -p /content && sudo chown \$(whoami) /content && \
      (test -d /content/razorstrike && cd /content/razorstrike && git fetch origin && git reset --hard origin/main) || \
      git clone --depth 1 https://github.com/lancejames221b/razorstrike.git /content/razorstrike; \
      cd /content/razorstrike && python3 -u scripts/vm_setup.py 2>&1 | tail -40"

    if [ -z "${HF_TOKEN:-}" ]; then
      echo "[gce] ERROR: HF_TOKEN not set locally - cannot inject into the VM" >&2
      exit 1
    fi

    if [ -n "$GCS_KEY_FILE_LOCAL" ]; then
      echo "[gce] uploading GCS service-account key (never committed to the repo)"
      _scp_iap_flag=(); if [ "$USE_IAP" = "1" ]; then _scp_iap_flag=(--tunnel-through-iap); fi
      _gcloud compute scp "$GCS_KEY_FILE_LOCAL" "$VM_NAME:/content/gcs-key.json" \
        --zone="$ZONE" --project="$PROJECT" "${_scp_iap_flag[@]}"
    fi

    # Base model staging: prefer a GCS-staged copy (in-region rsync runs at
    # GB/s vs ~70min pulling ~70GB from HF Hub) and fall back to HF only if
    # none is staged. Either way, this happens ONCE before torchrun spins
    # up N ranks - N-1 GPUs would otherwise idle through hf_hub's download
    # lock serializing the fetch across all N processes.
    #
    # Run DETACHED on the VM (nohup + background, sentinel file for
    # completion) rather than as one long blocking foreground SSH call -
    # confirmed empirically this session: a ~70GB rsync tied to a single
    # SSH/IAP session dies on tunnel drop (`[/usr/bin/ssh] exited with
    # return code [255]`) and the whole bootstrap fails, even though the
    # VM itself stayed healthy throughout. Detaching means a dropped
    # tunnel only loses the POLL, not the transfer in progress.
    BASE_MODEL_GCS="${BASE_MODEL_GCS:-gs://hawq-training-us-central1/models/HAWQ-v1}"
    echo "[gce] staging base model (GCS-staged preferred: $BASE_MODEL_GCS)"
    _stage_local=$(mktemp)
    cat > "$_stage_local" <<STAGESCRIPT
#!/bin/bash
cd /content/razorstrike
rm -f /content/stage_done.txt
_gcs_ok=0
if gcloud storage ls "$BASE_MODEL_GCS" >/dev/null 2>&1; then
  mkdir -p /content/base_model
  gcloud storage rsync -r --no-ignore-symlinks "$BASE_MODEL_GCS" /content/base_model && _gcs_ok=1
fi
if [ "\$_gcs_ok" = 1 ]; then
  echo "BASE_MODEL_SOURCE=gcs:/content/base_model" > /content/stage_done.txt
else
  HF_HOME=/content/hf_home HF_TOKEN='$HF_TOKEN' python3 -u -c "
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id='lancejames221b/HAWQ-v1', token='$HF_TOKEN')
print('BASE_MODEL_SOURCE=hf:' + p)
" > /tmp/hf_stage_out.txt 2>&1 && grep -oE 'BASE_MODEL_SOURCE=hf:.*' /tmp/hf_stage_out.txt > /content/stage_done.txt
fi
# Must be the LAST line: bash reads scripts incrementally, so any line
# appended below this would be read from an already-shredded file.
shred -u /tmp/stage_model.sh 2>/dev/null || rm -f /tmp/stage_model.sh
STAGESCRIPT
    _scp_iap_flag=(); if [ "$USE_IAP" = "1" ]; then _scp_iap_flag=(--tunnel-through-iap); fi
    _gcloud compute scp "$_stage_local" "$VM_NAME:/tmp/stage_model.sh" --zone="$ZONE" --project="$PROJECT" "${_scp_iap_flag[@]}"
    rm -f "$_stage_local"
    # Run DETACHED on the VM (nohup + background, sentinel file for
    # completion) rather than as one long blocking foreground SSH call -
    # confirmed empirically this session: a ~70GB rsync tied to a single
    # SSH/IAP session dies on tunnel drop (`[/usr/bin/ssh] exited with
    # return code [255]`) and the whole bootstrap fails, even though the
    # VM itself stayed healthy throughout. Detaching means a dropped
    # tunnel only loses the POLL, not the transfer in progress.
    _ssh "nohup bash /tmp/stage_model.sh > /content/stage.log 2>&1 & disown; sleep 2; echo STAGE_LAUNCHED"
    echo "[gce] polling staging progress (detached on VM, survives SSH drops)..."
    _stage_result=""
    for i in $(seq 1 180); do
      _poll="$(_ssh "cat /content/stage_done.txt 2>/dev/null; echo ---; tail -3 /content/stage.log 2>/dev/null" 2>/dev/null)"
      if [ "$i" = "1" ]; then
        echo "[gce] first poll (checking for early script errors):"
        echo "$_poll"
      fi
      if echo "$_poll" | grep -q "BASE_MODEL_SOURCE="; then
        _stage_result="$_poll"
        break
      fi
      sleep 10
    done
    echo "$_stage_result" | tail -10
    _base_repo_resolved="$(echo "$_stage_result" | grep -oE 'BASE_MODEL_SOURCE=(gcs|hf):.*' | head -1 | sed -E 's/BASE_MODEL_SOURCE=(gcs|hf)://')"
    if [ -z "$_base_repo_resolved" ]; then
      echo "[gce] ERROR: could not resolve base model source (GCS stage and HF download both failed, or polling timed out)" >&2
      exit 1
    fi
    echo "[gce] base model resolved -> $_base_repo_resolved"

    if [ "$FSDP" = "1" ]; then
      echo "[gce] launching FSDP (real data-parallel, --use_fsdp across $GPU_COUNT GPUs, GRAD_ACCUM=$GRAD_ACCUM, MAXLEN=$MAXLEN)"
      _fsdp_local=$(mktemp)
      cat > "$_fsdp_local" <<FSDPLAUNCH
#!/bin/bash
set -e
cd /content/razorstrike
git fetch origin -q && git reset --hard origin/main -q
echo "[launch] at commit: \$(git log -1 --format='%H %s')"

# Robust teardown before relaunch: pkill alone + a fixed sleep is not
# enough to reclaim GPU memory from a process holding tens of GB - poll
# until the process is actually gone, then verify every GPU is near-0 MiB
# before handing off to FSDP. Bracket trick ([x]name) so the pattern
# doesn't match this very shell's own cmdline (confirmed self-match bug:
# an inline SSH --command containing the literal pattern text kills its
# own remote shell and gcloud reports exit 255, misread as connectivity
# flakiness rather than the self-inflicted kill it actually is). Must
# key off \$TRAIN_MODULE, not a hardcoded script name - a crashed
# train_dpo/accelerate rank otherwise survives every relaunch attempt.
_kill_pat="${TRAIN_MODULE##*.}"
pkill -f "[\${_kill_pat:0:1}]\${_kill_pat:1}" 2>/dev/null || true
pkill -f "[a]ccelerate" 2>/dev/null || true
for i in \$(seq 1 30); do
  pgrep -f "[\${_kill_pat:0:1}]\${_kill_pat:1}" >/dev/null || pgrep -f "[a]ccelerate" >/dev/null || break
  sleep 2
done
if pgrep -f "[\${_kill_pat:0:1}]\${_kill_pat:1}" >/dev/null || pgrep -f "[a]ccelerate" >/dev/null; then
  echo "[launch] WARNING: prior training process still alive after 60s, force-killing"
  pkill -9 -f "[\${_kill_pat:0:1}]\${_kill_pat:1}" 2>/dev/null || true
  pkill -9 -f "[a]ccelerate" 2>/dev/null || true
  sleep 3
fi
echo "[launch] GPU memory before FSDP launch:"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader

HF_HOME=/content/hf_home \\
HF_TOKEN='$HF_TOKEN' \\
BASE_REPO='$_base_repo_resolved' \\
DATA_REPO='$DATA_REPO' \\
ADAPTER_REPO='$ADAPTER_FULL' \\
CKPT_GCS='$CKPT_GCS' \\
OUT_DIR=/content/adapter \\
MAXLEN=$MAXLEN LORA_R=$LORA_R LORA_ALPHA=$LORA_ALPHA \\
TARGET_MLP=$TARGET_MLP SAVE_STEPS=$SAVE_STEPS EVAL_STEPS=$EVAL_STEPS MAX_STEPS=$MAX_STEPS FORCE_CAUSAL_LM=1 \\
QLORA_4BIT=0 GRAD_ACCUM=$GRAD_ACCUM \\
SMOKE_LONGEST_N=$SMOKE_LONGEST_N \\
GCS_KEY_FILE=/content/gcs-key.json GCS_PROJECT='$PROJECT' \\
DPO_DATA_GCS='$DPO_DATA_GCS' MAX_PROMPT_LEN='$MAX_PROMPT_LEN' REF_LOGPROBS_PREPASS='$REF_LOGPROBS_PREPASS' DPO_BETA='$DPO_BETA' \\
RESUME='$RESUME' \\
PYTHONUNBUFFERED=1 \\
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
nohup python3 -m accelerate.commands.launch \\
  --num_processes $GPU_COUNT --num_machines 1 --mixed_precision bf16 \\
  --use_fsdp \\
  --fsdp_version 1 \\
  --fsdp_sharding_strategy FULL_SHARD \\
  --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \\
  --fsdp_transformer_layer_cls_to_wrap Qwen3_5MoeDecoderLayer \\
  --fsdp_use_orig_params true \\
  --fsdp_cpu_ram_efficient_loading true \\
  --fsdp_sync_module_states true \\
  --fsdp_state_dict_type SHARDED_STATE_DICT \\
  --fsdp_backward_prefetch $FSDP_BACKWARD_PREFETCH \\
  -m $TRAIN_MODULE > /content/train.log 2>&1 &
disown
sleep 3
echo LAUNCHED
pgrep -af "[a]ccelerate" || echo NOT_RUNNING
FSDPLAUNCH
      _scp_iap_flag=(); if [ "$USE_IAP" = "1" ]; then _scp_iap_flag=(--tunnel-through-iap); fi
      _gcloud compute scp "$_fsdp_local" "$VM_NAME:/tmp/fsdp_launch.sh" --zone="$ZONE" --project="$PROJECT" "${_scp_iap_flag[@]}"
      rm -f "$_fsdp_local"
      _ssh "bash /tmp/fsdp_launch.sh"
    else
      echo "[gce] launching single-process bf16 pipeline (device_map=auto across $GPU_COUNT GPUs, GRAD_ACCUM=$GRAD_ACCUM)"
      launch_cmd="cd /content/razorstrike && pkill -f ${TRAIN_MODULE##*.} 2>/dev/null; sleep 2; \
HF_HOME=/content/hf_home \
HF_TOKEN='$HF_TOKEN' \
BASE_REPO='$_base_repo_resolved' \
DATA_REPO='$DATA_REPO' \
ADAPTER_REPO='$ADAPTER_FULL' \
OUT_DIR=/content/adapter \
MAXLEN=$MAXLEN LORA_R=$LORA_R LORA_ALPHA=$LORA_ALPHA \
TARGET_MLP=0 SAVE_STEPS=$SAVE_STEPS EVAL_STEPS=$EVAL_STEPS MAX_STEPS=${MAX_STEPS:--1} FORCE_CAUSAL_LM=1 \
QLORA_4BIT=0 DEVICE_MAP=auto MAX_MEMORY_GIB=$MAX_MEMORY_GIB GRAD_ACCUM=$GRAD_ACCUM \
CKPT_GCS='$CKPT_GCS' GCS_KEY_FILE=/content/gcs-key.json GCS_PROJECT='$PROJECT' \
DPO_DATA_GCS='$DPO_DATA_GCS' MAX_PROMPT_LEN='$MAX_PROMPT_LEN' REF_LOGPROBS_PREPASS='$REF_LOGPROBS_PREPASS' DPO_BETA='$DPO_BETA' RESUME='$RESUME' \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup python3 -u -m $TRAIN_MODULE > /content/train.log 2>&1 &
sleep 5
pgrep -af ${TRAIN_MODULE##*.} || echo NOT_RUNNING"
      _ssh "$launch_cmd"
    fi
    echo "[gce] launched. Check with: $0 status"
}

cmd="${1:-}"

case "$cmd" in
  create)
    echo "[gce] creating $VM_NAME ($GPU_COUNT x $GPU_TYPE, $MACHINE_TYPE, zone=$ZONE, preemptible=$PREEMPTIBLE)"
    extra_flags=()
    if [ "$PREEMPTIBLE" = "1" ]; then
      extra_flags+=(--preemptible --no-restart-on-failure)
    else
      extra_flags+=(--restart-on-failure)
    fi
    _gcloud compute instances create "$VM_NAME" \
      --project="$PROJECT" --zone="$ZONE" \
      --machine-type="$MACHINE_TYPE" \
      --accelerator="type=$GPU_TYPE,count=$GPU_COUNT" \
      --image-family="$IMAGE_FAMILY" --image-project="$IMAGE_PROJECT" \
      --boot-disk-size="$BOOT_DISK_SIZE" --boot-disk-type=pd-ssd \
      --maintenance-policy=TERMINATE \
      "${extra_flags[@]}"
    _bootstrap_and_launch
    ;;

  resume)
    # Skips VM creation - reuses an ALREADY-RUNNING instance (e.g. one that
    # was provisioned by a `create` call whose local process died/was
    # killed after gcloud accepted the request but before bootstrap ran;
    # gcloud's own async creation is NOT cancelled by killing the local
    # script, so re-running `create` would just fail on "already exists"
    # after burning through _gcloud's retry loop). Requires $VM_NAME to
    # already be RUNNING in $ZONE.
    echo "[gce] resuming bootstrap+launch against existing $VM_NAME (zone=$ZONE) - skipping VM creation"
    _status="$(_gcloud compute instances describe "$VM_NAME" --zone="$ZONE" --project="$PROJECT" --format='value(status)')"
    if [ "$_status" != "RUNNING" ]; then
      echo "[gce] ERROR: $VM_NAME is not RUNNING (status=$_status) - wait for it or use 'create' for a fresh VM" >&2
      exit 1
    fi
    _bootstrap_and_launch
    ;;

  status)
    _ssh "tail -50 /content/train.log 2>/dev/null; echo ---; pgrep -af '[t]rain_lora|[a]ccelerate' >/dev/null && echo STILL_RUNNING || echo PROCESS_EXITED"
    ;;

  teardown)
    echo "[gce] deleting $VM_NAME"
    _gcloud compute instances delete "$VM_NAME" --zone="$ZONE" --project="$PROJECT" --quiet
    ;;

  *)
    echo "usage: $0 {create|resume|status|teardown}" >&2
    exit 1
    ;;
esac
