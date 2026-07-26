#!/usr/bin/env bash
# HAWQ-SEC-RE v3 - bf16 device_map=auto pipeline-parallel training launch
# on a GCE on-demand VM (single process, no DDP). Chosen after confirming:
# (1) QLoRA-4bit OOMs on a single 40GB A100 for this MoE arch (transformers
#     materializes full-precision before quantizing); (2) an offline
#     pre-quantized checkpoint didn't help either - the 256 expert-MLP
#     layers aren't standard nn.Linear so bitsandbytes' quantizer skips
#     them, checkpoint stayed ~63GB, barely smaller than bf16; (3) naive
#     pipeline-parallel gives the SAME throughput on 2 vs 8 GPUs (layers
#     execute sequentially either way) - 2x40GB A100 (80GB, fits the 65GB
#     model with headroom) is the same speed as 8x for 4x less cost.
#     User's explicit choice over the (untested, riskier) DDP-hybrid path.
#
# Usage:
#   VM_NAME=hawq-v3-train ZONE=us-central1-a GPU_COUNT=8 \
#   ADAPTER_FULL=lancejames221b/HAWQ-SEC-RE-lora-v3 \
#   DATA_REPO=gs://hawq-training-us-central1/datasets/hawq-re-v3 \
#   CKPT_GCS=gs://hawq-training-us-central1/checkpoints \
#   GCS_KEY_FILE_LOCAL=/Volumes/SeXternal/hawq_v3/hawq-training-vm-key.json \
#       ./scripts/gce_cluster_train.sh create      # provision + launch
#   ./scripts/gce_cluster_train.sh status           # check training log
#   ./scripts/gce_cluster_train.sh teardown         # delete the VM
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
# it does not create multiple replicas to divide grad_accum across.
GRAD_ACCUM="${GRAD_ACCUM:-16}"
# Per-GPU memory budget for device_map=auto's placement planner. Left
# generous (35GiB of 40GiB) since with only 1 replica (not N as in DDP)
# there's ample headroom versus the ~65GB model across 2x40GB=80GB.
MAX_MEMORY_GIB="${MAX_MEMORY_GIB:-35}"

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

cmd="${1:-}"

case "$cmd" in
  create)
    echo "[gce] creating $VM_NAME ($GPU_COUNT x $GPU_TYPE, $MACHINE_TYPE, zone=$ZONE, preemptible=$PREEMPTIBLE)"
    extra_flags=()
    if [ "$PREEMPTIBLE" = "1" ]; then extra_flags+=(--preemptible); fi
    _gcloud compute instances create "$VM_NAME" \
      --project="$PROJECT" --zone="$ZONE" \
      --machine-type="$MACHINE_TYPE" \
      --accelerator="type=$GPU_TYPE,count=$GPU_COUNT" \
      --image-family="$IMAGE_FAMILY" --image-project="$IMAGE_PROJECT" \
      --boot-disk-size="$BOOT_DISK_SIZE" --boot-disk-type=pd-ssd \
      --maintenance-policy=TERMINATE --restart-on-failure \
      "${extra_flags[@]}"

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
    BASE_MODEL_GCS="${BASE_MODEL_GCS:-gs://hawq-training-us-central1/models/HAWQ-v1}"
    echo "[gce] staging base model (GCS-staged preferred: $BASE_MODEL_GCS)"
    _stage_out="$(_ssh "cd /content/razorstrike && _gcs_ok=0
if gcloud storage ls '$BASE_MODEL_GCS' >/dev/null 2>&1; then
  mkdir -p /content/base_model
  gcloud storage rsync -r --no-ignore-symlinks '$BASE_MODEL_GCS' /content/base_model && _gcs_ok=1
fi
if [ \"\$_gcs_ok\" = 1 ]; then
  echo 'BASE_MODEL_SOURCE=gcs:/content/base_model'
else
  HF_HOME=/content/hf_home HF_TOKEN='$HF_TOKEN' python3 -u -c \"
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id='lancejames221b/HAWQ-v1', token='$HF_TOKEN')
print('BASE_MODEL_SOURCE=hf:' + p)
\"
fi")"
    echo "$_stage_out" | tail -20
    _base_repo_resolved="$(echo "$_stage_out" | grep -oE 'BASE_MODEL_SOURCE=(gcs|hf):.*' | tail -1 | sed -E 's/BASE_MODEL_SOURCE=(gcs|hf)://')"
    if [ -z "$_base_repo_resolved" ]; then
      echo "[gce] ERROR: could not resolve base model source (GCS stage and HF download both failed)" >&2
      exit 1
    fi
    echo "[gce] base model resolved -> $_base_repo_resolved"

    echo "[gce] launching single-process bf16 pipeline (device_map=auto across $GPU_COUNT GPUs, GRAD_ACCUM=$GRAD_ACCUM)"
    launch_cmd="cd /content/razorstrike && pkill -f train_lora 2>/dev/null; sleep 2; \
HF_HOME=/content/hf_home \
HF_TOKEN='$HF_TOKEN' \
BASE_REPO='$_base_repo_resolved' \
DATA_REPO='$DATA_REPO' \
ADAPTER_REPO='$ADAPTER_FULL' \
OUT_DIR=/content/adapter \
MAXLEN=$MAXLEN LORA_R=$LORA_R LORA_ALPHA=$LORA_ALPHA \
TARGET_MLP=0 SAVE_STEPS=250 EVAL_STEPS=250 MAX_STEPS=${MAX_STEPS:--1} FORCE_CAUSAL_LM=1 \
QLORA_4BIT=0 DEVICE_MAP=auto MAX_MEMORY_GIB=$MAX_MEMORY_GIB GRAD_ACCUM=$GRAD_ACCUM \
CKPT_GCS='$CKPT_GCS' GCS_KEY_FILE=/content/gcs-key.json GCS_PROJECT='$PROJECT' \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup python3 -u -m scripts.train_lora > /content/train.log 2>&1 &
sleep 5
pgrep -af train_lora || echo NOT_RUNNING"
    _ssh "$launch_cmd"
    echo "[gce] launched. Check with: $0 status"
    ;;

  status)
    _ssh "tail -50 /content/train.log 2>/dev/null; echo ---; pgrep -af train_lora >/dev/null && echo STILL_RUNNING || echo PROCESS_EXITED"
    ;;

  teardown)
    echo "[gce] deleting $VM_NAME"
    _gcloud compute instances delete "$VM_NAME" --zone="$ZONE" --project="$PROJECT" --quiet
    ;;

  *)
    echo "usage: $0 {create|status|teardown}" >&2
    exit 1
    ;;
esac
