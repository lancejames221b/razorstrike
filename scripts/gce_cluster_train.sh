#!/usr/bin/env bash
# HAWQ-SEC-RE v3 - multi-GPU DDP training launch on a GCE spot/on-demand VM.
# Replaces the Colab autodrive.py path for this run per explicit steering:
# "cluster this baby up" / "faster is better" - real data-parallel speedup
# (not Colab's single-GPU G4) via torchrun + QLoRA-4bit DDP across N A100s
# (see train_lora.py's LOCAL_RANK-aware device_map + GRAD_ACCUM scaling).
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
set -euo pipefail

VM_NAME="${VM_NAME:-hawq-v3-train}"
ZONE="${ZONE:-us-central1-a}"
PROJECT="${PROJECT:-ewitness-dev}"
MACHINE_TYPE="${MACHINE_TYPE:-a2-highgpu-8g}"
GPU_TYPE="${GPU_TYPE:-nvidia-tesla-a100}"
GPU_COUNT="${GPU_COUNT:-8}"
IMAGE_FAMILY="${IMAGE_FAMILY:-common-cu129-ubuntu-2204-nvidia-580}"
IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"
BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-400GB}"
PREEMPTIBLE="${PREEMPTIBLE:-0}"  # 0 = on-demand (reliability > cost for an 8-GPU DDP run)

ADAPTER_FULL="${ADAPTER_FULL:?set ADAPTER_FULL}"
DATA_REPO="${DATA_REPO:?set DATA_REPO}"
CKPT_GCS="${CKPT_GCS:-}"
GCS_KEY_FILE_LOCAL="${GCS_KEY_FILE_LOCAL:-}"
LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
MAXLEN="${MAXLEN:-3072}"
# Global effective batch stays 16 regardless of GPU_COUNT (per-GPU grad_accum
# scaled down so DDP parallelism cuts wall-clock, not the training dynamics).
GRAD_ACCUM="$(( 16 / GPU_COUNT > 0 ? 16 / GPU_COUNT : 1 ))"

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

    echo "[gce] launching torchrun (nproc_per_node=$GPU_COUNT, GRAD_ACCUM=$GRAD_ACCUM)"
    launch_cmd="cd /content/razorstrike && pkill -f train_lora 2>/dev/null; sleep 2; \
HF_HOME=/content/hf_home \
HF_TOKEN='$HF_TOKEN' \
BASE_REPO='$_base_repo_resolved' \
DATA_REPO='$DATA_REPO' \
ADAPTER_REPO='$ADAPTER_FULL' \
OUT_DIR=/content/adapter \
MAXLEN=$MAXLEN LORA_R=$LORA_R LORA_ALPHA=$LORA_ALPHA \
TARGET_MLP=0 SAVE_STEPS=250 EVAL_STEPS=250 MAX_STEPS=${MAX_STEPS:--1} FORCE_CAUSAL_LM=1 \
QLORA_4BIT=1 GRAD_ACCUM=$GRAD_ACCUM \
CKPT_GCS='$CKPT_GCS' GCS_KEY_FILE=/content/gcs-key.json GCS_PROJECT='$PROJECT' \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup python3 -m torch.distributed.run --nproc_per_node=$GPU_COUNT -m scripts.train_lora > /content/train.log 2>&1 &
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
