#!/bin/bash

# Experiment 5: capacity / FLOP-matched control for the Engram+mHC ablation family.
#
# Trains one variant identical to the layers-2,12,18 ablation sweep EXCEPT the
# Engram payload comes from a learned projection of the hidden state instead of an
# n-gram lookup (ablation_mode=mlp). It keeps the full branch machinery
# (value_proj, gate, short conv, mHC routing) but has NO memory table and NO
# n-gram addressing. Completes the decomposition:
#   real       = content + n-gram addressing + branch capacity
#   randomize  = random n-gram fingerprint + branch capacity   (no content)
#   uniform    = constant payload + branch capacity            (no content/address)
#   mlp (this) = branch capacity only                          (no lookup at all)
#
# Pipeline (via run_train.sh): base pretrain -> base_eval -> SFT -> chat_eval.
# Step-matched to the other ablations (same target-param-data ratio = 9.5).
# Intended for the engram1 H200 box, GPUs 4-7.

set -euo pipefail

REPO_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"
cd "$REPO_DIR"

# --- environment: sibling nanochat venv + its staged data/checkpoints ---
export NANO_SCALEMB_BASE_DIR="${NANO_SCALEMB_BASE_DIR:-$REPO_DIR/../nanochat/data}"
export NANOCHAT_BASE_DIR="$NANO_SCALEMB_BASE_DIR"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
# shellcheck disable=SC1091
source "$REPO_DIR/../nanochat/.venv/bin/activate"

MODEL_TAG="${MODEL_TAG:-nano-engram-d24-mlpctrl-21218-mhc}"
RUN_SUFFIX="${RUN_SUFFIX:-}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/runs/logs/engram_mlp_control_21218_mhc${RUN_SUFFIX}}"
REPORT_DIR="${REPORT_DIR:-$REPO_DIR/runs/reports/engram_mlp_control_21218_mhc${RUN_SUFFIX}}"
mkdir -p "$LOG_DIR" "$REPORT_DIR"

# --- knobs matched to runs/run_engram_ablation_sweep_21218_mhc.sh::run_case ---
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"          # GPUs 4-7
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-8}"
BASE_EVAL_DEVICE_BATCH_SIZE="${BASE_EVAL_DEVICE_BATCH_SIZE:-8}"
SFT_DEVICE_BATCH_SIZE="${SFT_DEVICE_BATCH_SIZE:-8}"
DYNAMO_CACHE_LIMIT="${DYNAMO_CACHE_LIMIT:-64}"

# Optional smoke run: set DRY_ITERS=20 to validate the GPU path end-to-end fast.
DRY_ITERS="${DRY_ITERS:-0}"
BASE_TRAIN_BASE_ARGS="--fp8 --eval-every=-1 --core-metric-every=-1 --sample-every=-1 --save-every=-1 --window-pattern=SSSL"
if [ "$DRY_ITERS" != "0" ]; then
  BASE_TRAIN_BASE_ARGS="$BASE_TRAIN_BASE_ARGS --num-iterations=$DRY_ITERS"
fi

LOG_FILE="$LOG_DIR/${MODEL_TAG}.log" \
REPORT_ARCHIVE_DIR="$REPORT_DIR/${MODEL_TAG}" \
REPORT_RESET="1" \
NPROC_PER_NODE="$NPROC_PER_NODE" \
TORCHDYNAMO_CACHE_SIZE_LIMIT="$DYNAMO_CACHE_LIMIT" \
NANOCHAT_DYNAMO_RECOMPILE_LIMIT="64" \
NANOCHAT_DYNAMO_ACCUMULATED_RECOMPILE_LIMIT="1024" \
MODEL_TAG="$MODEL_TAG" \
DEVICE_BATCH_SIZE="$DEVICE_BATCH_SIZE" \
BASE_EVAL_DEVICE_BATCH_SIZE="$BASE_EVAL_DEVICE_BATCH_SIZE" \
SFT_DEVICE_BATCH_SIZE="$SFT_DEVICE_BATCH_SIZE" \
MHC_ENABLED="1" \
MHC_NUM_STREAMS="4" \
MHC_SINKHORN_ITERS="20" \
ENGRAM_ENABLED="1" \
ENGRAM_LAYERS="2,12,18" \
ENGRAM_MEMORY_DIM="1280" \
ENGRAM_EMBEDDING_LR_MULT="5.0" \
ENGRAM_SEED="48" \
ENGRAM_ABLATION_MODE="mlp" \
ENGRAM_MHC_NUM_STREAMS="4" \
CHAT_EVAL_ARGS="--dist-timeout-minutes 120" \
BASE_TRAIN_EXTRA_ARGS="$BASE_TRAIN_BASE_ARGS" \
bash runs/run_train.sh

echo "DONE mlp-control: $MODEL_TAG (report in $REPORT_DIR)"
