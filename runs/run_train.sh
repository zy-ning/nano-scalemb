#!/bin/bash

# This script is PART 2 of an offline nano-engram workflow.
# It is designed to be run on a GPU machine WITHOUT internet access.
#
# Prerequisites:
# 1. The repository has already been copied to this machine.
# 2. A populated Python environment is already available locally.
#    By default this script expects .venv/ in the repo root.
# 3. The local data directory has already been staged under ./data, including:
#    - data/base_data_climbmix/
#    - data/tokenizer/
#    - data/identity_conversations.jsonl
#    - data/eval_bundle/
#
# 1) Example launch (simplest):
# bash runs/run_train.sh
# 2) Example launch in a screen session:
# screen -L -Logfile runs/run_train.log -S nano-engram-offline bash runs/run_train.sh
# 3) Example launch with wandb naming (still local-only unless your environment says otherwise):
# WANDB_RUN=nano-engram-offline bash runs/run_train.sh

set -euo pipefail

REPO_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"

# Default intermediate artifacts directory is in current directory/data
export OMP_NUM_THREADS=1
export NANO_SCALEMB_BASE_DIR="${NANO_SCALEMB_BASE_DIR:-${NANOCHAT_BASE_DIR:-$REPO_DIR/data}}"
export NANOCHAT_BASE_DIR="$NANO_SCALEMB_BASE_DIR"
mkdir -p "$NANO_SCALEMB_BASE_DIR"

# Keep caches local to the staged data directory if downstream libraries consult them.
export HF_HOME="${HF_HOME:-$NANO_SCALEMB_BASE_DIR/hf_home}"

LOG_FILE="${LOG_FILE:-$REPO_DIR/run_train.log}"
exec > >(tee -a "$LOG_FILE") 2>&1

REPORT_RESET="${REPORT_RESET:-1}"
REPORT_ARCHIVE_DIR="${REPORT_ARCHIVE_DIR:-}"

# NCCL debugging can be enabled manually if needed.
# export NCCL_DEBUG=INFO
# export NCCL_ASYNC_ERROR_HANDLING=1

# -----------------------------------------------------------------------------
# Offline prerequisites

cd "$REPO_DIR"

if [ ! -f ".venv/bin/activate" ]; then
    echo "WARNING: .venv/bin/activate was not found in $REPO_DIR"
    echo "Using the currently active Python environment."
fi

REQUIRED_PATHS=(
    "$NANO_SCALEMB_BASE_DIR/base_data_climbmix"
    "$NANO_SCALEMB_BASE_DIR/tokenizer/tokenizer.pkl"
    "$NANO_SCALEMB_BASE_DIR/tokenizer/token_bytes.pt"
    "$NANO_SCALEMB_BASE_DIR/identity_conversations.jsonl"
    "$NANO_SCALEMB_BASE_DIR/eval_bundle"
)

for path in "${REQUIRED_PATHS[@]}"; do
    if [ ! -e "$path" ]; then
        echo "ERROR: required offline asset missing: $path"
        exit 1
    fi
done

# -----------------------------------------------------------------------------
# Python environment setup

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# -----------------------------------------------------------------------------
# Run configuration

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
WANDB_RUN="${WANDB_RUN:-dummy}"

# Use a unique tag by default so offline retries do not silently reuse or overwrite
# the existing nano-engram-d24 checkpoints.
MODEL_TAG="${MODEL_TAG:-nano-engram-d24-offline-2-6-L}"

DEPTH="${DEPTH:-24}"
TARGET_PARAM_DATA_RATIO="${TARGET_PARAM_DATA_RATIO:-9.5}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"
BASE_EVAL_DEVICE_BATCH_SIZE="${BASE_EVAL_DEVICE_BATCH_SIZE:-16}"
SFT_DEVICE_BATCH_SIZE="${SFT_DEVICE_BATCH_SIZE:-16}"

# Recommended first retry based on prior evidence:
# - lighter two-layer Engram placement
# - force full-context attention to avoid SDPA + SSSL fallback issues
ENGRAM_LAYERS="${ENGRAM_LAYERS:-2,6}"
ENGRAM_MEMORY_DIM="${ENGRAM_MEMORY_DIM:-1280}"
ENGRAM_MAX_NGRAM_SIZE="${ENGRAM_MAX_NGRAM_SIZE:-3}"
ENGRAM_HEADS_PER_NGRAM="${ENGRAM_HEADS_PER_NGRAM:-8}"
ENGRAM_SLOT_MULTIPLIER="${ENGRAM_SLOT_MULTIPLIER:-18}"
ENGRAM_KERNEL_SIZE="${ENGRAM_KERNEL_SIZE:-4}"
ENGRAM_EMBEDDING_LR_MULT="${ENGRAM_EMBEDDING_LR_MULT:-2.0}"
ENGRAM_PAD_ID="${ENGRAM_PAD_ID:-0}"
ENGRAM_SEED="${ENGRAM_SEED:-42}"
ENGRAM_ABLATION_MODE="${ENGRAM_ABLATION_MODE:-none}"
ENGRAM_MHC_NUM_STREAMS="${ENGRAM_MHC_NUM_STREAMS:-4}"
ENGRAM_ENABLED="${ENGRAM_ENABLED:-1}"
MHC_ENABLED="${MHC_ENABLED:-0}"
MHC_NUM_STREAMS="${MHC_NUM_STREAMS:-4}"
MHC_SINKHORN_ITERS="${MHC_SINKHORN_ITERS:-20}"

BASE_TRAIN_EXTRA_ARGS="${BASE_TRAIN_EXTRA_ARGS:---fp8 --eval-every=-1 --core-metric-every=-1 --sample-every=-1 --save-every=-1 --window-pattern=L}"
BASE_EVAL_ARGS="${BASE_EVAL_ARGS:---eval=core,bpb,sample}"
SFT_EXTRA_ARGS="${SFT_EXTRA_ARGS:---eval-every=-1 --chatcore-every=-1}"
CHAT_EVAL_ARGS="${CHAT_EVAL_ARGS:-}"

# -----------------------------------------------------------------------------
# wandb setup

if command -v wandb >/dev/null 2>&1; then
    wandb offline || true
fi

if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

export WANDB_RUN

if [ "$REPORT_RESET" = "1" ]; then
    echo "Resetting report artifacts..."
    python -m nano_scalemb.report reset
fi

archive_provenance() {
    local provenance_dir="$1"
    mkdir -p "$provenance_dir"

    {
        echo "timestamp=$(date -Is)"
        echo "repo_dir=$REPO_DIR"
        echo "model_tag=$MODEL_TAG"
        echo "mhc_enabled=$MHC_ENABLED"
        echo "mhc_num_streams=$MHC_NUM_STREAMS"
        echo "mhc_sinkhorn_iters=$MHC_SINKHORN_ITERS"
        echo "engram_enabled=$ENGRAM_ENABLED"
        echo "engram_ablation_mode=$ENGRAM_ABLATION_MODE"
    } > "$provenance_dir/run-env.txt"

    git rev-parse HEAD > "$provenance_dir/git-head.txt" 2>&1 || true
    git status --short > "$provenance_dir/git-status-short.txt" 2>&1 || true
    git diff --stat > "$provenance_dir/git-diff-stat.txt" 2>&1 || true
    git diff > "$provenance_dir/git-diff.patch" 2>&1 || true

    {
        for path in AGENTS.md nano_scalemb/*.py scripts/*.py runs/*.sh tests/*.py docs/*.md; do
            [ -f "$path" ] || continue
            sha256sum "$path"
        done
    } > "$provenance_dir/source-sha256.txt"
}

if [ -n "$REPORT_ARCHIVE_DIR" ]; then
    archive_provenance "$REPORT_ARCHIVE_DIR/provenance"
fi

# -----------------------------------------------------------------------------
# Base model (pretraining)

echo "Starting offline nano-engram base pretraining..."
BASE_TRAIN_CMD=(
    torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_train --
    --depth="$DEPTH"
    --target-param-data-ratio="$TARGET_PARAM_DATA_RATIO"
    --device-batch-size="$DEVICE_BATCH_SIZE"
    --run="$WANDB_RUN"
    --model-tag="$MODEL_TAG"
)

if [ "$ENGRAM_ENABLED" = "1" ]; then
    BASE_TRAIN_CMD+=(
        --engram
        --engram-layers="$ENGRAM_LAYERS"
        --engram-memory-dim="$ENGRAM_MEMORY_DIM"
        --engram-max-ngram-size="$ENGRAM_MAX_NGRAM_SIZE"
        --engram-heads-per-ngram="$ENGRAM_HEADS_PER_NGRAM"
        --engram-slot-multiplier="$ENGRAM_SLOT_MULTIPLIER"
        --engram-kernel-size="$ENGRAM_KERNEL_SIZE"
        --engram-embedding-lr-mult="$ENGRAM_EMBEDDING_LR_MULT"
        --engram-pad-id="$ENGRAM_PAD_ID"
        --engram-seed="$ENGRAM_SEED"
        --engram-ablation-mode="$ENGRAM_ABLATION_MODE"
        --engram-mhc-num-streams="$ENGRAM_MHC_NUM_STREAMS"
    )
fi

if [ "$MHC_ENABLED" = "1" ]; then
    BASE_TRAIN_CMD+=(
        --mhc
        --mhc-num-streams="$MHC_NUM_STREAMS"
        --mhc-sinkhorn-iters="$MHC_SINKHORN_ITERS"
    )
fi

# shellcheck disable=SC2206
EXTRA_ARGS=( $BASE_TRAIN_EXTRA_ARGS )
BASE_TRAIN_CMD+=("${EXTRA_ARGS[@]}")
"${BASE_TRAIN_CMD[@]}"

echo "Starting base model evaluation..."
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_eval -- \
    --model-tag="$MODEL_TAG" \
    --device-batch-size="$BASE_EVAL_DEVICE_BATCH_SIZE" \
    $BASE_EVAL_ARGS

# -----------------------------------------------------------------------------
# SFT

echo "Starting SFT..."
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.chat_sft -- \
    --model-tag="$MODEL_TAG" \
    --device-batch-size="$SFT_DEVICE_BATCH_SIZE" \
    --run="$WANDB_RUN" \
    $SFT_EXTRA_ARGS

echo "Starting SFT evaluation..."
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.chat_eval -- \
    -i sft \
    -g "$MODEL_TAG" \
    $CHAT_EVAL_ARGS

# -----------------------------------------------------------------------------
# Generate report

echo "Generating final report..."
python -m nano_scalemb.report generate

if [ -n "$REPORT_ARCHIVE_DIR" ]; then
    mkdir -p "$REPORT_ARCHIVE_DIR"
    cp "$REPO_DIR/report.md" "$REPORT_ARCHIVE_DIR/report.md"
    cp -r "$NANO_SCALEMB_BASE_DIR/report/." "$REPORT_ARCHIVE_DIR/"
fi

echo "Offline nano-engram training complete. Check report.md and $LOG_FILE for results."

# To avoid container being killed due to inactivity after the script finishes, we can run a simple GPU stress test.
# python /inspire/hdd/project/exploration-topic/heziwei-25044/projects_zyning/gpu_stress.py
