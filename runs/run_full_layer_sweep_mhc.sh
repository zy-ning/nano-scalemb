#!/bin/bash

# Full nano-scalemb Engram+mHC layer sweep.
# This follows the nanochat layer-sweep style but uses the fixed learned MHC head.

set -euo pipefail

REPO_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"
cd "$REPO_DIR"

RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/runs/logs/full_layer_sweep_mhc-$RUN_SUFFIX}"
REPORT_DIR="${REPORT_DIR:-$REPO_DIR/runs/reports/full_layer_sweep_mhc-$RUN_SUFFIX}"
DYNAMO_CACHE_LIMIT="${DYNAMO_CACHE_LIMIT:-64}"
ENGRAM_MHC_NUM_STREAMS="${ENGRAM_MHC_NUM_STREAMS:-4}"
MHC_NUM_STREAMS="${MHC_NUM_STREAMS:-$ENGRAM_MHC_NUM_STREAMS}"
ENGRAM_MHC_SINKHORN_ITERS="${ENGRAM_MHC_SINKHORN_ITERS:-20}"
MHC_SINKHORN_ITERS="${MHC_SINKHORN_ITERS:-$ENGRAM_MHC_SINKHORN_ITERS}"
MHC_HEAD_MODE="${MHC_HEAD_MODE:-learned}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-8}"
BASE_EVAL_DEVICE_BATCH_SIZE="${BASE_EVAL_DEVICE_BATCH_SIZE:-8}"
SFT_DEVICE_BATCH_SIZE="${SFT_DEVICE_BATCH_SIZE:-8}"
CHAT_EVAL_ARGS="${CHAT_EVAL_ARGS:---dist-timeout-minutes 120}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

mkdir -p "$LOG_DIR"
mkdir -p "$REPORT_DIR"

run_case() {
    local layer_label="$1"
    local engram_layers="$2"
    local engram_memory_dim="$3"
    local engram_seed="$4"
    local model_tag="nano-scalemb-d24-layer-${layer_label}-mhc-${RUN_SUFFIX}"
    local log_file="$LOG_DIR/${model_tag}.log"
    local report_archive_dir="$REPORT_DIR/${model_tag}"

    if [ -f "$report_archive_dir/report.md" ]; then
        echo "Skipping $model_tag because archived report already exists at $report_archive_dir/report.md"
        return
    fi

    echo "================================================================"
    echo "Starting $model_tag"
    echo "  engram_layers=$engram_layers"
    echo "  engram_memory_dim=$engram_memory_dim"
    echo "  engram_seed=$engram_seed"
    echo "  engram_fusion_mode=mhc"
    echo "  mhc_enabled=1"
    echo "  mhc_head_mode=$MHC_HEAD_MODE"
    echo "  nproc_per_node=$NPROC_PER_NODE"
    echo "  log_file=$log_file"
    echo "  report_archive_dir=$report_archive_dir"
    echo "================================================================"

    LOG_FILE="$log_file" \
    REPORT_ARCHIVE_DIR="$report_archive_dir" \
    REPORT_RESET="1" \
    TORCHDYNAMO_CACHE_SIZE_LIMIT="$DYNAMO_CACHE_LIMIT" \
    NANOCHAT_DYNAMO_RECOMPILE_LIMIT="64" \
    NANOCHAT_DYNAMO_ACCUMULATED_RECOMPILE_LIMIT="1024" \
    NPROC_PER_NODE="$NPROC_PER_NODE" \
    MODEL_TAG="$model_tag" \
    DEVICE_BATCH_SIZE="$DEVICE_BATCH_SIZE" \
    BASE_EVAL_DEVICE_BATCH_SIZE="$BASE_EVAL_DEVICE_BATCH_SIZE" \
    SFT_DEVICE_BATCH_SIZE="$SFT_DEVICE_BATCH_SIZE" \
    MHC_ENABLED="1" \
    MHC_NUM_STREAMS="$MHC_NUM_STREAMS" \
    MHC_SINKHORN_ITERS="$MHC_SINKHORN_ITERS" \
    MHC_HEAD_MODE="$MHC_HEAD_MODE" \
    ENGRAM_ENABLED="1" \
    ENGRAM_LAYERS="$engram_layers" \
    ENGRAM_MEMORY_DIM="$engram_memory_dim" \
    ENGRAM_EMBEDDING_LR_MULT="5.0" \
    ENGRAM_SEED="$engram_seed" \
    ENGRAM_ABLATION_MODE="none" \
    ENGRAM_FUSION_MODE="mhc" \
    ENGRAM_MHC_NUM_STREAMS="$ENGRAM_MHC_NUM_STREAMS" \
    ENGRAM_MHC_SINKHORN_ITERS="$ENGRAM_MHC_SINKHORN_ITERS" \
    CHAT_EVAL_ARGS="$CHAT_EVAL_ARGS" \
    BASE_TRAIN_EXTRA_ARGS="--fp8 --eval-every=-1 --core-metric-every=-1 --sample-every=-1 --save-every=-1 --window-pattern=SSSL" \
    bash runs/run_train.sh
}

# Start-layer resolution sweep around the previously strong 12,17 anchor.
run_case "21217" "2,12,17" "1280" "56"
run_case "31217" "3,12,17" "1280" "56"
run_case "81217" "8,12,17" "1280" "56"
run_case "91217" "9,12,17" "1280" "56"

echo "Full layer sweep complete. Reports: $REPORT_DIR"
