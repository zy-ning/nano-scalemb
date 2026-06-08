#!/bin/bash

# Engram payload ablation sweep at layers 2,12,18 on the faithful Engram+mHC
# backbone (the only Engram+mHC path; see docs/engram_mhc_fusion_paths.md).
# Variants: mHC-only baseline, real Engram, randomized table, uniform table.

set -euo pipefail

REPO_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"
cd "$REPO_DIR"

RUN_SUFFIX="${RUN_SUFFIX:-}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/runs/logs/engram_ablation_sweep_21218_mhc${RUN_SUFFIX}}"
REPORT_DIR="${REPORT_DIR:-$REPO_DIR/runs/reports/engram_ablation_sweep_21218_mhc${RUN_SUFFIX}}"
DYNAMO_CACHE_LIMIT="${DYNAMO_CACHE_LIMIT:-64}"
ENGRAM_MHC_NUM_STREAMS="${ENGRAM_MHC_NUM_STREAMS:-4}"
MHC_NUM_STREAMS="${MHC_NUM_STREAMS:-$ENGRAM_MHC_NUM_STREAMS}"
MHC_SINKHORN_ITERS="${MHC_SINKHORN_ITERS:-20}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-8}"
BASE_EVAL_DEVICE_BATCH_SIZE="${BASE_EVAL_DEVICE_BATCH_SIZE:-8}"
SFT_DEVICE_BATCH_SIZE="${SFT_DEVICE_BATCH_SIZE:-8}"
CHAT_EVAL_ARGS="${CHAT_EVAL_ARGS:---dist-timeout-minutes 120}"

mkdir -p "$LOG_DIR"
mkdir -p "$REPORT_DIR"

run_case() {
    local model_tag="$1"
    local engram_enabled="$2"
    local ablation_mode="$3"
    local log_file="$LOG_DIR/${model_tag}.log"
    local report_archive_dir="$REPORT_DIR/${model_tag}"

    if [ -f "$report_archive_dir/report.md" ]; then
        echo "Skipping $model_tag because archived report already exists at $report_archive_dir/report.md"
        return
    fi

    echo "================================================================"
    echo "Starting $model_tag"
    echo "  engram_enabled=$engram_enabled"
    echo "  ablation_mode=$ablation_mode"
    echo "  engram_layers=2,12,18"
    echo "  mhc_enabled=1 (faithful Engram+mHC)"
    echo "  mhc_num_streams=$MHC_NUM_STREAMS"
    echo "  mhc_sinkhorn_iters=$MHC_SINKHORN_ITERS"
    echo "  device_batch_size=$DEVICE_BATCH_SIZE"
    echo "  log_file=$log_file"
    echo "  report_archive_dir=$report_archive_dir"
    echo "================================================================"

    LOG_FILE="$log_file" \
    REPORT_ARCHIVE_DIR="$report_archive_dir" \
    REPORT_RESET="1" \
    TORCHDYNAMO_CACHE_SIZE_LIMIT="$DYNAMO_CACHE_LIMIT" \
    NANOCHAT_DYNAMO_RECOMPILE_LIMIT="64" \
    NANOCHAT_DYNAMO_ACCUMULATED_RECOMPILE_LIMIT="1024" \
    MODEL_TAG="$model_tag" \
    DEVICE_BATCH_SIZE="$DEVICE_BATCH_SIZE" \
    BASE_EVAL_DEVICE_BATCH_SIZE="$BASE_EVAL_DEVICE_BATCH_SIZE" \
    SFT_DEVICE_BATCH_SIZE="$SFT_DEVICE_BATCH_SIZE" \
    MHC_ENABLED="1" \
    MHC_NUM_STREAMS="$MHC_NUM_STREAMS" \
    MHC_SINKHORN_ITERS="$MHC_SINKHORN_ITERS" \
    ENGRAM_ENABLED="$engram_enabled" \
    ENGRAM_LAYERS="2,12,18" \
    ENGRAM_MEMORY_DIM="1280" \
    ENGRAM_EMBEDDING_LR_MULT="5.0" \
    ENGRAM_SEED="48" \
    ENGRAM_ABLATION_MODE="$ablation_mode" \
    ENGRAM_MHC_NUM_STREAMS="$ENGRAM_MHC_NUM_STREAMS" \
    CHAT_EVAL_ARGS="$CHAT_EVAL_ARGS" \
    BASE_TRAIN_EXTRA_ARGS="--fp8 --eval-every=-1 --core-metric-every=-1 --sample-every=-1 --save-every=-1 --window-pattern=SSSL" \
    bash runs/run_train.sh
}

run_case "nano-engram-d24-mhc-baseline-21218" "0" "none"
run_case "nano-engram-d24-real-21218-mhc" "1" "none"
run_case "nano-engram-d24-randomize-21218-mhc" "1" "randomize"
run_case "nano-engram-d24-uniform-21218-mhc" "1" "uniform"

echo "Engram ablation sweep complete. Reports: $REPORT_DIR"
