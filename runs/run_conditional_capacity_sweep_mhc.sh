#!/bin/bash

# Conditional-capacity sweep on the mHC backbone.
#
# The question: extra conditional capacity helps -- but does it matter HOW the
# capacity is addressed, and does each layer need its own copy of it?
#
#   baseline       dense MLP, no conditional capacity
#   engram         n-gram hash memory, per-layer tables (the existing arm)
#   engram-shared  same, but ONE memory table + one hash addressing scheme for
#                  every Engram layer
#   moe            sparse MoE, per-layer routed expert pools
#   mobius         same experts, ONE routed pool shared across depth, plus a
#                  per-layer gated dense expert (Intern-S2-Mobius style)
#
# The two sharing arms are the same ablation applied to two different memories,
# so engram-vs-engram-shared and moe-vs-mobius answer the same question about
# each mechanism.
#
# The MoE arms use the G/A/X ladder at A=1, i.e. active expert FLOPs == one dense
# FFN, so every arm is compared at matched *active* params. base_train prints
# total/active/ratio per run; read those before reading the losses.

set -euo pipefail

REPO_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"
cd "$REPO_DIR"

RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/runs/logs/conditional_capacity_mhc-${RUN_SUFFIX}}"
REPORT_DIR="${REPORT_DIR:-$REPO_DIR/runs/reports/conditional_capacity_mhc-${RUN_SUFFIX}}"
DYNAMO_CACHE_LIMIT="${DYNAMO_CACHE_LIMIT:-64}"
ENGRAM_MHC_NUM_STREAMS="${ENGRAM_MHC_NUM_STREAMS:-4}"
MHC_NUM_STREAMS="${MHC_NUM_STREAMS:-$ENGRAM_MHC_NUM_STREAMS}"
MHC_SINKHORN_ITERS="${MHC_SINKHORN_ITERS:-20}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-8}"
BASE_EVAL_DEVICE_BATCH_SIZE="${BASE_EVAL_DEVICE_BATCH_SIZE:-8}"
SFT_DEVICE_BATCH_SIZE="${SFT_DEVICE_BATCH_SIZE:-8}"
CHAT_EVAL_ARGS="${CHAT_EVAL_ARGS:---dist-timeout-minutes 120}"

# Shared placement: MoE and Engram both land on every other layer, so the two
# mechanisms get comparable depth coverage.
ENGRAM_LAYERS="${ENGRAM_LAYERS:-2,12,18}"
MOE_EVERY="${MOE_EVERY:-2}"
MOE_GRANULARITY="${MOE_GRANULARITY:-8}"
MOE_EXPANSION="${MOE_EXPANSION:-8}"
MOE_ACTIVE_MULT="${MOE_ACTIVE_MULT:-1}"
MOE_BALANCER="${MOE_BALANCER:-loss_free}"
MOE_BACKEND="${MOE_BACKEND:-scattermoe}"
# Mobius pairs a shared routed pool with a private per-layer dense expert.
MOBIUS_SHARE_BLOCKS="${MOBIUS_SHARE_BLOCKS:-1}"
MOBIUS_SHARED_D_FF="${MOBIUS_SHARED_D_FF:-512}"

mkdir -p "$LOG_DIR"
mkdir -p "$REPORT_DIR"

# args: model_tag engram_enabled engram_share_memory moe_enabled moe_share_blocks moe_shared_d_ff
run_case() {
    local model_tag="$1"
    local engram_enabled="$2"
    local engram_share_memory="$3"
    local moe_enabled="$4"
    local moe_share_blocks="$5"
    local moe_shared_d_ff="$6"
    local log_file="$LOG_DIR/${model_tag}.log"
    local report_archive_dir="$REPORT_DIR/${model_tag}"

    if [ -f "$report_archive_dir/report.md" ]; then
        echo "Skipping $model_tag because archived report already exists at $report_archive_dir/report.md"
        return
    fi

    echo "================================================================"
    echo "Starting $model_tag"
    echo "  engram_enabled=$engram_enabled"
    echo "  engram_share_memory=$engram_share_memory"
    echo "  engram_layers=$ENGRAM_LAYERS"
    echo "  moe_enabled=$moe_enabled"
    echo "  moe_share_blocks=$moe_share_blocks"
    echo "  moe_shared_d_ff=$moe_shared_d_ff"
    echo "  moe_every=$MOE_EVERY G=$MOE_GRANULARITY X=$MOE_EXPANSION A=$MOE_ACTIVE_MULT"
    echo "  moe_balancer=$MOE_BALANCER moe_backend=$MOE_BACKEND"
    echo "  mhc_enabled=1 mhc_num_streams=$MHC_NUM_STREAMS"
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
    ENGRAM_LAYERS="$ENGRAM_LAYERS" \
    ENGRAM_MEMORY_DIM="1280" \
    ENGRAM_EMBEDDING_LR_MULT="5.0" \
    ENGRAM_SEED="48" \
    ENGRAM_ABLATION_MODE="none" \
    ENGRAM_SHARE_MEMORY="$engram_share_memory" \
    ENGRAM_MHC_NUM_STREAMS="$ENGRAM_MHC_NUM_STREAMS" \
    MOE_ENABLED="$moe_enabled" \
    MOE_EVERY="$MOE_EVERY" \
    MOE_SHARE_BLOCKS="$moe_share_blocks" \
    MOE_SHARED_D_FF="$moe_shared_d_ff" \
    MOE_GRANULARITY="$MOE_GRANULARITY" \
    MOE_EXPANSION="$MOE_EXPANSION" \
    MOE_ACTIVE_MULT="$MOE_ACTIVE_MULT" \
    MOE_BALANCER="$MOE_BALANCER" \
    MOE_BACKEND="$MOE_BACKEND" \
    CHAT_EVAL_ARGS="$CHAT_EVAL_ARGS" \
    BASE_TRAIN_EXTRA_ARGS="--fp8 --eval-every=-1 --core-metric-every=-1 --sample-every=-1 --save-every=-1 --window-pattern=SSSL" \
    bash runs/run_train.sh
}

#         model_tag                                  engram  eshare  moe  mshare              mshared_dff
run_case "nano-scalemb-d24-mhc-baseline-${RUN_SUFFIX}"      "0" "0" "0" "0"                     "0"
run_case "nano-scalemb-d24-engram-${RUN_SUFFIX}"            "1" "0" "0" "0"                     "0"
run_case "nano-scalemb-d24-engram-shared-${RUN_SUFFIX}"     "1" "1" "0" "0"                     "0"
run_case "nano-scalemb-d24-moe-${RUN_SUFFIX}"               "0" "0" "1" "0"                     "0"
run_case "nano-scalemb-d24-mobius-${RUN_SUFFIX}"            "0" "0" "1" "$MOBIUS_SHARE_BLOCKS"  "$MOBIUS_SHARED_D_FF"

echo "Conditional-capacity sweep complete. Reports: $REPORT_DIR"
