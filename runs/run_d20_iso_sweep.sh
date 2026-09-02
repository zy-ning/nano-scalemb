#!/bin/bash

# d20 conditional-capacity sweep, matched on ACTIVE params across all arms and
# additionally on TOTAL params across the four non-dense arms.
#
#   arm            total     active   what it varies
#   dense          901.7M    901.7M   control (no conditional capacity)
#   moe           1820.0M    902.5M   64 experts, top-8, per-layer pools
#   mobius        1820.0M    909.9M   640 experts, top-8, ONE pool shared by all
#                                     10 MoE layers (same total, same reads)
#   engram        1828.4M    918.3M   n-gram hash memory, per-layer tables
#   engram_shared 1828.0M    918.3M   same, ONE table + one addressing scheme
#
# Active is the read view: conditional weight counts once per *reading* layer,
# so a shared pool and per-layer pools of equal width are equal on active and
# differ only on total. That is what makes "does sharing cost anything?" a fair
# question here.
#
# The two sharing arms are the same ablation applied to two different memories.
# Held fixed: depth 20, d_model 1280, seq 2048, SSSL windows, mHC 4 streams,
# and a shared 2B-token budget (NOT compute-optimal -- a compute-optimal d20 is
# ~9.5B tokens / ~13h per arm; this is a 2.7h/arm slice sized to separate the
# arms, and losses should be read as a mid-training snapshot, not a converged
# result).
#
# Sizing was solved with equal totals to within +0.46% (engram is quantized by
# slot_multiplier, which only takes integer values).

set -euo pipefail

REPO_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"
cd "$REPO_DIR"

RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/runs/logs/d20_iso-${RUN_SUFFIX}}"
NPROC="${NPROC:-4}"
DEPTH="${DEPTH:-20}"
SEQ="${SEQ:-2048}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-8}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-524288}"
NUM_ITERATIONS="${NUM_ITERATIONS:-3814}"     # ~2.0B tokens
MHC_STREAMS="${MHC_STREAMS:-4}"
# The venv's torchrun: the system one resolves to a python whose torch import
# dies on libucc.
TORCHRUN="${TORCHRUN:-$REPO_DIR/../nanochat-moe/.venv/bin/torchrun}"

export TORCHDYNAMO_CACHE_SIZE_LIMIT="${TORCHDYNAMO_CACHE_SIZE_LIMIT:-64}"
export PYTHONPATH="${PYTHONPATH:-$REPO_DIR}"

mkdir -p "$LOG_DIR"

run_arm () {
    local tag="$1"; shift
    local log_file="$LOG_DIR/${tag}.log"

    if grep -q "^Total training time" "$log_file" 2>/dev/null; then
        echo "Skipping $tag (already completed: $log_file)"
        return
    fi

    echo "================================================================"
    echo "Starting $tag  ($(date -Is))"
    echo "  depth=$DEPTH seq=$SEQ iters=$NUM_ITERATIONS batch=$TOTAL_BATCH_SIZE"
    echo "  extra: $*"
    echo "  log=$log_file"
    echo "================================================================"

    "$TORCHRUN" --standalone --nproc_per_node="$NPROC" -m scripts.base_train -- \
        --depth="$DEPTH" \
        --max-seq-len="$SEQ" \
        --device-batch-size="$DEVICE_BATCH_SIZE" \
        --total-batch-size="$TOTAL_BATCH_SIZE" \
        --num-iterations="$NUM_ITERATIONS" \
        --target-param-data-ratio=-1 \
        --window-pattern=SSSL \
        --mhc --mhc-num-streams="$MHC_STREAMS" \
        --eval-every=500 --eval-tokens=10485760 \
        --core-metric-every=-1 --sample-every=-1 --save-every=-1 \
        --model-tag="d20-${tag}-${RUN_SUFFIX}" \
        "$@" > "$log_file" 2>&1 || {
            echo "  !! $tag FAILED (exit $?), see $log_file"
            tail -20 "$log_file"
            return
        }
    echo "  $tag done: $(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$log_file" | tail -1)"
}

# --- dense control ---------------------------------------------------------
run_arm dense

# --- MoE: 64 experts top-8, private pool per MoE layer ---------------------
run_arm moe \
    --moe --moe-backend=scattermoe \
    --moe-experts=64 --moe-top-k=8 --moe-d-ff-expert=640 --moe-every=2 \
    --moe-balancer=loss_free --moe-log-every=100

# --- Mobius: ONE pool of 640 experts shared across all 10 MoE layers -------
# 10x the experts in a single bank reproduces the per-layer arm's total; top_k
# and d_ff are unchanged so the per-token read is the same.
run_arm mobius \
    --moe --moe-backend=scattermoe \
    --moe-experts=640 --moe-top-k=8 --moe-d-ff-expert=640 --moe-every=2 \
    --moe-share-blocks=1 --moe-balancer=loss_free --moe-log-every=100

# --- Engram: per-layer memory tables ---------------------------------------
run_arm engram \
    --engram --engram-layers=2,6 --engram-memory-dim=1280 \
    --engram-slot-multiplier=15 --engram-mhc-num-streams="$MHC_STREAMS"

# --- Engram: ONE shared table + one hash addressing scheme -----------------
# Double slot_multiplier so a single table matches the two-table total.
run_arm engram_shared \
    --engram --engram-layers=2,6 --engram-memory-dim=1280 \
    --engram-slot-multiplier=30 --engram-share-memory \
    --engram-mhc-num-streams="$MHC_STREAMS"

echo
echo "d20 iso sweep complete. Logs: $LOG_DIR"
echo
printf "%-16s %-12s %-12s %s\n" arm min_bpb final_bpb tok/sec
for tag in dense moe mobius engram engram_shared; do
    f="$LOG_DIR/${tag}.log"
    [ -f "$f" ] || continue
    printf "%-16s %-12s %-12s %s\n" "$tag" \
        "$(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$f" | tail -1)" \
        "$(grep -oP 'Validation bpb: \K[0-9.]+' "$f" | tail -1)" \
        "$(grep -oP 'tok/sec: \K[0-9,]+' "$f" | tail -1)"
done
