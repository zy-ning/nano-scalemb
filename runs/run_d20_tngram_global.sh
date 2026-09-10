#!/bin/bash
#
# TN-gram GLOBAL READOUT (paper-faithful, arXiv 2606.08347 Eq. 8): does removing
# the per-head [16,80] output partition close the gap to native Engram?
#
# THE IMPL DIFF THIS TESTS. Our original port forced the CP read into the native
# [B,T,16,80] layout so value_proj/gate/short-conv stayed byte-identical. That
# imposes 16 ISOLATED rank-R bottlenecks: each 80-dim head is reconstructed from
# only its own R coordinates through its own F_k in (80,R). The paper never does
# this -- it CONCATENATES all (N-1)*R CP coordinates and maps the whole vector to
# d_model with a SINGLE learned M_V = F^T W_V. Every output dim sees every CP
# coord. --engram-tngram-global-readout drops the explicit F and lets the Engram
# value_proj be that global M_V (read width n_orders*K*R instead of memory_dim).
#
# Prior per-head result: R=80 -> 0.770665, R=400 -> 0.767101 (native slot15
# 0.76486). The R80 full-rank probe showed rank was NOT the per-head binding
# constraint, which points at the PARTITION itself.
#
# ARMS (iso-A-param to the per-head runs, so this isolates readout, not budget):
#   R=80  global: A-params 45.5M ~ per-head R80 (0.770665); value_proj 1280x1280
#                 (identical size to native), so ONLY the partition differs.
#   R=400 global: A-params 228M = 19.4% of model (paper's regime), faithful readout.
# Read: if global R80 << per-head R80 (0.770665), the partition was the liability.
# Anchors: native slot15 ~0.76486 (token floor 0.764396) ; native slot30 ~0.76393.

set -euo pipefail
REPO_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"
cd "$REPO_DIR"

RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/runs/logs/d20_tngram_global-${RUN_SUFFIX}}"
NPROC="${NPROC:-4}"
NUM_ITERATIONS="${NUM_ITERATIONS:-3814}"
MHC_STREAMS="${MHC_STREAMS:-4}"
PYTHON="${PYTHON:-$REPO_DIR/../nanochat-moe/.venv/bin/python}"
TORCHRUN="${TORCHRUN:-$REPO_DIR/../nanochat-moe/.venv/bin/torchrun}"

ARMS="${ARMS:-80 400}"
SEEDS="${SEEDS:-0}"

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/mnt/3fs/dots-pretrain/ningzhiyuan/local_exps/nsmb_out}"
export TORCHDYNAMO_CACHE_SIZE_LIMIT="${TORCHDYNAMO_CACHE_SIZE_LIMIT:-64}"
export PYTHONPATH="${PYTHONPATH:-$REPO_DIR}"

mkdir -p "$LOG_DIR"

run_arm () {
    local rank="$1" seed="$2"
    local tag="tngramG_R${rank}_s${seed}"
    local log_file="$LOG_DIR/${tag}.log"
    if grep -q "^Total training time" "$log_file" 2>/dev/null; then
        echo "Skipping $tag (already completed)"; return
    fi
    echo "================================================================"
    echo "Starting $tag  ($(date -Is))  cp_rank=$rank seed=$seed global-readout"
    echo "================================================================"
    "$TORCHRUN" --standalone --nproc_per_node="$NPROC" -m scripts.base_train -- \
        --depth=20 --max-seq-len=2048 --device-batch-size=8 --total-batch-size=524288 \
        --num-iterations="$NUM_ITERATIONS" --target-param-data-ratio=-1 \
        --window-pattern=SSSL --mhc --mhc-num-streams="$MHC_STREAMS" \
        --engram --engram-layers=2,6 --engram-memory-dim=1280 \
        --engram-mhc-num-streams="$MHC_STREAMS" --engram-share-memory \
        --engram-value-table=tngram --engram-cp-rank="$rank" \
        --engram-tngram-global-readout \
        --engram-seed="$seed" \
        --eval-every=500 --eval-tokens=10485760 \
        --core-metric-every=-1 --sample-every=-1 --save-every=-1 \
        --model-tag="d20-tngramG-${tag}-${RUN_SUFFIX}" \
        > "$log_file" 2>&1 || {
            echo "  !! $tag FAILED (exit $?)"; tail -20 "$log_file"; return
        }
    echo "  $tag done: $(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$log_file" | tail -1)"
}

for rank in $ARMS; do
    for s in $SEEDS; do run_arm "$rank" "$s"; done
done

echo
echo "TN-gram GLOBAL-readout sweep complete. Logs: $LOG_DIR"
printf "%-24s %s\n" arm min_bpb
for rank in $ARMS; do
    for s in $SEEDS; do
        tag="tngramG_R${rank}_s${s}"; f="$LOG_DIR/${tag}.log"; [ -f "$f" ] || continue
        printf "%-24s %s\n" "$tag" \
            "$(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$f" | tail -1)"
    done
done
echo
echo "vs per-head: R80 0.770665, R400 0.767101 ; native slot15 0.76486 (floor 0.764396)"
