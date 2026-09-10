#!/bin/bash
#
# TN-gram N=5 DEPTH (the paper's n-gram order): does deeper cross-order sharing
# help? The paper uses N=5 (orders {2,3,4,5}); our runs so far used N=3 ({2,3}).
# Theorem 4.1's shared-factor benefit is strongest with MORE orders to share
# across, so N=3 may simply be too shallow to show the paper's effect. This is
# the last untested paper divergence after readout (global vs per-head) came back
# null and the full rank/budget sweep came back negative.
#
# HEAD GEOMETRY. N=5 -> orders {2,3,4,5}, n_orders=4. memory_dim=1280 with K=8
# gives num_heads=4*8=32, head_dim=40 (vs N=3's 16 heads x 80). Shared A-params
# = 5*V*K*R = 947,440*R.
#
# ARMS (per-head readout, the stronger config from the R80 comparison):
#   R=48  -> 45.5M A-params, ISO-PARAM with the best prior N=3 CP (per-head R80,
#            0.770665) -> clean DEPTH isolation: same param budget, N=3->N=5.
#   R=80  -> 75.8M A-params (7.5% of model), deeper AND bigger.
# Read: if N=5 R48 << N=3 R80 (0.770665), depth was the missing lever and the
# paper's shared-factor win needs the extra orders. If ~same/worse, N=3 was not
# the handicap and the CP line is fully closed.
# Anchors: N=3 per-head R80 0.770665 ; native slot15 0.76486 (floor 0.764396).

set -euo pipefail
REPO_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"
cd "$REPO_DIR"

RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/runs/logs/d20_tngram_n5-${RUN_SUFFIX}}"
NPROC="${NPROC:-4}"
NUM_ITERATIONS="${NUM_ITERATIONS:-3814}"
MHC_STREAMS="${MHC_STREAMS:-4}"
PYTHON="${PYTHON:-$REPO_DIR/../nanochat-moe/.venv/bin/python}"
TORCHRUN="${TORCHRUN:-$REPO_DIR/../nanochat-moe/.venv/bin/torchrun}"

ARMS="${ARMS:-48 80}"
SEEDS="${SEEDS:-0}"

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/mnt/3fs/dots-pretrain/ningzhiyuan/local_exps/nsmb_out}"
export TORCHDYNAMO_CACHE_SIZE_LIMIT="${TORCHDYNAMO_CACHE_SIZE_LIMIT:-64}"
export PYTHONPATH="${PYTHONPATH:-$REPO_DIR}"

mkdir -p "$LOG_DIR"

run_arm () {
    local rank="$1" seed="$2"
    local tag="tngramN5_R${rank}_s${seed}"
    local log_file="$LOG_DIR/${tag}.log"
    if grep -q "^Total training time" "$log_file" 2>/dev/null; then
        echo "Skipping $tag (already completed)"; return
    fi
    echo "================================================================"
    echo "Starting $tag  ($(date -Is))  cp_rank=$rank seed=$seed N=5"
    echo "================================================================"
    "$TORCHRUN" --standalone --nproc_per_node="$NPROC" -m scripts.base_train -- \
        --depth=20 --max-seq-len=2048 --device-batch-size=8 --total-batch-size=524288 \
        --num-iterations="$NUM_ITERATIONS" --target-param-data-ratio=-1 \
        --window-pattern=SSSL --mhc --mhc-num-streams="$MHC_STREAMS" \
        --engram --engram-layers=2,6 --engram-memory-dim=1280 \
        --engram-max-ngram-size=5 \
        --engram-mhc-num-streams="$MHC_STREAMS" --engram-share-memory \
        --engram-value-table=tngram --engram-cp-rank="$rank" \
        --engram-seed="$seed" \
        --eval-every=500 --eval-tokens=10485760 \
        --core-metric-every=-1 --sample-every=-1 --save-every=-1 \
        --model-tag="d20-tngramN5-${tag}-${RUN_SUFFIX}" \
        > "$log_file" 2>&1 || {
            echo "  !! $tag FAILED (exit $?)"; tail -20 "$log_file"; return
        }
    echo "  $tag done: $(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$log_file" | tail -1)"
}

for rank in $ARMS; do
    for s in $SEEDS; do run_arm "$rank" "$s"; done
done

echo
echo "TN-gram N=5 sweep complete. Logs: $LOG_DIR"
printf "%-24s %s\n" arm min_bpb
for rank in $ARMS; do
    for s in $SEEDS; do
        tag="tngramN5_R${rank}_s${s}"; f="$LOG_DIR/${tag}.log"; [ -f "$f" ] || continue
        printf "%-24s %s\n" "$tag" \
            "$(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$f" | tail -1)"
    done
done
echo
echo "anchors: N=3 per-head R80 0.770665 ; native slot15 0.76486 (floor 0.764396)"
