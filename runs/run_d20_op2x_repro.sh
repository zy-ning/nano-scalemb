#!/bin/bash

# Reproduce op2x on CURRENT code across seeds.
#
# QUESTION. The historical op2x (slot30, merge_frac0.5, merge_at0.5, lsh, seed0) scored
# 0.764795 on Aug-28 code. The same config re-run as the timing sweep's at05 control
# scored 0.765826 on current code (n=1) -- ~1e-3 worse. The ONLY config-block diff is
# two new fields added since (merge_source="value", semantic_prior_path="") from the
# semantic-prior refactor of merge_rows/_cluster_head. Old code is uncommitted+gone, so
# we cannot check it out. Instead: is current code actually worse, or was at05 a hot draw?
#
# DESIGN. Run the EXACT op2x config on current code, seeds {0,1,2}. seed0 == a same-code
# replicate of at05 (0.765826). If any seed lands near 0.764795, the gap is run variance,
# not a code regression. If all three sit ~0.7658+, current code has a real ~1e-3
# regression on the value-merge path and the refactor is the suspect.
#
# READ (pre-registered):
#   min over seeds <= ~0.7650 (near historical op2x)  => variance; no regression.
#   all seeds >= ~0.7655 (clustered near at05)        => real current-code regression.
#   spread ~1e-3 straddling both                       => merge path is seed-sensitive;
#                                                         single-run anchors untrustworthy.
# GATE: every arm must fire the merge and print 5,685,536 effective rows.

set -euo pipefail
REPO_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"
cd "$REPO_DIR"

RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/runs/logs/d20_op2x_repro-${RUN_SUFFIX}}"
NPROC="${NPROC:-4}"
NUM_ITERATIONS="${NUM_ITERATIONS:-3814}"
MHC_STREAMS="${MHC_STREAMS:-4}"
SLOT="${SLOT:-30}"
MERGE_FRAC="${MERGE_FRAC:-0.5}"
MERGE_AT="${MERGE_AT:-0.5}"
MERGE_METRIC="${MERGE_METRIC:-lsh}"
PYTHON="${PYTHON:-$REPO_DIR/../nanochat-moe/.venv/bin/python}"
TORCHRUN="${TORCHRUN:-$REPO_DIR/../nanochat-moe/.venv/bin/torchrun}"

# space-separated seeds.
ARMS="${ARMS:-0 1 2}"

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/mnt/3fs/dots-pretrain/ningzhiyuan/local_exps/nsmb_out}"
export TORCHDYNAMO_CACHE_SIZE_LIMIT="${TORCHDYNAMO_CACHE_SIZE_LIMIT:-64}"
export PYTHONPATH="${PYTHONPATH:-$REPO_DIR}"

mkdir -p "$LOG_DIR"

run_arm () {
    local seed="$1"
    local tag="s${seed}"
    local log_file="$LOG_DIR/${tag}.log"
    if grep -q "^Total training time" "$log_file" 2>/dev/null; then
        echo "Skipping $tag (already completed)"; return
    fi
    echo "================================================================"
    echo "Starting $tag  ($(date -Is))  slot=$SLOT frac=$MERGE_FRAC at=$MERGE_AT seed=$seed"
    echo "================================================================"
    "$TORCHRUN" --standalone --nproc_per_node="$NPROC" -m scripts.base_train -- \
        --depth=20 --max-seq-len=2048 --device-batch-size=8 --total-batch-size=524288 \
        --num-iterations="$NUM_ITERATIONS" --target-param-data-ratio=-1 \
        --window-pattern=SSSL --mhc --mhc-num-streams="$MHC_STREAMS" \
        --engram --engram-layers=2,6 --engram-memory-dim=1280 \
        --engram-mhc-num-streams="$MHC_STREAMS" --engram-share-memory \
        --engram-slot-multiplier="$SLOT" \
        --engram-merge-frac="$MERGE_FRAC" --engram-merge-at-frac="$MERGE_AT" \
        --engram-merge-metric="$MERGE_METRIC" \
        --engram-seed="$seed" \
        --eval-every=500 --eval-tokens=10485760 \
        --core-metric-every=-1 --sample-every=-1 --save-every=-1 \
        --model-tag="d20-op2xrepro-${tag}-${RUN_SUFFIX}" \
        > "$log_file" 2>&1 || {
            echo "  !! $tag FAILED (exit $?)"; tail -20 "$log_file"; return
        }
    echo "  $tag done: $(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$log_file" | tail -1)"
}

for s in $ARMS; do run_arm "$s"; done

echo
echo "op2x reproduction complete. Logs: $LOG_DIR"
printf "%-8s %-12s %s\n" seed min_bpb eff_rows
for s in $ARMS; do
    tag="s${s}"; f="$LOG_DIR/${tag}.log"; [ -f "$f" ] || continue
    printf "%-8s %-12s %s\n" "$s" \
        "$(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$f" | tail -1)" \
        "$(grep -oaE '\-> [0-9,]+ effective rows' "$f" | tail -1 | grep -oE '[0-9,]+' | head -1)"
done
echo
echo "anchors: historical op2x (old code) = 0.764795 ; at05 (current code, seed0) = 0.765826"
