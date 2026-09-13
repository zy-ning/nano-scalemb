#!/bin/bash

# Token floor at n=3 -- properly-calibrated baseline for the row-merge comparison.
#
# WHY. The merge line's "wins" were all judged against a SINGLE floor number
# (shared_slot15 = 0.764396, n=1). But run_d20_op2x_repro showed the merge path has
# a ~6e-4 sd / ~1e-3 spread even for bit-identical configs. Comparing a noisy n=1
# floor to a noisy n=1 merge arm is meaningless at the 1e-3 scale. Run the floor at
# 3 seeds so BOTH sides carry the same error bar; then "does merge beat the floor?"
# becomes a real question (compare cluster means +- sd, not point vs point).
#
# CONFIG = shared_slot15 floor recipe exactly: slot15, NO MERGE, layers 2,6,
# memory_dim 1280, share_memory, address tokens, mhc 4. Only --engram-seed varies.
#
# READ: floor mean vs op2x-repro mean (0.76544, sd~6e-4). If floor mean is >~1sd
# below op2x mean, plain slot15 training genuinely beats the over-provision->merge
# schedule and the whole merge line is a wash. If they overlap within sd, merge
# neither helps nor hurts at matched effective capacity. GATE: NO merge line should
# print (floor arms never merge); each arm ~5.686M native rows at slot15.

set -euo pipefail
REPO_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"
cd "$REPO_DIR"

RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/runs/logs/d20_floor_n3-${RUN_SUFFIX}}"
NPROC="${NPROC:-4}"
NUM_ITERATIONS="${NUM_ITERATIONS:-3814}"
MHC_STREAMS="${MHC_STREAMS:-4}"
SLOT="${SLOT:-15}"
PYTHON="${PYTHON:-$REPO_DIR/../nanochat-moe/.venv/bin/python}"
TORCHRUN="${TORCHRUN:-$REPO_DIR/../nanochat-moe/.venv/bin/torchrun}"

ARMS="${ARMS:-0 1 2}"   # seeds

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
    echo "Starting $tag  ($(date -Is))  slot=$SLOT NO-MERGE seed=$seed"
    echo "================================================================"
    "$TORCHRUN" --standalone --nproc_per_node="$NPROC" -m scripts.base_train -- \
        --depth=20 --max-seq-len=2048 --device-batch-size=8 --total-batch-size=524288 \
        --num-iterations="$NUM_ITERATIONS" --target-param-data-ratio=-1 \
        --window-pattern=SSSL --mhc --mhc-num-streams="$MHC_STREAMS" \
        --engram --engram-layers=2,6 --engram-memory-dim=1280 \
        --engram-mhc-num-streams="$MHC_STREAMS" --engram-share-memory \
        --engram-slot-multiplier="$SLOT" \
        --engram-seed="$seed" \
        --eval-every=500 --eval-tokens=10485760 \
        --core-metric-every=-1 --sample-every=-1 --save-every=-1 \
        --model-tag="d20-floorn3-${tag}-${RUN_SUFFIX}" \
        > "$log_file" 2>&1 || {
            echo "  !! $tag FAILED (exit $?)"; tail -20 "$log_file"; return
        }
    echo "  $tag done: $(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$log_file" | tail -1)"
}

for s in $ARMS; do run_arm "$s"; done

echo
echo "Token floor n=3 complete. Logs: $LOG_DIR"
printf "%-8s %s\n" seed min_bpb
for s in $ARMS; do
    tag="s${s}"; f="$LOG_DIR/${tag}.log"; [ -f "$f" ] || continue
    printf "%-8s %s\n" "$s" "$(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$f" | tail -1)"
done
echo
echo "anchors: historical floor (n=1) = 0.764396 ; op2x-repro current-code mean = 0.76544 (sd~6e-4)"
