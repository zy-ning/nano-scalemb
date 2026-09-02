#!/bin/bash

# d20 idea: OVER-PROVISION RATIO sweep -- "pretrain a large (low-collision) table,
# then value-merge back to the reference size." How large should the pretrain table be?
#
# The over-provision+merge schedule already wins at 2x (op_10to5 0.764795 vs op_floor
# 0.765322, token, n=1). The user's hypothesis: a BIGGER pretrain table has FEWER hash
# collisions, so each n-gram's learned value is cleaner, so the value-similarity
# partition the merge clusters on is more genuinely semantic -> a better collapse.
# Test by pushing the over-provision RATIO while holding FINAL effective capacity fixed
# at the slot=15 floor (~5.69M rows). Every arm ends the same size; only the pretrain
# table size (and thus pretrain-phase collision rate) differs.
#
#   arm        slot  over-prov  merge_frac  final eff. rows   pretrain collisions
#   op1x       15    1x         none        5.69M             (= floor, done elsewhere)
#   op2x       30    2x         0.5         ~5.69M            (= op_10to5, done elsewhere)
#   op4x       60    4x         0.75        ~5.69M            fewer
#   op8x       120   8x         0.875       ~5.69M            fewest
#
# merge_frac per arm = 1 - (floor_rows / slot_rows). slot_rows ~ linear in slot but the
# per-head PRIME sizes are not EXACTLY 2x, so the script computes nothing -- it uses the
# nominal frac and the run prints the achieved effective-row count; we compare arms on
# the ACHIEVED effective rows (all should land within ~0.1% of 5.69M).
#
# READ: if bpb keeps dropping as over-provision rises, "pretrain large then merge" is a
# real collision-limited knob and bigger is better. If it flattens after 2x, the 2x
# prior is already clean enough. If it RISES at 8x, over-provision has a sweet spot
# (too-sparse a pretrain table learns weaker values before the merge).
#
# NOT PARAM-ISO across arms (bigger slot = more params + optimizer state during the
# pretrain phase; dead rows stay allocated post-merge). Iso on FINAL effective capacity
# + inference FLOPs only. Memory: slot=120 adds ~11GB/gpu opt+param over the base --
# safe on 186GB GB200. Within-sweep (same seed, same code) comparison is clean.
#
# GATE: op4x/op8x must FIRE the merge and print an effective-row count within ~0.5% of
# 5.69M, else the frac is mis-set for that slot and the arm is not capacity-matched.

set -euo pipefail

REPO_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"
cd "$REPO_DIR"

RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/runs/logs/d20_op_ratio-${RUN_SUFFIX}}"
NPROC="${NPROC:-4}"
NUM_ITERATIONS="${NUM_ITERATIONS:-3814}"
MHC_STREAMS="${MHC_STREAMS:-4}"
SEED="${SEED:-0}"
MERGE_AT="${MERGE_AT:-0.5}"
MERGE_METRIC="${MERGE_METRIC:-lsh}"
PYTHON="${PYTHON:-$REPO_DIR/../nanochat-moe/.venv/bin/python}"
TORCHRUN="${TORCHRUN:-$REPO_DIR/../nanochat-moe/.venv/bin/torchrun}"

# arms to run: default the two NEW ratios (1x/2x already exist in run_d20_overprovision_merge)
# space-separated "slot:frac" pairs.
ARMS="${ARMS:-60:0.75 120:0.875}"

WAIT_FOR_LOG="${WAIT_FOR_LOG:-}"
WAIT_FOR_MARKER="${WAIT_FOR_MARKER:-Total training time}"
WAIT_MAX_HOURS="${WAIT_MAX_HOURS:-24}"

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/mnt/3fs/dots-pretrain/ningzhiyuan/local_exps/nsmb_out}"
export TORCHDYNAMO_CACHE_SIZE_LIMIT="${TORCHDYNAMO_CACHE_SIZE_LIMIT:-64}"
export PYTHONPATH="${PYTHONPATH:-$REPO_DIR}"

mkdir -p "$LOG_DIR"

if [ -n "$WAIT_FOR_LOG" ]; then
    echo "Waiting for '$WAIT_FOR_MARKER' in $WAIT_FOR_LOG (max ${WAIT_MAX_HOURS}h)"
    deadline=$(( $(date +%s) + WAIT_MAX_HOURS * 3600 ))
    until grep -q "$WAIT_FOR_MARKER" "$WAIT_FOR_LOG" 2>/dev/null; do
        if [ "$(date +%s)" -ge "$deadline" ]; then
            echo "!! timed out waiting; NOT starting"; exit 1
        fi
        sleep 60
    done
    echo "Marker found; starting."
fi

run_arm () {
    local tag="$1"; local slot="$2"; local frac="$3"; shift 3
    local log_file="$LOG_DIR/${tag}.log"
    if grep -q "^Total training time" "$log_file" 2>/dev/null; then
        echo "Skipping $tag (already completed)"; return
    fi
    echo "================================================================"
    echo "Starting $tag  ($(date -Is))  slot=$slot merge_frac=$frac"
    echo "================================================================"
    "$TORCHRUN" --standalone --nproc_per_node="$NPROC" -m scripts.base_train -- \
        --depth=20 --max-seq-len=2048 --device-batch-size=8 --total-batch-size=524288 \
        --num-iterations="$NUM_ITERATIONS" --target-param-data-ratio=-1 \
        --window-pattern=SSSL --mhc --mhc-num-streams="$MHC_STREAMS" \
        --engram --engram-layers=2,6 --engram-memory-dim=1280 \
        --engram-mhc-num-streams="$MHC_STREAMS" --engram-share-memory \
        --engram-slot-multiplier="$slot" \
        --engram-merge-frac="$frac" --engram-merge-at-frac="$MERGE_AT" \
        --engram-merge-metric="$MERGE_METRIC" \
        --engram-seed="$SEED" \
        --eval-every=500 --eval-tokens=10485760 \
        --core-metric-every=-1 --sample-every=-1 --save-every=-1 \
        --model-tag="d20-opratio-${tag}-${RUN_SUFFIX}" \
        "$@" > "$log_file" 2>&1 || {
            echo "  !! $tag FAILED (exit $?)"; tail -20 "$log_file"; return
        }
    echo "  $tag done: $(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$log_file" | tail -1)"
}

for pair in $ARMS; do
    slot="${pair%%:*}"; frac="${pair##*:}"
    run_arm "op${slot}x_f${frac/./}" "$slot" "$frac"
done

echo
echo "Over-provision ratio sweep complete. Logs: $LOG_DIR"
printf "%-16s %-8s %-8s %-12s %s\n" arm slot frac min_bpb eff_rows
for pair in $ARMS; do
    slot="${pair%%:*}"; frac="${pair##*:}"; tag="op${slot}x_f${frac/./}"
    f="$LOG_DIR/${tag}.log"; [ -f "$f" ] || continue
    printf "%-16s %-8s %-8s %-12s %s\n" "$tag" "$slot" "$frac" \
        "$(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$f" | tail -1)" \
        "$(grep -oaE '\-> [0-9,]+ effective rows' "$f" | tail -1 | grep -oE '[0-9,]+')"
done
echo
echo "compare vs: op1x floor 0.765322 ; op2x 0.764795 (token, this session). All arms"
echo "should land at ~5.69M effective rows -- verify eff_rows before reading bpb."
