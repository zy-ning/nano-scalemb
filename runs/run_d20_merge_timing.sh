#!/bin/bash

# d20 idea: MERGE-TIMING sweep -- WHEN should the value-aware row merge fire?
#
# BACKGROUND. Every merge in this whole thread fired at merge_at_frac=0.5, never
# swept (op2x/op4x/op8x, semantic, all of it). But the thread's central mechanism
# claim is about VALUE QUALITY AT MERGE TIME: "2x is the sweet spot because beyond
# it the pretrain table is undertrained BEFORE the merge, so the value-similarity
# partition the merge clusters on is noisier" (docs/conditional_capacity_study.md;
# engram-overprovision-merge-schedule memory). Merge TIMING is the most direct lever
# on that value quality, and it is the one variable held fixed the entire time.
#
# THE SWEEP. Hold the known 2x sweet spot FIXED (slot30, merge_frac=0.5 = the op2x
# arm, 0.764795) and vary ONLY the step at which the alias fires:
#
#   arm     merge_at_frac   values at merge      post-merge adaptation budget
#   at25    0.25            rawest               most (2861 steps)
#   at50    0.50            = op2x control       3814*0.5 = 1907 steps
#   at75    0.75            cleaner              954 steps
#   at90    0.90            near-fully-trained   381 steps
#
# This is a clean single-variable sweep: identical params / FLOPs / final effective
# rows across all four arms; only the merge STEP changes. merge_at_frac already
# exists and is tested (run_d20_op_ratio.sh), so no new code.
#
# PRE-REGISTERED READS (decide before looking):
#   bpb keeps DROPPING toward late merges (at90 best) = the partition is value-
#     quality-limited exactly as the mechanism story predicts, and 0.5 was arbitrary.
#     Reopens the schedule with a cleaner lever than over-provisioning (which
#     plateaued past 2x).
#   U-SHAPED, minimum near 50-75% = a genuine tension between value maturity and
#     post-merge adaptation time; 0.5 was luckily near-optimal.
#   FLAT (all within the seed bar) = merge timing does not matter, which itself
#     bounds the "undertrained values" explanation for the 2x plateau.
#
# NOT a new mechanism and NOT param-iso to the token floor (slot30 carries 2x engram
# embeds + optimizer state; merged rows stay allocated but dead). Iso on FINAL
# effective capacity + inference FLOPs across the four arms. Within-sweep (same seed,
# same code) comparison is clean. n=1 seed-0 scout; a resolving replicate follows
# only if a gap clears the ~5e-4 token-arm bar.
#
# GATE: every arm must FIRE the merge and print an effective-row count within ~0.5%
# of the op2x reference (~5.69M... actually ~6.06M at slot30 lsh, the achieved op2x
# number), else the merge mis-fired and the arm is not capacity-matched.

set -euo pipefail

REPO_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"
cd "$REPO_DIR"

RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/runs/logs/d20_merge_timing-${RUN_SUFFIX}}"
NPROC="${NPROC:-4}"
NUM_ITERATIONS="${NUM_ITERATIONS:-3814}"
MHC_STREAMS="${MHC_STREAMS:-4}"
SEED="${SEED:-0}"
SLOT="${SLOT:-30}"
MERGE_FRAC="${MERGE_FRAC:-0.5}"
MERGE_METRIC="${MERGE_METRIC:-lsh}"
PYTHON="${PYTHON:-$REPO_DIR/../nanochat-moe/.venv/bin/python}"
TORCHRUN="${TORCHRUN:-$REPO_DIR/../nanochat-moe/.venv/bin/torchrun}"

# space-separated merge_at_frac values.
ARMS="${ARMS:-0.25 0.5 0.75 0.9}"

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
    local at="$1"
    local tag="at${at/./}"
    local log_file="$LOG_DIR/${tag}.log"
    if grep -q "^Total training time" "$log_file" 2>/dev/null; then
        echo "Skipping $tag (already completed)"; return
    fi
    echo "================================================================"
    echo "Starting $tag  ($(date -Is))  slot=$SLOT merge_frac=$MERGE_FRAC merge_at=$at"
    echo "================================================================"
    "$TORCHRUN" --standalone --nproc_per_node="$NPROC" -m scripts.base_train -- \
        --depth=20 --max-seq-len=2048 --device-batch-size=8 --total-batch-size=524288 \
        --num-iterations="$NUM_ITERATIONS" --target-param-data-ratio=-1 \
        --window-pattern=SSSL --mhc --mhc-num-streams="$MHC_STREAMS" \
        --engram --engram-layers=2,6 --engram-memory-dim=1280 \
        --engram-mhc-num-streams="$MHC_STREAMS" --engram-share-memory \
        --engram-slot-multiplier="$SLOT" \
        --engram-merge-frac="$MERGE_FRAC" --engram-merge-at-frac="$at" \
        --engram-merge-metric="$MERGE_METRIC" \
        --engram-seed="$SEED" \
        --eval-every=500 --eval-tokens=10485760 \
        --core-metric-every=-1 --sample-every=-1 --save-every=-1 \
        --model-tag="d20-mtiming-${tag}-${RUN_SUFFIX}" \
        > "$log_file" 2>&1 || {
            echo "  !! $tag FAILED (exit $?)"; tail -20 "$log_file"; return
        }
    echo "  $tag done: $(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$log_file" | tail -1)"
}

for at in $ARMS; do
    run_arm "$at"
done

echo
echo "Merge-timing sweep complete. Logs: $LOG_DIR"
printf "%-10s %-14s %-12s %s\n" arm merge_at min_bpb eff_rows
for at in $ARMS; do
    tag="at${at/./}"
    f="$LOG_DIR/${tag}.log"; [ -f "$f" ] || continue
    printf "%-10s %-14s %-12s %s\n" "$tag" "$at" \
        "$(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$f" | tail -1)" \
        "$(grep -oaE '\-> [0-9,]+ effective rows' "$f" | tail -1 | grep -oE '[0-9,]+' | head -1)"
done
echo
echo "anchor: op2x (slot30, frac0.5, merge_at0.5) = 0.764795 (this session) == the at50 arm."
echo "READ: monotone-down toward at90 = value-quality-limited (reopens schedule); U-shaped"
echo "min near 50-75% = maturity-vs-adaptation tension; flat = timing irrelevant."
