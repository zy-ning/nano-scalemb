#!/bin/bash

# d20 round 13: decompose the one big effect that is left, and close the dbs hole.
#
# After 12 rounds the survivors are exactly two: token addressing over
# hidden-state addressing (0.0040, 20x), and CROSS-LAYER SHARING (-0.0037 to
# -0.0039, 9-12x, three framings). Round 11 showed sharing is NOT a capacity
# effect -- at matched rows-per-read it still wins 0.00372 at 9.3x while using 67%
# of the storage -- so the mechanism is unexplained. "One jointly-trained memory"
# is a story, not a measurement.
#
# It bundles two separable things, and --engram-share-hash now opens the 2x2:
#
#                     per-layer hash              shared hash
#   per-layer table   `engram` 0.767555           ARM 3: same INDEX, own content
#   shared table      ARM 2: one POOL, own rows   `engram_shared` 0.763884
#
#   ARM 2 near 0.7639 -> the benefit is having ONE POOL of rows; consistent
#                        cross-depth addressing is incidental.
#   ARM 2 near 0.7675 -> the benefit is that an n-gram hits the SAME ROW at every
#                        depth, so one row accumulates gradient from all readers.
#   ARM 3 is the mirror: same row NUMBER in separate tables, so consistent
#                        addressing WITHOUT shared content. If it also lands near
#                        0.7675, shared content is required and indexing alone is
#                        worthless -- which would make ARM 2 the whole story.
#
# Mechanics (implemented and tested): per-head PRIME sizes fix the table geometry
# and must be canonical whenever the table or the addressing is shared; per-layer
# MULTIPLIERS decide which row inside that geometry an n-gram lands on and are
# canonical only when the addressing is shared. `share_hash=auto` (default)
# follows share_memory and reproduces every earlier run bit-for-bit -- verified by
# tests/test_engram.py::TestShareHashDecoupled::test_default_follows_share_memory,
# and indices are asserted in range for the shared-geometry cell.
#
# WELL POWERED AT n=1: the two hypotheses are 0.0037 apart and the shared-arm bar
# is 3.45e-4, so a single seed against the n=3 anchors resolves them at ~9x. Two
# seeds each anyway, given the history.
#
# Sizing verified on meta, all four cells iso: 1828.0-1828.4M total (+0.02%),
# 918.34M active identical, same 909.5M of table bytes.
#
# ARM 1 IS A METHODOLOGICAL FIX, not science. Round 11 mixed device-batch-size 4
# and 8 inside one capacity sweep; both dbs=4 points came in 0.0022-0.0044 better
# than the fit predicted, a single additive dbs effect does not fit them, and a
# prior dbs=4 arm showed no benefit at all -- so two of five arms are unusable and
# the top of the shared capacity curve is unattributable. Re-running
# perlayer_slot30 at dbs=4 isolates dbs at fixed rows (its dbs=8 value is
# 0.767602) and says whether any earlier cross-dbs comparison needs revisiting.
# Everything else here is pinned at dbs=8.

set -euo pipefail

REPO_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"
cd "$REPO_DIR"

RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/runs/logs/d20_shdec-${RUN_SUFFIX}}"
NPROC="${NPROC:-4}"
NUM_ITERATIONS="${NUM_ITERATIONS:-3814}"
MHC_STREAMS="${MHC_STREAMS:-4}"
TORCHRUN="${TORCHRUN:-$REPO_DIR/../nanochat-moe/.venv/bin/torchrun}"

WAIT_FOR_LOG="${WAIT_FOR_LOG:-}"
WAIT_FOR_MARKER="${WAIT_FOR_MARKER:-Round 12 complete}"
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
            echo "!! timed out waiting; NOT starting (GPUs may be busy)"; exit 1
        fi
        sleep 60
    done
    echo "Predecessor finished at $(date -Is); starting round 13"
fi

run_arm () {
    local tag="$1"; local dbs="$2"; shift 2
    local log_file="$LOG_DIR/${tag}.log"
    if grep -q "^Total training time" "$log_file" 2>/dev/null; then
        echo "Skipping $tag (already completed)"; return
    fi
    echo "================================================================"
    echo "Starting $tag  ($(date -Is))  device-batch-size=$dbs"
    echo "  extra: $*"
    echo "================================================================"
    "$TORCHRUN" --standalone --nproc_per_node="$NPROC" -m scripts.base_train -- \
        --depth=20 --max-seq-len=2048 --device-batch-size="$dbs" \
        --total-batch-size=524288 \
        --num-iterations="$NUM_ITERATIONS" --target-param-data-ratio=-1 \
        --window-pattern=SSSL --mhc --mhc-num-streams="$MHC_STREAMS" \
        --engram --engram-layers=2,6 --engram-memory-dim=1280 \
        --engram-mhc-num-streams="$MHC_STREAMS" \
        --eval-every=500 --eval-tokens=10485760 \
        --core-metric-every=-1 --sample-every=-1 --save-every=-1 \
        --model-tag="d20-shdec-${tag}-${RUN_SUFFIX}" \
        "$@" > "$log_file" 2>&1 || {
            echo "  !! $tag FAILED (exit $?)"; tail -20 "$log_file"; return
        }
    echo "  $tag done: $(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$log_file" | tail -1)"
}

# --- 1. the dbs control: identical to perlayer_slot30 except dbs 8 -> 4 -----
run_arm perlayer_slot30_dbs4 4 --engram-slot-multiplier=30

# --- 2/4. shared table, PER-LAYER addressing: is it just one pool? ---------
for seed in 0 1; do
    run_arm "share_tbl_s${seed}" 8 --engram-slot-multiplier=30 \
        --engram-share-memory --engram-share-hash=no --engram-seed="$seed"
done

# --- 3/5. per-layer tables, SHARED addressing: index without content -------
for seed in 0 1; do
    run_arm "share_hash_s${seed}" 8 --engram-slot-multiplier=15 \
        --engram-share-hash=yes --engram-seed="$seed"
done

echo
echo "Round 13 complete. Logs: $LOG_DIR"
printf "%-22s %-12s %s\n" arm min_bpb tok/sec
for tag in perlayer_slot30_dbs4 share_tbl_s0 share_tbl_s1 share_hash_s0 share_hash_s1; do
    f="$LOG_DIR/${tag}.log"; [ -f "$f" ] || continue
    printf "%-22s %-12s %s\n" "$tag" \
        "$(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$f" | tail -1)" \
        "$(grep -oP 'tok/sec: \K[0-9,]+' "$f" | tail -1)"
done
echo
echo "the 2x2 (all iso, 909.5M table bytes, 918.34M active):"
echo "                    per-layer hash          shared hash"
echo "  per-layer table   engram 0.767555         share_hash (arm 3/5)"
echo "  shared table      share_tbl (arm 2/4)     engram_shared 0.763884 (n=3, sd 3.5e-4)"
echo
echo "arm 1: perlayer_slot30 at dbs=8 was 0.767602. A large gap here means the two"
echo "  round-11 dbs=4 arms were batch-geometry artifacts, and every cross-dbs"
echo "  comparison in this study needs re-checking."
