#!/bin/bash

# d20 round 15: collisions are pervasive and cannot be hashed away -- so decorrelate
# them across more independent hashes instead.
#
# WHERE THIS COMES FROM. Round 14 established that the count gate's decoupled
# collision term is real and consistently signed: `collide_scale` is negative with
# 16/16 head agreement in all seven gates measured, and it is 3-4x LARGER on
# private tables (-0.26 to -0.37) than on the shared one (-0.087) -- private tables
# hold half the rows per layer, so they collide more, and the gate responds by
# penalising crowding harder. Collisions cost real bpb.
#
# THE DIAGNOSTIC SAYS "COLLIDE LESS" IS NOT AVAILABLE DIRECTLY. Reading the
# hit_rate buffers out of a round-14 checkpoint, per head, per n-gram order:
#
#   order-2 heads:  0.1% of rows empty,  p99 hit rate  8.6,  max 1670
#   order-3 heads:  0.0% of rows empty,  p99 hit rate  3.8,  max  575
#
# The tables are SATURATED -- essentially every row is occupied, and the implied
# occupancy is ~7 (order 2) to ~21 (order 3) hits per row per EMA window. The
# number of distinct n-grams vastly exceeds the row count, so no hash function can
# reduce collisions: uniform hashing already minimises max load up to the
# power-of-choices log-log improvement. And round 11 showed extra rows barely help
# (-0.000533 per doubling shared, flat per-layer), so buying capacity is not the
# answer either. Order-2 carries a far heavier tail (max 1670 vs 575), i.e. a few
# rows where frequent bigrams collide with each other -- exactly the damaging case.
#
# WHAT IS AVAILABLE: MORE INDEPENDENT DRAWS. Each head hashes the same n-gram under
# a different prime, so a given n-gram's collision *partners* differ per head. The
# signal (what this n-gram wants) is common across heads; the interference is
# independent. A linear readout over the concatenation can therefore average the
# interference down, and the averaging improves with head count.
#
# AND IT IS EXACTLY FREE. Table bytes = num_heads x rows_per_head x head_dim, and
# head_dim = memory_dim / num_heads, so at fixed memory_dim and fixed
# slot_multiplier the table size is INDEPENDENT of head count. Verified on meta:
#
#   heads/order   heads   head_dim   total       active      FLOPs/token
#         4          8       160     1827.94M    918.340M    3.135e9
#         8         16        80     1828.02M    918.340M    3.135e9   <- current
#        16         32        40     1828.16M    918.340M    3.135e9
#
# Total varies 0.025%; active params and FLOPs are identical. A genuinely
# single-variable sweep, which this study has rarely managed.
#
# ONE PRIOR DATA POINT, AND WHY IT DOES NOT SETTLE THIS. `cg_n5` (round 8) ran 32
# heads and came in 1.8x worse -- but it got there via max_ngram_size=5, so it
# confounded head count with two extra and much sparser n-gram orders (4 and 5).
# This is the clean test at orders 2-3 only.
#
# PREDICTION, genuinely two-sided. More heads gives more independent interference
# samples to average (good) but narrows each head's payload from 80 dims to 40
# (worse per-head resolution). The total read is 1280 dims either way. I lean
# toward more heads helping, because interference decorrelates while signal does
# not -- but the payload narrowing is a real cost and eight mechanism ideas in this
# study have already come back neutral or worse. heads4 is included precisely to
# give a slope rather than a point: if 32 heads help, 8 should hurt.

set -euo pipefail

REPO_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"
cd "$REPO_DIR"

RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/runs/logs/d20_hheads-${RUN_SUFFIX}}"
NPROC="${NPROC:-4}"
NUM_ITERATIONS="${NUM_ITERATIONS:-3814}"
MHC_STREAMS="${MHC_STREAMS:-4}"
TORCHRUN="${TORCHRUN:-$REPO_DIR/../nanochat-moe/.venv/bin/torchrun}"

WAIT_FOR_LOG="${WAIT_FOR_LOG:-}"
WAIT_FOR_MARKER="${WAIT_FOR_MARKER:-Round 14 complete}"
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
    echo "Predecessor finished at $(date -Is); starting round 15"
fi

run_arm () {
    local tag="$1"; local hpn="$2"; shift 2
    local log_file="$LOG_DIR/${tag}.log"
    if grep -q "^Total training time" "$log_file" 2>/dev/null; then
        echo "Skipping $tag (already completed)"; return
    fi
    echo "================================================================"
    echo "Starting $tag  ($(date -Is))  heads_per_ngram=$hpn"
    echo "  extra: $*"
    echo "================================================================"
    "$TORCHRUN" --standalone --nproc_per_node="$NPROC" -m scripts.base_train -- \
        --depth=20 --max-seq-len=2048 --device-batch-size=8 --total-batch-size=524288 \
        --num-iterations="$NUM_ITERATIONS" --target-param-data-ratio=-1 \
        --window-pattern=SSSL --mhc --mhc-num-streams="$MHC_STREAMS" \
        --engram --engram-layers=2,6 --engram-memory-dim=1280 \
        --engram-mhc-num-streams="$MHC_STREAMS" --engram-share-memory \
        --engram-slot-multiplier=30 --engram-heads-per-ngram="$hpn" \
        --eval-every=500 --eval-tokens=10485760 \
        --core-metric-every=-1 --sample-every=-1 --save-every=-1 \
        --model-tag="d20-hheads-${tag}-${RUN_SUFFIX}" \
        "$@" > "$log_file" 2>&1 || {
            echo "  !! $tag FAILED (exit $?)"; tail -20 "$log_file"; return
        }
    echo "  $tag done: $(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$log_file" | tail -1)"
}

# interleaved so a truncated run leaves both directions readable
run_arm heads16_s0 16 --engram-seed=0
run_arm heads4_s0   4 --engram-seed=0
run_arm heads16_s1 16 --engram-seed=1
run_arm heads4_s1   4 --engram-seed=1
run_arm heads16_s2 16 --engram-seed=2

echo
echo "Round 15 complete. Logs: $LOG_DIR"
printf "%-16s %-12s %s\n" arm min_bpb tok/sec
for tag in heads4_s0 heads4_s1 heads16_s0 heads16_s1 heads16_s2; do
    f="$LOG_DIR/${tag}.log"; [ -f "$f" ] || continue
    printf "%-16s %-12s %s\n" "$tag" \
        "$(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$f" | tail -1)" \
        "$(grep -oP 'tok/sec: \K[0-9,]+' "$f" | tail -1)"
done
echo
echo "baseline: 8 heads/order (16 heads x 80 dims) = engram_shared 0.763884"
echo "          n=3, sd 3.45e-4, SE 1.99e-4. A new n=3 arm resolves ~0.0005 at 2x."
echo
echo "expected shape if interference-averaging is the mechanism:"
echo "  heads16 < baseline < heads4   (monotone in head count)"
echo "if heads16 ~= heads4 ~= baseline, the H-head averaging was already saturated"
echo "at 16 and collisions are simply tolerated rather than cancelled."
echo
echo "also worth reading from a heads16 checkpoint: the per-head hit_rate tails."
echo "  order-2 max was 1670x mean at 8 heads/order. Does 16 change the tail, or"
echo "  only how much the READ cares about it?"
