#!/bin/bash

# d20 round 11: map the row-capacity curve, the only axis that has ever paid.
#
# WHY THIS AND NOT ANOTHER MECHANISM. After 46 runs the scoreboard is:
#   SURVIVED  conditional capacity vs dense (25-50x); token vs hidden-state
#             addressing (0.0040, 20x); cross-layer sharing's MEAN (-0.0039, 12x);
#             row capacity, 0.00195 per doubling (12x, three confirmations).
#   DIED      13 hidden-state addressing mechanisms; rank-k values; separate k/v;
#             fixed-exponent whitening; longer n-gram orders; the count gate
#             (-0.00076 pooled at 1.94x, with a 15x variance penalty).
# Six mechanism ideas in a row have come back neutral or worse. Row capacity is
# the one thing with a measured law, and its CURVE has never been mapped -- the
# 0.00195 figure comes from a single pair of points on a contextual arm.
#
# THIS IS A PURE STORAGE SWEEP AT CONSTANT COMPUTE. Verified on meta: active
# params 918.34M and FLOPs 3.135e9 are IDENTICAL for every arm below, from 1373M
# to 4557M total. An Engram read pulls 16 heads x 80 dims however many rows the
# table has, so slot_multiplier moves *storage only*. That makes this the
# total/active decoupling taken to its limit: 3.3x the parameters for zero extra
# compute.
#
# IT DELIBERATELY BREAKS ISO-TOTAL, and that is the point -- the resource is the
# independent variable. Do NOT compare these numbers to the mechanism arms in §1-§9
# as if they were matched. An arm here that wins by using 2.5x the parameters has
# not beaten anything; it has measured a slope.
#
# Two anchors already exist at n=3:
#   shared,   slot 30, 710,580 rows/head, 1828.0M -> 0.763884 (sd 3.5e-4)
#   perlayer, slot 15, 355,290 rows/head, 1828.4M -> 0.767606 (sd 8.1e-5)
#
# THE HEADLINE ARM IS perlayer_slot30: IS SHARING JUST ROWS?
# shared_slot30 and perlayer_slot15 have identical total, active and FLOPs, and
# differ by -0.00372 in favour of sharing. Sharing gives 2x the rows PER READ
# (one 710k table read twice, vs two 355k tables read once each), and the row law
# predicts only -0.00195 of that. So half the biggest surviving effect in the
# study is unexplained.
#   perlayer_slot30 has the SAME 710,580 rows per read as shared_slot30, but in
#   two private tables -- so 1.5x the total parameters, a deliberate advantage.
#     lands near 0.7639  -> sharing IS just rows; the effect fully reduces to
#                           capacity and the "shared memory" story is unnecessary
#     stays clearly worse -> sharing does something beyond capacity, even when the
#                           per-layer arm is handed 1.5x the parameters
#
# WELL POWERED AT n=1, unlike rounds 5-10. Expected effects are ~0.002 per
# doubling against bars of 3.5e-4 (shared) and 8.1e-5 (per-layer), so a new n=1
# arm resolves 0.0008 (shared) / 0.0002 (per-layer) at 2x -- i.e. 5x and 21x
# headroom on the expected effect. This is the first round in a while where one
# seed per point is the right allocation rather than a corner cut.
#
# Sizing (all verified on meta; active and FLOPs constant throughout):
#   arm               slot  rows/head  total     dbs
#   perlayer_slot30    30    710,580   2738.0M    8
#   shared_slot60      60   1,421,160  2737.6M    8
#   shared_slot15      15    355,290   1373.2M    8
#   shared_slot120    120   2,842,320  4556.7M    4   <- near the 3707M that OOM'd
#   perlayer_slot60    60   1,421,160  4557.2M    4      on d26 Mobius at dbs=8
# device-batch-size 4 keeps total batch at 524288, so those arms are numerically
# identical, just ~25% slower.

set -euo pipefail

REPO_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"
cd "$REPO_DIR"

RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/runs/logs/d20_rowcap-${RUN_SUFFIX}}"
NPROC="${NPROC:-4}"
NUM_ITERATIONS="${NUM_ITERATIONS:-3814}"
MHC_STREAMS="${MHC_STREAMS:-4}"
TORCHRUN="${TORCHRUN:-$REPO_DIR/../nanochat-moe/.venv/bin/torchrun}"

# Marker string in a log, never a pid or process name (§6: kill -0 succeeds on
# zombies; pgrep/ps self-match, including the bracketed base_tra[i]n form).
WAIT_FOR_LOG="${WAIT_FOR_LOG:-}"
WAIT_FOR_MARKER="${WAIT_FOR_MARKER:-Round 10 complete}"
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
    echo "Predecessor finished at $(date -Is); starting round 11"
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
        --model-tag="d20-rowcap-${tag}-${RUN_SUFFIX}" \
        "$@" > "$log_file" 2>&1 || {
            echo "  !! $tag FAILED (exit $?)"; tail -20 "$log_file"; return
        }
    echo "  $tag done: $(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$log_file" | tail -1)"
}

# --- 1. IS SHARING JUST ROWS? matched rows/read, per-layer gets 1.5x params ---
run_arm perlayer_slot30 8 --engram-slot-multiplier=30

# --- 2. does storage keep paying? (law predicts -0.00195) --------------------
run_arm shared_slot60 8 --engram-slot-multiplier=60 --engram-share-memory

# --- 3. the law downward (predicts +0.00195) --------------------------------
run_arm shared_slot15 8 --engram-slot-multiplier=15 --engram-share-memory

# --- 4. saturation: 4x rows, 2.5x total params, same compute ---------------
run_arm shared_slot120 4 --engram-slot-multiplier=120 --engram-share-memory

# --- 5. per-layer slope at the same storage as shared_slot120 -------------
run_arm perlayer_slot60 4 --engram-slot-multiplier=60

echo
echo "Round 11 complete. Logs: $LOG_DIR"
printf "%-20s %-12s %s\n" arm min_bpb tok/sec
for tag in shared_slot15 shared_slot60 shared_slot120 perlayer_slot30 perlayer_slot60; do
    f="$LOG_DIR/${tag}.log"; [ -f "$f" ] || continue
    printf "%-20s %-12s %s\n" "$tag" \
        "$(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$f" | tail -1)" \
        "$(grep -oP 'tok/sec: \K[0-9,]+' "$f" | tail -1)"
done
echo
echo "anchors (n=3): shared slot30 0.763884 (sd 3.5e-4)"
echo "               perlayer slot15 0.767606 (sd 8.1e-5)"
echo
echo "fit bpb vs log2(rows/head) SEPARATELY for shared and per-layer:"
echo "  same slope  -> capacity is capacity; sharing only supplies more of it"
echo "  different   -> sharing changes how well capacity is USED"
echo "and check whether the shared curve flattens between slot 60 and 120."
echo "REMINDER: these arms are NOT iso-total. A win here bought with 2.5x the"
echo "parameters has measured a slope, not beaten a baseline."
