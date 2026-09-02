#!/bin/bash

# d20 idea 1B-v2: OVER-PROVISION then MERGE BACK to the reference size.
#
# The first merge scout (run_d20_merge.sh) started every arm at slot=15 (5.69M rows)
# and merged DOWN to 2.84M/1.42M -- so the merge arm always ended with LESS final
# capacity than its control. That measured "how much does compressing a table hurt"
# and found a monotone dose-dependent penalty (+0.002 @2x, +0.0035 @4x, n=1). It did
# NOT test whether merging is a useful capacity SCHEDULE, because final capacity was
# confounded with the merge.
#
# THIS run fixes the confound. All arms end at the SAME final effective capacity
# (~5.69M rows); the only difference is the PATH there:
#
#   arm            slot  merge      final eff. rows   role
#   op_floor       15    none       5,686,228         the target size, trained there all along
#   op_10to5       30    0.5 @50%   ~5,686,228         over-provision 2x, merge back to floor  <-- THE QUESTION
#   op_ceiling     30    none       ~11,372,000        how much the big table is worth at all
#
# THE QUESTION: does spending the first half of training with 2x the rows, then
# collapsing similar-VALUE rows onto survivors, beat training at the target size from
# the start? If op_10to5 < op_floor by >= the seed bar, early over-provisioning +
# value-aware merge is a real capacity schedule. If it lands at op_floor, merging
# perfectly recovers the target-size model (no gain, no loss -- still a clean null).
# op_10to5 should sit between op_floor and op_ceiling; how close to op_ceiling =
# how much of the big table's learned structure the value-aware merge preserves.
#
# PRIOR ANCHORS (study, param-matched): slot=15 = 0.764396 ; slot=30 = 0.763884.
# The ceiling is only ~5e-4 below the floor -- doubling rows barely helps (saturation,
# section 9.2d). So headroom for op_10to5 to beat the floor is SMALL by construction;
# the honest pre-registered expectation is op_10to5 ~ op_floor (merge recovers the
# target model). A win toward 0.7639 would say the merge captured the big table's edge.
#
# NOT PARAM-ISO. slot=30 arms carry ~2x engram_embeds params + optimizer state during
# the first half (dead rows stay allocated post-merge). This is iso on FINAL EFFECTIVE
# CAPACITY and on inference FLOPs/token (Engram reads are gathers, not matmuls) -- NOT
# on param count or training memory. State that in the writeup.
#
# GATE: op_floor is the same recipe as the prior control (mg_f0). It MUST reproduce
# the clean shared_slot15 baseline 0.764396 to <=1 SE, else the plumbing regressed;
# do not read op_10to5 / op_ceiling until it does.

set -euo pipefail

REPO_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"
cd "$REPO_DIR"

RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/runs/logs/d20_op_merge-${RUN_SUFFIX}}"
NPROC="${NPROC:-4}"
NUM_ITERATIONS="${NUM_ITERATIONS:-3814}"
MHC_STREAMS="${MHC_STREAMS:-4}"
SEED="${SEED:-0}"                                 # single-seed scout; genuine bar (no MoE router)
MERGE_AT="${MERGE_AT:-0.5}"                        # fire the merge at 50% of training
MERGE_METRIC="${MERGE_METRIC:-lsh}"               # lsh scales past 8192 rows/head
PYTHON="${PYTHON:-$REPO_DIR/../nanochat-moe/.venv/bin/python}"
TORCHRUN="${TORCHRUN:-$REPO_DIR/../nanochat-moe/.venv/bin/torchrun}"

# Optional gate: wait for a marker in another log before starting (so we don't
# contend for GPUs with a still-running scout). E.g. WAIT_FOR_LOG=.../mg_f075_cg_s0.log
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
            echo "!! timed out waiting; NOT starting (GPUs may be busy)"; exit 1
        fi
        sleep 60
    done
    echo "Marker found; starting."
fi

# $1 tag, $2 slot, $3 merge_frac ("0" -> no merge). Everything else is engram_shared.
run_arm () {
    local tag="$1"; local slot="$2"; local frac="$3"; shift 3
    local log_file="$LOG_DIR/${tag}.log"
    if grep -q "^Total training time" "$log_file" 2>/dev/null; then
        echo "Skipping $tag (already completed)"; return
    fi
    local merge_flags=()
    if [ "$frac" != "0" ]; then
        merge_flags=(--engram-merge-frac="$frac" \
                     --engram-merge-at-frac="$MERGE_AT" \
                     --engram-merge-metric="$MERGE_METRIC")
    fi
    echo "================================================================"
    echo "Starting $tag  ($(date -Is))  slot=$slot merge_frac=$frac metric=$MERGE_METRIC"
    echo "================================================================"
    "$TORCHRUN" --standalone --nproc_per_node="$NPROC" -m scripts.base_train -- \
        --depth=20 --max-seq-len=2048 --device-batch-size=8 --total-batch-size=524288 \
        --num-iterations="$NUM_ITERATIONS" --target-param-data-ratio=-1 \
        --window-pattern=SSSL --mhc --mhc-num-streams="$MHC_STREAMS" \
        --engram --engram-layers=2,6 --engram-memory-dim=1280 \
        --engram-mhc-num-streams="$MHC_STREAMS" --engram-share-memory \
        --engram-slot-multiplier="$slot" \
        "${merge_flags[@]}" \
        --engram-seed="$SEED" \
        --eval-every=500 --eval-tokens=10485760 \
        --core-metric-every=-1 --sample-every=-1 --save-every=-1 \
        --model-tag="d20-opmerge-${tag}-${RUN_SUFFIX}" \
        "$@" > "$log_file" 2>&1 || {
            echo "  !! $tag FAILED (exit $?)"; tail -20 "$log_file"; return
        }
    echo "  $tag done: $(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$log_file" | tail -1)"
}

# floor FIRST: it is the plumbing gate (must reproduce clean slot=15 baseline 0.764396).
run_arm "op_floor"    15  "0"
run_arm "op_10to5"    30  "0.5"
run_arm "op_ceiling"  30  "0"

echo
echo "Idea 1B-v2 (over-provision + merge) complete. Logs: $LOG_DIR"
printf "%-14s %-8s %-8s %-12s %s\n" arm slot merge min_bpb tok/sec
for row in "op_floor 15 0" "op_10to5 30 0.5" "op_ceiling 30 0"; do
    set -- $row; tag="$1"; slot="$2"; frac="$3"
    f="$LOG_DIR/${tag}.log"; [ -f "$f" ] || continue
    printf "%-14s %-8s %-8s %-12s %s\n" "$tag" "$slot" "$frac" \
        "$(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$f" | tail -1)" \
        "$(grep -oP 'tok/sec: \K[0-9,]+' "$f" | tail -1)"
done
echo
echo "anchors: slot15 floor 0.764396 ; slot30 ceiling 0.763884 (study, param-matched)"
echo "GATE FIRST: op_floor must reproduce slot15 0.764396 to <=1 SE."
echo "READ: op_10to5 vs op_floor -- below by >= seed bar = merge is a useful capacity"
echo "schedule; ~equal = merge recovers the target model (clean null). Place it vs the"
echo "op_ceiling (~11.4M rows) to see how much of the big table the value-merge kept."
