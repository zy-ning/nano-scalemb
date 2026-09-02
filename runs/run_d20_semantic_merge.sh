#!/bin/bash

# d20 idea 2-fusion: OVER-PROVISION a SEMANTIC-HASH (LSH) table, then value-MERGE
# back to the reference size. Does today's over-provision+merge schedule RESCUE the
# semantic-hash liability?
#
# BACKGROUND. Two prior results point opposite ways:
#   - "idea 2" semantic hash (address the memory by hidden state via LSH/PQ codes,
#     colliding representation-similar n-grams): study measured lsh +0.0038, pq
#     +0.0052 vs the TOKEN baseline -- a net liability. Line was declared closed.
#   - This session's value-aware MERGE (cluster rows by LEARNED VALUE, alias similar
#     rows onto survivors mid-training): a small WIN as a capacity SCHEDULE --
#     over-provision 2x then merge back to the target size beat training at the
#     target size from the start (op_10to5 0.764795 vs op_floor 0.765322, n=1).
#
# THE FUSION. The merge clusters on the learned VALUE table only (engram.py
# _value_matrix -> embedding.weight[:, key_dim:]), never the address code, so it is
# address-source-AGNOSTIC. That lets us apply the winning SCHEDULE to the LOSING
# MECHANISM. Hypothesis: LSH hurt because it OVER-COLLIDES at the target size. Give
# it 2x rows (so it collides less) then value-merge the survivors back to the
# reference size -- clean up the bad collisions with a good (value-aware) one.
#
#   arm           address  slot  merge      final eff. rows   question
#   sem_floor     lsh      15    none       5.69M             reproduce the lsh liability (~+0.0038 vs token)
#   sem_10to5     lsh      30    0.5 @50%    ~5.69M            DOES over-provision+value-merge RESCUE lsh?  <--
#   sem_ceiling   lsh      30    none        ~11.37M           does 2x rows alone cut lsh over-collision?
#
# PRE-REGISTERED READS (decide before looking):
#   sem_10to5 < sem_floor by >= seed bar  = the schedule rescues the semantic hash.
#   sem_10to5 <= TOKEN baseline 0.7644    = REOPENS the closed idea-2 line (big deal).
#   sem_10to5 ~ sem_floor                 = merge is value-agnostic to the address
#                                           source; the lsh penalty is not about
#                                           collision COUNT -> stays closed.
#   sem_ceiling vs sem_floor              = is the lsh penalty just under-capacity
#                                           (ceiling recovers it) or intrinsic?
#
# LSH is FIXED (no learned params; discretize.py:70) -- cleanest interaction with the
# merge. PQ's EMA codebook would confound, so this scout uses lsh only.
#
# NOT PARAM-ISO across slots (same caveat as run_d20_overprovision_merge.sh): slot=30
# arms carry ~2x engram_embeds + optimizer state in the first half. Iso on FINAL
# effective capacity + inference FLOPs only. Within-bracket (same seed, same code)
# comparisons are clean regardless.
#
# GATE: sem_floor must reproduce the STUDY's lsh number (token baseline + ~0.0038),
# NOT the token baseline. If it lands near the token baseline instead, the address
# source did not take -- stop and check --engram-address-source wiring before reading
# the merge arms.

set -euo pipefail

REPO_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"
cd "$REPO_DIR"

RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/runs/logs/d20_sem_merge-${RUN_SUFFIX}}"
NPROC="${NPROC:-4}"
NUM_ITERATIONS="${NUM_ITERATIONS:-3814}"
MHC_STREAMS="${MHC_STREAMS:-4}"
SEED="${SEED:-0}"
MERGE_AT="${MERGE_AT:-0.5}"
MERGE_METRIC="${MERGE_METRIC:-lsh}"                # value-clustering metric (not the address source)
ADDRESS_SOURCE="${ADDRESS_SOURCE:-lsh}"            # the SEMANTIC hash under test
LSH_BITS="${LSH_BITS:-20}"
PYTHON="${PYTHON:-$REPO_DIR/../nanochat-moe/.venv/bin/python}"
TORCHRUN="${TORCHRUN:-$REPO_DIR/../nanochat-moe/.venv/bin/torchrun}"

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

# $1 tag, $2 slot, $3 merge_frac ("0" -> no merge). address_source fixed for all arms.
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
    echo "Starting $tag  ($(date -Is))  addr=$ADDRESS_SOURCE slot=$slot merge_frac=$frac"
    echo "================================================================"
    "$TORCHRUN" --standalone --nproc_per_node="$NPROC" -m scripts.base_train -- \
        --depth=20 --max-seq-len=2048 --device-batch-size=8 --total-batch-size=524288 \
        --num-iterations="$NUM_ITERATIONS" --target-param-data-ratio=-1 \
        --window-pattern=SSSL --mhc --mhc-num-streams="$MHC_STREAMS" \
        --engram --engram-layers=2,6 --engram-memory-dim=1280 \
        --engram-mhc-num-streams="$MHC_STREAMS" --engram-share-memory \
        --engram-slot-multiplier="$slot" \
        --engram-address-source="$ADDRESS_SOURCE" --engram-lsh-bits="$LSH_BITS" \
        "${merge_flags[@]}" \
        --engram-seed="$SEED" \
        --eval-every=500 --eval-tokens=10485760 \
        --core-metric-every=-1 --sample-every=-1 --save-every=-1 \
        --model-tag="d20-semmerge-${tag}-${RUN_SUFFIX}" \
        "$@" > "$log_file" 2>&1 || {
            echo "  !! $tag FAILED (exit $?)"; tail -20 "$log_file"; return
        }
    echo "  $tag done: $(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$log_file" | tail -1)"
}

# floor FIRST: the gate. It must reproduce the STUDY's lsh liability, else the
# address source did not wire in -- stop before reading the merge arms.
run_arm "sem_floor"    15  "0"
run_arm "sem_10to5"    30  "0.5"
run_arm "sem_ceiling"  30  "0"

echo
echo "Idea 2-fusion (over-provision semantic-hash + merge) complete. Logs: $LOG_DIR"
printf "%-14s %-8s %-8s %-12s %s\n" arm slot merge min_bpb tok/sec
for row in "sem_floor 15 0" "sem_10to5 30 0.5" "sem_ceiling 30 0"; do
    set -- $row; tag="$1"; slot="$2"; frac="$3"
    f="$LOG_DIR/${tag}.log"; [ -f "$f" ] || continue
    printf "%-14s %-8s %-8s %-12s %s\n" "$tag" "$slot" "$frac" \
        "$(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$f" | tail -1)" \
        "$(grep -oP 'tok/sec: \K[0-9,]+' "$f" | tail -1)"
done
echo
echo "anchors: TOKEN slot15 0.764396 ; study lsh ~ token +0.0038 ; slot15/30 op_floor"
echo "0.765322 / op_ceiling 0.763810 (token, this session)."
echo "GATE FIRST: sem_floor must reproduce study lsh (NOT the token baseline)."
echo "READ: sem_10to5 < sem_floor = schedule rescues the semantic hash; sem_10to5 <="
echo "0.7644 = REOPENS idea 2; sem_10to5 ~ sem_floor = lsh penalty is not collision-count."
