#!/bin/bash

# d20 round 7: the two axes the study has never touched, both built on the best
# arm (engram_shared, n=3, error bar 1.85e-4 -- the tightest baseline available,
# which makes these arms unusually well powered: even a 0.0005 effect is 2.7x).
#
# Why these two, from what rounds 1-6 established:
#   - ROW COUNT is the binding constraint: 0.00195 per doubling at 12x its bar.
#   - ROW CONTENT expressiveness is dead: rank measured 0.01x and 0.09x of its
#     bar at fixed rows, and -23x (harmful) on this very arm.
#   - ADDRESS INFORMATION is what pays: token n-grams beat hidden-state
#     addressing by 0.0040 because they carry information h has discarded.
# So: do not make rows richer. Make addressing better, and stop wasting row
# capacity on collisions.
#
# 1. count_gate -- Kneser-Ney backoff, which the Engram lacks. It reads every
#    n-gram order in parallel and concatenates with FIXED weights, so a 3-gram
#    seen twice is trusted exactly like one seen ten thousand times, and a rare
#    row -- the likeliest collision victim -- is trusted like a clean one. This
#    learns w = 2*sigmoid(a*log1p(hit_rate) + b) per head, a=b=0 at init so it
#    starts exactly neutral. Backoff was the single largest win in classical
#    n-gram LMing; it attacks the collision axis directly.
#    COST: 32 scalars. Zero change to rows, total params, active params or FLOPs.
#    This is the cheapest arm in the entire study.
#
# 2. key_dim -- separate k and v per row. Today ONE stored vector serves both
#    roles: the read is projected by stream_key_proj for the relevance gate and
#    by value_proj for the payload, and both projections are shared across every
#    row. So a row's key is a fixed linear function of its value -- two rows with
#    similar content are FORCED to have similar relevance, and a row cannot be
#    sharply selective about context while carrying arbitrary content. Attention,
#    MoE routers and product-key memories all separate k from v; the Engram does
#    not. Two sizings, because they bracket the trade differently:
#
#      kv16 slot25  key_dim=16: rows -17% (predicted cost ~0.0005 from the
#                   row-count law). Total -0.57%, and active -1.14% because a
#                   narrower gate input shrinks stream_key_proj. The arm is
#                   therefore HANDICAPPED on active params, so a win is
#                   conservative; a loss is partly confounded by it.
#      kv80 slot15  key_dim=80 = head_dim, the symmetric split. Gate projection
#                   is unchanged, so total +0.01% and active +0.00% -- exactly
#                   iso, the clean comparison. But rows HALVE, so it must beat
#                   ~0.00195 to show at all.
#
# Sizing verified by building each config on meta:
#   engram_shared  1828.0M total / 918.3M active / 710,580 rows per head
#   count_gate     1828.0M        / 918.3M       / 710,580   (+0.00% / +0.00%)
#   kv16 slot25    1817.6M        / 907.9M       / 592,150   (-0.57% / -1.14%)
#   kv80 slot15    1828.1M        / 918.3M       / 355,290   (+0.01% / +0.00%)
#
# PRE-REGISTERED, and deliberately modest: I have now been wrong twice on this
# family (§8.4 predicted a gain from rank-k rows and got the opposite sign; the
# follow-up asymmetry claim was retracted). So:
#   count_gate: direction unknown. It is strictly more expressive than the fixed
#     weights it replaces and starts neutral, so it should not HURT beyond noise.
#     I am not predicting a large gain. AdamW can partly absorb any reweighting
#     by rescaling rows, the same caveat that applies to readout whitening.
#   kv16: needs to beat ~0.0005 (its row cost).
#   kv80: needs to beat ~0.00195 (its row cost).
#   If all three land within 2x of 1.85e-4, the honest reading is that the gate
#   and the k/v coupling were not binding, and row capacity is the only lever
#   left on this architecture.

set -euo pipefail

REPO_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"
cd "$REPO_DIR"

RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/runs/logs/d20_kvbo-${RUN_SUFFIX}}"
NPROC="${NPROC:-4}"
NUM_ITERATIONS="${NUM_ITERATIONS:-3814}"   # ~2.0B tokens, same as rounds 1-6
MHC_STREAMS="${MHC_STREAMS:-4}"
TORCHRUN="${TORCHRUN:-$REPO_DIR/../nanochat-moe/.venv/bin/torchrun}"

# Wait on a marker STRING IN A LOG, never a pid: §6 records kill -0 succeeding on
# a zombie, and three separate pgrep -f self-matches -- plus a fourth where the
# bracketed base_tra[i]n workaround self-matched because the pattern appeared in
# the checking shell's own eval'd command line. A file marker has neither failure
# mode, and run_arm never exits early so the marker always prints.
WAIT_FOR_LOG="${WAIT_FOR_LOG:-}"
WAIT_FOR_MARKER="${WAIT_FOR_MARKER:-Round 6 complete}"
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
    echo "Predecessor finished at $(date -Is); starting round 7"
fi

run_arm () {
    local tag="$1"; shift
    local log_file="$LOG_DIR/${tag}.log"
    if grep -q "^Total training time" "$log_file" 2>/dev/null; then
        echo "Skipping $tag (already completed)"; return
    fi
    echo "================================================================"
    echo "Starting $tag  ($(date -Is))"
    echo "  extra: $*"
    echo "================================================================"
    "$TORCHRUN" --standalone --nproc_per_node="$NPROC" -m scripts.base_train -- \
        --depth=20 --max-seq-len=2048 --device-batch-size=8 --total-batch-size=524288 \
        --num-iterations="$NUM_ITERATIONS" --target-param-data-ratio=-1 \
        --window-pattern=SSSL --mhc --mhc-num-streams="$MHC_STREAMS" \
        --engram --engram-layers=2,6 --engram-memory-dim=1280 \
        --engram-mhc-num-streams="$MHC_STREAMS" --engram-share-memory \
        --eval-every=500 --eval-tokens=10485760 \
        --core-metric-every=-1 --sample-every=-1 --save-every=-1 \
        --model-tag="d20-kvbo-${tag}-${RUN_SUFFIX}" \
        "$@" > "$log_file" 2>&1 || {
            echo "  !! $tag FAILED (exit $?)"; tail -20 "$log_file"; return
        }
    echo "  $tag done: $(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$log_file" | tail -1)"
}

# --- 1. the free arm, two seeds --------------------------------------------
# Nothing changes but 32 scalars, so this is the only arm in the study whose
# comparison to its baseline has no sizing caveat at all. Two seeds because the
# effect could be small and this arm class has a 1.85e-4 bar.
for seed in 0 1; do
    run_arm "count_gate_s${seed}" --engram-slot-multiplier=30 \
        --engram-count-gate --engram-seed="$seed"
done

# --- 2. k/v split, cheap sizing (rows nearly held, active handicapped) -----
run_arm kv16_s0 --engram-slot-multiplier=25 --engram-key-dim=16 --engram-seed=0

# --- 3. k/v split, symmetric sizing (exactly iso, pays a row doubling) -----
run_arm kv80_s0 --engram-slot-multiplier=15 --engram-key-dim=80 --engram-seed=0

echo
echo "Round 7 complete. Logs: $LOG_DIR"
printf "%-18s %-12s %s\n" arm min_bpb tok/sec
for tag in count_gate_s0 count_gate_s1 kv16_s0 kv80_s0; do
    f="$LOG_DIR/${tag}.log"; [ -f "$f" ] || continue
    printf "%-18s %-12s %s\n" "$tag" \
        "$(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$f" | tail -1)" \
        "$(grep -oP 'tok/sec: \K[0-9,]+' "$f" | tail -1)"
done
echo
echo "baseline engram_shared (n=3): 0.763784 / 0.763599 / see round 6 s2"
echo "  bar 1.85e-4. count_gate is iso on every axis; kv16 must beat ~0.0005"
echo "  (its row cost) and kv80 must beat ~0.00195 (a full row doubling)."
