#!/bin/bash

# d20 round 9: does the count gate generalize beyond the shared table?
#
# --engram-count-gate has only ever been tested on engram_shared, where it won:
# 0.762644 / 0.762765 (n=2, mean 0.762705) vs 0.763884 (n=3, sd 3.5e-4), i.e.
# -0.00118 at 3.7x with complete separation across all five runs, on 32 scalars,
# iso on rows / total / active / FLOPs / throughput. Best result in the study.
#
# Everything §9 says rests on that one configuration. If someone reads §9 and adds
# a count gate to a per-layer n-gram memory, does it work? This is the arm that
# answers it, and it is the cheapest way to find out whether the headline claim is
# about hash memories in general or about *shared* hash memories in particular.
#
# There is a real reason to expect an interaction. The shared table is read by
# BOTH Engram layers, so its rows see ~2x the hit rate and a different frequency
# distribution than a per-layer table's; and sharing already collapses seed
# variance 27x, so the gate might be doing something the sharing had partly done.
#
# PRE-REGISTERED: the gate should help per-layer AT LEAST as much as it helped
# shared, and plausibly more. Mechanism: a per-layer table has half the rows of
# the shared one (355k vs 710k per head at iso-total), so it collides more, and
# backoff is worth more where collisions are worse. Row count is the measured
# binding constraint (0.00195 per doubling, 12x its bar), and the gate's learned
# behaviour was frequency-dependent downweighting -- exactly a collision remedy.
#   delta ~= -0.0012 or larger  -> the gate is a general property of hash memories
#   delta ~= 0                  -> it only works with cross-layer sharing, and §9
#                                  must say so
#   delta clearly smaller       -> partial; sharing did some of the work
#
# Baseline is `engram` 0.767555 / `tokens` 0.767512 -- the same config run twice
# from different scripts, so 4.3e-5 apart, which pins the same-config noise but
# NOT the seed variance. Treat the comparison against an assumed
# engram_shared-class sd of 3.5e-4: a -0.0012 delta would land at ~3x.
#
# Sizing verified on meta: 1828.4M total / 918.3M active, identical to `engram`.
# The gate adds 32 scalars and one float buffer; nothing else moves.

set -euo pipefail

REPO_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"
cd "$REPO_DIR"

RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/runs/logs/d20_cggen-${RUN_SUFFIX}}"
NPROC="${NPROC:-4}"
NUM_ITERATIONS="${NUM_ITERATIONS:-3814}"
MHC_STREAMS="${MHC_STREAMS:-4}"
TORCHRUN="${TORCHRUN:-$REPO_DIR/../nanochat-moe/.venv/bin/torchrun}"

# Marker string in the predecessor's log, never a pid or process name (§6).
WAIT_FOR_LOG="${WAIT_FOR_LOG:-}"
WAIT_FOR_MARKER="${WAIT_FOR_MARKER:-Round 8 complete}"
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
    echo "Predecessor finished at $(date -Is); starting round 9"
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
        --engram-mhc-num-streams="$MHC_STREAMS" \
        --eval-every=500 --eval-tokens=10485760 \
        --core-metric-every=-1 --sample-every=-1 --save-every=-1 \
        --model-tag="d20-cggen-${tag}-${RUN_SUFFIX}" \
        "$@" > "$log_file" 2>&1 || {
            echo "  !! $tag FAILED (exit $?)"; tail -20 "$log_file"; return
        }
    echo "  $tag done: $(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$log_file" | tail -1)"
}

# Per-layer tables (no --engram-share-memory), slot 15 so total matches `engram`.
run_arm count_gate_perlayer --engram-slot-multiplier=15 \
    --engram-count-gate --engram-seed=0

echo
echo "Round 9 complete. Logs: $LOG_DIR"
printf "%-22s %-12s %s\n" arm min_bpb tok/sec
for tag in count_gate_perlayer; do
    f="$LOG_DIR/${tag}.log"; [ -f "$f" ] || continue
    printf "%-22s %-12s %s\n" "$tag" \
        "$(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$f" | tail -1)" \
        "$(grep -oP 'tok/sec: \K[0-9,]+' "$f" | tail -1)"
done
echo
echo "per-layer baseline: engram 0.767555 / tokens 0.767512  (same config twice)"
echo "shared-table delta for reference: count_gate 0.762705 - engram_shared 0.763884"
echo "                                = -0.00118"
echo "read the learned gate from the checkpoint and compare the per-head scales:"
echo "  \$NANOCHAT_BASE_DIR/base_checkpoints/d20-cggen-count_gate_perlayer-*/model_003814.pt"
echo "  keys engram_modules.{2,6}.multi_head_embedding.count_gate_{scale,bias}"
echo "  (per-layer, so TWO independent gates -- do both layers learn a<0?)"
