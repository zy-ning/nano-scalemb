#!/bin/bash

# d20 continuous-Engram, round 3: compose the two things that actually won, and
# tune the one knob that was never tuned.
#
# Where rounds 1-2 landed (2B tokens, d20, mHC-4, matched active params 918.3M):
#   engram_shared 0.763784   <- best overall: ONE memory table shared across layers
#   hybrid        0.766783   <- beat token addressing by -0.00073 (all 8 checkpoints)
#   tokens        0.767512
#   lsh           0.771347   |
#   pkm           0.771378   |  every arm that REPLACED token addressing
#   pq            0.772703   |  lost by +0.0038..+0.0052
#   dense         0.772614
#   lsh_n1        0.773514   <- removing the n-gram window is worse than no memory
#
# The pattern: contextual addressing as a REPLACEMENT loses (three different
# mechanisms, all ~+0.004), but as an ADDITION alongside token n-grams it wins.
# Consistent with the token n-gram injecting information the residual stream has
# discarded, which contextual addressing cannot substitute for -- only supplement.
#
# Arms:
#   hybrid_shared   The two winners composed: hybrid addressing + share_memory.
#                   Their effects were measured in different dimensions (what the
#                   memory is addressed BY vs whether layers share one memory), so
#                   they may add. NOTE this shares the DISCRETIZER as well as the
#                   table -- with per-layer projections the same hidden state
#                   would hit unrelated rows of the shared table at different
#                   depths, which is a bigger table, not a shared memory.
#   hybrid_frac25   Only 25% of heads keep token addressing (75% contextual).
#   hybrid_frac75   75% token, 25% contextual.
#                   frac=0.5 was a guess and beat tokens, so the optimum is
#                   somewhere in (0,1) rather than at the token end; these two
#                   bracket it. frac changes only WHICH head reads WHICH code
#                   source, so all three share identical geometry -- an unusually
#                   clean single-variable sweep.
#
# Sizing: every arm here is 1841.8M total / 918.3M active, identical to the
# `hybrid` reference (hybrid_shared uses slot=22 to offset having one table
# instead of two; -0.03%).

set -euo pipefail

REPO_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"
cd "$REPO_DIR"

RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/runs/logs/d20_ce_v3-${RUN_SUFFIX}}"
NPROC="${NPROC:-4}"
NUM_ITERATIONS="${NUM_ITERATIONS:-3814}"   # ~2.0B tokens, same as rounds 1-2
MHC_STREAMS="${MHC_STREAMS:-4}"
TORCHRUN="${TORCHRUN:-$REPO_DIR/../nanochat-moe/.venv/bin/torchrun}"

export TORCHDYNAMO_CACHE_SIZE_LIMIT="${TORCHDYNAMO_CACHE_SIZE_LIMIT:-64}"
export PYTHONPATH="${PYTHONPATH:-$REPO_DIR}"

mkdir -p "$LOG_DIR"

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
        --engram-address-source=hybrid --engram-lsh-bits=15 \
        --eval-every=500 --eval-tokens=10485760 \
        --core-metric-every=-1 --sample-every=-1 --save-every=-1 \
        --model-tag="d20-ce3-${tag}-${RUN_SUFFIX}" \
        "$@" > "$log_file" 2>&1 || {
            echo "  !! $tag FAILED (exit $?)"; tail -20 "$log_file"; return
        }
    echo "  $tag done: $(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$log_file" | tail -1)"
}

# the composition of the two winners -- highest-value arm remaining
run_arm hybrid_shared --engram-hybrid-token-head-frac=0.5 \
    --engram-share-memory --engram-slot-multiplier=22

# bracket the token/contextual head split
run_arm hybrid_frac25 --engram-hybrid-token-head-frac=0.25 --engram-slot-multiplier=11
run_arm hybrid_frac75 --engram-hybrid-token-head-frac=0.75 --engram-slot-multiplier=11

echo
echo "Round 3 complete. Logs: $LOG_DIR"
printf "%-16s %-12s %s\n" arm min_bpb tok/sec
for tag in hybrid_shared hybrid_frac25 hybrid_frac75; do
    f="$LOG_DIR/${tag}.log"; [ -f "$f" ] || continue
    printf "%-16s %-12s %s\n" "$tag" \
        "$(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$f" | tail -1)" \
        "$(grep -oP 'tok/sec: \K[0-9,]+' "$f" | tail -1)"
done
echo "reference: engram_shared 0.763784 | hybrid 0.766783 | tokens 0.767512 | dense 0.772614"
