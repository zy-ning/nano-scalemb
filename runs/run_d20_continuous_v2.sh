#!/bin/bash

# d20 continuous-Engram, round 2. Round 1 found that ALL contextual address
# sources lose to token hashing by a tight +0.0038..+0.0052 bpb:
#
#   tokens 0.767512 | lsh 0.771347 | pkm 0.771378 | pq 0.772703 | dense 0.772614
#
# and that pkm ~= lsh to 3e-5 despite pkm having fully learned differentiable
# addressing with a soft top-k read. Learning the addressing bought nothing.
# Diagnostics on the round-1 checkpoints found:
#   * LSH already spreads BETTER than tokens (row-uniqueness 0.994 vs 0.862) yet
#     loses -- so address capacity/entropy is not the binding constraint.
#   * LSH realizes only 9.2-9.8 of its 15 nominal bits (correlated sign bits).
#   * PQ collapsed the residual stream to eff_dim 1/1280. Self-inflicted: the
#     commitment loss pulls the hidden state toward its centroid. pq ~= dense,
#     i.e. a ~926M-param memory contributed nothing.
#   * Non-uniform VQ allocation would gain ~0.00 dB: subspace variances are
#     already balanced to within 1-20%.
#
# Each arm here targets one of those findings.
#
#   lsh_balanced  Re-centre each LSH bit on an EMA of its own projection mean.
#                 Attacks the wasted-entropy finding directly; measured to lift
#                 realized entropy 8.6 -> 9.9 of 15 bits (+59% distinct codes).
#   lsh_perhead   One independent latent per hash head, so the read is
#                 conditioned on num_heads x bits of h instead of the same bits
#                 broadcast to every head. NOTE: my original motivation for this
#                 ("heads are redundant") was WRONG -- distinct primes per head
#                 already decorrelate them by CRT. It still increases how much of
#                 h the code captures, which is the real (weaker) rationale.
#   pq_detached   PQ addressing off a stop-gradient hidden state, no commitment
#                 loss, per-head codes (which removes the 31-bit packing limit
#                 that forced the very coarse 2x64 quantizer). Tests whether
#                 round-1 PQ failed because of the collapse or because of VQ.
#   hybrid        Half the heads keep TOKEN addressing, half address by hidden
#                 state. This is the arm that tests the leading hypothesis: the
#                 token path INJECTS information h has discarded (exact recent
#                 token identity), whereas contextual addressing only re-reads
#                 what is already in h. If that is right, semantics should ADD
#                 to exact match rather than replace it -- so hybrid is the only
#                 arm here with a real chance of BEATING tokens.
#
# Sizing: matched on active params (918.3M, identical across all arms) and
# total to within +1.20% (the three LSH-geometry arms) / -0.00% (pq_detached).
# Baselines to compare against, same conditions, same 2B-token budget:
#   tokens 0.767512 | lsh 0.771347 | dense 0.772614 | engram_shared 0.763784

set -euo pipefail

REPO_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"
cd "$REPO_DIR"

RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/runs/logs/d20_ce_v2-${RUN_SUFFIX}}"
NPROC="${NPROC:-4}"
NUM_ITERATIONS="${NUM_ITERATIONS:-3814}"   # ~2.0B tokens, same as round 1
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
        --eval-every=500 --eval-tokens=10485760 \
        --core-metric-every=-1 --sample-every=-1 --save-every=-1 \
        --model-tag="d20-ce2-${tag}-${RUN_SUFFIX}" \
        "$@" > "$log_file" 2>&1 || {
            echo "  !! $tag FAILED (exit $?)"; tail -20 "$log_file"; return
        }
    echo "  $tag done: $(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$log_file" | tail -1)"
}

# The arm most likely to beat tokens: keep exact-match, add semantics.
run_arm hybrid --engram-address-source=hybrid --engram-lsh-bits=15 \
    --engram-hybrid-token-head-frac=0.5 --engram-slot-multiplier=11

# Fix the wasted-entropy finding.
run_arm lsh_balanced --engram-address-source=lsh --engram-lsh-bits=15 \
    --engram-lsh-balance --engram-slot-multiplier=11

# Was round-1 PQ's failure the collapse, or VQ itself?
run_arm pq_detached --engram-address-source=pq --engram-pq-codebook-size=256 \
    --engram-address-latents-per-head --engram-pq-detach-encoder \
    --engram-slot-multiplier=1375

# More of h captured per read.
run_arm lsh_perhead --engram-address-source=lsh --engram-lsh-bits=15 \
    --engram-address-latents-per-head --engram-slot-multiplier=11

echo
echo "Round 2 complete. Logs: $LOG_DIR"
printf "%-14s %-12s %s\n" arm min_bpb tok/sec
for tag in hybrid lsh_balanced pq_detached lsh_perhead; do
    f="$LOG_DIR/${tag}.log"; [ -f "$f" ] || continue
    printf "%-14s %-12s %s\n" "$tag" \
        "$(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$f" | tail -1)" \
        "$(grep -oP 'tok/sec: \K[0-9,]+' "$f" | tail -1)"
done
echo "round-1: tokens 0.767512 | lsh 0.771347 | pkm 0.771378 | pq 0.772703 | dense 0.772614"
