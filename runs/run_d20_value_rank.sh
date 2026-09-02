#!/bin/bash

# d20 round 5: does a memory row need to be a FUNCTION of the hidden state?
#
# Every arm in rounds 1-4 varied what the memory is addressed BY. None varied
# what a row CONTAINS. Reading the study through the Hebbian-kernel-memory frame
# of arXiv:2607.10034 (docs/conditional_capacity_study.md §8.1) says that was the
# wrong axis, and that the study already measured the right one:
#
#   MoE / Mobius      values are functions of h    0.764727   beats dense by 0.0079
#   token Engram      constant rows, exact address 0.763599
#   10 contextual     constant rows, noisy address 0.771347   beats dense by 0.0013
#   dense                                          0.772614
#
# 0.0066 between the two contextual families, 7.5x the contextual error bar and
# larger than every effect §4 and §5.1 argue over. A constant-valued memory over
# a continuous key space discards all within-cell variation, and the cell is the
# only resolution a noisy address has.
#
# --engram-value-rank=1 makes row r a rank-1 map, read = W_out[r] @ act(W_in[r] q)
# with q = query_proj(rms_norm(h)) shared across rows. rank 0 is the Engram end of
# the ladder, large rank with few rows is the MoE end.
#
# THE PREDICTION IS DIFFERENTIAL, NOT "BIGGER IS BETTER". Rank costs addresses
# (a row is 2.6x the bytes, so 2.6x fewer rows at fixed total):
#   lsh_rank1    vs lsh    should improve a LOT   (pre-registered: > 0.003)
#   tokens_rank1 vs tokens should improve LITTLE or lose
# because a token n-gram is an exact discrete key for which a constant is the
# right payload, while a quantized hidden state is not. If BOTH gain equally then
# this is merely "expressive values help" and the §8.1 account is wrong. That is
# the outcome that would falsify it, and it is worth stating before the runs.
#
# Two seeds per arm, not four arms at one seed. §5.1 retracted three claims built
# on single-seed gaps; the contextual error bar is 8.8e-4, so a one-seed
# difference-of-differences would be uninterpretable. --engram-seed changes the
# hash multipliers and (for lsh) the LSH projections.
#
# Sizing, verified by building each config on meta rather than by arithmetic:
#   arm            total     active   flops/tok   vs its baseline
#   tokens         1828.4M   918.3M   3.135e9     --
#   tokens_rank1   1829.5M   918.7M   3.137e9     +0.06% / +0.04% / +0.06%
#   lsh            1841.8M   918.3M   3.135e9     --
#   lsh_rank1      1843.0M   918.7M   3.137e9     +0.07% / +0.04% / +0.07%
# query_dim differs between the arms (120 vs 140) because integer slot_multiplier
# granularity is coarse at these table sizes and iso-total mattered more; each arm
# is compared against its OWN baseline, so this is not in the differential, but it
# is an asymmetry worth knowing about.

set -euo pipefail

REPO_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"
cd "$REPO_DIR"

RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/runs/logs/d20_vrank-${RUN_SUFFIX}}"
NPROC="${NPROC:-4}"
NUM_ITERATIONS="${NUM_ITERATIONS:-3814}"   # ~2.0B tokens, same as rounds 1-4
MHC_STREAMS="${MHC_STREAMS:-4}"
TORCHRUN="${TORCHRUN:-$REPO_DIR/../nanochat-moe/.venv/bin/torchrun}"
# Optional extra rungs. Off by default: each is another n=1 arm, and §5.1's
# lesson is that breadth at one seed is what forced the retractions.
RUN_RANK4="${RUN_RANK4:-0}"        # lsh_rank4: is the effect monotone in rank?
RUN_SHARED="${RUN_SHARED:-0}"      # shared_rank1: the arm that could take rank 1

# /tmp gets wiped periodically and destroyed a previous run mid-flight
# (docs/conditional_capacity_study.md §6).
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/mnt/3fs/dots-pretrain/ningzhiyuan/local_exps/nsmb_out}"
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
        --model-tag="d20-vr-${tag}-${RUN_SUFFIX}" \
        "$@" > "$log_file" 2>&1 || {
            echo "  !! $tag FAILED (exit $?)"; tail -20 "$log_file"; return
        }
    echo "  $tag done: $(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$log_file" | tail -1)"
}

# --- the contextual side: predicted to gain a lot ---------------------------
# Run first. If the prediction fails here, nothing else in the script matters.
for seed in 0 1; do
    run_arm "lsh_rank1_s${seed}" \
        --engram-address-source=lsh --engram-lsh-bits=15 \
        --engram-slot-multiplier=4 \
        --engram-value-rank=1 --engram-value-query-dim=140 \
        --engram-seed="$seed"
done

# --- the token side: predicted to gain little or lose -----------------------
for seed in 0 1; do
    run_arm "tokens_rank1_s${seed}" \
        --engram-slot-multiplier=6 \
        --engram-value-rank=1 --engram-value-query-dim=120 \
        --engram-seed="$seed"
done

# --- opt-in rungs ----------------------------------------------------------
if [ "$RUN_RANK4" = "1" ]; then
    # rank 4 at iso-total: 4*(140+80) = 880 bytes/row vs 208, so slot 4 -> 1.
    run_arm lsh_rank4_s0 \
        --engram-address-source=lsh --engram-lsh-bits=15 \
        --engram-slot-multiplier=1 \
        --engram-value-rank=4 --engram-value-query-dim=140 \
        --engram-seed=0
fi
if [ "$RUN_SHARED" = "1" ]; then
    # Composes rank 1 with the one effect that survived repricing at 4.3x its
    # bar (cross-layer sharing). Only worth running if lsh_rank1 lands.
    for seed in 0 1; do
        run_arm "shared_rank1_s${seed}" \
            --engram-slot-multiplier=12 --engram-share-memory \
            --engram-value-rank=1 --engram-value-query-dim=120 \
            --engram-seed="$seed"
    done
fi

echo
echo "Round 5 complete. Logs: $LOG_DIR"
printf "%-22s %-12s %s\n" arm min_bpb tok/sec
for tag in lsh_rank1_s0 lsh_rank1_s1 tokens_rank1_s0 tokens_rank1_s1 \
           lsh_rank4_s0 shared_rank1_s0 shared_rank1_s1; do
    f="$LOG_DIR/${tag}.log"; [ -f "$f" ] || continue
    printf "%-22s %-12s %s\n" "$tag" \
        "$(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$f" | tail -1)" \
        "$(grep -oP 'tok/sec: \K[0-9,]+' "$f" | tail -1)"
done
echo
echo "baselines (rank 0, iso-total, same conditions):"
echo "  lsh 0.771347 | tokens 0.767512 | engram_shared 0.763784/0.763599 | dense 0.772614"
echo
echo "the test is the DIFFERENCE OF DIFFERENCES:"
echo "  (lsh_rank1 - 0.771347)  should be < -0.003"
echo "  (tokens_rank1 - 0.767512) should be ~0 or positive"
echo "  if both are equally negative, the §8.1 account is wrong"
