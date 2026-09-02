#!/bin/bash

# d20 continuous-Engram, round 4: the open questions from
# docs/conditional_capacity_study.md §5, in priority order.
#
# Standings after 17 arms (2B tokens, d20, mHC-4, matched active 918.3M):
#   engram_shared 0.763784 (best) | mobius 0.764727 | hybrid_frac75 0.765593
#   moe 0.766529 | hybrid_frac50 0.766783 | hybrid_shared 0.767041
#   tokens 0.767512 | hybrid_frac25 0.768086 | lsh 0.771347 | dense 0.772614
#
# The account being tested: address granularity has an interior optimum, driven by
# write reuse. Zipfian token n-grams hit the same rows thousands of times so those
# slots learn a lot; high-entropy codes spread writes thin so no slot learns much.
# Every arm so far pushed toward FINER addressing and got worse; the token/hidden
# mix peaks at 75/25.
#
#   lsh_b8      LSH with only 256 codes/position (vs 32768 at b15) -- COARSER than
#   lsh_b12     the token alphabet (23686). THE FALSIFICATION TEST: nothing has
#               tested coarser addressing, so the account has never been at risk.
#               If bpb improves as bits drop, reuse is confirmed as the mechanism.
#               If it degrades monotonically, the account is wrong and the token
#               advantage is about something else (probably information content).
#   frac875     87.5% token heads (7 of 8), i.e. between the measured optimum at
#               75% and the turn at 100%. The three interior points are near-linear
#               (-0.0013, -0.0012 per 25%) so the true minimum may not be at 75%.
#   shared_f75  engram_shared + frac75. hybrid_shared FAILED to compose at frac50
#               (§4.2): contextual heads write layer-specific content into the
#               shared table and dilute the property that made sharing work. That
#               mechanism predicts the damage shrinks with the contextual share --
#               at 25% it may compose where it did not at 50%. This is the only
#               untested route to beating 0.763784. NOTE it is a post-hoc rescue of
#               a failed prediction, so treat a win here more sceptically than the
#               others.
#   *_seed1     Replicates of the two best arms with --engram-seed=1, which changes
#               the hash multipliers AND the LSH projections (seed + 1009*layer_id).
#               There is no global torch seed in base_train, so a plain re-run only
#               samples CUDA nondeterminism (that is what the 4.3e-5 noise floor
#               measured). Varying the hash gives a structurally different sample.
#
# Sizing: all arms 1841.2-1842.3M total (within +/-0.03%) and 918.3M active.

set -euo pipefail

REPO_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"
cd "$REPO_DIR"

RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/runs/logs/d20_ce_v4-${RUN_SUFFIX}}"
NPROC="${NPROC:-4}"
NUM_ITERATIONS="${NUM_ITERATIONS:-3814}"   # ~2.0B tokens, same as rounds 1-3
MHC_STREAMS="${MHC_STREAMS:-4}"
TORCHRUN="${TORCHRUN:-$REPO_DIR/../nanochat-moe/.venv/bin/torchrun}"

# Durable base dir: /tmp gets wiped periodically and destroyed a previous run
# mid-flight (see docs/conditional_capacity_study.md §6).
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
        --model-tag="d20-ce4-${tag}-${RUN_SUFFIX}" \
        "$@" > "$log_file" 2>&1 || {
            echo "  !! $tag FAILED (exit $?)"; tail -20 "$log_file"; return
        }
    echo "  $tag done: $(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$log_file" | tail -1)"
}

# --- 1. THE FALSIFICATION TEST: coarser-than-token addressing ---------------
run_arm lsh_b8 --engram-address-source=lsh --engram-lsh-bits=8 \
    --engram-slot-multiplier=1409
run_arm lsh_b12 --engram-address-source=lsh --engram-lsh-bits=12 \
    --engram-slot-multiplier=88

# --- 2. refine the token/hidden mix optimum ---------------------------------
run_arm hybrid_frac875 --engram-address-source=hybrid --engram-lsh-bits=15 \
    --engram-hybrid-token-head-frac=0.875 --engram-slot-multiplier=11

# --- 3. the only untested route to a new best ------------------------------
run_arm hybrid_shared_f75 --engram-address-source=hybrid --engram-lsh-bits=15 \
    --engram-hybrid-token-head-frac=0.75 --engram-share-memory \
    --engram-slot-multiplier=22

# --- 4. replicates of the two best arms (structural: different hash) -------
run_arm engram_shared_seed1 --engram-slot-multiplier=30 --engram-share-memory \
    --engram-seed=1
run_arm hybrid_frac75_seed1 --engram-address-source=hybrid --engram-lsh-bits=15 \
    --engram-hybrid-token-head-frac=0.75 --engram-slot-multiplier=11 --engram-seed=1

echo
echo "Round 4 complete. Logs: $LOG_DIR"
printf "%-22s %-12s %s\n" arm min_bpb tok/sec
for tag in lsh_b8 lsh_b12 hybrid_frac875 hybrid_shared_f75 engram_shared_seed1 hybrid_frac75_seed1; do
    f="$LOG_DIR/${tag}.log"; [ -f "$f" ] || continue
    printf "%-22s %-12s %s\n" "$tag" \
        "$(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$f" | tail -1)" \
        "$(grep -oP 'tok/sec: \K[0-9,]+' "$f" | tail -1)"
done
echo
echo "reference: engram_shared 0.763784 | mobius 0.764727 | hybrid_frac75 0.765593"
echo "           tokens 0.767512 | lsh(b15) 0.771347 | dense 0.772614"
