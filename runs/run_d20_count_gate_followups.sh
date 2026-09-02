#!/bin/bash

# d20 round 8: follow up the one thing that worked.
#
# Round 7 result: --engram-count-gate on engram_shared gave 0.762644 / 0.762765
# (n=2, mean 0.762705) against engram_shared's 0.763884 (n=3, sd 3.5e-4). That is
# -0.00118 at 3.7x the conservative combined SE, with COMPLETE SEPARATION -- both
# count_gate seeds sit below all three baseline seeds, 0.00083 clear. New best in
# the study, on 32 scalars, iso on rows / total / active / FLOPs / throughput.
#
# The final checkpoint let us read what it learned, per head:
#   count_gate_scale (a)  order-2 heads -0.32   order-3 heads -0.25
#   count_gate_bias  (b)  order-2 heads -1.36   order-3 heads -2.04
# All 16 negative, i.e. the more often a row is addressed the LESS it is trusted
# -- Kneser-Ney backoff, learned from scratch. In the sigmoid tail
# w ~ 2*exp(b)*(1+rate)^a, so the learned inverse-frequency EXPONENT is ~ -0.3,
# and the sparser order-3 heads are discounted ~1.7x harder than order-2 at every
# frequency. Only `a` is meaningful: value_proj can apply its own per-head scale,
# so it could reproduce any fixed b, but not a frequency-DEPENDENT weight.
#
# Three arms, all exactly iso (verified on meta: total +0.00/+0.00/+0.01%,
# active +0.00% each), in decreasing expected value:
#
# 1. cg_n5 -- count_gate + n-gram orders 2,3,4,5 instead of 2,3.
#    §9.3 ranked longer orders as the top thing to build: it is the ONLY axis that
#    pays (does the address carry information h has discarded), and orders 4-5 are
#    exact surface patterns no 3-gram window can reach. Pairing it with the count
#    gate is the point -- with fixed weights a sparse 5-gram order dilutes the
#    read, but the gate demonstrably learns to discount the sparser order, so it
#    should be able to allocate trust across four orders instead of two.
#    Sizing accident worth recording: 32 heads x 40 dims at slot 30 is iso with
#    16 x 80 at slot 30, because doubling heads and halving head_dim leaves the
#    table bytes unchanged. max_ngram_size=4 is IMPOSSIBLE at memory_dim=1280
#    (24 heads does not divide 1280); 5 works because 32 does.
#
# 2. whiten_p03 -- --engram-readout-whiten --engram-readout-whiten-power=0.3.
#    Whitening imposes a FIXED inverse-frequency exponent with no per-head or
#    per-order freedom. I shipped it defaulting to power 1.0; the learned gate
#    says ~0.3, so 1.0 was ~3x too aggressive. This asks how much of the win is
#    just the exponent: if whiten_p03 ~= count_gate then the value is the
#    exponent and it costs ZERO parameters; if it falls short, the per-head and
#    per-order adaptivity is what matters. Either answer is worth having, and it
#    is the cheapest possible ablation of the winning arm.
#
# 3. count_gate_s2 -- third seed. §0's lesson is that n=2 underestimates sd
#    (engram_shared read 1.3e-4 at n=2 and 3.5e-4 at n=3, a factor of 2.6).
#    count_gate's n=2 range is 1.21e-4, suspiciously tight; if that survives a
#    third seed it is itself a finding (the gate would be stabilizing training,
#    like cross-layer sharing does), and if it does not, the 5.7x figure needs to
#    stay at the conservative 3.7x.
#
# NOT run, and why: count_gate + hybrid_shared_f75. The axes do not overlap
# (address information vs read reliability), so composition was tempting, but
# hybrid_shared_f75 starts 0.0012 BEHIND engram_shared, so the composition would
# have to beat that deficit before it says anything about composing. Test it on
# the better base or not at all.

set -euo pipefail

REPO_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"
cd "$REPO_DIR"

RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/runs/logs/d20_cgf-${RUN_SUFFIX}}"
NPROC="${NPROC:-4}"
NUM_ITERATIONS="${NUM_ITERATIONS:-3814}"   # ~2.0B tokens, same as rounds 1-7
MHC_STREAMS="${MHC_STREAMS:-4}"
TORCHRUN="${TORCHRUN:-$REPO_DIR/../nanochat-moe/.venv/bin/torchrun}"

# Poll a marker STRING IN A LOG, never a pid or a process name: §6 records kill -0
# succeeding on zombies and four separate pgrep/ps self-matches, including the
# bracketed base_tra[i]n form (the pattern is always in the checker's own argv).
WAIT_FOR_LOG="${WAIT_FOR_LOG:-}"
WAIT_FOR_MARKER="${WAIT_FOR_MARKER:-Round 7 complete}"
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
    echo "Predecessor finished at $(date -Is); starting round 8"
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
        --engram-slot-multiplier=30 \
        --eval-every=500 --eval-tokens=10485760 \
        --core-metric-every=-1 --sample-every=-1 --save-every=-1 \
        --model-tag="d20-cgf-${tag}-${RUN_SUFFIX}" \
        "$@" > "$log_file" 2>&1 || {
            echo "  !! $tag FAILED (exit $?)"; tail -20 "$log_file"; return
        }
    echo "  $tag done: $(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$log_file" | tail -1)"
}

# --- 1. the highest-upside arm: more n-gram orders, gated ------------------
run_arm cg_n5 --engram-count-gate --engram-max-ngram-size=5 --engram-seed=0

# --- 2. is the win just the exponent? (zero extra parameters) -------------
run_arm whiten_p03 --engram-readout-whiten --engram-readout-whiten-power=0.3 \
    --engram-seed=0

# --- 3. third seed on the new best ---------------------------------------
run_arm count_gate_s2 --engram-count-gate --engram-seed=2

echo
echo "Round 8 complete. Logs: $LOG_DIR"
printf "%-18s %-12s %s\n" arm min_bpb tok/sec
for tag in cg_n5 whiten_p03 count_gate_s2; do
    f="$LOG_DIR/${tag}.log"; [ -f "$f" ] || continue
    printf "%-18s %-12s %s\n" "$tag" \
        "$(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$f" | tail -1)" \
        "$(grep -oP 'tok/sec: \K[0-9,]+' "$f" | tail -1)"
done
echo
echo "references (all shared-table, slot 30, iso):"
echo "  count_gate    0.762644 / 0.762765  (n=2, mean 0.762705)  <- best"
echo "  engram_shared 0.763784 / 0.763599 / 0.764268 (n=3, mean 0.763884, sd 3.5e-4)"
echo "  mobius 0.764727 | dense 0.772614"
echo
echo "cg_n5 beats count_gate  -> longer orders add address information (§9.3 #1)"
echo "whiten_p03 ~= count_gate -> the win is the exponent, and it is free"
echo "whiten_p03 << count_gate -> per-head/per-order adaptivity is what matters"
