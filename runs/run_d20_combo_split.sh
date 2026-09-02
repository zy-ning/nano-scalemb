#!/bin/bash

# d20 round 18: sweep the table/expert budget split inside the combo.
#
# WHY THIS IS THE LAST CHEAP QUESTION IN THE COMBO. Round 16 found the largest
# iso-parameter effect in the study by putting the hash memory and the routed
# experts in one model (combo_iso, -0.00236 vs cg_dec at 4.66x, complete seed
# separation). Round 17 then tried the two obvious STRUCTURAL follow-ups and both
# came back negative: the real Mobius component (a private always-on expert per
# layer) at 0.65x, and removing the count gate at 2.08x confirming the gate merely
# survives rather than adding. What is left is not a new mechanism but a knob:
# combo_iso tested exactly ONE budget division and it was the first one tried.
#
# THIS IS INTERPOLATION, NOT EXTRAPOLATION -- both endpoints are already measured:
#
#   split (table/experts)   arm            min bpb              n
#   100 / 0                 cg_dec         0.762987 (sd 4.9e-4)  5
#    73 / 27                split_tbl      <- this round
#    44 / 56                combo_iso      0.760625 (sd 7.9e-4)  3
#    26 / 74                split_exp      <- this round
#     0 / 100               mobius         0.764727 (no seed knob) 1
#
# The interior point beats BOTH endpoints, so the curve provably has an interior
# optimum. The only question is whether 44/56 is near its peak or merely the first
# point sampled. Two flanking points turn one measurement into a shape.
#
# SIZING, verified on meta. Active params are almost invariant here by
# construction: the Engram's active read depends on memory_dim (not on how many
# rows the table has), and expert_active depends on top_k (not on how many experts
# exist). So moving the split moves TOTAL storage while leaving the per-token read
# essentially fixed -- which is exactly the comparison we want.
#
#   arm         slot  experts   total       vs ref     active      vs ref    split
#   combo_iso     15      357   2330.85M    ref        1426.23M    ref       43.7/56.3
#   split_tbl     25      172   2330.72M    -0.005%    1423.86M    -0.166%   72.9/27.1
#   split_exp      9      468   2330.92M    +0.003%    1427.65M    +0.100%   26.3/73.7
#
# (Absolute totals are the num_scaling_params view; the run's own header prints a
# figure lower by a constant ~503M. The constant cancels in every delta.)
#
# THE ONE CONFOUND THAT CANNOT BE REMOVED. `router_active` is 1280 x n_experts read
# by 10 layers, and n_experts IS the independent variable. So the arm with more
# experts necessarily carries slightly more active params. Quoted in ABSOLUTE terms,
# because the percentage depends on which param convention is used:
#
#   arm         experts   active (run header)   vs combo_iso
#   split_tbl       172   920.5M                -2.4M
#   combo_iso       357   922.9M                ref
#   split_exp       468   924.3M (predicted)    +1.4M
#
# Spread ~3.8M active across the sweep. It is small -- for scale, +910M of table
# bought only -0.00056 (§9.2b), so 3.8M should be worth ~1e-6 -- but it is
# systematic and it favours split_exp. Note it when reading any split_exp win.
#
# *** POWER LIMIT. READ THIS BEFORE READING THE RESULT. *** combo_iso's sd is
# 7.92e-4, so a new n=3 arm against it has a Welch SE of ~6.5e-4 and resolves only
# ~0.0013 at 2x. Interpolating the five points above, a flank one step from the
# optimum plausibly costs only 0.0005-0.0010 -- BELOW that. So:
#
#   - the question "is either flank BETTER than 44/56" IS answerable, and it is the
#     actionable one (it would say move the split);
#   - the question "how much worse is each flank" probably is NOT resolvable per
#     arm, and must not be reported as if it were;
#   - the SHAPE across five points (two of them far-away endpoints with their own
#     n=5 and n=3 bars) is more informative than any single flank-vs-centre gap,
#     and is the intended read.
#
# Seed sd remains a LOWER BOUND on every arm: --engram-seed permutes the Engram
# hash multipliers but does NOT reseed the MoE router (no --moe-seed exists, §0).
# Round 17 demonstrated this directly -- combo_nogate_s0's first logged router
# stats were byte-identical to combo_iso_s0's (§9.2g).
#
# PRE-REGISTERED. Reference combo_iso = 0.760625 (n=3, sd 7.92e-4).
#
#   EITHER flank clearly below 0.7593  -> 44/56 is NOT the optimum; the split is a
#                                         live knob and the curve should be walked
#                                         further in that direction
#   BOTH flanks ~0.7606               -> the optimum is broad and flat; the split
#                                         does not need tuning, and "have both
#                                         mechanisms" is the whole of the effect.
#                                         This retires the split as a knob.
#   BOTH flanks clearly above 0.7619  -> 44/56 sits on a sharp peak, which would be
#                                         surprising for a first guess and would
#                                         itself deserve a finer sweep
#   ASYMMETRIC (one flank flat, one worse) -> the informative outcome even without
#                                         per-arm resolution: it says which
#                                         direction the peak lies in

set -euo pipefail

REPO_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"
cd "$REPO_DIR"

RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/runs/logs/d20_split-${RUN_SUFFIX}}"
NPROC="${NPROC:-4}"
NUM_ITERATIONS="${NUM_ITERATIONS:-3814}"
MHC_STREAMS="${MHC_STREAMS:-4}"
TORCHRUN="${TORCHRUN:-$REPO_DIR/../nanochat-moe/.venv/bin/torchrun}"

WAIT_FOR_LOG="${WAIT_FOR_LOG:-}"
WAIT_FOR_MARKER="${WAIT_FOR_MARKER:-Round 17 complete}"
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
    echo "Predecessor finished at $(date -Is); starting round 18"
fi

# $1 tag, $2 engram slot_multiplier, $3 n_experts, rest -> extra flags.
# Everything else is combo_iso's recipe held fixed; dbs=8 on every arm (§6: dbs is
# worth -0.00184 by itself and must never vary within a comparison).
run_arm () {
    local tag="$1"; local slot="$2"; local nexp="$3"; shift 3
    local log_file="$LOG_DIR/${tag}.log"
    if grep -q "^Total training time" "$log_file" 2>/dev/null; then
        echo "Skipping $tag (already completed)"; return
    fi
    echo "================================================================"
    echo "Starting $tag  ($(date -Is))  engram slot=$slot  moe experts=$nexp"
    echo "  extra: $*"
    echo "================================================================"
    "$TORCHRUN" --standalone --nproc_per_node="$NPROC" -m scripts.base_train -- \
        --depth=20 --max-seq-len=2048 --device-batch-size=8 --total-batch-size=524288 \
        --num-iterations="$NUM_ITERATIONS" --target-param-data-ratio=-1 \
        --window-pattern=SSSL --mhc --mhc-num-streams="$MHC_STREAMS" \
        --engram --engram-layers=2,6 --engram-memory-dim=1280 \
        --engram-mhc-num-streams="$MHC_STREAMS" --engram-share-memory \
        --engram-slot-multiplier="$slot" \
        --engram-count-gate --engram-count-gate-decouple \
        --moe --moe-backend=scattermoe --moe-experts="$nexp" --moe-top-k=8 \
        --moe-d-ff-expert=640 --moe-every=2 --moe-share-blocks=1 \
        --moe-balancer=loss_free --moe-log-every=200 \
        --eval-every=500 --eval-tokens=10485760 \
        --core-metric-every=-1 --sample-every=-1 --save-every=-1 \
        --model-tag="d20-split-${tag}-${RUN_SUFFIX}" \
        "$@" > "$log_file" 2>&1 || {
            echo "  !! $tag FAILED (exit $?)"; tail -20 "$log_file"; return
        }
    echo "  $tag done: $(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$log_file" | tail -1)"
}

# Interleaved so a truncated run still leaves BOTH flanks readable -- which is the
# whole point of the round, since the shape needs both sides.
run_arm split_tbl_s0 25 172 --engram-seed=0
run_arm split_exp_s0  9 468 --engram-seed=0
run_arm split_tbl_s1 25 172 --engram-seed=1
run_arm split_exp_s1  9 468 --engram-seed=1
run_arm split_tbl_s2 25 172 --engram-seed=2
run_arm split_exp_s2  9 468 --engram-seed=2

echo
echo "Round 18 complete. Logs: $LOG_DIR"
printf "%-16s %-12s %s\n" arm min_bpb tok/sec
for tag in split_tbl_s0 split_tbl_s1 split_tbl_s2 split_exp_s0 split_exp_s1 split_exp_s2; do
    f="$LOG_DIR/${tag}.log"; [ -f "$f" ] || continue
    printf "%-16s %-12s %s\n" "$tag" \
        "$(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$f" | tail -1)" \
        "$(grep -oP 'tok/sec: \K[0-9,]+' "$f" | tail -1)"
done
echo
echo "the five-point curve (table% / expert% of the conditional budget):"
echo "  100/0   cg_dec     0.762987  (n=5, sd 4.9e-4)"
echo "   73/27  split_tbl  <- this round, slot 25 + 172 experts"
echo "   44/56  combo_iso  0.760625  (n=3, sd 7.9e-4)   <- the point to beat"
echo "   26/74  split_exp  <- this round, slot 9 + 468 experts"
echo "    0/100 mobius     0.764727  (n=1, no seed knob)"
echo
echo "READ THE SHAPE, NOT THE INDIVIDUAL GAPS. A new n=3 arm resolves only ~0.0013"
echo "against combo_iso, and a one-step flank plausibly costs 0.0005-0.0010. The"
echo "answerable question is 'is either flank BETTER'; 'how much worse' is not."
echo
echo "confound that cannot be removed: router_active scales with n_experts, so"
echo "split_exp carries +0.100% active and split_tbl -0.166% (spread 0.27%, ~3.8M)."
echo "It systematically favours split_exp. For scale, +910M of table bought 0.00056."
