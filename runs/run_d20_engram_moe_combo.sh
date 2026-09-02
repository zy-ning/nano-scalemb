#!/bin/bash

# d20 round 16: put the hash memory and the routed experts in the SAME model.
#
# §1 compared them and concluded "the spread between mechanisms is smaller than the
# gap to dense" -- all four conditional arms beat dense by 0.005-0.009 and landed
# within 0.0038 of each other. Then 14 rounds went by and **they were never
# combined**. That is the largest untested question the study's own framing implies.
#
#   mobius  0.764727   routed experts, one shared pool of 640
#   cg_dec  0.762987   hash memory, shared table + decoupled count gate (n=5)
#   dense   0.772614
#
# The two mechanisms should be complementary if the account in §9.1-§9.2 is right:
# a token n-gram address carries *external* information (exact surface form that the
# residual stream has discarded), while an MoE expert is a learned input-dependent
# *function* of the hidden state. Different information, different machinery. If
# they are instead two ways of buying the same conditional capacity, combining them
# buys nothing.
#
# WHY THIS AND NOT AN ELEVENTH MECHANISM. Ten new mechanisms have been tried inside
# the "hash memory with learned values" design -- better hash, more heads, more
# rows, rank-k values, separate k/v, Fourier gates, longer orders, consistent
# addressing, fixed-exponent whitening, per-head latents -- and exactly one
# resolved. The base rate for an eleventh is poor. This round instead composes two
# things already known to work.
#
# TWO ARMS, TWO DIFFERENT QUESTIONS.
#
#   combo_iso  (slot 15 + 357 experts, 1827.5M total)
#     Matched to cg_dec's 1828.0M within 0.03%. Splits the same parameter budget
#     44/56 between table and experts instead of putting all of it in one.
#     THE DISCIPLINED QUESTION: at a fixed budget, is a mix better than either pure?
#     Directly comparable to cg_dec 0.762987 (n=5, sd 4.9e-4) and mobius 0.764727.
#
#   combo_full (slot 30 + 640 experts, 2746.3M total)
#     Both mechanisms at full reference size, so +50% total over either alone.
#     THE CEILING QUESTION: do the effects ADD when neither is starved? A win here
#     is partly "more parameters help" and must be read that way, but it bounds how
#     much is available from the combination at all.
#
# SIZING, verified on meta:
#   arm          total     active     FLOPs/token
#   cg_dec       1828.0M   918.34M    3.135e9
#   mobius       1820.0M   909.87M    3.084e9
#   combo_iso    1827.5M   922.91M    3.162e9   (-0.03% / +0.50% / +0.87% vs cg_dec)
#   combo_full   2746.3M   926.53M    3.184e9
#
# NOTE THE COMBO'S SMALL ADVANTAGE, and read results accordingly: +0.50% active and
# +0.87% FLOPs over cg_dec, all of it the MoE router (1280 x n_experts, read by 10
# layers). It cannot be removed without changing moe_every away from the mobius
# reference. So a combo WIN is partly confounded by that half-percent; a combo LOSS
# is decisive.
#
# THE ERROR BAR IS A LOWER BOUND. `--engram-seed` permutes the Engram's hash
# multipliers but does NOT reseed the MoE router -- there is no `--moe-seed` and no
# global manual_seed in base_train (§0). So seed replicates here sample the Engram's
# draw plus CUDA nondeterminism, and NOT a structurally different MoE. The measured
# sd therefore understates the true variance, and any "Nx the bar" figure for these
# arms is optimistic. Fixing that is the `--moe-seed` item in §9.3.
#
# PRE-REGISTERED. cg_dec is the reference to beat, at 0.762987.
#   combo_iso clearly below 0.7630   -> the mechanisms are complementary, and the
#                                       best use of a budget is a mix
#   combo_iso ~ 0.7630               -> interchangeable; capacity is capacity, and
#                                       §1's "mechanism matters less than having it"
#                                       extends to combinations
#   combo_iso clearly above 0.7630   -> the memory alone is the better use of the
#                                       budget, and splitting it costs
#   combo_full below combo_iso by ~the amount its extra parameters would buy on
#     either mechanism alone -> nothing new; the combination is not synergistic

set -euo pipefail

REPO_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"
cd "$REPO_DIR"

RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/runs/logs/d20_combo-${RUN_SUFFIX}}"
NPROC="${NPROC:-4}"
NUM_ITERATIONS="${NUM_ITERATIONS:-3814}"
MHC_STREAMS="${MHC_STREAMS:-4}"
TORCHRUN="${TORCHRUN:-$REPO_DIR/../nanochat-moe/.venv/bin/torchrun}"

WAIT_FOR_LOG="${WAIT_FOR_LOG:-}"
WAIT_FOR_MARKER="${WAIT_FOR_MARKER:-Round 15 complete}"
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
    echo "Predecessor finished at $(date -Is); starting round 16"
fi

# $1 tag, $2 engram slot, $3 n_experts, rest -> extra flags
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
    # dbs=8 on every arm (§6: dbs is worth -0.00184 and must never vary within a
    # comparison). Watch combo_full for OOM at 2746M.
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
        --model-tag="d20-combo-${tag}-${RUN_SUFFIX}" \
        "$@" > "$log_file" 2>&1 || {
            echo "  !! $tag FAILED (exit $?)"; tail -20 "$log_file"; return
        }
    echo "  $tag done: $(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$log_file" | tail -1)"
}

# interleaved so a truncated run leaves both questions readable
run_arm combo_iso_s0  15 357 --engram-seed=0
run_arm combo_full_s0 30 640 --engram-seed=0
run_arm combo_iso_s1  15 357 --engram-seed=1
run_arm combo_full_s1 30 640 --engram-seed=1
run_arm combo_iso_s2  15 357 --engram-seed=2

echo
echo "Round 16 complete. Logs: $LOG_DIR"
printf "%-18s %-12s %s\n" arm min_bpb tok/sec
for tag in combo_iso_s0 combo_iso_s1 combo_iso_s2 combo_full_s0 combo_full_s1; do
    f="$LOG_DIR/${tag}.log"; [ -f "$f" ] || continue
    printf "%-18s %-12s %s\n" "$tag" \
        "$(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$f" | tail -1)" \
        "$(grep -oP 'tok/sec: \K[0-9,]+' "$f" | tail -1)"
done
echo
echo "references, all shared-pool:"
echo "  cg_dec  0.762987  (n=5, sd 4.9e-4)  1828.0M total, 918.34M active  <- to beat"
echo "  mobius  0.764727  (n=1, no seed knob) 1820.0M / 909.87M"
echo "  dense   0.772614"
echo
echo "combo_iso is matched to cg_dec on total (-0.03%) but carries +0.50% active and"
echo "+0.87% FLOPs from the router, so a WIN is partly confounded and a LOSS is not."
echo "Its seed sd is a LOWER BOUND: --engram-seed does not reseed the MoE router."
echo
echo "also worth reading: the MoE balance stats (maxvio / regret, logged every 200"
echo "steps). If the router collapses when an Engram is present, that is a"
echo "mechanism-interaction finding independent of the bpb."
