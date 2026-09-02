#!/bin/bash

# d20 round 17: two follow-ups on the round-16 composition result.
#
# Round 16 put the hash memory and the routed experts in one model and got the
# largest iso-parameter effect in the study:
#
#   combo_iso  0.760625  (n=3, sd 7.92e-4)  <- THE REFERENCE FOR BOTH ARMS
#   cg_dec     0.762987  (n=5, sd 4.89e-4)  = combo_iso - 0.00236, 4.66x
#   mobius     0.764727  (n=1, no seed knob)
#   dense      0.772614
#
# Two things the round could not tell us, one per arm.
#
# ARM 1: combo_mobius -- THE REAL MOBIUS RECIPE, WHICH HAS NEVER BEEN RUN HERE.
#
#   §1's "mobius" arm and round 16's combo BOTH used only `--moe-share-blocks=1`.
#   Neither used `--moe-shared-d-ff`. But the per-layer *private gated dense
#   expert* is precisely what upstream Intern-S2-Mobius adds on top of a shared
#   pool -- it is what makes a shared pool more than plain weight tying (see the
#   SharedExpert docstring in nano_scalemb/moe/block.py). So the study has been
#   calling an arm "mobius" for 16 rounds while omitting its defining component.
#
#   This arm adds it, and pays for it exactly:
#     one always-on private expert of width 640 = 10 layers x 2 x 1280 x 640
#                                              = 16.40M params
#     one routed expert from the shared pool    = 2 x 1280 x 640 = 1.6384M
#     so top_k 8->7 frees exactly one expert's worth of ACTIVE params, and
#        n_experts 357->347 frees exactly ten experts' worth of TOTAL params.
#
#   Verified on meta against combo_iso: total +0.000%, active -0.008%,
#   FLOPs/token -0.020%. This is the cleanest single-variable arm in the study --
#   cleaner than the round-15 head sweep (0.025% total).
#
#   THE QUESTION, in one line: at identical total, active and FLOPs, is one
#   ALWAYS-ON PRIVATE per-layer expert better than one more ROUTED expert from
#   the shared pool?
#
# ARM 2: combo_nogate -- does cg_dec still contribute once the MoE is present?
#
#   combo_iso already carries the decoupled count gate (round 16 used the full
#   cg_dec recipe). So the -0.00236 was measured as "cg_dec + MoE" vs "cg_dec",
#   and the gate's own -0.00090 has never been re-measured inside the combo. If
#   the MoE absorbs it, the combo's win is really "engram_shared + MoE" and §9.3
#   item 5 (cg_dec on a third base) is answered in the negative.
#
#   Removes only --engram-count-gate{,-decouple}: 64 scalars and one
#   non-persistent buffer. Iso to 0.000% on all three axes.
#
#   *** READ THE POWER LIMIT BEFORE READING THE RESULT. *** combo_iso's own sd is
#   7.92e-4 -- 1.6x cg_dec's 4.89e-4, i.e. adding the MoE INFLATED the arm's
#   sensitivity to the Engram hash draw. Against that bar, a new n=3 arm has a
#   Welch SE of 6.5e-4 and can only resolve ~0.0013 at 2x. The gate's standalone
#   effect is -0.00090, which is BELOW that. So:
#
#     - this arm CAN detect the gate mattering MORE inside the combo (>=0.0013)
#     - this arm CAN detect the gate HURTING inside the combo
#     - this arm CANNOT distinguish "absorbed by the MoE" from "still worth
#       0.0009". A flat result is genuinely uninformative on that question and
#       must be reported as such, not as "the gate does nothing".
#
#   It is queued anyway because the two outcomes it *can* resolve are both
#   interesting, and because it is the only way to stop quoting the combo as
#   "cg_dec + MoE" without ever having checked the "cg_dec" half.
#
# SIZING, all verified on meta (deltas vs combo_iso; the absolute totals below
# are the run-script convention, which differs from num_scaling_params() by a
# constant that cancels in every delta):
#   arm            total     active     FLOPs/token   vs combo_iso
#   combo_iso      1827.5M   922.91M    3.162e9       reference
#   combo_nogate   1827.5M   922.91M    3.162e9       0.000% / -0.000% / -0.000%
#   combo_mobius   1827.5M   922.83M    3.162e9       0.000% / -0.008% / -0.020%
#
# THE ERROR BAR IS STILL A LOWER BOUND on every arm here: --engram-seed permutes
# the Engram hash multipliers but does NOT reseed the MoE router (there is no
# --moe-seed and no global manual_seed in base_train, §0). All three seeds of
# each arm share one router draw. Building --moe-seed is §9.3 item 1 and is the
# thing that would make this whole family resolvable.
#
# PRE-REGISTERED. Reference combo_iso = 0.760625, n=3, sd 7.92e-4.
#
#   combo_mobius clearly below 0.7593  -> the private always-on expert beats a
#                                         routed one at equal cost; the study's
#                                         "mobius" arm was under-specified for 16
#                                         rounds and §1's ranking needs a footnote
#   combo_mobius ~ 0.7606              -> where the capacity lives does not
#                                         matter, only how much there is; this
#                                         also retires --moe-shared-d-ff as a knob
#   combo_mobius clearly above 0.7619  -> routed beats always-on at this budget,
#                                         and omitting the shared expert was the
#                                         right accident
#
#   combo_nogate clearly above 0.7619  -> the gate contributes MORE with the MoE
#                                         present than the -0.00090 it was worth
#                                         alone: a positive mechanism interaction
#   combo_nogate ~ 0.7606              -> UNRESOLVED by construction (see the
#                                         power limit above). Report as "cannot
#                                         see an effect of 0.0009 on this base",
#                                         NOT as "the gate is absorbed"
#   combo_nogate clearly below 0.7593  -> the gate HURTS once the MoE is there,
#                                         which would be the first negative
#                                         interaction found in the study

set -euo pipefail

REPO_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"
cd "$REPO_DIR"

RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/runs/logs/d20_combo2-${RUN_SUFFIX}}"
NPROC="${NPROC:-4}"
NUM_ITERATIONS="${NUM_ITERATIONS:-3814}"
MHC_STREAMS="${MHC_STREAMS:-4}"
TORCHRUN="${TORCHRUN:-$REPO_DIR/../nanochat-moe/.venv/bin/torchrun}"

WAIT_FOR_LOG="${WAIT_FOR_LOG:-}"
WAIT_FOR_MARKER="${WAIT_FOR_MARKER:-Round 16 complete}"
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
    echo "Predecessor finished at $(date -Is); starting round 17"
fi

# $1 tag, rest -> the arm's distinguishing flags. Everything shared by both arms
# lives here; dbs=8 on every arm (§6: dbs is worth -0.00184 by itself and must
# never vary within a comparison).
run_arm () {
    local tag="$1"; shift
    local log_file="$LOG_DIR/${tag}.log"
    if grep -q "^Total training time" "$log_file" 2>/dev/null; then
        echo "Skipping $tag (already completed)"; return
    fi
    echo "================================================================"
    echo "Starting $tag  ($(date -Is))"
    echo "  arm flags: $*"
    echo "================================================================"
    "$TORCHRUN" --standalone --nproc_per_node="$NPROC" -m scripts.base_train -- \
        --depth=20 --max-seq-len=2048 --device-batch-size=8 --total-batch-size=524288 \
        --num-iterations="$NUM_ITERATIONS" --target-param-data-ratio=-1 \
        --window-pattern=SSSL --mhc --mhc-num-streams="$MHC_STREAMS" \
        --engram --engram-layers=2,6 --engram-memory-dim=1280 \
        --engram-mhc-num-streams="$MHC_STREAMS" --engram-share-memory \
        --engram-slot-multiplier=15 \
        --moe --moe-backend=scattermoe --moe-d-ff-expert=640 \
        --moe-every=2 --moe-share-blocks=1 \
        --moe-balancer=loss_free --moe-log-every=200 \
        --eval-every=500 --eval-tokens=10485760 \
        --core-metric-every=-1 --sample-every=-1 --save-every=-1 \
        --model-tag="d20-combo2-${tag}-${RUN_SUFFIX}" \
        "$@" > "$log_file" 2>&1 || {
            echo "  !! $tag FAILED (exit $?)"; tail -20 "$log_file"; return
        }
    echo "  $tag done: $(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$log_file" | tail -1)"
}

# combo_mobius: 347 routed experts, top-7, + a private always-on 640-wide expert
#               per MoE layer. Exactly iso to combo_iso (see header).
MOBIUS_FLAGS=(--moe-experts=347 --moe-top-k=7 --moe-shared-d-ff=640
              --engram-count-gate --engram-count-gate-decouple)

# combo_nogate: combo_iso with the decoupled count gate removed. Nothing else
#               differs -- same 357 experts, same top-8, no shared expert.
NOGATE_FLAGS=(--moe-experts=357 --moe-top-k=8)

# Interleaved so a truncated run leaves both questions readable.
run_arm combo_mobius_s0 "${MOBIUS_FLAGS[@]}" --engram-seed=0
run_arm combo_nogate_s0 "${NOGATE_FLAGS[@]}" --engram-seed=0
run_arm combo_mobius_s1 "${MOBIUS_FLAGS[@]}" --engram-seed=1
run_arm combo_nogate_s1 "${NOGATE_FLAGS[@]}" --engram-seed=1
run_arm combo_mobius_s2 "${MOBIUS_FLAGS[@]}" --engram-seed=2
run_arm combo_nogate_s2 "${NOGATE_FLAGS[@]}" --engram-seed=2

echo
echo "Round 17 complete. Logs: $LOG_DIR"
printf "%-20s %-12s %s\n" arm min_bpb tok/sec
for tag in combo_mobius_s0 combo_mobius_s1 combo_mobius_s2 \
           combo_nogate_s0 combo_nogate_s1 combo_nogate_s2; do
    f="$LOG_DIR/${tag}.log"; [ -f "$f" ] || continue
    printf "%-20s %-12s %s\n" "$tag" \
        "$(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$f" | tail -1)" \
        "$(grep -oP 'tok/sec: \K[0-9,]+' "$f" | tail -1)"
done
echo
echo "reference: combo_iso 0.760625 (n=3, sd 7.92e-4: .760619/.759836/.761419)"
echo "           cg_dec    0.762987 (n=5, sd 4.89e-4)"
echo "           dense     0.772614"
echo
echo "Both arms are iso to combo_iso on total (0.000%), active (<=0.008%) and"
echo "FLOPs (<=0.020%). A new n=3 arm against combo_iso resolves only ~0.0013 at"
echo "2x, so combo_nogate CANNOT rule the gate's 0.00090 in or out -- read a flat"
echo "result there as 'below resolution', not as 'the gate is absorbed'."
echo
echo "also worth reading:"
echo "  - MoE balance stats (every 200 steps). combo_mobius routes top-7 of 347"
echo "    with an always-on expert alongside; if the private expert soaks up the"
echo "    load, max_vio should IMPROVE on combo_iso's 1.94 -> 8.20."
echo "  - the learned shared-expert gate: sigmoid(x . gate) starts at 0.5. Where"
echo "    it settles per layer says how much the model wants always-on capacity,"
echo "    and is readable from the checkpoint with no forward pass."
