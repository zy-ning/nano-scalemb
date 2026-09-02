#!/bin/bash

# d20 round 14: harden the only mechanism that has ever resolved on this
# architecture, and check whether it transfers off the arm it was found on.
#
# `cg_dec` (--engram-count-gate --engram-count-gate-decouple) on the shared table:
#   n=3, mean 0.762826, sd 4.07e-4, -0.00106 vs engram_shared at 3.43x.
# It is the only resolved mechanism in 63 runs, and the way it won is unusual: the
# mean is indistinguishable from the plain count gate (0.63x), but the sd is 1.8x
# tighter (7.50e-4 -> 4.07e-4), which is what lifted a 1.57x non-result to 3.43x.
# Decoupling did not make the gate better, it made it measurable.
#
# TWO REASONS THAT NEEDS HARDENING BEFORE ANYONE BUILDS ON IT.
#
# 1. The claim rests on an sd measured at n=3, and this study has watched n=2 sd
#    estimates move by 2.6x, 8.8x, 12.8x and 1.4x when a third seed landed (§0).
#    n=3 is better but not safe. Seeds 3 and 4 take the shared arm to **n=5**,
#    where the sd estimate is finally worth quoting. Note what is really being
#    tested: not the mean (3.43x -> 3.92x is a modest gain) but whether
#    sd = 4.07e-4 SURVIVES. If it inflates to gate-class 7.5e-4 the result reverts
#    to ~1.9x and joins the pile.
# 2. The plain count gate turned out to be **shared-table-specific**: -0.00075 at
#    1.57x on the shared table but -0.00080 at only 1.14x on the per-layer one,
#    because its per-layer sd was 1.21e-3 (15x the base). If decoupling is genuinely
#    a conditioning fix rather than a shared-table quirk, it should tighten the
#    per-layer arm too. Three seeds on the per-layer base measure both the mean and,
#    more importantly, **its sd** -- the prediction is ~4e-4, not 1.21e-3.
#
# Sizing verified on meta, both bases iso to their own baselines:
#   engram_shared 1828.0M / 918.34M   cg_dec (shared)  1828.0M / 918.34M
#   engram        1828.4M / 918.34M   cg_dec_perlayer  1828.4M / 918.34M
# The decoupling adds 64 scalars and one min-reduce over 8 heads. dbs=8 throughout
# (§6: dbs is worth -0.00184 and must never vary inside a comparison).
#
# Power, from the measured sds:
#   shared, n=5 : combSE 2.70e-4, so the -0.00106 effect reads 3.92x
#   per-layer, n=3: combSE 2.40e-4, resolves 0.00048 at 2x -- ample for a ~-0.001
#     effect, and the per-layer base is the tightest anchor in the study (sd 8.1e-5)
#
# Arms are INTERLEAVED across the two tracks so a truncated run leaves both
# readable rather than one complete and one absent.
#
# PRE-REGISTERED:
#   - shared sd at n=5 stays near 4e-4  -> the conditioning story holds
#   - shared sd at n=5 inflates to ~7.5e-4 -> retract to ~1.9x, unresolved
#   - per-layer cg_dec sd ~4e-4 and mean ~-0.001 -> decoupling is a general
#     conditioning fix, and the plain gate's per-layer failure was variance
#   - per-layer sd stays ~1.2e-3 -> decoupling only works with a shared table, and
#     the mechanism is narrower than it looks

set -euo pipefail

REPO_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"
cd "$REPO_DIR"

RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/runs/logs/d20_cghard-${RUN_SUFFIX}}"
NPROC="${NPROC:-4}"
NUM_ITERATIONS="${NUM_ITERATIONS:-3814}"
MHC_STREAMS="${MHC_STREAMS:-4}"
TORCHRUN="${TORCHRUN:-$REPO_DIR/../nanochat-moe/.venv/bin/torchrun}"

WAIT_FOR_LOG="${WAIT_FOR_LOG:-}"
WAIT_FOR_MARKER="${WAIT_FOR_MARKER:-Round 13 complete}"
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
    echo "Predecessor finished at $(date -Is); starting round 14"
fi

# $1 tag, $2 slot, $3 "shared"|"perlayer", rest -> extra flags
run_arm () {
    local tag="$1"; local slot="$2"; local mode="$3"; shift 3
    local log_file="$LOG_DIR/${tag}.log"
    if grep -q "^Total training time" "$log_file" 2>/dev/null; then
        echo "Skipping $tag (already completed)"; return
    fi
    local share=()
    [ "$mode" = "shared" ] && share=(--engram-share-memory)
    echo "================================================================"
    echo "Starting $tag  ($(date -Is))  slot=$slot  $mode"
    echo "  extra: $*"
    echo "================================================================"
    "$TORCHRUN" --standalone --nproc_per_node="$NPROC" -m scripts.base_train -- \
        --depth=20 --max-seq-len=2048 --device-batch-size=8 --total-batch-size=524288 \
        --num-iterations="$NUM_ITERATIONS" --target-param-data-ratio=-1 \
        --window-pattern=SSSL --mhc --mhc-num-streams="$MHC_STREAMS" \
        --engram --engram-layers=2,6 --engram-memory-dim=1280 \
        --engram-mhc-num-streams="$MHC_STREAMS" --engram-slot-multiplier="$slot" \
        "${share[@]}" \
        --engram-count-gate --engram-count-gate-decouple \
        --eval-every=500 --eval-tokens=10485760 \
        --core-metric-every=-1 --sample-every=-1 --save-every=-1 \
        --model-tag="d20-cghard-${tag}-${RUN_SUFFIX}" \
        "$@" > "$log_file" 2>&1 || {
            echo "  !! $tag FAILED (exit $?)"; tail -20 "$log_file"; return
        }
    echo "  $tag done: $(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$log_file" | tail -1)"
}

# interleaved: does it transfer? / does the sd hold?
run_arm cg_dec_perlayer_s0 15 perlayer --engram-seed=0
run_arm cg_dec_s3          30 shared   --engram-seed=3
run_arm cg_dec_perlayer_s1 15 perlayer --engram-seed=1
run_arm cg_dec_s4          30 shared   --engram-seed=4
run_arm cg_dec_perlayer_s2 15 perlayer --engram-seed=2

echo
echo "Round 14 complete. Logs: $LOG_DIR"
printf "%-22s %-12s %s\n" arm min_bpb tok/sec
for tag in cg_dec_s3 cg_dec_s4 cg_dec_perlayer_s0 cg_dec_perlayer_s1 cg_dec_perlayer_s2; do
    f="$LOG_DIR/${tag}.log"; [ -f "$f" ] || continue
    printf "%-22s %-12s %s\n" "$tag" \
        "$(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$f" | tail -1)" \
        "$(grep -oP 'tok/sec: \K[0-9,]+' "$f" | tail -1)"
done
echo
echo "SHARED track, existing seeds 0/1/2: 0.762866 / 0.762400 / 0.763211"
echo "  (n=3 mean 0.762826, sd 4.07e-4) vs engram_shared 0.763884 (n=3, sd 3.45e-4)"
echo "  -> recompute the n=5 mean AND sd. The sd is the point."
echo
echo "PER-LAYER track, baseline engram 0.767606 (n=3, sd 8.12e-5 -- tightest in the study)"
echo "  plain count gate on this base was -0.00080 at only 1.14x, sd 1.21e-3."
echo "  -> if cg_dec's sd here is ~4e-4 rather than ~1.2e-3, decoupling is a general"
echo "     conditioning fix and the plain gate's per-layer failure was variance."
echo
echo "also read the learned coefficients from any checkpoint:"
echo "  shared:    engram_shared_memory.count_gate_{scale,freq_scale,collide_scale}"
echo "  per-layer: engram_modules.{2,6}.multi_head_embedding.count_gate_*"
echo "  the plain shared arms gave a=-0.62 (16/16 neg), e=+0.42 (16/16 pos),"
echo "  d=-0.065 (16/16 neg). Does the per-layer base reproduce those signs?"
