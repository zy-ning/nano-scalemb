#!/bin/bash
#
# TN-gram at our REAL 32k stack, HIGH RANK: the untested window where the
# vocab-8192 cell says the crossover has already happened.
#
# WHY THIS IS THE DECISIVE EXPERIMENT FOR OUR STACK.
# The small-vocab sweep established that the CP-vs-native sign is governed by
# R/V, and that the flip point SLIDES DOWN as vocab grows. Measured (N=5,
# CP-native, W=CP wins):
#     R/V | v1024     | v4096     | v8192
#     1.00| -0.0020 W | -0.0058 W |  (infeasible)
#     0.50| -0.0006 W | -0.0009 W | -0.0069 W
#     0.25| +0.0011 L | +0.0010 L | -0.0039 W   <-- SIGN FLIP 4096->8192
#     0.22|    --     |    --     | -0.0041 W
# Holding R/V FIXED at 0.25, CP goes from LOSING to WINNING purely by raising V.
# Interpolated flip: ~R/V 0.41 (v1024), ~0.38 (v4096), well BELOW 0.22 (v8192).
# Extrapolating that trend to V~23686 predicts the flip sits at a LOW R/V --
# plausibly inside the range we can actually afford.
#
# Our entire 32k ladder only ever reached R=400 (R/V 0.017), and the gap was
# already NARROWING fast with rank: R10 +0.0061, R80 +0.0058, R400 +0.0022.
# Everything between R/V 0.017 and the memory ceiling is UNTESTED -- and the
# standing practical conclusion ("native wins for our 32k stack, R/V~1 needs
# R~32k and is infeasible") was written assuming we needed R/V~1. The v8192 cell
# shows we do NOT: R/V 0.22 wins there. So the real question is whether a
# FEASIBLE rank suffices at 32k.
#
# MEMORY -- fit on the ACTUAL 32k runs (same backbone as the target), not the
# small-vocab cells. A-params = N*V*K*R with N=3 (orders {2,3}), K=8, Vc=23686:
#     peak_MiB = 65662 + 25.5 B per A-param
# Validated against all three measured 32k peaks to within 6 MiB:
#     R=10  pred 65801 / actual 65806
#     R=80  pred 66769 / actual 66763
#     R=400 pred 71196 / actual 71197
# Projections (GB200 cap 189471 MiB):
#     R=1024  R/V 0.043   582M A-params   ~78GB   OK
#     R=2048  R/V 0.086  1164M A-params   ~92GB   OK
#     R=4096  R/V 0.173  2328M A-params  ~120GB   OK
#     R=6144  R/V 0.259  3492M A-params  ~149GB   OK (~40GB headroom)
# R=8192 (R/V 0.35) would exceed the 90%-of-cap guard and is excluded.
#
# ANCHOR. Do NOT re-run native: the native slot15 anchor is well calibrated at
# n=5, mean 0.765091, sd 4.36e-4 (see engram-overprovision-merge-schedule).
# The 32k noise floor is sd~6e-4, and the cross-vocab CP wins we are chasing are
# -0.004..-0.007, comfortably above it.
#
# READ (pre-registered). Gap = CP - 0.765091.
#   Any arm <= 0.7645 (i.e. gap <= -6e-4, one sd below the anchor)  => CP WINS at
#     32k with a feasible rank. This OVERTURNS the standing practical verdict and
#     means CP is usable on our production stack.
#   Monotone-decreasing gap that stays positive => flip is real but sits above the
#     memory ceiling; native remains the practical choice, now for a QUANTIFIED
#     reason rather than an assumed one.
#   Gap flat/rising with R => the small-vocab crossover does NOT transfer to 32k
#     and the R/V story is vocab-local, not universal.
#
# KILL RESILIENCE. Same as run_tngram_v8192.sh: arms on this node get SIGTERMed
# as a whole process group at unpredictable intervals (6 observed, 20m-85m in),
# which kills the driver too, so in-script retry alone cannot help and setsid
# does not prevent it. --save-every=500 + auto-resume from the newest COMPLETE
# checkpoint means a kill costs <=500 steps instead of a whole multi-hour arm.
# Safe for the metric: bpb falls monotonically here, so the reported minimum is
# the final eval, which a resumed run still reaches.

set -euo pipefail
REPO_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"
cd "$REPO_DIR"

RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/runs/logs/tngram_32k_highrank-${RUN_SUFFIX}}"
NPROC="${NPROC:-4}"
NUM_ITERATIONS="${NUM_ITERATIONS:-3814}"
MHC_STREAMS="${MHC_STREAMS:-4}"
TORCHRUN="${TORCHRUN:-$REPO_DIR/../nanochat-moe/.venv/bin/torchrun}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/mnt/3fs/dots-pretrain/ningzhiyuan/local_exps/nsmb_out}"

# ascending rank: cheapest/fastest first so the trend is visible early
ARMS="${ARMS:-1024 2048 4096 6144}"
N="${N:-3}"                       # orders {2,3}, matching the existing 32k ladder
DBS="${DBS:-8}"
SAVE_EVERY="${SAVE_EVERY:-500}"
NATIVE_ANCHOR="${NATIVE_ANCHOR:-0.765091}"

export TORCHDYNAMO_CACHE_SIZE_LIMIT="${TORCHDYNAMO_CACHE_SIZE_LIMIT:-64}"
export PYTHONPATH="${PYTHONPATH:-$REPO_DIR}"

mkdir -p "$LOG_DIR"

COMMON=(
    --depth=20 --max-seq-len=2048 --total-batch-size=524288
    --num-iterations="$NUM_ITERATIONS" --target-param-data-ratio=-1
    --window-pattern=SSSL --mhc --mhc-num-streams="$MHC_STREAMS"
    --engram --engram-layers=2,6 --engram-memory-dim=1280
    --engram-mhc-num-streams="$MHC_STREAMS" --engram-share-memory
    --engram-seed=0
    --eval-every=500 --eval-tokens=10485760
    --core-metric-every=-1 --sample-every=-1 --save-every="$SAVE_EVERY"
)

# Newest step with a COMPLETE checkpoint (model + meta + all NPROC optim shards).
# An arm killed mid-save can leave a partial set; resuming from that would fail.
latest_complete_ckpt () {
    local ckpt_dir="$1" step f ok r
    [ -d "$ckpt_dir" ] || { echo ""; return; }
    for f in $(ls "$ckpt_dir"/model_*.pt 2>/dev/null | sort -r); do
        step="$(basename "$f" .pt)"; step="${step#model_}"
        [ -f "$ckpt_dir/meta_${step}.json" ] || continue
        ok=1
        for r in $(seq 0 $((NPROC-1))); do
            [ -f "$ckpt_dir/optim_${step}_rank${r}.pt" ] || { ok=0; break; }
        done
        [ "$ok" = 1 ] && { echo "$((10#$step))"; return; }
    done
    echo ""
}

attempt () {
    local rank="$1" dbs="$2" log_file="$3"
    local model_tag="d20-tngram32k-R${rank}-${RUN_SUFFIX}"
    local extra=() resume_step
    resume_step="$(latest_complete_ckpt "$NANOCHAT_BASE_DIR/base_checkpoints/$model_tag")"
    if [ -n "$resume_step" ] && [ "$resume_step" -gt 0 ] 2>/dev/null; then
        echo "  resuming $model_tag from step $resume_step"
        echo "=== RESUME from step $resume_step ($(date -Is)) ===" >> "$log_file"
        extra+=(--resume-from-step="$resume_step")
    fi
    "$TORCHRUN" --standalone --nproc_per_node="$NPROC" \
        -m scripts.base_train -- \
        "${COMMON[@]}" --device-batch-size="$dbs" \
        --engram-max-ngram-size="$N" \
        --engram-value-table=tngram --engram-cp-rank="$rank" \
        "${extra[@]}" --model-tag="$model_tag" \
        >> "$log_file" 2>&1
}

run_arm () {
    local rank="$1"
    local tag="tngram32k_R${rank}_s0"
    local log_file="$LOG_DIR/${tag}.log"
    local rv; rv=$(awk "BEGIN{printf \"%.3f\", $rank/23686}")
    if grep -q "^Total training time" "$log_file" 2>/dev/null; then
        echo "Skipping $tag (already completed)"; return
    fi
    echo "================================================================"
    echo "Starting $tag  ($(date -Is))  R=$rank  R/V~$rv  N=$N  dbs=$DBS"
    echo "================================================================"
    if attempt "$rank" "$DBS" "$log_file"; then
        echo "  $tag done: $(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$log_file" | tail -1)"
        return
    fi
    if tail -200 "$log_file" | grep -qiE "out of memory"; then
        local half=$((DBS/2))
        if [ "$half" -ge 1 ]; then
            echo "  !! $tag OOMed at dbs=$DBS; retrying at dbs=$half (same total batch, identical math)"
            echo "=== OOM RETRY at device-batch-size=$half ($(date -Is)) ===" >> "$log_file"
            if attempt "$rank" "$half" "$log_file"; then
                echo "  $tag done (dbs=$half): $(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$log_file" | tail -1)"
                return
            fi
        fi
    fi
    echo "  !! $tag FAILED (see $log_file)"; tail -15 "$log_file"
}

for rank in $ARMS; do run_arm "$rank"; done

echo
echo "32k high-rank ladder complete. Logs: $LOG_DIR"
echo "native slot15 anchor = $NATIVE_ANCHOR (n=5, sd 4.36e-4)"
printf "%-22s %-8s %-11s %s\n" arm R/V min_bpb gap_vs_native
for rank in $ARMS; do
    tag="tngram32k_R${rank}_s0"; f="$LOG_DIR/${tag}.log"; [ -f "$f" ] || continue
    b=$(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$f" | tail -1)
    rv=$(awk "BEGIN{printf \"%.3f\", $rank/23686}")
    gap=$([ -n "$b" ] && awk "BEGIN{printf \"%+.4f\", $b-$NATIVE_ANCHOR}" || echo "-")
    printf "%-22s %-8s %-11s %s\n" "$tag" "$rv" "${b:-—}" "$gap"
done
echo
echo "READ: any arm <= 0.7645 (gap <= -6e-4) => CP WINS at 32k with a FEASIBLE rank,"
echo "overturning the standing 'native wins for our 32k stack' verdict. A monotone but"
echo "still-positive gap => flip is real but above the memory ceiling. Flat/rising gap"
echo "=> the small-vocab R/V crossover does not transfer to 32k."
