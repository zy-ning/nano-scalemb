#!/bin/bash
#
# NATIVE iso-param partners for the 32k high-rank TN-gram ladder.
#
# WHY THIS EXISTS. run_tngram_32k_highrank.sh compares CP arms against the
# native slot15 anchor 0.765091. That is the WRONG BAR for the big arms, because
# native value params = rows * head_dim(80), so the native table scales too:
#     slot15   5,686,228 rows ->  454.9M params  bpb 0.765091 (n=5, sd 4.4e-4)
#     slot30  11,371,072 rows ->  909.7M params  bpb ~0.76393
#     slot60  22,740,718 rows -> 1819.3M params  bpb 0.763329
#     slot120 45,479,426 rows -> 3638.4M params  bpb 0.761128 (rowcap shared_slot120)
# Native gets MONOTONICALLY BETTER with table size (0.765091 -> 0.761128), so a
# CP arm with 5-8x the params of slot15 must clear a much lower bar than 0.7645.
# (An earlier note in memory recorded native slot15 as "5.69M params" -- that was
# the ROW COUNT, not the param count. The original TN-gram "iso-param" matching
# matched CP A-params to native ROWS, i.e. gave CP ~80x FEWER params than native.)
#
# CORRECT ISO-PARAM PAIRING (32k, N=3, K=8, Vc=23686; CP A-params = N*Vc*K*R):
#     CP R=1024   582M  <->  slot15   455M  (1.28x)   must beat 0.765091
#     CP R=2048  1164M  <->  slot30   910M  (1.28x)   must beat ~0.76393
#     CP R=4096  2328M  <->  slot60  1819M  (1.28x)   must beat 0.763329
#     CP R=6144  3493M  <->  slot120 3638M  (0.96x)   must beat 0.761128
#
# WHY RE-RUN RATHER THAN REUSE. The existing native anchors come from different
# lineages and dates (slot120 from d20_rowcap 2026-08-16, slot60/op_ratio
# 2026-08-29, slot15 floor_n3 2026-09-13) while the CP arms run on 2026-09-17
# code. This thread has already been burned once by exactly that: the "op2x is
# better than at05" scare turned out to be a cross-CODE-VERSION comparison, not a
# regression. With run-to-run sd~6e-4 and the CP-vs-native effects of interest in
# the 1-4e-3 range, same-code partners are worth the GPU time. slot60 and slot120
# are the two that matter (they pair with the biggest, most param-advantaged CP
# arms); slot15 is already solid at n=5 and slot30 has multiple draws.
#
# MEMORY (measured on prior runs of the same sizes, GB200 cap 189471 MiB):
#     slot60  peak ~100.6GB   slot120 peak ~123.7GB   -> both fit comfortably.
# For reference the CP side tops out at R=6144 (~149GB); R=7168 would be ~91% of
# cap and R=8192 ~99%, so R=6144 is the practical ceiling of the CP ladder.
#
# GPU DISCIPLINE. Waits for WAIT_PID (the CP ladder driver) so the two never
# contend for the 4 GPUs. Same checkpoint+resume kill resilience as the other
# runners: this node SIGTERMs whole process groups at unpredictable intervals.
#
# READ. Pair each CP arm with its native partner above. If CP loses to its
# iso-param native partner at every rung, the small-vocab R/V crossover does NOT
# transfer to 32k once the comparison is made fair, and "native wins on our
# production stack" stands -- now for a properly param-matched reason. If CP beats
# its partner at the big rungs, CP is genuinely preferable at scale on 32k.

set -euo pipefail
REPO_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"
cd "$REPO_DIR"

RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/runs/logs/native_isoparam-${RUN_SUFFIX}}"
NPROC="${NPROC:-4}"
NUM_ITERATIONS="${NUM_ITERATIONS:-3814}"
MHC_STREAMS="${MHC_STREAMS:-4}"
TORCHRUN="${TORCHRUN:-$REPO_DIR/../nanochat-moe/.venv/bin/torchrun}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/mnt/3fs/dots-pretrain/ningzhiyuan/local_exps/nsmb_out}"

# slot multipliers to run; 60 and 120 are the iso-param partners that matter
ARMS="${ARMS:-60 120}"
N="${N:-3}"
DBS="${DBS:-8}"
SAVE_EVERY="${SAVE_EVERY:-500}"
WAIT_PID="${WAIT_PID:-}"

export TORCHDYNAMO_CACHE_SIZE_LIMIT="${TORCHDYNAMO_CACHE_SIZE_LIMIT:-64}"
export PYTHONPATH="${PYTHONPATH:-$REPO_DIR}"

mkdir -p "$LOG_DIR"

if [ -n "$WAIT_PID" ]; then
    echo "Waiting for pid $WAIT_PID (CP ladder) to finish before starting native arms..."
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 120; done
    echo "pid $WAIT_PID finished; starting native iso-param arms ($(date -Is))"
fi

# EXACTLY the CP ladder's flags, minus value-table/cp-rank, plus slot-multiplier.
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
    local slot="$1" dbs="$2" log_file="$3"
    local model_tag="d20-natiso-slot${slot}-${RUN_SUFFIX}"
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
        --engram-slot-multiplier="$slot" \
        "${extra[@]}" --model-tag="$model_tag" \
        >> "$log_file" 2>&1
}

run_arm () {
    local slot="$1"
    local tag="natiso_slot${slot}_s0"
    local log_file="$LOG_DIR/${tag}.log"
    if grep -q "^Total training time" "$log_file" 2>/dev/null; then
        echo "Skipping $tag (already completed)"; return
    fi
    echo "================================================================"
    echo "Starting $tag  ($(date -Is))  slot=$slot  N=$N  dbs=$DBS"
    echo "================================================================"
    if attempt "$slot" "$DBS" "$log_file"; then
        echo "  $tag done: $(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$log_file" | tail -1)"
        return
    fi
    if tail -200 "$log_file" | grep -qiE "out of memory"; then
        local half=$((DBS/2))
        if [ "$half" -ge 1 ]; then
            echo "  !! $tag OOMed at dbs=$DBS; retrying at dbs=$half (same total batch, identical math)"
            echo "=== OOM RETRY at device-batch-size=$half ($(date -Is)) ===" >> "$log_file"
            if attempt "$slot" "$half" "$log_file"; then
                echo "  $tag done (dbs=$half): $(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$log_file" | tail -1)"
                return
            fi
        fi
    fi
    echo "  !! $tag FAILED (see $log_file)"; tail -15 "$log_file"
}

for slot in $ARMS; do run_arm "$slot"; done

echo
echo "Native iso-param arms complete. Logs: $LOG_DIR"
printf "%-24s %-12s %-11s %s\n" arm value_params min_bpb historical
for slot in $ARMS; do
    tag="natiso_slot${slot}_s0"; f="$LOG_DIR/${tag}.log"; [ -f "$f" ] || continue
    b=$(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$f" | tail -1)
    em=$(grep -oP 'engram_embeds\s+: \K[0-9,]+' "$f" | head -1)
    case "$slot" in 60) h=0.763329;; 120) h=0.761128;; *) h="-";; esac
    printf "%-24s %-12s %-11s %s\n" "$tag" "${em:-—}" "${b:-—}" "$h"
done
echo
echo "READ: pair with CP arms -- R=4096 (2328M) vs slot60, R=6144 (3493M) vs slot120."
echo "Also check each fresh native vs its historical value: a large drift would mean"
echo "the old cross-lineage anchors were not safe to compare against."
