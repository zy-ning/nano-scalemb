#!/bin/bash
#
# vocab-8192 NATIVE with a LARGE table: does CP's small-vocab win survive
# parameter matching?
#
# THE PROBLEM THIS CLOSES. Every small-vocab cell compared CP against native at
# the DEFAULT slot_multiplier=18, which is a much SMALLER table than the CP arms
# it was scored against. Measured value params (engram_embeds) at vocab 8192, N=5:
#     native (slot18)   189.0M   0.783098
#     CP R=1802         590.5M   0.779004   -0.0041  but 3.12x the params
#     CP R=2048         671.1M   0.779202   -0.0039  but 3.55x the params
#     CP R=4096        1342.2M   0.776223   -0.0069  but 7.10x the params
# So CP "won" the small-vocab cells while holding 3-7x more parameters. The same
# confound runs through the whole R/V story: raising R raises params, so
# "CP wins at high R/V" is partly just "CP wins when given more params".
# (At vocab 4096 CP R=1024 actually LOSES with 1.77x the params; the only clean
# point anywhere is vocab1024 R=512 at 0.88x params, winning by just -0.0006.)
#
# At 32k this was settled by running iso-param natives (slot60/slot120): CP lost
# at EVERY rung by +0.0021..+0.0032 (run_native_isoparam.sh). The small-vocab
# regime never got that test -- nobody ran native with a big table there.
#
# ARMS. rows scale linearly in slot_multiplier (default 18 -> 2,362,342 rows,
# 189.0M params at vocab 8192), so:
#     slot 56  -> ~3.1x -> ~588M params  <-> CP R=1802 (590.5M)   [~1.00x]
#     slot 128 -> ~7.1x -> ~1344M params <-> CP R=4096 (1342.2M)  [~1.00x]
# Both are near-exact param matches to the CP arms already measured in
# runs/logs/tngram_v8192-20260914-123144, so no CP re-runs are needed.
#
# FLAGS. Identical to the small-vocab sweep (N=5, --engram-no-tokenizer-compression,
# layers 2,6, memory_dim 1280, mhc4, share-memory) so the only difference from the
# existing native arm is slot_multiplier. Same base dir + tokenizer, so step-0 bpb
# must come out at 3.213328 exactly, as every other arm in this cell did.
#
# READ (pre-registered).
#   If native slot56 <= 0.779004 and native slot128 <= 0.776223, CP's small-vocab
#     wins were PURELY a parameter-count artifact and the TN-gram line closes
#     completely -- consistent with the param-fair 32k result.
#   If native stays ABOVE the CP arms at matched params, then CP genuinely is the
#     better value representation at small vocab, and the 32k negative is about the
#     32k REGIME (R/V unreachable) rather than about CP itself. That would be a real
#     (if impractical) positive and worth stating precisely.
#   Noise: 32k run-to-run sd~6e-4; the gaps at stake here are 4-7e-3, well clear.
#
# MEMORY. native 189M params peaked ~63GB; ~25.5 B/param scaling puts slot56 at
# ~73GB and slot128 at ~91GB, both far under the GB200's 189GB.
#
# Same checkpoint+resume kill resilience as the other runners (this node SIGTERMs
# whole process groups at unpredictable intervals).

set -euo pipefail
REPO_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"
cd "$REPO_DIR"

RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/runs/logs/v8192_native_bigtable-${RUN_SUFFIX}}"
NPROC="${NPROC:-4}"
NUM_ITERATIONS="${NUM_ITERATIONS:-3814}"
MHC_STREAMS="${MHC_STREAMS:-4}"
TORCHRUN="${TORCHRUN:-$REPO_DIR/../nanochat-moe/.venv/bin/torchrun}"
BASE_ROOT="${BASE_ROOT:-/mnt/3fs/dots-pretrain/ningzhiyuan/local_exps}"
VOCAB="${VOCAB:-8192}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$BASE_ROOT/nsmb_out_v${VOCAB}}"

ARMS="${ARMS:-56 128}"
N="${N:-5}"
DBS="${DBS:-8}"
SAVE_EVERY="${SAVE_EVERY:-500}"
WAIT_PID="${WAIT_PID:-}"

export TORCHDYNAMO_CACHE_SIZE_LIMIT="${TORCHDYNAMO_CACHE_SIZE_LIMIT:-64}"
export PYTHONPATH="${PYTHONPATH:-$REPO_DIR}"

mkdir -p "$LOG_DIR"

if [ -n "$WAIT_PID" ]; then
    echo "Waiting for pid $WAIT_PID before starting..."
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 120; done
fi

COMMON=(
    --depth=20 --max-seq-len=2048 --total-batch-size=524288
    --num-iterations="$NUM_ITERATIONS" --target-param-data-ratio=-1
    --window-pattern=SSSL --mhc --mhc-num-streams="$MHC_STREAMS"
    --engram --engram-layers=2,6 --engram-memory-dim=1280
    --engram-mhc-num-streams="$MHC_STREAMS" --engram-share-memory
    --engram-no-tokenizer-compression
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
    local model_tag="d20-sv${VOCAB}-N${N}-natslot${slot}-${RUN_SUFFIX}"
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
        --engram-max-ngram-size="$N" --engram-slot-multiplier="$slot" \
        "${extra[@]}" --model-tag="$model_tag" \
        >> "$log_file" 2>&1
}

run_arm () {
    local slot="$1"
    local tag="sv${VOCAB}_N${N}_natslot${slot}_s0"
    local log_file="$LOG_DIR/${tag}.log"
    if grep -q "^Total training time" "$log_file" 2>/dev/null; then
        echo "Skipping $tag (already completed)"; return
    fi
    echo "================================================================"
    echo "Starting $tag  ($(date -Is))  vocab=$VOCAB N=$N slot=$slot dbs=$DBS"
    echo "================================================================"
    if attempt "$slot" "$DBS" "$log_file"; then
        echo "  $tag done: $(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$log_file" | tail -1)"
        return
    fi
    if tail -200 "$log_file" | grep -qiE "out of memory"; then
        local half=$((DBS/2))
        if [ "$half" -ge 1 ]; then
            echo "  !! $tag OOMed at dbs=$DBS; retrying at dbs=$half"
            echo "=== OOM RETRY at device-batch-size=$half ($(date -Is)) ===" >> "$log_file"
            attempt "$slot" "$half" "$log_file" && {
                echo "  $tag done (dbs=$half): $(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$log_file" | tail -1)"; return; }
        fi
    fi
    echo "  !! $tag FAILED"; tail -15 "$log_file"
}

for slot in $ARMS; do run_arm "$slot"; done

echo
echo "vocab-$VOCAB native big-table arms complete. Logs: $LOG_DIR"
printf "%-28s %-14s %-11s %s\n" arm value_params min_bpb "CP counterpart"
for slot in $ARMS; do
    tag="sv${VOCAB}_N${N}_natslot${slot}_s0"; f="$LOG_DIR/${tag}.log"; [ -f "$f" ] || continue
    b=$(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$f" | tail -1)
    em=$(grep -oP 'engram_embeds\s+: \K[0-9,]+' "$f" | head -1)
    case "$slot" in 56) cp="CP R=1802 590.5M 0.779004";; 128) cp="CP R=4096 1342.2M 0.776223";; *) cp="-";; esac
    printf "%-28s %-14s %-11s %s\n" "$tag" "${em:-—}" "${b:-—}" "$cp"
done
echo
echo "READ: native <= its CP counterpart => CP's small-vocab wins were a param-count"
echo "artifact and the TN-gram line closes completely. Native still above => CP is"
echo "genuinely better at small vocab and the 32k negative is a regime limit, not CP."
