#!/bin/bash
#
# TN-gram vocab-8192 cell: reproduce the paper's SECOND data point and get the
# third vocab rung needed to extrapolate the R/V crossover toward our 32k stack.
#
# WHY THIS CELL. The small-vocab sweep (run_tngram_smallvocab.sh + _rv05.sh)
# reproduced the paper's R/V~1 tie/win at vocab {1024, 4096}. Two things are
# still missing:
#   (1) The paper's own larger-vocab point is R=1800 / vocab=8192 -> R/V~0.22,
#       where TN-gram ALREADY LOSES BPB (1.071 vs 1.070). We have never run at
#       that vocab, so we have reproduced only the paper's easy point. R=1802
#       below lands exactly on it.
#   (2) The practical question for OUR stack is "is there any feasible rank at
#       which CP wins at a 32k vocab?". That depends on how the CROSSOVER rank
#       scales with V. We measured the flip at ~R/V 0.3 (vocab 1024) and already
#       below R/V 0.25 (vocab 4096) -- two points cannot fit a trend. vocab 8192
#       is the third rung and makes the extrapolation to 32k meaningful.
#
# MEMORY CEILING (measured, important). CP A-params = N*V*K*R (K=8), so at
# R/V~1 they scale as V^2. Measured peaks at vocab 4096 N=5: R/V .25 (168M
# A-params) 71.9GB, R/V .50 (336M) 82.5GB, R/V 1.0 (671M) 103.7GB -> ~60 B per
# A-param. Extrapolating to vocab 8192 at R/V~1 = 2.68B A-params gives ~224GB
# per GPU, ABOVE the GB200's 189GB -> R/V~1 IS NOT RUNNABLE at vocab 8192 and is
# deliberately NOT in the ladder. The arms below top out at R/V~0.5 (1.34B
# A-params, ~110GB projected).
#
# OOM POLICY. device-batch-size is a pure grad-accum knob: total-batch-size is
# pinned at 524288 either way, so halving it changes throughput ONLY, never the
# math or the result. If an arm OOMs we retry at half the micro-batch, so a
# borderline arm degrades in speed instead of dying. Retries are logged.
#
# LADDER (N=5, matching the paper's depth; the cheap R/V .22 point first so the
# paper-reproduction lands even if the big arms run long):
#   R=1802 (R/V~0.22)  <- PAPER'S second point; paper reports CP LOSING here
#   R=2048 (R/V~0.25)  <- matches the R/V .25 rung measured at 1024/4096
#   R=4096 (R/V~0.50)  <- matches the .50 midpoint rung
#   native             <- the in-cell anchor (shared by all ranks)
#
# READ. Within the cell, native-vs-CP sign at each R/V. Prediction from the
# existing trend (flip point slides DOWN as vocab grows): at vocab 8192 even
# R/V~0.22-0.25 should WIN, whereas the paper reports a LOSS at R/V~0.22. If we
# instead reproduce the paper's loss at R=1802, the flip point is NOT sliding
# down monotonically and the "CP could work at 32k with feasible rank"
# extrapolation dies. Either outcome is decisive.

set -euo pipefail
REPO_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"
cd "$REPO_DIR"

RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/runs/logs/tngram_v8192-${RUN_SUFFIX}}"
NPROC="${NPROC:-4}"
NUM_ITERATIONS="${NUM_ITERATIONS:-3814}"
MHC_STREAMS="${MHC_STREAMS:-4}"
TORCHRUN="${TORCHRUN:-$REPO_DIR/../nanochat-moe/.venv/bin/torchrun}"
BASE_ROOT="${BASE_ROOT:-/mnt/3fs/dots-pretrain/ningzhiyuan/local_exps}"

VOCAB="${VOCAB:-8192}"
N="${N:-5}"
# cheap paper-point first, then ascending rank
ARMS="${ARMS:-native 1802 2048 4096}"
DBS="${DBS:-8}"            # device-batch-size; halved once on OOM
WAIT_PID="${WAIT_PID:-}"   # optional pid to wait on so we never contend for GPUs

export TORCHDYNAMO_CACHE_SIZE_LIMIT="${TORCHDYNAMO_CACHE_SIZE_LIMIT:-64}"
export PYTHONPATH="${PYTHONPATH:-$REPO_DIR}"

mkdir -p "$LOG_DIR"

if [ -n "$WAIT_PID" ]; then
    echo "Waiting for pid $WAIT_PID to finish before starting vocab-$VOCAB arms..."
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
    echo "pid $WAIT_PID finished; starting vocab-$VOCAB arms ($(date -Is))"
fi

COMMON=(
    --depth=20 --max-seq-len=2048 --total-batch-size=524288
    --num-iterations="$NUM_ITERATIONS" --target-param-data-ratio=-1
    --window-pattern=SSSL --mhc --mhc-num-streams="$MHC_STREAMS"
    --engram --engram-layers=2,6 --engram-memory-dim=1280
    --engram-mhc-num-streams="$MHC_STREAMS" --engram-share-memory
    --engram-no-tokenizer-compression
    --eval-every=500 --eval-tokens=10485760
    --core-metric-every=-1 --sample-every=-1 --save-every=-1
)

# one attempt at a given micro-batch; returns non-zero on failure
attempt () {
    local base_dir="$1" n="$2" arm="$3" dbs="$4" log_file="$5"
    local extra=()
    [ "$arm" = "native" ] || extra=(--engram-value-table=tngram --engram-cp-rank="$arm")
    NANOCHAT_BASE_DIR="$base_dir" "$TORCHRUN" --standalone --nproc_per_node="$NPROC" \
        -m scripts.base_train -- \
        "${COMMON[@]}" --device-batch-size="$dbs" \
        --engram-max-ngram-size="$n" "${extra[@]}" \
        --model-tag="d20-sv${VOCAB}-N${n}-${arm}-${RUN_SUFFIX}" \
        >> "$log_file" 2>&1
}

run_arm () {
    local arm="$1"
    local base_dir="$BASE_ROOT/nsmb_out_v${VOCAB}"
    local tag rv
    if [ "$arm" = "native" ]; then tag="sv${VOCAB}_N${N}_native_s0"; rv="-"
    else tag="sv${VOCAB}_N${N}_R${arm}_s0"; rv=$(awk "BEGIN{printf \"%.3f\", $arm/$VOCAB}"); fi
    local log_file="$LOG_DIR/${tag}.log"
    if grep -q "^Total training time" "$log_file" 2>/dev/null; then
        echo "Skipping $tag (already completed)"; return
    fi
    if [ ! -f "$base_dir/tokenizer/tokenizer.pkl" ]; then
        echo "  !! $tag: tokenizer missing at $base_dir -- run prep_smallvocab_tokenizers.sh (VOCABS=$VOCAB) first"
        return
    fi
    echo "================================================================"
    echo "Starting $tag  ($(date -Is))  vocab=$VOCAB N=$N arm=$arm R/V=$rv dbs=$DBS"
    echo "================================================================"
    if attempt "$base_dir" "$N" "$arm" "$DBS" "$log_file"; then
        echo "  $tag done: $(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$log_file" | tail -1)"
        return
    fi
    # OOM retry at half the micro-batch (same total batch -> identical math)
    if grep -qiE "out of memory|CUDA error: out of memory" "$log_file"; then
        local half=$((DBS/2))
        if [ "$half" -ge 1 ]; then
            echo "  !! $tag OOMed at dbs=$DBS; retrying at dbs=$half (same total batch, identical math)"
            echo "=== OOM RETRY at device-batch-size=$half ($(date -Is)) ===" >> "$log_file"
            if attempt "$base_dir" "$N" "$arm" "$half" "$log_file"; then
                echo "  $tag done (dbs=$half): $(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$log_file" | tail -1)"
                return
            fi
        fi
    fi
    echo "  !! $tag FAILED"; tail -20 "$log_file"
}

for arm in $ARMS; do run_arm "$arm"; done

echo
echo "vocab-$VOCAB cell complete. Logs: $LOG_DIR"
printf "%-26s %-8s %s\n" arm R/V min_bpb
for arm in $ARMS; do
    if [ "$arm" = "native" ]; then tag="sv${VOCAB}_N${N}_native_s0"; rv="-"
    else tag="sv${VOCAB}_N${N}_R${arm}_s0"; rv=$(awk "BEGIN{printf \"%.3f\", $arm/$VOCAB}"); fi
    f="$LOG_DIR/${tag}.log"; [ -f "$f" ] || continue
    printf "%-26s %-8s %s\n" "$tag" "$rv" \
        "$(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$f" | tail -1)"
done
echo
echo "READ: R=1802 is the PAPER'S R/V~0.22 point (they report CP LOSING there)."
echo "Our trend predicts a WIN. Whichever way it lands settles whether the"
echo "crossover rank keeps sliding down with vocab -- i.e. whether CP could ever"
echo "be practical at our 32k tokenizer."
