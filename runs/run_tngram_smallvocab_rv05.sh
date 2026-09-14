#!/bin/bash
#
# TN-gram small-vocab crossover: fill in the R/V ~ 0.5 midpoint.
#
# The main sweep (run_tngram_smallvocab.sh) measured R/V in {1.0, 0.25} per cell
# and found the CP-vs-native sign flips somewhere between them (native always sits
# between CP R=V and CP R=V/4). R/V~0.5 sits right in the suspected flip region
# (~0.3) and sharpens the crossover curve. One CP arm per (vocab,N) cell; the
# native anchors already exist in the same LOG_DIR from the main sweep.
#
#   vocab 1024 -> R=512   (R/V~0.50)
#   vocab 4096 -> R=2048  (R/V~0.50)
#
# Same flags as the main sweep (paper-faithful: --engram-no-tokenizer-compression
# so A indexes the full padded vocab). Logs into the SAME dir so the summary table
# is unified. Idempotent skip on completed arms. One arm at a time.
#
# LAUNCH ORDER: waits for the main sweep's last arm to finish (WAIT_PID) so the two
# never contend for the 4 GPUs.

set -euo pipefail
REPO_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"
cd "$REPO_DIR"

# reuse the main sweep's log dir so all arms share one summary
LOG_DIR="${LOG_DIR:-$REPO_DIR/runs/logs/tngram_smallvocab-20260911-073314}"
RUN_SUFFIX="${RUN_SUFFIX:-20260911-073314}"
NPROC="${NPROC:-4}"
NUM_ITERATIONS="${NUM_ITERATIONS:-3814}"
MHC_STREAMS="${MHC_STREAMS:-4}"
TORCHRUN="${TORCHRUN:-$REPO_DIR/../nanochat-moe/.venv/bin/torchrun}"
BASE_ROOT="${BASE_ROOT:-/mnt/3fs/dots-pretrain/ningzhiyuan/local_exps}"

VOCABS="${VOCABS:-1024 4096}"
NS="${NS:-5 3}"
WAIT_PID="${WAIT_PID:-}"   # pid of the main sweep to wait on before starting

export TORCHDYNAMO_CACHE_SIZE_LIMIT="${TORCHDYNAMO_CACHE_SIZE_LIMIT:-64}"
export PYTHONPATH="${PYTHONPATH:-$REPO_DIR}"

mkdir -p "$LOG_DIR"

if [ -n "$WAIT_PID" ]; then
    echo "Waiting for main sweep pid $WAIT_PID to finish before starting R/V~0.5 arms..."
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
    echo "Main sweep pid $WAIT_PID finished; starting R/V~0.5 arms ($(date -Is))"
fi

COMMON=(
    --depth=20 --max-seq-len=2048 --device-batch-size=8 --total-batch-size=524288
    --num-iterations="$NUM_ITERATIONS" --target-param-data-ratio=-1
    --window-pattern=SSSL --mhc --mhc-num-streams="$MHC_STREAMS"
    --engram --engram-layers=2,6 --engram-memory-dim=1280
    --engram-mhc-num-streams="$MHC_STREAMS" --engram-share-memory
    --engram-no-tokenizer-compression
    --eval-every=500 --eval-tokens=10485760
    --core-metric-every=-1 --sample-every=-1 --save-every=-1
)

run_arm () {
    local vocab="$1" n="$2" rank="$3"
    local base_dir="$BASE_ROOT/nsmb_out_v${vocab}"
    local tag="sv${vocab}_N${n}_R${rank}_s0"
    local log_file="$LOG_DIR/${tag}.log"
    if grep -q "^Total training time" "$log_file" 2>/dev/null; then
        echo "Skipping $tag (already completed)"; return
    fi
    if [ ! -f "$base_dir/tokenizer/tokenizer.pkl" ]; then
        echo "  !! $tag: tokenizer missing at $base_dir"; return
    fi
    echo "================================================================"
    echo "Starting $tag  ($(date -Is))  vocab=$vocab N=$n R=$rank (R/V~0.5)"
    echo "================================================================"
    NANOCHAT_BASE_DIR="$base_dir" "$TORCHRUN" --standalone --nproc_per_node="$NPROC" \
        -m scripts.base_train -- \
        "${COMMON[@]}" --engram-max-ngram-size="$n" \
        --engram-value-table=tngram --engram-cp-rank="$rank" \
        --model-tag="d20-sv${vocab}-N${n}-${rank}-${RUN_SUFFIX}" \
        > "$log_file" 2>&1 || {
            echo "  !! $tag FAILED (exit $?)"; tail -20 "$log_file"; return
        }
    echo "  $tag done: $(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$log_file" | tail -1)"
}

for vocab in $VOCABS; do
    rank=$((vocab/2))          # R/V ~ 0.5
    for n in $NS; do
        run_arm "$vocab" "$n" "$rank"
    done
done

echo
echo "R/V~0.5 arms complete. Logs: $LOG_DIR"
printf "%-26s %-8s %s\n" arm R/V min_bpb
for vocab in $VOCABS; do
    rank=$((vocab/2))
    for n in $NS; do
        tag="sv${vocab}_N${n}_R${rank}_s0"
        f="$LOG_DIR/${tag}.log"; [ -f "$f" ] || continue
        printf "%-26s %-8s %s\n" "$tag" "0.500" \
            "$(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$f" | tail -1)"
    done
done
