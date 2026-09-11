#!/bin/bash
#
# TN-gram small-vocab reproduction: does CP reach native parity at the PAPER'S
# rank/vocab ratio R/V ~ 1?
#
# BACKGROUND. Our TN-gram (CP-factorized Engram value table, arXiv 2606.08347)
# lost to the native per-order hash table on BPB at every axis we swept -- rank,
# budget, init, cross-order sharing, global readout, N=5 depth. But EVERY one of
# those runs sat at R/V << 1 (R=48..400 vs compressed vocab ~23,686 -> R/V ~
# 0.002-0.017). The CP token factor is A in R^{V x R}: at R~V each token gets an
# ~uncompressed row; at R<<V it is a hard bottleneck. The paper's headline TIE is
# at R/V ~ 1 (R=1024, vocab=1024); at its one larger-vocab point (R=1800,
# vocab=8192, R/V~0.22) TN-gram already LOSES BPB. So our negative is the
# continuation of the paper's own trend, not a contradiction. This sweep tests
# the missing regime directly with genuinely small BACKBONE vocabs.
#
# FIDELITY. --engram-no-tokenizer-compression on ALL arms so A indexes the full
# (padded) vocab directly, exactly like the paper (raw token IDs). R=V then means
# R/V ~ 1. Both native and CP arms share every flag except the value table, so the
# comparison is apples-to-apples. BPB is vocab-invariant (token_bytes.pt), so cells
# are internally comparable; do NOT compare bpb ACROSS vocab except as an R/V trend.
#
# GRID (user-selected): vocab {1024,4096} x N {3,5} x { native , CP R=V , CP R=V/4 }.
#   vocab 1024 -> CP ranks {1024, 256};  vocab 4096 -> CP ranks {4096, 1024}.
# Native anchor is shared across the two ranks within a (vocab,N) cell.
# 4 native + 8 CP = 12 runs, ~2.1h each, one at a time.
#
# DECISION POINT: run the most paper-faithful cell first (vocab1024 N5 R=1024 vs
# native). If CP still loses at R/V~1 there, the rest are unlikely to flip.
#
# READ: within each (vocab,N) cell, does CP min-bpb at R=V TIE native (paper) or
# still lose (our trend extended)? Does R=V/4 lose more than R=V (R/V dependence)?

set -euo pipefail
REPO_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"
cd "$REPO_DIR"

RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/runs/logs/tngram_smallvocab-${RUN_SUFFIX}}"
NPROC="${NPROC:-4}"
NUM_ITERATIONS="${NUM_ITERATIONS:-3814}"
MHC_STREAMS="${MHC_STREAMS:-4}"
PYTHON="${PYTHON:-$REPO_DIR/../nanochat-moe/.venv/bin/python}"
TORCHRUN="${TORCHRUN:-$REPO_DIR/../nanochat-moe/.venv/bin/torchrun}"
BASE_ROOT="${BASE_ROOT:-/mnt/3fs/dots-pretrain/ningzhiyuan/local_exps}"

# Ordered so the decision-point cell (vocab1024, N5) runs first.
VOCABS="${VOCABS:-1024 4096}"
NS="${NS:-5 3}"

export TORCHDYNAMO_CACHE_SIZE_LIMIT="${TORCHDYNAMO_CACHE_SIZE_LIMIT:-64}"
export PYTHONPATH="${PYTHONPATH:-$REPO_DIR}"

mkdir -p "$LOG_DIR"

# common flags shared by every arm (paper-faithful: no tokenizer compression)
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
    local vocab="$1" n="$2" arm="$3"    # arm = "native" | rank number
    local base_dir="$BASE_ROOT/nsmb_out_v${vocab}"
    local tag extra=()
    if [ "$arm" = "native" ]; then
        tag="sv${vocab}_N${n}_native_s0"
    else
        tag="sv${vocab}_N${n}_R${arm}_s0"
        extra=(--engram-value-table=tngram --engram-cp-rank="$arm")
    fi
    local log_file="$LOG_DIR/${tag}.log"
    if grep -q "^Total training time" "$log_file" 2>/dev/null; then
        echo "Skipping $tag (already completed)"; return
    fi
    if [ ! -f "$base_dir/tokenizer/tokenizer.pkl" ]; then
        echo "  !! $tag: tokenizer missing at $base_dir -- run prep_smallvocab_tokenizers.sh first"; return
    fi
    echo "================================================================"
    echo "Starting $tag  ($(date -Is))  vocab=$vocab N=$n arm=$arm"
    echo "================================================================"
    NANOCHAT_BASE_DIR="$base_dir" "$TORCHRUN" --standalone --nproc_per_node="$NPROC" \
        -m scripts.base_train -- \
        "${COMMON[@]}" --engram-max-ngram-size="$n" "${extra[@]}" \
        --model-tag="d20-sv${vocab}-N${n}-${arm}-${RUN_SUFFIX}" \
        > "$log_file" 2>&1 || {
            echo "  !! $tag FAILED (exit $?)"; tail -20 "$log_file"; return
        }
    echo "  $tag done: $(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$log_file" | tail -1)"
}

for vocab in $VOCABS; do
    if   [ "$vocab" = "1024" ]; then ranks="1024 256"
    elif [ "$vocab" = "4096" ]; then ranks="4096 1024"
    else ranks="$vocab $((vocab/4))"; fi
    for n in $NS; do
        run_arm "$vocab" "$n" native          # anchor first
        for r in $ranks; do run_arm "$vocab" "$n" "$r"; done
    done
done

echo
echo "TN-gram small-vocab sweep complete. Logs: $LOG_DIR"
printf "%-26s %-8s %s\n" arm R/V min_bpb
for vocab in $VOCABS; do
    if   [ "$vocab" = "1024" ]; then ranks="1024 256"
    elif [ "$vocab" = "4096" ]; then ranks="4096 1024"
    else ranks="$vocab $((vocab/4))"; fi
    for n in $NS; do
        for arm in native $ranks; do
            if [ "$arm" = "native" ]; then tag="sv${vocab}_N${n}_native_s0"; rv="-"
            else tag="sv${vocab}_N${n}_R${arm}_s0"; rv=$(awk "BEGIN{printf \"%.3f\", $arm/$vocab}"); fi
            f="$LOG_DIR/${tag}.log"; [ -f "$f" ] || continue
            printf "%-26s %-8s %s\n" "$tag" "$rv" \
                "$(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$f" | tail -1)"
        done
    done
done
echo
echo "READ: within each (vocab,N) cell, CP R=V (R/V~1) vs native. Paper ties at R/V~1;"
echo "our prior 32k runs (R/V~0.002-0.017) all lost. Does small vocab reproduce the tie?"
