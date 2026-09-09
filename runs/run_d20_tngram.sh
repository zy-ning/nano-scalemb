#!/bin/bash
#
# TN-gram (Tensorized Engram, arXiv 2606.08347): does ONE CP-factorized tensor
# whose token-position factors are SHARED across n-gram orders match the native
# per-order hash tables at the SAME value-param budget?
#
# CONTEXT. Our Engram keeps a SEPARATE hash table per n-gram order (orders {2,3},
# n_head_per_ngram=8 heads each), so nested n-grams share no latent structure and
# every order pays its own collisions -- exactly the design that paper critiques.
# TN-gram replaces the row tables with a rank-R CP tensor: shared factors A_1..A_N
# indexed by the window tokens directly, an explicit value factor F, learnable
# order-absorption vectors w, per-order RMSNorm/scale. Value params drop from
# O(V^n d) to O(n V R + d R) ~ 569,120*R.
#
# EXPERIMENT. Param-matched to the native slot tables (value params only):
#   R=10 -> 5,691,216 params ~ native slot15 (5,685,536; token floor 0.764396)
#   R=20 -> 11,382,416 params ~ native slot30 (11,370,359; ~0.76393)
# Everything else is the resolved d20 recipe (share_memory, tokens, mHC 4). No
# merge (a factorized table has no rows to cluster).
#
# READ (pre-registered):
#   R=10 at/below native slot15 0.76486  => CP factorization matches at 1x budget.
#   R=20 at/below native slot30 0.76393  => matches at 2x budget.
#   both above their anchors             => sharing structure across orders costs
#                                            more than the collisions it removes.
# Anchors: native slot15 ~0.76486 (token floor 0.764396) ; native slot30 ~0.76393.

set -euo pipefail
REPO_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"
cd "$REPO_DIR"

RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/runs/logs/d20_tngram-${RUN_SUFFIX}}"
NPROC="${NPROC:-4}"
NUM_ITERATIONS="${NUM_ITERATIONS:-3814}"
MHC_STREAMS="${MHC_STREAMS:-4}"
PYTHON="${PYTHON:-$REPO_DIR/../nanochat-moe/.venv/bin/python}"
TORCHRUN="${TORCHRUN:-$REPO_DIR/../nanochat-moe/.venv/bin/torchrun}"

# arms: "<cp_rank>:<anchor label>"   seeds: SEEDS
ARMS="${ARMS:-10 20}"
SEEDS="${SEEDS:-0}"

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/mnt/3fs/dots-pretrain/ningzhiyuan/local_exps/nsmb_out}"
export TORCHDYNAMO_CACHE_SIZE_LIMIT="${TORCHDYNAMO_CACHE_SIZE_LIMIT:-64}"
export PYTHONPATH="${PYTHONPATH:-$REPO_DIR}"

mkdir -p "$LOG_DIR"

run_arm () {
    local rank="$1" seed="$2"
    local tag="tngram_R${rank}_s${seed}"
    local log_file="$LOG_DIR/${tag}.log"
    if grep -q "^Total training time" "$log_file" 2>/dev/null; then
        echo "Skipping $tag (already completed)"; return
    fi
    echo "================================================================"
    echo "Starting $tag  ($(date -Is))  cp_rank=$rank seed=$seed"
    echo "================================================================"
    "$TORCHRUN" --standalone --nproc_per_node="$NPROC" -m scripts.base_train -- \
        --depth=20 --max-seq-len=2048 --device-batch-size=8 --total-batch-size=524288 \
        --num-iterations="$NUM_ITERATIONS" --target-param-data-ratio=-1 \
        --window-pattern=SSSL --mhc --mhc-num-streams="$MHC_STREAMS" \
        --engram --engram-layers=2,6 --engram-memory-dim=1280 \
        --engram-mhc-num-streams="$MHC_STREAMS" --engram-share-memory \
        --engram-value-table=tngram --engram-cp-rank="$rank" \
        --engram-seed="$seed" \
        --eval-every=500 --eval-tokens=10485760 \
        --core-metric-every=-1 --sample-every=-1 --save-every=-1 \
        --model-tag="d20-tngram-${tag}-${RUN_SUFFIX}" \
        > "$log_file" 2>&1 || {
            echo "  !! $tag FAILED (exit $?)"; tail -20 "$log_file"; return
        }
    echo "  $tag done: $(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$log_file" | tail -1)"
}

for rank in $ARMS; do
    for s in $SEEDS; do run_arm "$rank" "$s"; done
done

echo
echo "TN-gram sweep complete. Logs: $LOG_DIR"
printf "%-22s %s\n" arm min_bpb
for rank in $ARMS; do
    for s in $SEEDS; do
        tag="tngram_R${rank}_s${s}"; f="$LOG_DIR/${tag}.log"; [ -f "$f" ] || continue
        printf "%-22s %s\n" "$tag" \
            "$(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$f" | tail -1)"
    done
done
echo
echo "anchors: R10 vs native slot15 ~0.76486 (floor 0.764396) ; R20 vs native slot30 ~0.76393"
