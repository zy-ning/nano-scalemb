#!/bin/bash

# d20 idea: SEMANTIC-PRIOR row merge -- a token-indexed value-aware collision.
#
# BACKGROUND. The over-provision->value-merge schedule wins at 2x but PLATEAUS
# (op2x 0.764795 best; op4x 0.766214, op8x 0.765504 both WORSE than op1x floor
# 0.765322). Root cause (this thread): the merge clusters on HALF-TRAINED values,
# so beyond 2x the sparser pretrain table learns noisier values before the merge
# and the partition degrades. Bottleneck = value QUALITY, not collision count.
#
# THE IDEA (user's reframed "semantic hash"). Cluster on a STATIC feature that
# does NOT depend on mid-training value quality: a frozen token-embedding centroid
# per hash row, built offline from a fully-trained wte (scripts/
# build_engram_semantic_prior.py). Rows whose colliding n-grams are semantically
# similar merge; the table then trains ALREADY-collapsed onto that partition
# (base_train fires merge_source=semantic at STEP 0). Still TOKEN-indexed -- a row
# is a hash(token-window) bucket, exactly as the baseline -- so this is NOT the
# closed hidden-state idea-2 (lsh/pq addressing, +0.004-0.006 liabilities).
#
#   arm         slot  merge                          final eff. rows   question
#   sem_prior   30    0.5 @step0, source=semantic    ~5.69M            does a SEMANTIC collision beat random? <--
#   sem_value   30    0.5 @50%,  source=value        ~5.69M            same-session op2x control (0.764795)
#
# The floor anchor is the known TOKEN slot15 baseline 0.764396 (not re-run).
#
# PRE-REGISTERED READS (decide before looking):
#   sem_prior < 0.764396 by >= seed bar = a GOOD (semantic) collision structure
#     imposed from the start beats random hash collision. Reopens the semantic line
#     the RIGHT way (token-indexed, not hidden-state).
#   sem_prior ~ 0.764396 = the prior adds nothing over random collision -- consistent
#     with "collisions in a learned table are benign" (the whole thread's finding).
#   sem_prior > floor = the imposed partition destroys useful row diversity (bounds
#     how absorbed the superposition is).
#   sem_prior vs sem_value = does the static fully-trained prior beat the
#     half-trained-value partition at the SAME 2x-merge capacity?
#
# NOT PARAM-ISO across the floor (slot30 carries 2x engram embeds + optimizer state;
# merged rows stay allocated but dead). Iso on FINAL effective capacity + inference
# FLOPs only. The prior .pt is a frozen buffer, not a learnable param. Within-bracket
# (same seed, same code) comparisons are clean.
#
# GATE: the prior build must report row-coverage; a low coverage (<~90% of a head's
# rows hit) means --top-k is too small and most rows can't be merged. And sem_prior
# must print an effective-row count within ~0.5% of 5.69M (the merge fired).

set -euo pipefail

REPO_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"
cd "$REPO_DIR"

RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/runs/logs/d20_sem_prior-${RUN_SUFFIX}}"
NPROC="${NPROC:-4}"
NUM_ITERATIONS="${NUM_ITERATIONS:-3814}"
MHC_STREAMS="${MHC_STREAMS:-4}"
SEED="${SEED:-0}"
MERGE_METRIC="${MERGE_METRIC:-lsh}"          # cluster metric (lsh scales to millions of rows)
TOP_K="${TOP_K:-8000000}"                    # frequent order-3 windows to scatter into the prior
PROJ_DIM="${PROJ_DIM:-128}"
PYTHON="${PYTHON:-$REPO_DIR/../nanochat-moe/.venv/bin/python}"
TORCHRUN="${TORCHRUN:-$REPO_DIR/../nanochat-moe/.venv/bin/torchrun}"

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/mnt/3fs/dots-pretrain/ningzhiyuan/local_exps/nsmb_out}"
export TORCHDYNAMO_CACHE_SIZE_LIMIT="${TORCHDYNAMO_CACHE_SIZE_LIMIT:-64}"
export PYTHONPATH="${PYTHONPATH:-$REPO_DIR}"

COUNTS="${COUNTS:-$NANOCHAT_BASE_DIR/engram_backoff/counts_2b_o3.npz}"
# a completed d20 engram_shared run: its trained wte seeds the semantic centroids.
WTE_CKPT="${WTE_CKPT:-$NANOCHAT_BASE_DIR/base_checkpoints/d20-ce4-engram_shared_seed1-20260813-123441/model_003814.pt}"
PRIOR="${PRIOR:-$NANOCHAT_BASE_DIR/engram_semantic/prior_slot30_s${SEED}_k${TOP_K}.pt}"

WAIT_FOR_LOG="${WAIT_FOR_LOG:-}"
WAIT_FOR_MARKER="${WAIT_FOR_MARKER:-Total training time}"
WAIT_MAX_HOURS="${WAIT_MAX_HOURS:-24}"

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
    echo "Marker found; starting."
fi

# ----------------------------------------------------------------------------
# 0. Build the semantic prior once (idempotent: skip if it already exists).
# ----------------------------------------------------------------------------
if [ -f "$PRIOR" ]; then
    echo "Semantic prior exists: $PRIOR (skipping build)"
else
    echo "Building semantic prior -> $PRIOR (top_k=$TOP_K, proj_dim=$PROJ_DIM)"
    "$PYTHON" -m scripts.build_engram_semantic_prior \
        --counts "$COUNTS" --wte-ckpt "$WTE_CKPT" \
        --slot-multiplier 30 --engram-seed "$SEED" \
        --proj-dim "$PROJ_DIM" --top-k "$TOP_K" \
        --out "$PRIOR" 2>&1 | tee "$LOG_DIR/prior_build.log"
fi

# $1 tag, $2 merge_source, $3 extra flags
run_arm () {
    local tag="$1"; local source="$2"; shift 2
    local log_file="$LOG_DIR/${tag}.log"
    if grep -q "^Total training time" "$log_file" 2>/dev/null; then
        echo "Skipping $tag (already completed)"; return
    fi
    echo "================================================================"
    echo "Starting $tag  ($(date -Is))  merge_source=$source"
    echo "================================================================"
    "$TORCHRUN" --standalone --nproc_per_node="$NPROC" -m scripts.base_train -- \
        --depth=20 --max-seq-len=2048 --device-batch-size=8 --total-batch-size=524288 \
        --num-iterations="$NUM_ITERATIONS" --target-param-data-ratio=-1 \
        --window-pattern=SSSL --mhc --mhc-num-streams="$MHC_STREAMS" \
        --engram --engram-layers=2,6 --engram-memory-dim=1280 \
        --engram-mhc-num-streams="$MHC_STREAMS" --engram-share-memory \
        --engram-slot-multiplier=30 \
        --engram-merge-frac=0.5 --engram-merge-metric="$MERGE_METRIC" \
        --engram-merge-source="$source" \
        --engram-seed="$SEED" \
        --eval-every=500 --eval-tokens=10485760 \
        --core-metric-every=-1 --sample-every=-1 --save-every=-1 \
        --model-tag="d20-semprior-${tag}-${RUN_SUFFIX}" \
        "$@" > "$log_file" 2>&1 || {
            echo "  !! $tag FAILED (exit $?)"; tail -20 "$log_file"; return
        }
    echo "  $tag done: $(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$log_file" | tail -1)"
}

# sem_prior FIRST (the headline). semantic merge fires at step 0.
run_arm "sem_prior" "semantic" --engram-semantic-prior-path="$PRIOR"
# same-session control: the op2x value-merge at 50% (should reproduce ~0.764795).
run_arm "sem_value" "value" --engram-merge-at-frac=0.5

echo
echo "Semantic-prior merge scout complete. Logs: $LOG_DIR"
printf "%-14s %-10s %-12s %s\n" arm source min_bpb eff_rows
for row in "sem_prior semantic" "sem_value value"; do
    set -- $row; tag="$1"; src="$2"
    f="$LOG_DIR/${tag}.log"; [ -f "$f" ] || continue
    printf "%-14s %-10s %-12s %s\n" "$tag" "$src" \
        "$(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$f" | tail -1)" \
        "$(grep -oaE '\-> [0-9,]+ effective rows' "$f" | tail -1 | grep -oE '[0-9,]+' | head -1)"
done
echo
echo "anchors: TOKEN slot15 floor 0.764396 ; op2x value-merge 0.764795 (this session)."
echo "READ: sem_prior < 0.764396 = semantic collision beats random; ~floor = prior adds"
echo "nothing (collisions benign); sem_prior vs sem_value = static prior vs half-trained values."
