#!/bin/bash

# Tiny Engram layer-sweep smoke test.
# This is intentionally small: it exercises tokenizer/data loading, model build,
# Engram insertion at different layers, one train step, and checkpoint writing.

set -euo pipefail

REPO_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"
cd "$REPO_DIR"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NANO_SCALEMB_BASE_DIR="${NANO_SCALEMB_BASE_DIR:-${NANOCHAT_BASE_DIR:-$REPO_DIR/data}}"
export NANOCHAT_BASE_DIR="$NANO_SCALEMB_BASE_DIR"
mkdir -p "$NANO_SCALEMB_BASE_DIR"

LAYERS="${LAYERS:-0 1}"
DEPTH="${DEPTH:-2}"
ASPECT_RATIO="${ASPECT_RATIO:-64}"
HEAD_DIM="${HEAD_DIM:-64}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-32}"
NUM_ITERATIONS="${NUM_ITERATIONS:-1}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-32}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-1}"
ENGRAM_MEMORY_DIM="${ENGRAM_MEMORY_DIM:-32}"
ENGRAM_HEADS_PER_NGRAM="${ENGRAM_HEADS_PER_NGRAM:-1}"
ENGRAM_SLOT_MULTIPLIER="${ENGRAM_SLOT_MULTIPLIER:-1}"

for layer in $LAYERS; do
    model_tag="${MODEL_TAG_PREFIX:-smoke-layer}-${layer}"
    echo "================================================================"
    echo "Starting layer smoke case: layer=$layer model_tag=$model_tag"
    echo "NANO_SCALEMB_BASE_DIR=$NANO_SCALEMB_BASE_DIR"
    echo "================================================================"

    python -m scripts.base_train \
        --depth="$DEPTH" \
        --aspect-ratio="$ASPECT_RATIO" \
        --head-dim="$HEAD_DIM" \
        --max-seq-len="$MAX_SEQ_LEN" \
        --num-iterations="$NUM_ITERATIONS" \
        --target-param-data-ratio=-1 \
        --total-batch-size="$TOTAL_BATCH_SIZE" \
        --device-batch-size="$DEVICE_BATCH_SIZE" \
        --eval-every=-1 \
        --core-metric-every=-1 \
        --sample-every=-1 \
        --save-every=-1 \
        --window-pattern=L \
        --engram \
        --engram-layers="$layer" \
        --engram-fusion-mode=legacy \
        --engram-no-tokenizer-compression \
        --engram-memory-dim="$ENGRAM_MEMORY_DIM" \
        --engram-heads-per-ngram="$ENGRAM_HEADS_PER_NGRAM" \
        --engram-slot-multiplier="$ENGRAM_SLOT_MULTIPLIER" \
        --run=dummy \
        --model-tag="$model_tag"
done

echo "Layer sweep smoke complete. Check $NANO_SCALEMB_BASE_DIR/base_checkpoints/."
