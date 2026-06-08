#!/bin/bash

set -euo pipefail

REPO_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"

export OMP_NUM_THREADS=1
export NANO_SCALEMB_BASE_DIR="${NANO_SCALEMB_BASE_DIR:-${NANOCHAT_BASE_DIR:-$REPO_DIR/data}}"
export NANOCHAT_BASE_DIR="$NANO_SCALEMB_BASE_DIR"
mkdir -p "$NANO_SCALEMB_BASE_DIR"

LOG_FILE="${LOG_FILE:-$REPO_DIR/engram.out}"
exec > >(tee -a "$LOG_FILE") 2>&1

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
WANDB_RUN="${WANDB_RUN:-dummy}"
MODEL_TAG="${MODEL_TAG:-nano-engram-d24}"

TOKENIZER_BOOTSTRAP_SHARDS="${TOKENIZER_BOOTSTRAP_SHARDS:-8}"
PRETRAIN_SHARDS="${PRETRAIN_SHARDS:-170}"

DEPTH="${DEPTH:-24}"
TARGET_PARAM_DATA_RATIO="${TARGET_PARAM_DATA_RATIO:-9.5}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-16}"
BASE_EVAL_DEVICE_BATCH_SIZE="${BASE_EVAL_DEVICE_BATCH_SIZE:-16}"
SFT_DEVICE_BATCH_SIZE="${SFT_DEVICE_BATCH_SIZE:-16}"

ENGRAM_LAYERS="${ENGRAM_LAYERS:-2,4,6,8,10}"
ENGRAM_MEMORY_DIM="${ENGRAM_MEMORY_DIM:-1280}"
ENGRAM_MAX_NGRAM_SIZE="${ENGRAM_MAX_NGRAM_SIZE:-3}"
ENGRAM_HEADS_PER_NGRAM="${ENGRAM_HEADS_PER_NGRAM:-8}"
ENGRAM_SLOT_MULTIPLIER="${ENGRAM_SLOT_MULTIPLIER:-18}"
ENGRAM_KERNEL_SIZE="${ENGRAM_KERNEL_SIZE:-4}"
ENGRAM_EMBEDDING_LR_MULT="${ENGRAM_EMBEDDING_LR_MULT:-2.0}"
ENGRAM_PAD_ID="${ENGRAM_PAD_ID:-0}"
ENGRAM_SEED="${ENGRAM_SEED:-42}"
ENGRAM_FUSION_MODE="${ENGRAM_FUSION_MODE:-legacy}"
ENGRAM_MHC_NUM_STREAMS="${ENGRAM_MHC_NUM_STREAMS:-4}"
ENGRAM_MHC_SINKHORN_ITERS="${ENGRAM_MHC_SINKHORN_ITERS:-20}"

BASE_TRAIN_EXTRA_ARGS="${BASE_TRAIN_EXTRA_ARGS:---fp8 --eval-every=-1 --core-metric-every=-1 --sample-every=-1 --save-every=-1}"
BASE_EVAL_ARGS="${BASE_EVAL_ARGS:---eval=core,bpb,sample}"
SFT_EXTRA_ARGS="${SFT_EXTRA_ARGS:---eval-every=-1 --chatcore-every=-1}"
CHAT_EVAL_ARGS="${CHAT_EVAL_ARGS:-}"
REUSE_BASE_CKPT="${REUSE_BASE_CKPT:-1}"

BASE_CHECKPOINT_DIR="$NANO_SCALEMB_BASE_DIR/base_checkpoints/$MODEL_TAG"
if [ "$REUSE_BASE_CKPT" = "1" ] && compgen -G "$BASE_CHECKPOINT_DIR/model_*.pt" > /dev/null; then
  HAVE_BASE_CKPT=1
else
  HAVE_BASE_CKPT=0
fi

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

python -m nano_scalemb.report reset

# if [ "$HAVE_BASE_CKPT" -eq 0 ]; then
python -m nano_scalemb.dataset -n "$TOKENIZER_BOOTSTRAP_SHARDS"
python -m nano_scalemb.dataset -n "$PRETRAIN_SHARDS" &
DATASET_DOWNLOAD_PID=$!
python -m scripts.tok_train
python -m scripts.tok_eval

wait "$DATASET_DOWNLOAD_PID"

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_train -- \
    --depth="$DEPTH" \
    --target-param-data-ratio="$TARGET_PARAM_DATA_RATIO" \
    --device-batch-size="$DEVICE_BATCH_SIZE" \
    --run="$WANDB_RUN" \
    --model-tag="$MODEL_TAG" \
    --engram \
    --engram-layers="$ENGRAM_LAYERS" \
    --engram-memory-dim="$ENGRAM_MEMORY_DIM" \
    --engram-max-ngram-size="$ENGRAM_MAX_NGRAM_SIZE" \
    --engram-heads-per-ngram="$ENGRAM_HEADS_PER_NGRAM" \
    --engram-slot-multiplier="$ENGRAM_SLOT_MULTIPLIER" \
    --engram-kernel-size="$ENGRAM_KERNEL_SIZE" \
    --engram-embedding-lr-mult="$ENGRAM_EMBEDDING_LR_MULT" \
    --engram-pad-id="$ENGRAM_PAD_ID" \
    --engram-seed="$ENGRAM_SEED" \
    --engram-fusion-mode="$ENGRAM_FUSION_MODE" \
    --engram-mhc-num-streams="$ENGRAM_MHC_NUM_STREAMS" \
    --engram-mhc-sinkhorn-iters="$ENGRAM_MHC_SINKHORN_ITERS" \
    $BASE_TRAIN_EXTRA_ARGS
# fi

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_eval -- \
  --model-tag="$MODEL_TAG" \
  --device-batch-size="$BASE_EVAL_DEVICE_BATCH_SIZE" \
  $BASE_EVAL_ARGS

curl -L -o "$NANO_SCALEMB_BASE_DIR/identity_conversations.jsonl" \
  https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.chat_sft -- \
  --model-tag="$MODEL_TAG" \
  --device-batch-size="$SFT_DEVICE_BATCH_SIZE" \
  --run="$WANDB_RUN" \
  $SFT_EXTRA_ARGS

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.chat_eval -- \
  -i sft \
  -g "$MODEL_TAG" \
  $CHAT_EVAL_ARGS

python -m nano_scalemb.report generate
