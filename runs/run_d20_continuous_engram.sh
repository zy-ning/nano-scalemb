#!/bin/bash

# d20 continuous-Engram sweep: what should the memory be addressed BY?
#
# The token Engram hashes raw token ids, so it can only key on surface form --
# "New York" and "NYC" land on unrelated rows. These arms derive the address
# from the layer's hidden state instead, so the memory is keyed by what the
# model currently represents.
#
#   arm            address source                       learned?  differentiable?
#   tokens         hash token ids (paper baseline)       no        --
#   lsh            frozen random projection -> sign bits no        --
#   pq             product-quantization codebooks        yes       via STE
#   pkm            product keys, soft top-k read         yes       fully
#   lsh_n1         LSH, unigram (no temporal window)     no        --
#
# READ THE LADDER IN THIS ORDER. `lsh` is the arm that carries the argument: it
# changes ONLY the address source, holding "fixed addressing, learned contents"
# constant, so lsh-vs-tokens isolates semantics from surface form. `pq` also
# makes addressing learnable, and `pkm` additionally makes the READ soft (a
# mixture of topk rows rather than one hard row), so a win there confounds two
# or three changes at once.
#
# `lsh_n1` ablates the temporal window: with contextual addressing the hidden
# state already carries context, so the n-gram shift may be redundant. Compare
# only against `lsh`, not across sources.
#
# Sizing: matched on ACTIVE params (918.3M, same as the completed iso sweep) and
# on TOTAL where achievable.
#   tokens slot=15   1828.4M (+0.46%)
#   lsh    slot=11   1841.8M (+1.20%)   b15: 2^15 ~ the 23686-symbol compressed
#                                       token alphabet the baseline used
#   pq     slot=86   1820.5M (+0.03%)   64^2 codes
#   pkm    nk=640    1982.6M (+8.94%)   n_keys moves total QUADRATICALLY, so
#                                       iso-total is not reachable; 640 is the
#                                       closest rung. Report the gap.
#
# Baselines for comparison come from runs/logs/d20_iso-20260811-042109:
#   dense 0.772614 | moe 0.766529 | mobius 0.764727
#   engram 0.767555 | engram_shared 0.763784
# The `tokens` arm here re-runs the engram baseline as a same-conditions control.

set -euo pipefail

REPO_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"
cd "$REPO_DIR"

RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/runs/logs/d20_continuous-${RUN_SUFFIX}}"
NPROC="${NPROC:-4}"
DEPTH="${DEPTH:-20}"
SEQ="${SEQ:-2048}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-8}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-524288}"
NUM_ITERATIONS="${NUM_ITERATIONS:-3814}"   # ~2.0B tokens, same budget as the iso sweep
MHC_STREAMS="${MHC_STREAMS:-4}"
ENGRAM_LAYERS="${ENGRAM_LAYERS:-2,6}"
# The venv's torchrun: the system one resolves to a python whose torch import
# dies on libucc.
TORCHRUN="${TORCHRUN:-$REPO_DIR/../nanochat-moe/.venv/bin/torchrun}"

export TORCHDYNAMO_CACHE_SIZE_LIMIT="${TORCHDYNAMO_CACHE_SIZE_LIMIT:-64}"
export PYTHONPATH="${PYTHONPATH:-$REPO_DIR}"

mkdir -p "$LOG_DIR"

run_arm () {
    local tag="$1"; shift
    local log_file="$LOG_DIR/${tag}.log"

    if grep -q "^Total training time" "$log_file" 2>/dev/null; then
        echo "Skipping $tag (already completed)"
        return
    fi

    echo "================================================================"
    echo "Starting $tag  ($(date -Is))"
    echo "  extra: $*"
    echo "  log=$log_file"
    echo "================================================================"

    "$TORCHRUN" --standalone --nproc_per_node="$NPROC" -m scripts.base_train -- \
        --depth="$DEPTH" \
        --max-seq-len="$SEQ" \
        --device-batch-size="$DEVICE_BATCH_SIZE" \
        --total-batch-size="$TOTAL_BATCH_SIZE" \
        --num-iterations="$NUM_ITERATIONS" \
        --target-param-data-ratio=-1 \
        --window-pattern=SSSL \
        --mhc --mhc-num-streams="$MHC_STREAMS" \
        --engram --engram-layers="$ENGRAM_LAYERS" \
        --engram-memory-dim=1280 \
        --engram-mhc-num-streams="$MHC_STREAMS" \
        --eval-every=500 --eval-tokens=10485760 \
        --core-metric-every=-1 --sample-every=-1 --save-every=-1 \
        --model-tag="d20-ce-${tag}-${RUN_SUFFIX}" \
        "$@" > "$log_file" 2>&1 || {
            echo "  !! $tag FAILED (exit $?)"; tail -20 "$log_file"; return
        }
    echo "  $tag done: $(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$log_file" | tail -1)"
}

# control: token addressing, same conditions as everything else
run_arm tokens --engram-slot-multiplier=15

# the clean control: hidden-state addressing, NO learned addressing
run_arm lsh --engram-address-source=lsh --engram-lsh-bits=15 --engram-slot-multiplier=11

# learned quantized addressing
run_arm pq --engram-address-source=pq --engram-pq-subspaces=2 \
    --engram-pq-codebook-size=64 --engram-slot-multiplier=86

# learned differentiable addressing + soft read
run_arm pkm --engram-address-source=pkm --engram-pkm-n-keys=640 \
    --engram-pkm-topk=32 --engram-pkm-query-dim=256 --engram-slot-multiplier=1

# does the temporal n-gram window still earn its place under contextual addressing?
run_arm lsh_n1 --engram-address-source=lsh --engram-lsh-bits=15 \
    --engram-slot-multiplier=11 --engram-max-ngram-size=1

echo
echo "d20 continuous-Engram sweep complete. Logs: $LOG_DIR"
echo
printf "%-10s %-12s %-10s %s\n" arm min_bpb tok/sec commit
for tag in tokens lsh pq pkm lsh_n1; do
    f="$LOG_DIR/${tag}.log"
    [ -f "$f" ] || continue
    printf "%-10s %-12s %-10s %s\n" "$tag" \
        "$(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$f" | tail -1)" \
        "$(grep -oP 'tok/sec: \K[0-9,]+' "$f" | tail -1)" \
        "$(grep -oP 'vq_commitment: \K[0-9.]+' "$f" | tail -1)"
done
echo
echo "iso-sweep baselines: dense 0.772614 | engram 0.767555 | engram_shared 0.763784"
