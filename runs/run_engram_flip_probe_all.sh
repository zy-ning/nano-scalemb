#!/bin/bash

# Engram CONTENT-sensitivity probe via in-distribution token flip.
#
# Backbone prompt fixed ("The capital of France is"); the Engram stream is a real
# re-tokenized sentence that either agrees (self: "...France is") or swaps the fact
# (flip: "...Japan is"). The Engram right-aligns its stream to the backbone and the
# entity sits in the order-2/3 n-gram window at the prediction position, so the flip
# lands exactly where the backbone queries -- a clean, position-aligned, in-distribution
# causal intervention (unlike the donor probe's tiled foreign document).
#
# Load-bearing readout = directional signal: flipping France->Japan should raise the
# matching capital (Tokyo) relative to the home capital (Paris). Run on real + the two
# content-free controls:
#   - uniform : identical table rows -> flip is a no-op -> signal ~0 (probe floor).
#   - randomize : distinct frozen rows, no learned meaning -> any signal is surface,
#     and should NOT be directional toward the matching capital.
#   - real : the test. Positive directional signal => the read injects factual content.
#
# CPU-only; uses the sibling nanochat venv + checkpoints (see project memory).

set -euo pipefail

REPO_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"
cd "$REPO_DIR"

PYTHON="${PYTHON:-$REPO_DIR/../nanochat/.venv/bin/python}"
export NANO_SCALEMB_BASE_DIR="${NANO_SCALEMB_BASE_DIR:-$REPO_DIR/../nanochat/data}"
export NANOCHAT_BASE_DIR="$NANO_SCALEMB_BASE_DIR"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export TOKENIZERS_PARALLELISM=false

CASES_FILE="${CASES_FILE:-runs/engram_flip_probe_21218_cases.json}"
SUFFIX="-ablation-true-paper-mhc-mhcheadfix-20260507"

declare -A TAGS=(
  [real]="nano-engram-d24-real-21218${SUFFIX}"
  [randomize]="nano-engram-d24-randomize-21218${SUFFIX}"
  [uniform]="nano-engram-d24-uniform-21218${SUFFIX}"
)

for variant in real randomize uniform; do
  tag="${TAGS[$variant]}"
  out="$REPO_DIR/runs/reports/engram_flip_probe_21218/$variant"
  mkdir -p "$out"
  echo "================ flip probe :: $variant ================"
  "$PYTHON" -m scripts.engram_flip_eval \
      --source base \
      --model-tag "$tag" \
      --cases-file "$CASES_FILE" \
      --output-dir "$out" \
      --device-type cpu \
      --dtype float32 \
      --top-k 10 | tail -3
  echo "wrote $out/results.json"
done
