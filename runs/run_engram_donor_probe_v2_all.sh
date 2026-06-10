#!/bin/bash

# Donor probe v2, expanded case set, run on the real checkpoint plus the two
# content-free controls (randomize, uniform). Controls validate the probe:
#   - uniform : every table row identical -> lookup returns the SAME vector for
#     any n-gram -> donor swap must have ~zero effect on anything (probe floor).
#   - randomize : distinct frozen rows -> donor swap changes retrieved vectors
#     (surface/addressing effect) but the rows carry no learned meaning.
#   - real : the test. If real ~ randomize (consistency/surface, no matched>adv
#     content effect) then its read is random-fingerprint-like, not fact retrieval.
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

CASES_FILE="${CASES_FILE:-runs/engram_donor_probe_21218_v2_cases.json}"
SUFFIX="-ablation-true-paper-mhc-mhcheadfix-20260507"

declare -A TAGS=(
  [real]="nano-engram-d24-real-21218${SUFFIX}"
  [randomize]="nano-engram-d24-randomize-21218${SUFFIX}"
  [uniform]="nano-engram-d24-uniform-21218${SUFFIX}"
)

for variant in real randomize uniform; do
  tag="${TAGS[$variant]}"
  out="$REPO_DIR/runs/reports/engram_donor_probe_21218_v2/$variant"
  mkdir -p "$out"
  echo "================ donor probe v2 :: $variant ================"
  "$PYTHON" -m scripts.engram_donor_eval \
      --source base \
      --model-tag "$tag" \
      --cases-file "$CASES_FILE" \
      --output-dir "$out" \
      --device-type cpu \
      --dtype float32 \
      --top-k 10 | tail -2
  echo "wrote $out/results.json"
done
