#!/bin/bash

# Donor probe v2 for the real Engram+mHC checkpoint at layers 2,12,18.
#
# Improves on v1 by separating CONTENT-sensitivity from CONSISTENCY-sensitivity.
# v1 measured every donor as a delta against `self` (engram stream = the true
# prompt), so every swap broke prompt/engram consistency and every delta was
# negative, conflating "the content is wrong" with "I changed the input".
#
# v2 adds donor conditions that all break consistency equally (matched /
# adversarial / unrelated / neutral), so the only thing varying among them is the
# donor's *content*. The load-bearing contrast is matched vs adversarial (same
# topic, same structure, only the fact differs): matched > adversarial means the
# read is sensitive to factual correctness, not merely to topic or to consistency.
# Cases span known facts, rare-real facts (decisiveness), and one synthetic fact
# (a learned-table memory should NOT be able to inject a novel token).
#
# CPU-only; uses the sibling nanochat venv and its checkpoints (see project memory).

set -euo pipefail

REPO_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"
cd "$REPO_DIR"

PYTHON="${PYTHON:-$REPO_DIR/../nanochat/.venv/bin/python}"
export NANO_SCALEMB_BASE_DIR="${NANO_SCALEMB_BASE_DIR:-$REPO_DIR/../nanochat/data}"
export NANOCHAT_BASE_DIR="$NANO_SCALEMB_BASE_DIR"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export TOKENIZERS_PARALLELISM=false

MODEL_TAG="${MODEL_TAG:-nano-engram-d24-real-21218-ablation-true-paper-mhc-mhcheadfix-20260507}"
CASES_FILE="${CASES_FILE:-runs/engram_donor_probe_21218_v2_cases.json}"
REPORT_DIR="${REPORT_DIR:-$REPO_DIR/runs/reports/engram_donor_probe_21218_v2}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/runs/logs/engram_donor_probe_21218_v2}"
mkdir -p "$REPORT_DIR" "$LOG_DIR"

LOG_FILE="$LOG_DIR/engram_donor_probe_21218_v2.log"

"$PYTHON" -m scripts.engram_donor_eval \
    --source base \
    --model-tag "$MODEL_TAG" \
    --cases-file "$CASES_FILE" \
    --output-dir "$REPORT_DIR" \
    --device-type cpu \
    --dtype float32 \
    --top-k 10 | tee "$LOG_FILE"

echo "Wrote results to $REPORT_DIR/results.{json,csv}"
