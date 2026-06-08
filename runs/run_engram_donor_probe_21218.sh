#!/bin/bash

# Donor probe for the real Engram+mHC checkpoint at layers 2,12,18 (the topology
# used throughout the blog's ablation study). Swaps the Engram donor stream
# (engram_input_ids) for matched / adversarial / unrelated text and measures the
# change in the target token's logit. Produces the data behind docs/blog/donor_probe.html.

set -euo pipefail

REPO_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"
cd "$REPO_DIR"

LOG_DIR="${LOG_DIR:-$REPO_DIR/runs/logs/engram_donor_probe_21218}"
REPORT_DIR="${REPORT_DIR:-$REPO_DIR/runs/reports/engram_donor_probe_21218}"
mkdir -p "$LOG_DIR"
mkdir -p "$REPORT_DIR"

LOG_FILE="$LOG_DIR/engram_donor_probe_21218.log"

source .venv/bin/activate
python -m nano_scalemb.report reset
python -m scripts.engram_donor_eval \
    --source base \
    --model-tag nano-engram-d24-real-21218-mhc \
    --cases-file runs/engram_donor_probe_21218_cases.json \
    --output-dir "$REPORT_DIR" \
    --top-k 10 | tee "$LOG_FILE"
python -m nano_scalemb.report generate
cp report.md "$REPORT_DIR/report.md"
