#!/bin/bash

# d20 round 12: decouple "this n-gram is frequent" from "this row is crowded",
# then re-run the Katz test that the conflation invalidated.
#
# WHY. The Fourier gate (§9.2, round 10) came back MONOTONE: given a basis that
# provably fits a unimodal trust curve (tests/test_engram.py::TestCountGateFourier
# fits one to 5e-3 while the linear gate cannot get below 5e-2), the model
# produced plain inverse-frequency downweighting with its peak at count 0. The
# likeliest reason is that the gate's only input -- a ROW's hit rate -- conflates
# two causes that both argue for downweighting:
#   a genuinely frequent n-gram   -> reliable content, but generic
#   a crowded row (many collisions) -> mush
# No non-monotonicity can appear in a signal where both causes push the same way.
# Katz discounting is about the reliability of a count for a SPECIFIC n-gram, and
# we never gave the gate that quantity.
#
# THE DECOMPOSITION IS FREE. Every head of an n-gram order hashes the SAME n-gram
# under a DIFFERENT prime. So frequency-driven heat is common to all of an order's
# heads, while collision-driven heat is independent across them:
#   f_o = min over the heads of order o   -> that n-gram's own rate (tightest bound)
#   c_h = rate_h - f_o                    -> head h's collision excess, >= 0
# rate_h == f_o + c_h exactly -- a decomposition, not an approximation, and it
# needs no cardinality sketch and no new buffers, because all H rates are already
# gathered per position. One min-reduce over 8 heads.
#
# Strict superset, again deliberately: the raw-rate term is kept, so zeroing the
# new coefficients recovers count_gate / count_gate_fourier bit-for-bit (tested).
# With --engram-count-gate-decouple the Fourier basis also MOVES to the frequency
# channel, since that is the quantity the hypothesis is about.
#
# Sizing verified on meta, iso to 4 decimals: 1828.0236M total / 918.340M active
# for base, +count_gate, and +decouple; 1828.0238M with Fourier. Extra parameters:
# 64 (decouple) or 192 (decouple + K=4). Zero graph breaks under torch.compile.
#
# THE DELIVERABLE IS THE CURVE, NOT THE BPB. Gate-class arms carry sd 7.5e-4
# (shared), so even n=3 only resolves ~0.00095, and the count gate's own effect
# sits at -0.00076 pooled and unresolved after 12 runs. Do not expect these arms
# to settle bpb; arms 1 and 3 exist to answer ONE question from their checkpoints:
#   with collision load in its own term, is trust NON-MONOTONE in f_o?
# If yes, the Katz mechanism is real and the earlier monotone result was an
# artifact of the conflated input -- a finding about what the memory needs,
# independent of the metric. If it is monotone again, the hypothesis is dead on
# its own terms and inverse-frequency is simply what an Engram wants.
#
# Arms 2/4/5 are the control: does decoupling alone (no Fourier) move anything?
# n=3 against engram_shared's n=3, resolving ~0.00095.
#
# HONEST PRIOR: seven mechanism ideas in a row have come back neutral or worse.
# I am not predicting a bpb win. The curve is the reason to run this.

set -euo pipefail

REPO_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"
cd "$REPO_DIR"

RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/runs/logs/d20_cgdec-${RUN_SUFFIX}}"
NPROC="${NPROC:-4}"
NUM_ITERATIONS="${NUM_ITERATIONS:-3814}"
MHC_STREAMS="${MHC_STREAMS:-4}"
FOURIER_BANDS="${FOURIER_BANDS:-4}"
TORCHRUN="${TORCHRUN:-$REPO_DIR/../nanochat-moe/.venv/bin/torchrun}"

# Marker string in a log, never a pid or process name (§6).
WAIT_FOR_LOG="${WAIT_FOR_LOG:-}"
WAIT_FOR_MARKER="${WAIT_FOR_MARKER:-Round 11 complete}"
WAIT_MAX_HOURS="${WAIT_MAX_HOURS:-24}"

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/mnt/3fs/dots-pretrain/ningzhiyuan/local_exps/nsmb_out}"
export TORCHDYNAMO_CACHE_SIZE_LIMIT="${TORCHDYNAMO_CACHE_SIZE_LIMIT:-64}"
export PYTHONPATH="${PYTHONPATH:-$REPO_DIR}"

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
    echo "Predecessor finished at $(date -Is); starting round 12"
fi

run_arm () {
    local tag="$1"; shift
    local log_file="$LOG_DIR/${tag}.log"
    if grep -q "^Total training time" "$log_file" 2>/dev/null; then
        echo "Skipping $tag (already completed)"; return
    fi
    echo "================================================================"
    echo "Starting $tag  ($(date -Is))"
    echo "  extra: $*"
    echo "================================================================"
    # device-batch-size fixed at 8 for EVERY arm. Round 11 mixed dbs=8 and dbs=4
    # inside one sweep and the dbs=4 point broke the fit; the bos-bestfit loader
    # packs documents per batch, so dbs changes the token stream, not just the
    # accumulation schedule. Never vary it within a comparison again.
    "$TORCHRUN" --standalone --nproc_per_node="$NPROC" -m scripts.base_train -- \
        --depth=20 --max-seq-len=2048 --device-batch-size=8 --total-batch-size=524288 \
        --num-iterations="$NUM_ITERATIONS" --target-param-data-ratio=-1 \
        --window-pattern=SSSL --mhc --mhc-num-streams="$MHC_STREAMS" \
        --engram --engram-layers=2,6 --engram-memory-dim=1280 \
        --engram-mhc-num-streams="$MHC_STREAMS" --engram-share-memory \
        --engram-slot-multiplier=30 --engram-count-gate \
        --engram-count-gate-decouple \
        --eval-every=500 --eval-tokens=10485760 \
        --core-metric-every=-1 --sample-every=-1 --save-every=-1 \
        --model-tag="d20-cgdec-${tag}-${RUN_SUFFIX}" \
        "$@" > "$log_file" 2>&1 || {
            echo "  !! $tag FAILED (exit $?)"; tail -20 "$log_file"; return
        }
    echo "  $tag done: $(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$log_file" | tail -1)"
}

# --- 1. the curve, as soon as possible -------------------------------------
run_arm cg_dec_four_s0 --engram-count-gate-fourier="$FOURIER_BANDS" --engram-seed=0
# --- 2. control: decoupling without the basis ------------------------------
run_arm cg_dec_s0 --engram-seed=0
# --- 3. does the curve SHAPE replicate? (what made the backoff finding solid)
run_arm cg_dec_four_s1 --engram-count-gate-fourier="$FOURIER_BANDS" --engram-seed=1
# --- 4-5. control to n=3 ---------------------------------------------------
run_arm cg_dec_s1 --engram-seed=1
run_arm cg_dec_s2 --engram-seed=2

echo
echo "Round 12 complete. Logs: $LOG_DIR"
printf "%-18s %-12s %s\n" arm min_bpb tok/sec
for tag in cg_dec_s0 cg_dec_s1 cg_dec_s2 cg_dec_four_s0 cg_dec_four_s1; do
    f="$LOG_DIR/${tag}.log"; [ -f "$f" ] || continue
    printf "%-18s %-12s %s\n" "$tag" \
        "$(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$f" | tail -1)" \
        "$(grep -oP 'tok/sec: \K[0-9,]+' "$f" | tail -1)"
done
echo
echo "references (shared, slot 30, all iso):"
echo "  engram_shared  0.763884  (n=3, sd 3.5e-4)"
echo "  count_gate     0.763136  (n=3, sd 7.5e-4)  -- -0.00076 pooled, UNRESOLVED"
echo
echo "THE DELIVERABLE: read the curve from a Fourier checkpoint and ask whether"
echo "trust is non-monotone in the DECOUPLED frequency channel f_o."
echo "  \$NANOCHAT_BASE_DIR/base_checkpoints/d20-cgdec-cg_dec_four_s0-*/model_003814.pt"
echo "  keys engram_shared_memory.count_gate_{scale,bias,freq_scale,collide_scale,fourier_coef}"
echo "  freqs w_k = pi*2^k/7, k=0..K-1 (non-persistent, regenerate)"
echo "  logit(f,c) = a*log1p(f+c) + b + e*log1p(f) + d*log1p(c)"
echo "               + sum_k [coef_k sin(w_k log1p(f)) + coef_{K+k} cos(w_k log1p(f))]"
echo "  sweep f over [0,1100] at several fixed c, and report monotonicity in f."
echo "  ALSO report the sign of collide_scale: negative = crowded rows distrusted,"
echo "  which is the prediction the whole decomposition rests on."
