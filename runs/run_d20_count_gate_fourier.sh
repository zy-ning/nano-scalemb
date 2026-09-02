#!/bin/bash

# d20 round 10: settle whether count gating works at all, then test whether the
# count-to-trust curve wants to be non-monotone.
#
# WHY THE FIRST FOUR ARMS ARE REPLICATES AND NOT NEW IDEAS.
# Round 8 killed the count_gate headline: at n=2 it read -0.00118 vs
# engram_shared at 3.7x with complete separation, and the third seed took its sd
# from 8.6e-5 to 7.5e-4 (8.8x) and the result to -0.00075 at 1.57x with the
# groups overlapping. The only unrefuted positive left is count_gate on the
# PER-LAYER table: 0.765645 vs engram/tokens 0.767534, i.e. -0.00189, but that is
# n=1 against a baseline that has never had a seed replicate either. Two
# unmeasured variances on a 1-vs-1 comparison is precisely the shape of all six
# retractions in this study. So arms 1-4 give BOTH sides n=3 before anything is
# built on top.
#
# With gate-class sd ~7.5e-4, n=3 on both sides gives combined SE ~6.1e-4 and
# resolves ~0.0012 at 2x. The effect to detect is -0.00189, so this design can
# actually see it -- unlike the n=1 comparison it replaces.
#
# Arms are INTERLEAVED base/variant so that a truncated run still has matched n.
#
# THE NEW IDEA (arms 5-6): Fourier features on log-count.
# The current gate is w = 2*sigmoid(a*u + b), u = log1p(rel_hit_rate) -- strictly
# MONOTONE in count. Classical smoothing is not. Katz discounting distrusts a
# count-1 row because it is an unreliable *estimate*, while a count-10000 row is
# reliable but generic; the optimal trust curve is plausibly unimodal, peaking at
# middling counts. The 2-parameter gate cannot express that shape at all -- this
# is verified in tests/test_engram.py, where the Fourier gate fits a unimodal
# target to 5e-3 and the linear gate cannot get below 5e-2.
#
#   logit = a*u + b + sum_k [ c_sin,k sin(w_k u) + c_cos,k cos(w_k u) ]
#   w_k = pi * 2^k / 7,  K=4 bands, so wavelengths 14 / 7 / 3.5 / 1.75 in u
#   (the finest resolves a ~5.8x change in count)
#
# ADDITIVE, not a replacement, and that is deliberate: the measured solution was
# strongly linear (a ~ -0.3 on all 64 heads across 4 independent gates), and a
# linear trend is what Fourier bases are worst at reconstructing. Keeping a*u + b
# makes this a strict SUPERSET -- zero the coefficients and it is bit-identical to
# count_gate -- so it can only lose through optimization or variance, never
# through expressiveness. Tested.
#
# Cost: 320 scalars total (2K per head x 16 heads x 2 layers, plus the linear
# terms). Sizing verified on meta: 1828.357M total / 918.34M active for ALL
# THREE configs, identical to 6 significant figures. Nothing else moves.
#
# HONEST PRIOR. Four consecutive read-path mechanisms have come back neutral or
# worse (rank-k values, separate k/v, fixed-exponent whitening, longer n-gram
# orders), and the count gate itself is at 1.57x on the shared table. So the
# expected outcome here is "no resolvable difference", and the value of the round
# is mostly in arms 1-4 finally putting an error bar on the one positive signal.
# I am not predicting the Fourier arm wins. What would make it interesting: if
# the learned curve comes out non-monotone (read it from the checkpoint), that is
# a finding about what the memory needs even if bpb does not move.

set -euo pipefail

REPO_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"
cd "$REPO_DIR"

RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/runs/logs/d20_cgfour-${RUN_SUFFIX}}"
NPROC="${NPROC:-4}"
NUM_ITERATIONS="${NUM_ITERATIONS:-3814}"
MHC_STREAMS="${MHC_STREAMS:-4}"
FOURIER_BANDS="${FOURIER_BANDS:-4}"
TORCHRUN="${TORCHRUN:-$REPO_DIR/../nanochat-moe/.venv/bin/torchrun}"

# Marker string in a log, never a pid or process name (§6: kill -0 succeeds on
# zombies; pgrep/ps self-match, including the bracketed base_tra[i]n form).
WAIT_FOR_LOG="${WAIT_FOR_LOG:-}"
WAIT_FOR_MARKER="${WAIT_FOR_MARKER:-Round 9 complete}"
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
    echo "Predecessor finished at $(date -Is); starting round 10"
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
    "$TORCHRUN" --standalone --nproc_per_node="$NPROC" -m scripts.base_train -- \
        --depth=20 --max-seq-len=2048 --device-batch-size=8 --total-batch-size=524288 \
        --num-iterations="$NUM_ITERATIONS" --target-param-data-ratio=-1 \
        --window-pattern=SSSL --mhc --mhc-num-streams="$MHC_STREAMS" \
        --engram --engram-layers=2,6 --engram-memory-dim=1280 \
        --engram-mhc-num-streams="$MHC_STREAMS" --engram-slot-multiplier=15 \
        --eval-every=500 --eval-tokens=10485760 \
        --core-metric-every=-1 --sample-every=-1 --save-every=-1 \
        --model-tag="d20-cgfour-${tag}-${RUN_SUFFIX}" \
        "$@" > "$log_file" 2>&1 || {
            echo "  !! $tag FAILED (exit $?)"; tail -20 "$log_file"; return
        }
    echo "  $tag done: $(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$log_file" | tail -1)"
}

# --- 1-4: put an error bar on the one unrefuted positive signal -------------
# Interleaved so a truncated run still has matched n on both sides.
run_arm base_s1        --engram-seed=1
run_arm cg_perlayer_s1 --engram-count-gate --engram-seed=1
run_arm base_s2        --engram-seed=2
run_arm cg_perlayer_s2 --engram-count-gate --engram-seed=2

# --- 5-6: the non-monotone gate --------------------------------------------
for seed in 0 1; do
    run_arm "cg_fourier_s${seed}" --engram-count-gate \
        --engram-count-gate-fourier="$FOURIER_BANDS" --engram-seed="$seed"
done

echo
echo "Round 10 complete. Logs: $LOG_DIR"
printf "%-18s %-12s %s\n" arm min_bpb tok/sec
for tag in base_s1 base_s2 cg_perlayer_s1 cg_perlayer_s2 cg_fourier_s0 cg_fourier_s1; do
    f="$LOG_DIR/${tag}.log"; [ -f "$f" ] || continue
    printf "%-18s %-12s %s\n" "$tag" \
        "$(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$f" | tail -1)" \
        "$(grep -oP 'tok/sec: \K[0-9,]+' "$f" | tail -1)"
done
echo
echo "existing seed-0 runs to fold in:"
echo "  base (engram/tokens)  0.767555 / 0.767512"
echo "  cg_perlayer_s0        0.765645"
echo "n=3 vs n=3 at gate-class sd 7.5e-4 resolves ~0.0012 at 2x; the effect to"
echo "detect is -0.00189."
echo
echo "then READ THE LEARNED CURVE from the fourier checkpoints:"
echo "  engram_modules.{2,6}.multi_head_embedding.count_gate_{scale,bias,fourier_coef}"
echo "  reconstruct w(u) = 2*sigmoid(a*u+b+sum c_k basis_k(u)) over u in [0,7]."
echo "  IS IT NON-MONOTONE? That is a finding about what the memory needs even if"
echo "  bpb does not move."
