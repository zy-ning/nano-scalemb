#!/bin/bash
#
# vocab-8192: one more ISO-PARAM rung (~2.68B value params) to test the
# "native saturates while CP keeps scaling" claim.
#
# WHERE THIS CAME FROM. Param-matching the small-vocab cells (run_v8192_native_bigtable.sh)
# removed most, but not all, of CP's apparent win:
#     native slot56   587.5M  0.778626  vs CP R=1802  590.5M  0.779004  (1.005x)  +0.0004 NATIVE
#     native slot128 1342.4M  0.777793  vs CP R=4096 1342.2M  0.776223  (1.000x)  -0.0016 CP
# Against the ORIGINAL unfair baseline (native slot18, 189.0M, 0.783098) CP R=1802
# "won" by -0.0041 and R=4096 by -0.0069; param matching erased 90% and 77% of
# those respectively, and flipped R=1802's sign. So the "R/V governs the sign"
# story was largely a PARAM-COUNT CONFOUND -- but a residual CP edge survives at
# the top rung and needs explaining rather than dismissing.
#
# THE HYPOTHESIS TO TEST. Marginal gain per param-doubling so far:
#     native:  189.0->587.5M (3.11x) -0.0045 ; 587.5->1342.4M (2.28x) -0.0008
#     CP:      671.1->1342.2M (2.00x) -0.0030
# i.e. native looks SATURATED (-0.0008 for 2.3x more params) while CP is still
# paying off (-0.0030 for 2x). If that is real, the two curves DIVERGE and CP is
# genuinely the better value representation for LARGE tables at small vocab --
# a narrow but real positive, and a different claim from the R/V story.
#
# BUT IT RESTS ON ONE PAIR OF POINTS. native's -0.0008 step could equally be a
# single unlucky draw: the 32k run-to-run sd is ~6e-4, so a -0.0008 marginal gain
# is barely above noise, and the -0.0016 CP edge at slot128 is under 3 sigma.
# This rung is the cheapest way to tell divergence from noise.
#
# ARMS (iso-param pair at ~2.68B, ~2x the current top rung):
#     native slot256  -> 2684M params   (slot128 measured 1342.4M, rows linear in slot)
#     CP R=8192       -> 2684M A-params (N*V*K*R = 5*8192*8*8192)
# Exactly iso-param with each other by construction.
#
# READ.
#   Gap WIDENS beyond -0.0016 => divergence confirmed; native really does saturate
#     and CP scales better on large small-vocab tables. Report as a real (if
#     32k-impractical) positive for CP.
#   Gap STAYS ~-0.0016 or NARROWS => the slot56->128 flatness was noise; both
#     curves scale alike and CP has no genuine representational edge. The TN-gram
#     line then closes completely.
#   Also diagnostic: native slot256's own marginal gain. If it resumes ~-0.003
#     (like CP) then slot128 was simply a low draw and "native saturates" is dead.
#
# MEMORY (measured at this vocab, GB200 cap 189471 MiB):
#     native slot128 87.8GB @1342M ; CP R=4096 113.9GB @1342M
#   -> +1342M params ~ +33GB => native slot256 ~121GB, CP R=8192 ~147GB. Both fit.
#   R=16384 (5.4B) would be ~215GB => out of reach; this is the LAST rung available.
#
# Flags identical to the rest of the vocab-8192 cell (N=5,
# --engram-no-tokenizer-compression, layers 2,6, memory_dim 1280, mhc4,
# share-memory), so step-0 bpb must be 3.213328 exactly as in every other arm.
# Same checkpoint+resume kill resilience (this node SIGTERMs whole process groups;
# it has happened 7 times, and the resume path has already proven itself).

set -euo pipefail
REPO_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"
cd "$REPO_DIR"

RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/runs/logs/v8192_isoparam_rung3-${RUN_SUFFIX}}"
NPROC="${NPROC:-4}"
NUM_ITERATIONS="${NUM_ITERATIONS:-3814}"
MHC_STREAMS="${MHC_STREAMS:-4}"
TORCHRUN="${TORCHRUN:-$REPO_DIR/../nanochat-moe/.venv/bin/torchrun}"
BASE_ROOT="${BASE_ROOT:-/mnt/3fs/dots-pretrain/ningzhiyuan/local_exps}"
VOCAB="${VOCAB:-8192}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$BASE_ROOT/nsmb_out_v${VOCAB}}"

# "nat256" = native slot_multiplier 256 ; "cp8192" = CP cp_rank 8192
ARMS="${ARMS:-nat256 cp8192}"
N="${N:-5}"
DBS="${DBS:-8}"
SAVE_EVERY="${SAVE_EVERY:-500}"
WAIT_PID="${WAIT_PID:-}"

export TORCHDYNAMO_CACHE_SIZE_LIMIT="${TORCHDYNAMO_CACHE_SIZE_LIMIT:-64}"
export PYTHONPATH="${PYTHONPATH:-$REPO_DIR}"

mkdir -p "$LOG_DIR"

if [ -n "$WAIT_PID" ]; then
    echo "Waiting for pid $WAIT_PID before starting..."
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 120; done
fi

COMMON=(
    --depth=20 --max-seq-len=2048 --total-batch-size=524288
    --num-iterations="$NUM_ITERATIONS" --target-param-data-ratio=-1
    --window-pattern=SSSL --mhc --mhc-num-streams="$MHC_STREAMS"
    --engram --engram-layers=2,6 --engram-memory-dim=1280
    --engram-mhc-num-streams="$MHC_STREAMS" --engram-share-memory
    --engram-no-tokenizer-compression
    --eval-every=500 --eval-tokens=10485760
    --core-metric-every=-1 --sample-every=-1 --save-every="$SAVE_EVERY"
)

latest_complete_ckpt () {
    local ckpt_dir="$1" step f ok r
    [ -d "$ckpt_dir" ] || { echo ""; return; }
    for f in $(ls "$ckpt_dir"/model_*.pt 2>/dev/null | sort -r); do
        step="$(basename "$f" .pt)"; step="${step#model_}"
        [ -f "$ckpt_dir/meta_${step}.json" ] || continue
        ok=1
        for r in $(seq 0 $((NPROC-1))); do
            [ -f "$ckpt_dir/optim_${step}_rank${r}.pt" ] || { ok=0; break; }
        done
        [ "$ok" = 1 ] && { echo "$((10#$step))"; return; }
    done
    echo ""
}

attempt () {
    local arm="$1" dbs="$2" log_file="$3"
    local model_tag="d20-sv${VOCAB}-N${N}-${arm}-${RUN_SUFFIX}"
    local spec=() extra=() resume_step
    case "$arm" in
        nat256) spec=(--engram-slot-multiplier=256) ;;
        cp8192) spec=(--engram-value-table=tngram --engram-cp-rank=8192) ;;
        *) echo "  !! unknown arm $arm"; return 1 ;;
    esac
    resume_step="$(latest_complete_ckpt "$NANOCHAT_BASE_DIR/base_checkpoints/$model_tag")"
    if [ -n "$resume_step" ] && [ "$resume_step" -gt 0 ] 2>/dev/null; then
        echo "  resuming $model_tag from step $resume_step"
        echo "=== RESUME from step $resume_step ($(date -Is)) ===" >> "$log_file"
        extra+=(--resume-from-step="$resume_step")
    fi
    "$TORCHRUN" --standalone --nproc_per_node="$NPROC" \
        -m scripts.base_train -- \
        "${COMMON[@]}" --device-batch-size="$dbs" \
        --engram-max-ngram-size="$N" "${spec[@]}" \
        "${extra[@]}" --model-tag="$model_tag" \
        >> "$log_file" 2>&1
}

run_arm () {
    local arm="$1"
    local tag="sv${VOCAB}_N${N}_${arm}_s0"
    local log_file="$LOG_DIR/${tag}.log"
    if grep -q "^Total training time" "$log_file" 2>/dev/null; then
        echo "Skipping $tag (already completed)"; return
    fi
    echo "================================================================"
    echo "Starting $tag  ($(date -Is))  vocab=$VOCAB N=$N arm=$arm dbs=$DBS"
    echo "================================================================"
    if attempt "$arm" "$DBS" "$log_file"; then
        echo "  $tag done: $(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$log_file" | tail -1)"
        return
    fi
    if tail -200 "$log_file" | grep -qiE "out of memory"; then
        local half=$((DBS/2))
        if [ "$half" -ge 1 ]; then
            echo "  !! $tag OOMed at dbs=$DBS; retrying at dbs=$half (same total batch, identical math)"
            echo "=== OOM RETRY at device-batch-size=$half ($(date -Is)) ===" >> "$log_file"
            attempt "$arm" "$half" "$log_file" && {
                echo "  $tag done (dbs=$half): $(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$log_file" | tail -1)"; return; }
        fi
    fi
    echo "  !! $tag FAILED"; tail -15 "$log_file"
}

for arm in $ARMS; do run_arm "$arm"; done

echo
echo "vocab-$VOCAB iso-param rung 3 complete. Logs: $LOG_DIR"
printf "%-30s %-16s %s\n" arm value_params min_bpb
for arm in $ARMS; do
    tag="sv${VOCAB}_N${N}_${arm}_s0"; f="$LOG_DIR/${tag}.log"; [ -f "$f" ] || continue
    printf "%-30s %-16s %s\n" "$tag" \
        "$(grep -oP 'engram_embeds\s+: \K[0-9,]+' "$f" | head -1)" \
        "$(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$f" | tail -1)"
done
echo
echo "PRIOR RUNGS (iso-param, CP-native): 590M +0.0004 NATIVE | 1342M -0.0016 CP"
echo "READ: gap widens beyond -0.0016 => real divergence, CP scales better on large"
echo "small-vocab tables. Gap flat/narrowing => slot128 flatness was noise and the"
echo "TN-gram line closes completely. Also check native slot256's marginal gain:"
echo "if it resumes ~-0.003 then 'native saturates' is dead."
