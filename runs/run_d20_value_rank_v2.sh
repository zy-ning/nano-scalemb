#!/bin/bash

# d20 round 6: disentangle ROW COUNT from ROW CONTENT, and put an error bar on
# the baseline that round 5 compares against.
#
# Round 5 result (n=2): lsh_rank1 0.775254 / 0.773538, mean 0.774396, vs lsh
# 0.771347. That is +0.0030 the WRONG WAY. The pre-registered prediction was
# < -0.003, so the sign is wrong; both seeds are worse than rank 0 and both are
# worse than dense (0.772614).
#
# But round 5 could not attribute it, because iso-total forced rank 1 to buy
# function-valued rows with 2.6x fewer addresses (slot 11 -> 4). Two readings fit
# the data equally well:
#   (a) function-valued rows do not help contextual addressing
#   (b) they do help, but losing 2.6x of the rows costs more
#
# This round completes the 2x2 that separates them. Two cells are already run:
#
#                rows 131k              rows 360k
#   rank 0       lsh_r0_slot4  (NEW)    lsh          0.771347
#   rank 1       lsh_rank1     0.774396 lsh_r1_isorow (NEW)
#
#   rank effect at fixed rows:  lsh_rank1 - lsh_r0_slot4  and  lsh_r1_isorow - lsh
#   row  effect at fixed rank:  lsh - lsh_r0_slot4        and  lsh_r1_isorow - lsh_rank1
#
# lsh_r1_isorow is deliberately NOT iso-total: 3458.3M vs lsh's 1841.8M, because
# holding rows AND total fixed while changing bytes/row is arithmetically
# impossible (total = rows x bytes/row). The param advantage biases toward the
# §8.1 account, which makes a null result there decisive AGAINST it: if
# function-valued rows cannot beat 0.771347 with 1.9x the parameters and the same
# addresses, the account is dead.
#
# Sizing verified by building each config on meta:
#   lsh            1841.8M total / 918.3M active / 360,448 rows per head
#   lsh_rank1      1843.0M        / 918.7M       / 131,072   (round 5)
#   lsh_r0_slot4   1254.5M        / 918.3M       / 131,072
#   lsh_r1_isorow  3458.3M        / 918.7M       / 360,448
#
# lsh_r1_isorow runs at device-batch-size 4, not 8: 3458M is close to the 3707M
# that OOM'd on the d26 Mobius arm. Total batch is unchanged (524288) so the arm
# is numerically identical, just ~25% slower.
#
# Seeds, per §5.1's "more seeds, not more arms":
#   lsh_s1            The lsh baseline is n=1 and the ENTIRE 2x2 is measured
#                     against it. Its own error bar is worth more than any new
#                     mechanism. Round 5 also measured a 1.72e-3 seed spread on
#                     lsh_rank1, ~2x the 8.84e-4 from hybrid_frac75, so the
#                     contextual bar is a range and needs more samples.
#   engram_shared_s2  Third seed on the best arm in the study (n=2 -> n=3).
#
# NOT run, and worth stating: a mobius replicate. There is no --moe-seed and no
# global manual_seed in base_train, so re-running an MoE arm only samples CUDA
# nondeterminism (the 4.3e-5 floor), not a structurally different draw. Giving
# mobius a real error bar needs a seed knob on the router init first, which is
# implementation work, not a run.

set -euo pipefail

REPO_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"
cd "$REPO_DIR"

RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/runs/logs/d20_vrank2-${RUN_SUFFIX}}"
NPROC="${NPROC:-4}"
NUM_ITERATIONS="${NUM_ITERATIONS:-3814}"   # ~2.0B tokens, same as rounds 1-5
MHC_STREAMS="${MHC_STREAMS:-4}"
TORCHRUN="${TORCHRUN:-$REPO_DIR/../nanochat-moe/.venv/bin/torchrun}"

# Wait for a round-5 driver to finish before touching the GPUs. Polls for the
# marker STRING IN ITS LOG, not a pid: §6 records that `kill -0` succeeds on a
# zombie and that `pgrep -f`/`ps | grep` self-match the checking shell (three
# separate multi-hour GPU idles, and the bracketed-pattern workaround failed too
# once the pattern appeared in the checker's own command line). A file marker has
# neither failure mode. run_arm never exits early, so the marker always prints
# even if individual arms fail.
WAIT_FOR_LOG="${WAIT_FOR_LOG:-}"
WAIT_FOR_MARKER="${WAIT_FOR_MARKER:-Round 5 complete}"
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
            echo "!! timed out waiting for round 5; NOT starting (GPUs may be busy)"
            exit 1
        fi
        sleep 60
    done
    echo "Round 5 finished at $(date -Is); starting round 6"
fi

run_arm () {
    local tag="$1"; local dbs="$2"; shift 2
    local log_file="$LOG_DIR/${tag}.log"
    if grep -q "^Total training time" "$log_file" 2>/dev/null; then
        echo "Skipping $tag (already completed)"; return
    fi
    echo "================================================================"
    echo "Starting $tag  ($(date -Is))  device-batch-size=$dbs"
    echo "  extra: $*"
    echo "================================================================"
    "$TORCHRUN" --standalone --nproc_per_node="$NPROC" -m scripts.base_train -- \
        --depth=20 --max-seq-len=2048 --device-batch-size="$dbs" \
        --total-batch-size=524288 \
        --num-iterations="$NUM_ITERATIONS" --target-param-data-ratio=-1 \
        --window-pattern=SSSL --mhc --mhc-num-streams="$MHC_STREAMS" \
        --engram --engram-layers=2,6 --engram-memory-dim=1280 \
        --engram-mhc-num-streams="$MHC_STREAMS" \
        --eval-every=500 --eval-tokens=10485760 \
        --core-metric-every=-1 --sample-every=-1 --save-every=-1 \
        --model-tag="d20-vr2-${tag}-${RUN_SUFFIX}" \
        "$@" > "$log_file" 2>&1 || {
            echo "  !! $tag FAILED (exit $?)"; tail -20 "$log_file"; return
        }
    echo "  $tag done: $(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$log_file" | tail -1)"
}

# --- 1. the 2x2 control: constant rows at lsh_rank1's row count -------------
# If this lands near 0.7744 then the row loss explains round 5 entirely and
# function-valued rows are neutral (and 589M params bought nothing). If it is
# clearly worse, they did help, just not enough to pay for the addresses.
run_arm lsh_r0_slot4 8 \
    --engram-address-source=lsh --engram-lsh-bits=15 \
    --engram-slot-multiplier=4 --engram-seed=0

# --- 2. the 2x2 control: rank-1 rows at lsh's row count --------------------
# The decisive one. Same addresses as lsh, 1.9x the parameters. If this does not
# beat 0.771347, §8.1 is dead on its own terms.
run_arm lsh_r1_isorow 4 \
    --engram-address-source=lsh --engram-lsh-bits=15 \
    --engram-slot-multiplier=11 \
    --engram-value-rank=1 --engram-value-query-dim=140 --engram-seed=0

# --- 3. error bar on the baseline the whole 2x2 is measured against --------
run_arm lsh_s1 8 \
    --engram-address-source=lsh --engram-lsh-bits=15 \
    --engram-slot-multiplier=11 --engram-seed=1

# --- 4. third seed on the best arm in the study ----------------------------
run_arm engram_shared_s2 8 \
    --engram-slot-multiplier=30 --engram-share-memory --engram-seed=2

echo
echo "Round 6 complete. Logs: $LOG_DIR"
printf "%-20s %-12s %s\n" arm min_bpb tok/sec
for tag in lsh_r0_slot4 lsh_r1_isorow lsh_s1 engram_shared_s2; do
    f="$LOG_DIR/${tag}.log"; [ -f "$f" ] || continue
    printf "%-20s %-12s %s\n" "$tag" \
        "$(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$f" | tail -1)" \
        "$(grep -oP 'tok/sec: \K[0-9,]+' "$f" | tail -1)"
done
echo
echo "the 2x2 (rows x rank), contextual addressing:"
echo "               rows 131k        rows 360k"
echo "  rank 0       lsh_r0_slot4     lsh 0.771347 (+ lsh_s1)"
echo "  rank 1       lsh_rank1 0.7744 lsh_r1_isorow"
echo "engram_shared seeds: 0.763784 / 0.763599 / (s2)"
