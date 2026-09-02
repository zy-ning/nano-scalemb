#!/bin/bash

# d20 idea 1B: mid-training value-aware ROW MERGE, pre-registered as a COMPRESSION
# test -- NOT a bpb win. At merge_at_frac of training, cluster each Engram table's
# rows by their LEARNED value and alias merged rows onto one survivor for the rest
# of the run, shrinking the EFFECTIVE table 2x (frac=0.5) / 4x (frac=0.75). Adds NO
# learnable params (row_alias is a buffer; merged rows go dead) -> ISO to the
# reference recipe by construction. Swept over merge_frac {0 control, 50%, 75%} x
# {+- count gate}.
#
# WHY THIS, AND WHY AS COMPRESSION. Three independent results now say collision in a
# LEARNED Engram is benign, not corruption:
#   - §9.2d table saturation: a better hash is impossible, tables are ~100% full.
#   - §9.2e more independent hashes do nothing (0.13x of an error bar); "nothing
#     available at this scale reduces collisions -- the gate's response is the fix".
#   - round-19 backoff: routing the rare-trigram tail to a per-token key lands
#     +0.002..+0.004 ABOVE baseline at every load; de-corrupting kept rows did not
#     help. The bo_all control (back off ONLY train-unseen windows) sits closest to
#     baseline (~+0.0006), confirming unseen-handling is ~neutral.
# The one genuinely UNTESTED mechanism is latent-aware collision: merge rows that
# ENDED UP similar in value, not rows that happen to hash together. Given the above
# we do NOT expect freed capacity to lower bpb; we ask the COMPRESSION question:
# does bpb stay FLAT (within the seed bar) while the effective table shrinks 2x/4x?
#
# THE MATRIX (each arm iso to its reference -- the merge adds no params):
#   arm         merge_frac   count gate   reference (measured)
#   mg_f0       0 (control)  off          engram_shared 0.763884 (n=3, sd 3.5e-4)
#   mg_f50      50%          off          engram_shared
#   mg_f75      75%          off          engram_shared
#   mg_f50_cg   50%          on (decoup)  cg_dec        0.762987 (n=5, sd 4.9e-4)
#   mg_f75_cg   75%          on           cg_dec
#
# mg_f0 keeps the identity alias -> MUST reproduce engram_shared to <=1 SE. That is
# the FIRST GATE: if it drifts, the alias plumbing is buggy; do not read f50/f75.
#
# UNLIKE the combo family, --engram-seed gives a GENUINE bar here (no MoE router).
#
# PRE-REGISTERED READS (decide before looking):
#   flat within pooled SE  = COMPRESSION CONFIRMED. Headline: "2x/4x fewer effective
#                            rows at equal bpb on a saturated table." A real, publish-
#                            able efficiency finding consistent with the saturation
#                            story. This is the expected outcome.
#   bpb drop >= 2x SE      = FALSIFIES "collisions benign"; freed capacity helped ->
#                            reopen §9.2e. Would be a genuine surprise.
#   bpb rise >= 2x SE      = merging destroyed real signal; row DIVERSITY mattered ->
#                            bounds how absorbed the superposition really is.
#   +-gate axis           = does the learned gate + a smaller effective table compound
#                            or substitute? Read per merge_frac against cg_dec.
#   metric cosine vs lsh  = cosine (exact, N<=8192/head) is default; lsh is the scale
#                            path. At slot=15, P=355,290 rows/head >> 8192, so the LSH
#                            path is what actually runs at d20 -- cosine here is the
#                            small-table unit-test path only. Set MERGE_METRIC=lsh.

set -euo pipefail

REPO_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"
cd "$REPO_DIR"

RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/runs/logs/d20_merge-${RUN_SUFFIX}}"
NPROC="${NPROC:-4}"
NUM_ITERATIONS="${NUM_ITERATIONS:-3814}"
MHC_STREAMS="${MHC_STREAMS:-4}"
SLOT="${SLOT:-15}"
SEEDS="${SEEDS:-0 1 2}"                          # set SEEDS=0 for a one-seed scout
FRACS="${FRACS:-0.5 0.75}"                        # non-control merge fractions
MERGE_AT="${MERGE_AT:-0.5}"                        # fire the merge at 50% of training
MERGE_METRIC="${MERGE_METRIC:-lsh}"               # lsh scales past 8192 rows/head
PYTHON="${PYTHON:-$REPO_DIR/../nanochat-moe/.venv/bin/python}"
TORCHRUN="${TORCHRUN:-$REPO_DIR/../nanochat-moe/.venv/bin/torchrun}"

WAIT_FOR_LOG="${WAIT_FOR_LOG:-}"
WAIT_FOR_MARKER="${WAIT_FOR_MARKER:-Round 19 complete}"
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
fi

# $1 tag, $2 merge_frac ("" -> control, no merge flags), rest -> extra flags (gate, seed).
# Everything else is the engram_shared recipe held fixed; dbs=8 on every arm (§6).
run_arm () {
    local tag="$1"; local frac="$2"; shift 2
    local log_file="$LOG_DIR/${tag}.log"
    if grep -q "^Total training time" "$log_file" 2>/dev/null; then
        echo "Skipping $tag (already completed)"; return
    fi
    local merge_flags=()
    if [ -n "$frac" ] && [ "$frac" != "0" ]; then
        merge_flags=(--engram-merge-frac="$frac" \
                     --engram-merge-at-frac="$MERGE_AT" \
                     --engram-merge-metric="$MERGE_METRIC")
    fi
    echo "================================================================"
    echo "Starting $tag  ($(date -Is))  merge_frac=${frac:-0} metric=$MERGE_METRIC"
    echo "  extra: $*"
    echo "================================================================"
    "$TORCHRUN" --standalone --nproc_per_node="$NPROC" -m scripts.base_train -- \
        --depth=20 --max-seq-len=2048 --device-batch-size=8 --total-batch-size=524288 \
        --num-iterations="$NUM_ITERATIONS" --target-param-data-ratio=-1 \
        --window-pattern=SSSL --mhc --mhc-num-streams="$MHC_STREAMS" \
        --engram --engram-layers=2,6 --engram-memory-dim=1280 \
        --engram-mhc-num-streams="$MHC_STREAMS" --engram-share-memory \
        --engram-slot-multiplier="$SLOT" \
        "${merge_flags[@]}" \
        --eval-every=500 --eval-tokens=10485760 \
        --core-metric-every=-1 --sample-every=-1 --save-every=-1 \
        --model-tag="d20-merge-${tag}-${RUN_SUFFIX}" \
        "$@" > "$log_file" 2>&1 || {
            echo "  !! $tag FAILED (exit $?)"; tail -20 "$log_file"; return
        }
    echo "  $tag done: $(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$log_file" | tail -1)"
}

CG=(--engram-count-gate --engram-count-gate-decouple)

# Seed-outer so seed 0 yields the full matrix (control + fracs x +-gate) at n=1
# first, then n=2, n=3. Control FIRST within each seed: it is the plumbing gate.
for s in $SEEDS; do
    run_arm "mg_f0_s${s}"    "0"  --engram-seed="$s"                 # identity-alias control
    for frac in $FRACS; do
        fp="${frac/./}"                                              # 0.5 -> 05, 0.75 -> 075
        run_arm "mg_f${fp}_s${s}"    "$frac"              --engram-seed="$s"
        run_arm "mg_f${fp}_cg_s${s}" "$frac"  "${CG[@]}"  --engram-seed="$s"
    done
done

echo
echo "Idea 1B complete. Logs: $LOG_DIR"
printf "%-18s %-12s %s\n" arm min_bpb tok/sec
tags=""
for s in $SEEDS; do
    tags="$tags mg_f0_s${s}"
    for frac in $FRACS; do fp="${frac/./}"; tags="$tags mg_f${fp}_s${s} mg_f${fp}_cg_s${s}"; done
done
for tag in $tags; do
    f="$LOG_DIR/${tag}.log"; [ -f "$f" ] || continue
    printf "%-18s %-12s %s\n" "$tag" \
        "$(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$f" | tail -1)" \
        "$(grep -oP 'tok/sec: \K[0-9,]+' "$f" | tail -1)"
done
echo
echo "refs: engram_shared 0.763884 (n=3, sd 3.5e-4) [no gate] ; cg_dec 0.762987 (n=5) [gate]"
echo "GATE FIRST: mg_f0 must reproduce engram_shared to <=1 SE, else the alias is buggy."
echo "Then read f50/f75 as COMPRESSION: flat within pooled SE = 2x/4x fewer effective"
echo "rows at equal bpb (the expected win). A bpb DROP would reopen §9.2e."
