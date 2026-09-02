#!/bin/bash

# d20 round 19: Engram hash BACKOFF -- route the rare n-gram tail to a per-token
# (*,*,t) key so the frequent, useful rows stay uncontaminated -- swept over the
# COVERAGE/COLLISION trade-off (the top-p budget), x {+-count gate}.
#
# WHY. The tables are 100% saturated (§9.2d) and a better/uniform hash does nothing
# (§9.2e). A CPU diagnostic this round localized the trigram channel's weakness: a
# top-1% frequent trigram's row is 64.5% OTHER n-grams, and the count gate can only
# discount the WHOLE blended row. Plain prune-to-null is a wash. BACKOFF is not:
# routing non-kept trigrams to their (*,*,t) key de-corrupts the kept rows while
# the tail keeps a coarse per-token signal. Adds NO learnable params -> ISO.
#
# THE TRADE-OFF THIS ROUND MAPS. At a FIXED table (slot=15, P=355,290 rows/head),
# keeping more keys (higher top-p budget) raises occurrence COVERAGE but also raises
# COLLISION among the kept keys -- you cannot have both. Counts are EXACT (one scan
# over all ~2B training tokens; no sample-vs-train mismatch). Measured:
#
#   load   kept keys   coverage   (exact counts over all 2B training tokens)
#   50%     177,644     10.6%      ks_topp_l50_2b_slot15.pt
#   100%    355,289     14.7%      ks_topp_l100_2b_slot15.pt
#   200%    710,580     19.8%      ks_topp_l200_2b_slot15.pt
#   (355,777,458 distinct trigrams total; no-backoff baseline = all of them, ~100000% load)
#
# The whole curve is >=5x cleaner than the no-backoff baseline; the question is
# where the interior optimum sits -- does more retained trigram signal (coverage)
# beat the collision it re-introduces? All three are token addressing, orders 2/3,
# order-2 kept full-resolution (its rows are already ~86% clean); backoff on order 3.
#
# THE MATRIX (each arm iso to its reference by construction -- backoff adds no params):
#   arm        load        count gate   reference (measured)
#   bo_l50     50%         off          engram_shared 0.763884 (n=3, sd 3.5e-4)
#   bo_l50_cg  50%         on (decoup)  cg_dec        0.762987 (n=5, sd 4.9e-4)
#   bo_l100    100%        off          engram_shared
#   bo_l100_cg 100%        on           cg_dec
#   bo_l200    200%        off          engram_shared
#   bo_l200_cg 200%        on           cg_dec
#   bo_all     all-seen*   off          engram_shared   <- CONTROL, see below
#   bo_all_cg  all-seen*   on           cg_dec
#
# *bo_all keeps every window seen in TRAINING and backs off ONLY the train-UNSEEN
#  (= eval-novel) ones. top-p=100% is NOT baseline: baseline hashes even unseen
#  windows to their own row; bo_all backs them off. So bo_all has ~baseline-level
#  collision on the frequent rows (all seen windows kept) and isolates the
#  unseen-backoff component alone. If bo_all ~= engram_shared, all the gain in the
#  bo_l* sweep is rare-seen DE-CORRUPTION; if bo_all beats it, unseen-handling is
#  itself a mechanism. Built on a LARGER sample so "seen" ~= "learnable".
#
# UNLIKE the combo family, --engram-seed gives a GENUINE bar here (no MoE router).
#
# PRE-REGISTERED READS (decide before looking):
#   the COVERAGE curve (l50->l100->l200, no gate) is the primary result: monotone
#     down = coverage dominates (keep more); interior min = a real optimum; monotone
#     up = collision dominates (keep fewer, the de-corruption end wins).
#   any arm clears its ref by >=2x pooled SE -> backoff is real -> next: order 5.
#   flat across the whole curve -> the count gate + parallel bigram table already
#     captured the soft backoff; report as "below resolution", not "does nothing".
#   +-gate axis: bo_lX ~= cg_dec means hard backoff SUBSTITUTES for the learned
#     gate; bo_lX_cg < cg_dec means they COMPOUND. Read this per load.

set -euo pipefail

REPO_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"
cd "$REPO_DIR"

RUN_SUFFIX="${RUN_SUFFIX:-$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/runs/logs/d20_backoff-${RUN_SUFFIX}}"
NPROC="${NPROC:-4}"
NUM_ITERATIONS="${NUM_ITERATIONS:-3814}"
MHC_STREAMS="${MHC_STREAMS:-4}"
SLOT="${SLOT:-15}"
KEYSET_TOKENS="${KEYSET_TOKENS:-2000000000}"   # all training tokens -> exact counts
SEEDS="${SEEDS:-0 1 2}"                         # set SEEDS=0 for a one-seed scout
LOADS="${LOADS:-25 50 100 200 400 800}"         # top-p budget as % of table rows/head
PYTHON="${PYTHON:-$REPO_DIR/../nanochat-moe/.venv/bin/python}"
TORCHRUN="${TORCHRUN:-$REPO_DIR/../nanochat-moe/.venv/bin/torchrun}"

WAIT_FOR_LOG="${WAIT_FOR_LOG:-}"
WAIT_FOR_MARKER="${WAIT_FOR_MARKER:-Round 18 complete}"
WAIT_MAX_HOURS="${WAIT_MAX_HOURS:-24}"

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/mnt/3fs/dots-pretrain/ningzhiyuan/local_exps/nsmb_out}"
export TORCHDYNAMO_CACHE_SIZE_LIMIT="${TORCHDYNAMO_CACHE_SIZE_LIMIT:-64}"
export PYTHONPATH="${PYTHONPATH:-$REPO_DIR}"

KS_DIR="${KS_DIR:-$NANOCHAT_BASE_DIR/engram_backoff}"
# All keysets are derived from ONE scan of all ~2B training tokens (exact counts,
# no sample-vs-train mismatch), cached in the counts dump.
COUNTS="$KS_DIR/counts_2b_o3.npz"
ks_path () { echo "$KS_DIR/ks_topp_l${1}_2b_slot${SLOT}.pt"; }   # $1 = load %
KS_ALL="$KS_DIR/ks_allseen_2b_slot${SLOT}.pt"    # control: keep ALL windows seen in
#   TRAINING, back off only train-unseen (=eval-novel) ones. Isolates the unseen-
#   backoff component: all seen windows still collide at ~baseline level, so any gap
#   vs engram_shared is unseen-handling, not rare-seen de-corruption.

mkdir -p "$LOG_DIR" "$KS_DIR"

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

# --- Step 0: derive all keysets from ONE scan of all training tokens (CPU) --------
# The scan is the only expensive part; cache it once, then every keyset is instant.
P_PER_HEAD=$(( 23686 * SLOT ))    # compressed vocab (23,686) * slot
if [ ! -f "$COUNTS" ]; then
    echo "Scanning ${KEYSET_TOKENS} training tokens -> counts dump $COUNTS"
    "$PYTHON" -m scripts.build_engram_backoff_keyset \
        --mode topp --target-keys "$(( P_PER_HEAD / 2 ))" --orders 3 \
        --n-tokens "$KEYSET_TOKENS" --slot-multiplier "$SLOT" \
        --dump-counts "$COUNTS" --out "$(ks_path 50)"
else
    echo "Counts dump exists: $COUNTS"
fi
build_from_counts () {   # $1 out, then extra args (--target-keys N | --mode allseen)
    local out="$1"; shift
    if [ -f "$out" ]; then echo "Keyset exists: $out"; return; fi
    echo "Building keyset -> $out  (from counts dump)"
    "$PYTHON" -m scripts.build_engram_backoff_keyset \
        --orders 3 --slot-multiplier "$SLOT" --from-counts "$COUNTS" \
        --out "$out" "$@"
}
for load in $LOADS; do
    build_from_counts "$(ks_path "$load")" --mode topp \
        --target-keys "$(( P_PER_HEAD * load / 100 ))"
done
build_from_counts "$KS_ALL"  --mode allseen

# $1 tag, $2 keyset path, $3 backoff mode, rest -> extra flags (count gate, seed).
# Everything else is the engram_shared recipe held fixed; dbs=8 on every arm (§6).
run_arm () {
    local tag="$1"; local keyset="$2"; local mode="$3"; shift 3
    local log_file="$LOG_DIR/${tag}.log"
    if grep -q "^Total training time" "$log_file" 2>/dev/null; then
        echo "Skipping $tag (already completed)"; return
    fi
    echo "================================================================"
    echo "Starting $tag  ($(date -Is))  keyset=$(basename "$keyset")"
    echo "  extra: $*"
    echo "================================================================"
    "$TORCHRUN" --standalone --nproc_per_node="$NPROC" -m scripts.base_train -- \
        --depth=20 --max-seq-len=2048 --device-batch-size=8 --total-batch-size=524288 \
        --num-iterations="$NUM_ITERATIONS" --target-param-data-ratio=-1 \
        --window-pattern=SSSL --mhc --mhc-num-streams="$MHC_STREAMS" \
        --engram --engram-layers=2,6 --engram-memory-dim=1280 \
        --engram-mhc-num-streams="$MHC_STREAMS" --engram-share-memory \
        --engram-slot-multiplier="$SLOT" \
        --engram-backoff-keyset="$keyset" --engram-backoff-mode="$mode" \
        --eval-every=500 --eval-tokens=10485760 \
        --core-metric-every=-1 --sample-every=-1 --save-every=-1 \
        --model-tag="d20-backoff-${tag}-${RUN_SUFFIX}" \
        "$@" > "$log_file" 2>&1 || {
            echo "  !! $tag FAILED (exit $?)"; tail -20 "$log_file"; return
        }
    echo "  $tag done: $(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$log_file" | tail -1)"
}

CG=(--engram-count-gate --engram-count-gate-decouple)

# Seed-outer so seed 0 yields the full curve (budget x +-gate) + control at n=1
# first, then n=2, n=3. Interleaved within a seed so a truncation stays readable.
for s in $SEEDS; do
    for load in $LOADS; do
        run_arm "bo_l${load}_s${s}"    "$(ks_path "$load")" topp               --engram-seed="$s"
        run_arm "bo_l${load}_cg_s${s}" "$(ks_path "$load")" topp   "${CG[@]}"   --engram-seed="$s"
    done
    # unseen-backoff control (keep all seen, back off only unseen); anchors the
    # curve and separates unseen-handling from rare-seen de-corruption.
    run_arm "bo_all_s${s}"      "$KS_ALL"   allseen             --engram-seed="$s"
    run_arm "bo_all_cg_s${s}"   "$KS_ALL"   allseen "${CG[@]}"  --engram-seed="$s"
done

echo
echo "Round 19 complete. Logs: $LOG_DIR"
printf "%-16s %-12s %s\n" arm min_bpb tok/sec
tags=""
for s in $SEEDS; do for load in $LOADS; do tags="$tags bo_l${load}_s${s} bo_l${load}_cg_s${s}"; done
    tags="$tags bo_all_s${s} bo_all_cg_s${s}"; done
for tag in $tags; do
    f="$LOG_DIR/${tag}.log"; [ -f "$f" ] || continue
    printf "%-16s %-12s %s\n" "$tag" \
        "$(grep -oP 'Minimum validation bpb: \K[0-9.]+' "$f" | tail -1)" \
        "$(grep -oP 'tok/sec: \K[0-9,]+' "$f" | tail -1)"
done
echo
echo "coverage/collision curve, loads = $LOADS % (topp); coverage rises sublinearly:"
echo "   50%->10.6%  100%->14.7%  200%->19.8%  400%->~24%  800%->~29%  all-seen->100%"
echo "refs: engram_shared 0.763884 (n=3, sd 3.5e-4) [no gate] ; cg_dec 0.762987 (n=5) [gate]"
echo
echo "READ THE CURVE, then +-gate per load. Monotone-down = coverage wins (keep more);"
echo "interior min = real optimum; monotone-up = de-corruption end wins (keep fewer)."
echo "bo_all vs engram_shared: if it beats baseline, unseen-handling is a mechanism;"
echo "if flat, all the gain in bo_l* is rare-seen de-corruption."
