#!/bin/bash
#
# Prepare small-vocab tokenizers for the TN-gram R/V-regime reproduction.
#
# WHY. The TN-gram CP value table lost to the native hash table on BPB at every
# axis we swept -- but always at rank/vocab ratio R/V << 1 (R=48..400 vs our
# compressed vocab ~23,686, so R/V ~ 0.002-0.017). The paper's headline TIE is
# at R/V ~ 1 (R=1024, vocab=1024); at its one larger-vocab point (R=1800,
# vocab=8192, R/V~0.22) TN-gram already LOSES BPB. To test the paper's actual
# regime we need a genuinely smaller BACKBONE vocab (the Engram's built-in
# tokenizer-compression bottoms out at ~23.7k and cannot reach 1024).
#
# WHAT. For each target vocab V in {1024, 4096}: make an isolated base dir,
# symlink the shared ClimbMix data + eval bundle into it (data is tokenized on
# the fly from parquet text, so NO re-tokenization of a dataset is needed), then
# train a RustBPE tokenizer of size V there. tok_train writes both tokenizer.pkl
# and token_bytes.pt (the latter makes BPB vocab-invariant, so native-vs-CP
# stays a fair comparison across vocab sizes). Minutes each, CPU only.
#
# Idempotent: skips a vocab whose tokenizer.pkl already exists.

set -euo pipefail
REPO_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"
cd "$REPO_DIR"

PYTHON="${PYTHON:-$REPO_DIR/../nanochat-moe/.venv/bin/python}"
# Shared source data/eval, mirrored from the default 32k base dir's symlinks.
SRC_CACHE="${SRC_CACHE:-/mnt/backup/ningzhiyuan/.cache/nanochat_bench/climbmix}"
BASE_ROOT="${BASE_ROOT:-/mnt/3fs/dots-pretrain/ningzhiyuan/local_exps}"

VOCABS="${VOCABS:-1024 4096}"

for V in $VOCABS; do
    BASE_DIR="$BASE_ROOT/nsmb_out_v${V}"
    echo "================================================================"
    echo "vocab=$V  base_dir=$BASE_DIR"
    echo "================================================================"
    mkdir -p "$BASE_DIR"
    # Symlink shared data + eval bundle (never copy; these are large + read-only).
    for d in base_data_climbmix eval_bundle; do
        if [ ! -e "$BASE_DIR/$d" ]; then
            ln -s "$SRC_CACHE/$d" "$BASE_DIR/$d"
            echo "  linked $d"
        fi
    done

    if [ -f "$BASE_DIR/tokenizer/tokenizer.pkl" ]; then
        echo "  Skipping tokenizer train (tokenizer.pkl exists)"
        continue
    fi

    echo "  Training vocab=$V tokenizer ..."
    NANOCHAT_BASE_DIR="$BASE_DIR" "$PYTHON" -m scripts.tok_train --vocab-size "$V"

    # Sanity: report the actual (padded) vocab and confirm token_bytes exists.
    NANOCHAT_BASE_DIR="$BASE_DIR" "$PYTHON" - <<'PY'
from nano_scalemb.tokenizer import get_tokenizer
import os
from nano_scalemb.common import get_base_dir
tok = get_tokenizer()
tb = os.path.join(get_base_dir(), "tokenizer", "token_bytes.pt")
print(f"  -> vocab_size={tok.get_vocab_size():,}  token_bytes={'OK' if os.path.exists(tb) else 'MISSING'}")
PY
done

echo
echo "Small-vocab tokenizers ready under $BASE_ROOT/nsmb_out_v{$VOCABS// /,}"
