"""
Train model. From root directory of the project, run as:

python -m scripts.base_train

or distributed as:

torchrun --nproc_per_node=8 -m scripts.base_train

If you are only on CPU/Macbook, you'll want to train a much much smaller LLM. Example:
python -m scripts.base_train --depth=4 --max-seq-len=512 --device-batch-size=1 --eval-tokens=512 --core-metric-every=-1 --total-batch-size=512 --num-iterations=20
"""

import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import json
import time
import math
import argparse
from dataclasses import asdict, replace
from contextlib import nullcontext, contextmanager

import wandb
import torch

from nano_scalemb.engram import EngramConfig, collect_engram_aux_loss
from nano_scalemb.gpt import GPT, GPTConfig
from nano_scalemb.mhc import MHCConfig
from nano_scalemb.moe.block import collect_aux_loss, collect_moe_stats
from nano_scalemb.moe.config import MoEConfig
from nano_scalemb.moe.ladder import (
    active_params,
    describe as describe_moe,
    moe_config_from_knobs,
    total_params,
)
from nano_scalemb.dataloader import (
    tokenizing_distributed_data_loader_bos_bestfit,
    tokenizing_distributed_data_loader_with_state_bos_bestfit,
)
from nano_scalemb.common import (
    compute_init,
    compute_cleanup,
    print0,
    DummyWandb,
    print_banner,
    get_base_dir,
    autodetect_device_type,
    get_peak_flops,
)
from nano_scalemb.tokenizer import get_tokenizer, get_token_bytes
from nano_scalemb.checkpoint_manager import save_checkpoint, load_checkpoint
from nano_scalemb.loss_eval import evaluate_bpb
from nano_scalemb.engine import Engine
from nano_scalemb.flash_attention import HAS_FA3
from scripts.base_eval import evaluate_core

print_banner()
if "TORCHDYNAMO_CACHE_SIZE_LIMIT" in os.environ:
    torch._dynamo.config.cache_size_limit = int(
        os.environ["TORCHDYNAMO_CACHE_SIZE_LIMIT"]
    )

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Pretrain base model")
# Logging
parser.add_argument(
    "--run",
    type=str,
    default="dummy",
    help="wandb run name ('dummy' disables wandb logging)",
)
# Runtime
parser.add_argument(
    "--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)"
)
# FP8 training
parser.add_argument(
    "--fp8",
    action="store_true",
    help="enable FP8 training (requires H100+ GPU and torchao)",
)
parser.add_argument(
    "--fp8-recipe",
    type=str,
    default="tensorwise",
    choices=["rowwise", "tensorwise"],
    help="FP8 scaling recipe: tensorwise (faster, recommended) or rowwise (more accurate but slower)",
)
# Model architecture
parser.add_argument(
    "--depth", type=int, default=20, help="depth of the Transformer model"
)
parser.add_argument(
    "--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio"
)
parser.add_argument(
    "--head-dim", type=int, default=128, help="target head dimension for attention"
)
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
parser.add_argument(
    "--window-pattern",
    type=str,
    default="SSSL",
    help="sliding window pattern tiled across layers: L=full, S=half context (e.g. 'SSL')",
)
parser.add_argument(
    "--engram",
    action="store_true",
    help="enable nano-engram modules during base training",
)
parser.add_argument(
    "--engram-layers",
    type=str,
    default="2,6",
    help="comma-separated layer ids for Engram insertion",
)
parser.add_argument(
    "--engram-max-ngram-size",
    type=int,
    default=3,
    help="maximum ngram size for Engram hashing",
)
parser.add_argument(
    "--engram-heads-per-ngram", type=int, default=8, help="hash heads per ngram order"
)
parser.add_argument(
    "--engram-memory-dim", type=int, default=1280, help="Engram memory dimension"
)
parser.add_argument(
    "--engram-slot-multiplier",
    type=int,
    default=18,
    help="Engram hash table size multiplier",
)
parser.add_argument(
    "--engram-kernel-size", type=int, default=4, help="Engram short-conv kernel size"
)
parser.add_argument(
    "--engram-no-tokenizer-compression",
    action="store_true",
    help="disable tokenizer compression inside Engram",
)
parser.add_argument(
    "--engram-embedding-lr-mult",
    type=float,
    default=5.0,
    help="learning-rate multiplier for Engram embedding tables",
)
parser.add_argument(
    "--engram-pad-id",
    type=int,
    default=0,
    help="pad token id used for Engram left padding",
)
parser.add_argument(
    "--engram-seed", type=int, default=0, help="seed for Engram hashing multipliers"
)
parser.add_argument(
    "--engram-ablation-mode",
    type=str,
    default="none",
    choices=["none", "randomize", "uniform", "mlp"],
    help="Engram payload ablation: none=trainable table, randomize=frozen Gaussian table, uniform=frozen shared vector table, mlp=capacity/FLOP-matched control (no n-gram lookup; payload is a learned projection of the hidden state)",
)
parser.add_argument(
    "--engram-address-source",
    type=str,
    default="tokens",
    choices=["tokens", "lsh", "pq", "pkm", "hybrid"],
    help="what the Engram memory is addressed by: tokens=hash token ids (paper), "
    "lsh=frozen random projection of the hidden state, pq=learned VQ codebooks, "
    "pkm=product-key memory (learned keys, soft top-k read)",
)
parser.add_argument("--engram-lsh-bits", type=int, default=20, help="LSH code width in bits")
parser.add_argument(
    "--engram-lsh-balance",
    action="store_true",
    help="re-centre each LSH bit on an EMA of its projection mean so bits are ~balanced "
    "(raises realized code entropy; measured 8.6 -> 9.9 of 15 bits)",
)
parser.add_argument(
    "--engram-hybrid-token-head-frac",
    type=float,
    default=0.5,
    help="fraction of heads keeping TOKEN addressing under --engram-address-source=hybrid",
)
parser.add_argument(
    "--engram-readout-whiten",
    action="store_true",
    help="scale each memory read by 1/(its hit rate), so hot rows stop dominating the sum",
)
parser.add_argument(
    "--engram-readout-whiten-power",
    type=float,
    default=1.0,
    help="exponent on the whitening correction; 1.0 is full 1/frequency, 0.0 disables it",
)
parser.add_argument(
    "--engram-readout-whiten-decay",
    type=float,
    default=0.99,
    help="EMA decay for the per-row hit-rate estimate used by readout whitening",
)
parser.add_argument(
    "--engram-readout-whiten-max",
    type=float,
    default=1.0,
    help="cap on the whitening weight; 1.0 downweights hot rows only, >1 also boosts cold ones",
)
parser.add_argument(
    "--engram-value-rank",
    type=int,
    default=0,
    help="0 = a memory row is a constant vector (the paper); k>0 = a rank-k map of the hidden state",
)
parser.add_argument(
    "--engram-value-table",
    type=str,
    default="rows",
    choices=["rows", "tngram"],
    help="'rows' (default) = hash-indexed row tables (MultiHeadEmbedding/LowRankMemory); "
    "'tngram' = one CP-factorized tensor with token-position factors shared across "
    "n-gram orders (arXiv 2606.08347), indexed by window tokens directly",
)
parser.add_argument(
    "--engram-cp-rank",
    type=int,
    default=0,
    help="CP rank R for --engram-value-table=tngram (>=1 required there; ignored otherwise)",
)
parser.add_argument(
    "--engram-tngram-independent-factors",
    action="store_true",
    help="tngram only: give each n-gram order its OWN token-position factors instead "
    "of sharing A across orders (arXiv 2606.08347 shares by default). Ablation "
    "isolating whether cross-order sharing is the liability; match params by scaling "
    "cp_rank down (orders {2,3}: --engram-cp-rank=6 independent == 10 shared)",
)
parser.add_argument(
    "--engram-value-query-dim",
    type=int,
    default=128,
    help="width of the shared projection of h that rank-k memory rows act on",
)
parser.add_argument(
    "--engram-key-dim",
    type=int,
    default=0,
    help="per-head key dims stored alongside the value and used only for the gate (0 = shared vector, the paper)",
)
parser.add_argument(
    "--engram-count-gate",
    action="store_true",
    help="weight each head by a learned function of its row's hit rate (n-gram backoff)",
)
parser.add_argument(
    "--engram-count-gate-decay",
    type=float,
    default=0.99,
    help="EMA decay for the hit-rate estimate the count gate reads",
)
parser.add_argument(
    "--engram-count-gate-fourier",
    type=int,
    default=0,
    help="Fourier bands on log-count added to the count gate (0 = linear only); "
    "lets the count-to-trust curve be non-monotone, as Katz discounting is",
)
parser.add_argument(
    "--engram-share-hash",
    type=str,
    default="auto",
    choices=["auto", "yes", "no"],
    help="share the hash multipliers across Engram layers independently of "
    "--engram-share-memory ('auto' follows it, reproducing earlier runs)",
)
parser.add_argument(
    "--engram-count-gate-decouple",
    action="store_true",
    help="split a row's hit rate into its n-gram's own frequency (min over the "
    "order's heads) and that head's collision excess, and gate on both",
)
parser.add_argument(
    "--engram-backoff-keyset",
    type=str,
    default="",
    help="path to a frozen backoff keyset (scripts/build_engram_backoff_keyset.py); "
    "windows NOT in it back off to their per-token (*,*,t) key so the kept, frequent "
    "rows stay uncontaminated. Adds no params (iso). Token addressing only. '' = off",
)
parser.add_argument(
    "--engram-backoff-mode",
    type=str,
    default="none",
    choices=["none", "topp", "fw_topp", "allseen"],
    help="selection rule the keyset was built with (recorded for provenance; the "
    "keyset itself is frozen). allseen = keep every observed window, back off only "
    "keyset-unseen ones (the unseen-backoff control)",
)
parser.add_argument(
    "--engram-backoff-p",
    type=float,
    default=0.0,
    help="nucleus threshold the keyset was built with (recorded for provenance)",
)
parser.add_argument(
    "--engram-backoff-context-keep",
    type=int,
    default=1,
    help="tokens kept when a window backs off: 1 -> (*,*,t), 2 -> (*,b,t)",
)
parser.add_argument(
    "--engram-merge-frac",
    type=float,
    default=0.0,
    help="mid-training value-aware row merge: fraction of rows/head merged onto a "
    "survivor at --engram-merge-at-frac of training, shrinking the EFFECTIVE table "
    "(iso -- row_alias is a buffer, merged rows go dead). Token addressing only. 0 = off",
)
parser.add_argument(
    "--engram-merge-at-frac",
    type=float,
    default=0.5,
    help="fraction of training at which the one-shot row merge fires",
)
parser.add_argument(
    "--engram-merge-metric",
    type=str,
    default="cosine",
    choices=["cosine", "lsh"],
    help="how rows are clustered by learned value: exact cosine (small tables) or "
    "sign-LSH bucketing (scales to the shared table's millions of rows)",
)
parser.add_argument(
    "--engram-merge-source",
    type=str,
    default="value",
    choices=["value", "semantic"],
    help="what the row-merge clusters on. 'value' (default) = learned value table "
    "mid-training. 'semantic' = a static external token-embedding centroid per row "
    "(--engram-semantic-prior-path), merged FROZEN at step 0 so the table trains "
    "already-collapsed onto a semantic partition. Token addressing only.",
)
parser.add_argument(
    "--engram-merge-writeback",
    type=str,
    default="survivor",
    choices=["survivor", "mean", "mean_freq"],
    help="what value the survivor keeps after merge. 'survivor' (default) discards "
    "merged-away rows' learned values (original behavior). 'mean' sets the survivor "
    "to the plain mean of its cluster, banking the merged rows' learning. 'mean_freq' "
    "uses the hit-weighted mean (frequent rows dominate); it tracks a hit_rate buffer "
    "without changing the forward read.",
)
parser.add_argument(
    "--engram-merge-cluster",
    type=str,
    default="chunk",
    choices=["chunk", "nn"],
    help="how the lsh path groups rows. 'chunk' (default) cuts code-sorted rows into "
    "equal-size contiguous groups. 'nn' places group boundaries at least-similar "
    "adjacent seams so groups hold value-similar rows (true nearest neighbours).",
)
parser.add_argument(
    "--engram-semantic-prior-path",
    type=str,
    default="",
    help="path to a semantic-prior .pt (feature/hit_mask) from "
    "scripts.build_engram_semantic_prior; required for --engram-merge-source=semantic",
)
parser.add_argument(
    "--engram-value-activation",
    type=str,
    default="gelu",
    choices=["gelu", "relu", "identity"],
    help="nonlinearity inside a rank-k memory row ('identity' keeps the row linear)",
)
parser.add_argument(
    "--engram-address-latents-per-head",
    action="store_true",
    help="emit one independent latent per hash head instead of one code shared by all",
)
parser.add_argument(
    "--engram-pq-detach-encoder",
    action="store_true",
    help="address off a stop-gradient hidden state and emit no commitment loss "
    "(the commitment term collapsed the residual stream to ~1 effective dim)",
)
parser.add_argument("--engram-pq-subspaces", type=int, default=4, help="PQ sub-spaces")
parser.add_argument(
    "--engram-pq-codebook-size", type=int, default=64, help="centroids per PQ sub-space"
)
parser.add_argument("--engram-pq-decay", type=float, default=0.99, help="PQ codebook EMA decay")
parser.add_argument(
    "--engram-pq-commitment-weight", type=float, default=0.25, help="PQ commitment loss weight"
)
parser.add_argument("--engram-pkm-n-keys", type=int, default=512, help="PKM sub-keys per half (rows/head = n_keys^2)")
parser.add_argument("--engram-pkm-topk", type=int, default=32, help="PKM rows mixed per head")
parser.add_argument("--engram-pkm-query-dim", type=int, default=256, help="PKM query/key width")
parser.add_argument(
    "--engram-no-discretize-normalize",
    action="store_true",
    help="skip RMSNorm before discretizing (default is to normalize, for scale invariance)",
)
parser.add_argument(
    "--engram-share-memory",
    action="store_true",
    help="share ONE memory table and one hash addressing scheme across all Engram layers "
    "(cross-layer sharing ablation; the Engram analogue of --moe-share-blocks)",
)
parser.add_argument(
    "--engram-mhc-num-streams",
    type=int,
    default=4,
    help="number of Engram per-stream branches; must match --mhc-num-streams when --mhc is set",
)
parser.add_argument(
    "--mhc",
    action="store_true",
    help="enable true backbone mHC persistent residual streams; with --engram this is the faithful Engram+mHC path",
)
parser.add_argument(
    "--mhc-num-streams",
    type=int,
    default=4,
    help="number of persistent residual streams for true backbone mHC",
)
parser.add_argument(
    "--mhc-sinkhorn-iters",
    type=int,
    default=20,
    help="Sinkhorn iterations for true backbone mHC",
)
# MoE / Mobius. --moe-share-blocks selects the arm: 0 = per-layer MoE (the
# conditional-capacity baseline), N > 0 = Mobius (N routed pools shared across
# depth, Intern-S2-Mobius style). Both compose with --mhc.
parser.add_argument(
    "--moe",
    action="store_true",
    help="replace the dense MLP with a sparse MoE block on every --moe-every-th layer",
)
parser.add_argument(
    "--moe-experts", type=int, default=8, help="number of routed experts per pool"
)
parser.add_argument(
    "--moe-top-k", type=int, default=2, help="experts activated per token"
)
parser.add_argument(
    "--moe-d-ff-expert", type=int, default=256, help="hidden width of each expert"
)
parser.add_argument(
    "--moe-every",
    type=int,
    default=1,
    help="MoE replaces the dense MLP where layer_idx %% moe_every == 0",
)
parser.add_argument(
    "--moe-share-blocks",
    type=int,
    default=0,
    help="0 = per-layer MoE; N > 0 = Mobius with N routed pools shared across depth",
)
parser.add_argument(
    "--moe-shared-d-ff",
    type=int,
    default=0,
    help="width of the per-layer gated dense expert (0 = off; Mobius uses > 0)",
)
parser.add_argument(
    "--moe-activation",
    type=str,
    default="gelu",
    choices=["gelu", "relu", "silu"],
    help="expert activation",
)
parser.add_argument(
    "--moe-gate",
    type=str,
    default="sigmoid",
    choices=["sigmoid", "softmax"],
    help="router gate score fn",
)
parser.add_argument(
    "--moe-balancer",
    type=str,
    default="loss_free",
    choices=["noop", "loss_free", "aux"],
    help="load balancer: noop=free top-k, loss_free=DeepSeek bias, aux=Switch aux loss",
)
parser.add_argument(
    "--moe-balancer-kwargs",
    type=str,
    default="",
    help='JSON kwargs for the balancer, e.g. \'{"gamma": 0.01}\' or \'{"alpha": 0.01}\'',
)
parser.add_argument(
    "--moe-backend",
    type=str,
    default="scattermoe",
    choices=["scattermoe", "fallback"],
    help="expert kernel: scattermoe (Triton, needs the gpu extra) or fallback (pure torch)",
)
parser.add_argument(
    "--moe-expert-lr",
    type=float,
    default=0.004,
    help="AdamW learning rate for the 3D expert weights (Muon is 2D-only)",
)
parser.add_argument(
    "--moe-log-every",
    type=int,
    default=100,
    help="log router diagnostics every N steps (-1 = disable; each log step costs a host sync)",
)
# Scaling-ladder knobs. When --moe-granularity > 0 these override the explicit
# --moe-experts / --moe-top-k / --moe-d-ff-expert flags.
parser.add_argument(
    "--moe-granularity",
    type=float,
    default=0.0,
    help="G: d_ff_expert = ff_mult*d_model/G (0 = use the explicit --moe-* sizes)",
)
parser.add_argument(
    "--moe-expansion", type=float, default=8.0, help="X: n_experts = X*G"
)
parser.add_argument(
    "--moe-active-mult",
    type=float,
    default=1.0,
    help="A: top_k = A*G (A=1 => active expert FLOPs == one dense FFN)",
)
parser.add_argument(
    "--moe-ff-mult",
    type=int,
    default=4,
    help="dense FFN width multiple used as the ladder's iso-FLOP base",
)
# Training horizon (only one used, in order of precedence)
parser.add_argument(
    "--num-iterations",
    type=int,
    default=-1,
    help="explicit number of optimization steps (-1 = disable)",
)
parser.add_argument(
    "--target-flops",
    type=float,
    default=-1.0,
    help="calculate num_iterations to reach target_flops (-1 = disable)",
)
parser.add_argument(
    "--target-param-data-ratio",
    type=float,
    default=10.5,
    help="calculate num_iterations to maintain data:param ratio (Chinchilla=20, -1 = disable)",
)
# Optimization
parser.add_argument(
    "--device-batch-size",
    type=int,
    default=32,
    help="per-device batch size. good number to reduce to 16,8,4,... if you OOM on VRAM.",
)
parser.add_argument(
    "--total-batch-size",
    type=int,
    default=-1,
    help="total batch size in tokens. decent numbers are e.g. 524288. (-1 = auto-compute optimal)",
)
parser.add_argument(
    "--embedding-lr",
    type=float,
    default=0.3,
    help="learning rate for embedding parameters (Adam)",
)
parser.add_argument(
    "--unembedding-lr",
    type=float,
    default=0.004,
    help="learning rate for unembedding parameters (Adam)",
)
parser.add_argument(
    "--weight-decay",
    type=float,
    default=0.2,
    help="cautious weight decay for the Muon optimizer (for weights)",
)
parser.add_argument(
    "--matrix-lr",
    type=float,
    default=0.02,
    help="learning rate for matrix parameters (Muon)",
)
parser.add_argument(
    "--scalar-lr",
    type=float,
    default=0.5,
    help="learning rate for scalars (resid_lambdas, x0_lambdas)",
)
parser.add_argument(
    "--adam-beta1", type=float, default=0.8, help="Adam beta1 for embedding/unembedding"
)
parser.add_argument(
    "--adam-beta2",
    type=float,
    default=0.95,
    help="Adam beta2 for embedding/unembedding",
)
parser.add_argument(
    "--warmup-ratio", type=float, default=0.0, help="ratio of iterations for LR warmup"
)
parser.add_argument(
    "--warmdown-ratio",
    type=float,
    default=0.5,
    help="ratio of iterations for LR warmdown",
)
parser.add_argument(
    "--final-lr-frac",
    type=float,
    default=0.0,
    help="final LR as fraction of initial LR",
)
parser.add_argument(
    "--resume-from-step",
    type=int,
    default=-1,
    help="resume training from this step (-1 = disable)",
)
# Evaluation
parser.add_argument(
    "--eval-every",
    type=int,
    default=250,
    help="evaluate val bpb every N steps (-1 = disable)",
)
parser.add_argument(
    "--eval-tokens",
    type=int,
    default=80 * 524288,
    help="number of tokens to evaluate val loss on",
)
parser.add_argument(
    "--core-metric-every",
    type=int,
    default=2000,
    help="evaluate CORE metric every N steps (-1 = disable)",
)
parser.add_argument(
    "--core-metric-max-per-task",
    type=int,
    default=500,
    help="examples per task for CORE metric",
)
parser.add_argument(
    "--sample-every",
    type=int,
    default=2000,
    help="sample from model every N steps (-1 = disable)",
)
parser.add_argument(
    "--save-every",
    type=int,
    default=-1,
    help="save checkpoints every N steps (-1 = only at end)",
)
# Output
parser.add_argument(
    "--model-tag",
    type=str,
    default=None,
    help="override model tag for checkpoint directory name",
)
args = parser.parse_args()
user_config = vars(args).copy()  # for logging
# -----------------------------------------------------------------------------
# Compute init and wandb logging

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
autocast_ctx = (
    torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
    if device_type == "cuda"
    else nullcontext()
)
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float("inf")  # MFU not meaningful for CPU/MPS

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = (
    DummyWandb()
    if use_dummy_wandb
    else wandb.init(project="nano_scalemb", name=args.run, config=user_config)
)

# Flash Attention status
if HAS_FA3:
    print0(
        "✓ Using Flash Attention 3 (Hopper GPU detected), efficient, new and awesome."
    )
else:
    print0("!" * 80)
    print0("WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback")
    print0("WARNING: Training will be less efficient without FA3")
    if args.window_pattern != "L":
        print0(
            f"WARNING: SDPA has no support for sliding window attention (window_pattern='{args.window_pattern}'). Your GPU utilization will be terrible."
        )
        print0(
            "WARNING: Recommend using --window-pattern L for full context attention without alternating sliding window patterns."
        )
    print0("!" * 80)

# -----------------------------------------------------------------------------
# Tokenizer will be useful for evaluation and also we need the vocab size to init the model
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# -----------------------------------------------------------------------------
# Initialize the Model


def build_engram_config() -> EngramConfig | None:
    if not args.engram:
        if args.engram_ablation_mode != "none":
            raise ValueError(
                "--engram-ablation-mode requires --engram because the dense baseline should omit Engram entirely"
            )
        if args.engram_share_memory:
            raise ValueError("--engram-share-memory requires --engram")
        if args.engram_address_source != "tokens":
            raise ValueError("--engram-address-source requires --engram")
        if args.engram_readout_whiten:
            raise ValueError("--engram-readout-whiten requires --engram")
        if args.engram_value_rank:
            raise ValueError("--engram-value-rank requires --engram")
        if args.engram_key_dim:
            raise ValueError("--engram-key-dim requires --engram")
        if args.engram_count_gate:
            raise ValueError("--engram-count-gate requires --engram")
        if args.engram_count_gate_fourier:
            raise ValueError("--engram-count-gate-fourier requires --engram")
        if args.engram_count_gate_decouple:
            raise ValueError("--engram-count-gate-decouple requires --engram")
        if args.engram_backoff_keyset:
            raise ValueError("--engram-backoff-keyset requires --engram")
        if args.engram_value_table != "rows":
            raise ValueError("--engram-value-table requires --engram")
        if args.engram_cp_rank:
            raise ValueError("--engram-cp-rank requires --engram")
        if args.engram_tngram_independent_factors:
            raise ValueError("--engram-tngram-independent-factors requires --engram")
        return None
    if args.engram_count_gate_decouple and not args.engram_count_gate:
        raise ValueError(
            "--engram-count-gate-decouple requires --engram-count-gate: it splits "
            "that gate's input channel, it is not a separate mechanism"
        )
    if args.engram_count_gate_fourier:
        if args.engram_count_gate_fourier < 0:
            raise ValueError("--engram-count-gate-fourier must be >= 0")
        if not args.engram_count_gate:
            raise ValueError(
                "--engram-count-gate-fourier requires --engram-count-gate: the "
                "Fourier terms are added to that gate's linear term, not a "
                "separate mechanism"
            )
    for flag, name in ((args.engram_key_dim, "--engram-key-dim"),
                       (args.engram_count_gate, "--engram-count-gate")):
        if not flag:
            continue
        # Neither has a MultiHeadEmbedding to act on, so the flag would be
        # silently inert rather than wrong -- which is harder to notice.
        if args.engram_address_source == "pkm":
            raise ValueError(
                f"{name} does not apply to --engram-address-source=pkm: the "
                "product-key memory owns its own keys and values"
            )
        if args.engram_ablation_mode == "mlp":
            raise ValueError(
                f"{name} does not apply to --engram-ablation-mode=mlp: that "
                "control has no memory table"
            )
    if args.engram_key_dim:
        if args.engram_key_dim < 0:
            raise ValueError(
                f"--engram-key-dim must be >= 0, got {args.engram_key_dim}"
            )
        if args.engram_value_rank:
            raise ValueError(
                "--engram-key-dim with --engram-value-rank is not supported: the "
                "rank axis measured 0.01x-0.09x of its error bar (study §8.4), so "
                "combining them would spend an arm on a dead variable"
            )
    if args.engram_value_rank:
        if args.engram_value_rank < 0:
            raise ValueError(
                f"--engram-value-rank must be >= 0, got {args.engram_value_rank}"
            )
        # Both run without a MultiHeadEmbedding, so the flag would silently do
        # nothing rather than be wrong -- which is harder to notice.
        if args.engram_address_source == "pkm":
            raise ValueError(
                "--engram-value-rank does not apply to "
                "--engram-address-source=pkm: the product-key memory owns its "
                "own values"
            )
        if args.engram_ablation_mode == "mlp":
            raise ValueError(
                "--engram-value-rank does not apply to "
                "--engram-ablation-mode=mlp: that control has no memory table"
            )
    if args.engram_readout_whiten:
        # Both of these run without a MultiHeadEmbedding, so the flag would be
        # silently inert rather than wrong -- which is worse.
        if args.engram_address_source == "pkm":
            raise ValueError(
                "--engram-readout-whiten does not apply to "
                "--engram-address-source=pkm: the product-key memory owns its "
                "own values and has no hash table to whiten"
            )
        if args.engram_ablation_mode == "mlp":
            raise ValueError(
                "--engram-readout-whiten does not apply to "
                "--engram-ablation-mode=mlp: that control has no memory table"
            )
        if not 0.0 <= args.engram_readout_whiten_power <= 1.0:
            raise ValueError(
                "--engram-readout-whiten-power must be in [0, 1], got "
                f"{args.engram_readout_whiten_power}"
            )
    if args.engram_backoff_keyset:
        if args.engram_address_source != "tokens":
            raise ValueError(
                "--engram-backoff-keyset only applies to "
                "--engram-address-source=tokens: the keyset is over token n-grams"
            )
        if not os.path.exists(args.engram_backoff_keyset):
            raise ValueError(
                f"--engram-backoff-keyset path does not exist: {args.engram_backoff_keyset}"
            )
        if args.engram_backoff_context_keep < 1:
            raise ValueError("--engram-backoff-context-keep must be >= 1")
    elif args.engram_backoff_mode != "none":
        raise ValueError(
            "--engram-backoff-mode is set but --engram-backoff-keyset is empty: the "
            "mode is provenance for a keyset that must be provided"
        )
    if args.engram_merge_frac:
        if not 0.0 < args.engram_merge_frac < 1.0:
            raise ValueError(
                f"--engram-merge-frac must be in (0, 1), got {args.engram_merge_frac}"
            )
        if not 0.0 < args.engram_merge_at_frac < 1.0:
            raise ValueError(
                f"--engram-merge-at-frac must be in (0, 1), got {args.engram_merge_at_frac}"
            )
        # Value-aware merge clusters on the learned VALUE table only (never the
        # address code), so it is address-source-agnostic for the flat-value
        # MultiHeadEmbedding used by tokens/lsh/pq. pkm/hybrid use a different
        # memory geometry (merge_rows would raise), so they stay excluded. This
        # opens the "over-provision a semantic-hash table then value-merge back to
        # the reference size" test -- does the schedule rescue the LSH liability?
        if args.engram_address_source not in ("tokens", "lsh", "pq"):
            raise ValueError(
                "--engram-merge-frac requires --engram-address-source in "
                "{tokens, lsh, pq} (pkm/hybrid use an incompatible memory table)"
            )
    if args.engram_merge_source == "semantic":
        # The semantic prior is a static token-embedding centroid per hash row, so
        # it is only defined for TOKEN addressing (rows are token-window buckets),
        # and it merges the over-provisioned table back down -- so it needs a merge
        # fraction to act on.
        if not args.engram_merge_frac:
            raise ValueError(
                "--engram-merge-source=semantic requires --engram-merge-frac > 0"
            )
        if args.engram_address_source != "tokens":
            raise ValueError(
                "--engram-merge-source=semantic requires --engram-address-source=tokens"
            )
        if not args.engram_semantic_prior_path:
            raise ValueError(
                "--engram-merge-source=semantic requires --engram-semantic-prior-path"
            )
    if args.engram_value_table == "tngram":
        # A CP-factorized tensor has no hash-indexed rows: it indexes the window
        # tokens directly, carries a single payload per read, and cannot be
        # value-merged (no rows to cluster). Forbid every option that assumes a
        # MultiHeadEmbedding/LowRankMemory row table, so a silent no-op is a hard
        # error instead.
        if args.engram_cp_rank < 1:
            raise ValueError(
                "--engram-value-table=tngram requires --engram-cp-rank >= 1"
            )
        if args.engram_value_rank:
            raise ValueError(
                "--engram-value-table=tngram is incompatible with "
                "--engram-value-rank (it replaces the row table entirely)"
            )
        if args.engram_key_dim:
            raise ValueError(
                "--engram-value-table=tngram is incompatible with --engram-key-dim "
                "(a factorized read carries no separate key)"
            )
        if args.engram_address_source != "tokens":
            raise ValueError(
                "--engram-value-table=tngram indexes window tokens directly, so it "
                "requires --engram-address-source=tokens"
            )
        if args.engram_merge_frac:
            raise ValueError(
                "--engram-value-table=tngram has no rows to merge; "
                "--engram-merge-frac is undefined"
            )
        if args.engram_readout_whiten or args.engram_count_gate:
            raise ValueError(
                "--engram-value-table=tngram has no per-row hit statistics; "
                "--engram-readout-whiten / --engram-count-gate do not apply"
            )
        if args.engram_ablation_mode == "mlp":
            raise ValueError(
                "--engram-value-table=tngram does not apply to "
                "--engram-ablation-mode=mlp: that control has no memory table"
            )
    elif args.engram_cp_rank:
        raise ValueError(
            "--engram-cp-rank only applies to --engram-value-table=tngram"
        )
    elif args.engram_tngram_independent_factors:
        raise ValueError(
            "--engram-tngram-independent-factors only applies to "
            "--engram-value-table=tngram"
        )
    layer_ids = tuple(
        int(layer.strip()) for layer in args.engram_layers.split(",") if layer.strip()
    )
    if not layer_ids:
        raise ValueError(
            "--engram-layers must provide at least one layer id when --engram is set"
        )
    return EngramConfig(
        layer_ids=layer_ids,
        max_ngram_size=args.engram_max_ngram_size,
        n_head_per_ngram=args.engram_heads_per_ngram,
        memory_dim=args.engram_memory_dim,
        slot_multiplier=args.engram_slot_multiplier,
        kernel_size=args.engram_kernel_size,
        use_tokenizer_compression=not args.engram_no_tokenizer_compression,
        embedding_lr_mult=args.engram_embedding_lr_mult,
        pad_id=args.engram_pad_id,
        seed=args.engram_seed,
        ablation_mode=args.engram_ablation_mode,
        mhc_num_streams=args.engram_mhc_num_streams,
        share_memory=args.engram_share_memory,
        share_hash={'auto': None, 'yes': True, 'no': False}[args.engram_share_hash],
        address_source=args.engram_address_source,
        discretize_normalize=not args.engram_no_discretize_normalize,
        lsh_bits=args.engram_lsh_bits,
        pq_subspaces=args.engram_pq_subspaces,
        pq_codebook_size=args.engram_pq_codebook_size,
        pq_decay=args.engram_pq_decay,
        pq_commitment_weight=args.engram_pq_commitment_weight,
        pkm_n_keys=args.engram_pkm_n_keys,
        pkm_topk=args.engram_pkm_topk,
        pkm_query_dim=args.engram_pkm_query_dim,
        lsh_balance=args.engram_lsh_balance,
        hybrid_token_head_frac=args.engram_hybrid_token_head_frac,
        address_latents_per_head=args.engram_address_latents_per_head,
        pq_detach_encoder=args.engram_pq_detach_encoder,
        readout_whiten=args.engram_readout_whiten,
        readout_whiten_power=args.engram_readout_whiten_power,
        readout_whiten_decay=args.engram_readout_whiten_decay,
        readout_whiten_max=args.engram_readout_whiten_max,
        value_rank=args.engram_value_rank,
        value_query_dim=args.engram_value_query_dim,
        value_activation=args.engram_value_activation,
        value_table=args.engram_value_table,
        cp_rank=args.engram_cp_rank,
        tngram_share_factors=not args.engram_tngram_independent_factors,
        key_dim=args.engram_key_dim,
        count_gate=args.engram_count_gate,
        count_gate_decay=args.engram_count_gate_decay,
        count_gate_fourier=args.engram_count_gate_fourier,
        count_gate_decouple=args.engram_count_gate_decouple,
        backoff_keyset_path=args.engram_backoff_keyset,
        backoff_mode=args.engram_backoff_mode,
        backoff_p=args.engram_backoff_p,
        backoff_context_keep=args.engram_backoff_context_keep,
        merge_frac=args.engram_merge_frac,
        merge_at_frac=args.engram_merge_at_frac,
        merge_metric=args.engram_merge_metric,
        merge_source=args.engram_merge_source,
        merge_writeback=args.engram_merge_writeback,
        merge_cluster=args.engram_merge_cluster,
        semantic_prior_path=args.engram_semantic_prior_path,
    )


engram_config = build_engram_config()


def build_mhc_config() -> MHCConfig | None:
    if not args.mhc:
        return None
    if engram_config is not None:
        if engram_config.mhc_num_streams != args.mhc_num_streams:
            raise ValueError(
                "--mhc with --engram requires --engram-mhc-num-streams == --mhc-num-streams"
            )
    return MHCConfig(
        num_streams=args.mhc_num_streams,
        sinkhorn_iters=args.mhc_sinkhorn_iters,
    )


mhc_config = build_mhc_config()


def build_moe_config(model_dim) -> MoEConfig | None:
    """Resolve the MoE/Mobius config for a given model dim.

    Depends on model_dim because the ladder knobs size the experts relative to
    the dense FFN width, so this is called from build_model_meta rather than once
    up front (mirroring how the d12 reference model is rebuilt for scaling laws).
    """
    if not args.moe:
        if args.moe_share_blocks:
            raise ValueError("--moe-share-blocks requires --moe")
        return None
    balancer_kwargs = json.loads(args.moe_balancer_kwargs) if args.moe_balancer_kwargs else {}
    shared = dict(
        moe_every=args.moe_every,
        share_blocks=args.moe_share_blocks,
        shared_d_ff=args.moe_shared_d_ff,
        activation=args.moe_activation,
        gate=args.moe_gate,
        backend=args.moe_backend,
        balancer=args.moe_balancer,
        balancer_kwargs=balancer_kwargs,
        expert_lr=args.moe_expert_lr,
    )
    if args.moe_granularity > 0:
        return moe_config_from_knobs(
            model_dim,
            args.moe_granularity,
            args.moe_expansion,
            active_mult=args.moe_active_mult,
            ff_mult=args.moe_ff_mult,
            **shared,
        )
    return MoEConfig(
        n_experts=args.moe_experts,
        top_k=args.moe_top_k,
        d_ff_expert=args.moe_d_ff_expert,
        **shared,
    )


def build_model_meta(depth):
    """Build a model on meta device for a given depth (shapes/dtypes only, no data)."""
    # Model dim is nudged up to nearest multiple of head_dim for clean division
    # (FA3 requires head_dim divisible by 8, and this guarantees head_dim == args.head_dim exactly)
    base_dim = depth * args.aspect_ratio
    model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    num_heads = model_dim // args.head_dim
    depth_engram_config = engram_config
    if engram_config is not None:
        valid_layer_ids = tuple(
            layer_id for layer_id in engram_config.layer_ids if layer_id < depth
        )
        if valid_layer_ids != engram_config.layer_ids:
            depth_engram_config = replace(engram_config, layer_ids=valid_layer_ids)

    config = GPTConfig(
        sequence_len=args.max_seq_len,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern=args.window_pattern,
        engram=depth_engram_config,
        mhc=mhc_config,
        moe=build_moe_config(model_dim),
    )
    with torch.device("meta"):
        model_meta = GPT(config)
    return model_meta


# Build the model, move to device, init the weights
model = build_model_meta(
    args.depth
)  # 1) Build on meta device (only shapes/dtypes, no data)
model_config = model.config
model_config_kwargs = asdict(model_config)
print0(f"Model config:\n{json.dumps(model_config_kwargs, indent=2)}")
if model_config.moe is not None:
    # An MoE-vs-dense comparison is uninterpretable without all three of these:
    # the arms are matched on *active*, and total/active is the sparsity actually
    # realized (which moe_every and share_blocks both move).
    d = describe_moe(model_config.n_embd, model_config.moe, ff_mult=args.moe_ff_mult)
    arm = (
        f"mobius (share_blocks={model_config.moe.share_blocks})"
        if model_config.moe.share_blocks > 0
        else "moe (per-layer)"
    )
    print0(
        f"MoE arm: {arm} | {d['n_experts']}a{d['top_k']} dff={d['d_ff_expert']} "
        f"every={model_config.moe.moe_every} shared_d_ff={model_config.moe.shared_d_ff} "
        f"balancer={model_config.moe.balancer} backend={model_config.moe.backend}"
    )
    print0(
        f"MoE ladder point: granularity G={d['G']:.3g} active_mult A={d['A']:.3g} "
        f"expansion X={d['X']:.3g} sparsity={d['sparsity']:.3g}"
    )
    _tot, _act = total_params(model), active_params(model)
    print0(
        f"MoE params: total {_tot / 1e6:.1f}M | active {_act / 1e6:.1f}M "
        f"({_tot / max(1, _act):.2f}x)"
    )
if model_config.engram is not None and getattr(model, "_ngram_hasher", None) is not None:
    ks = getattr(model._ngram_hasher, "_backoff_keysets", {})
    if ks:
        base = model._ngram_hasher._hash_base
        slot = model_config.engram.slot_multiplier
        parts = [
            f"n={o}: {ids.numel():,} keys (~{ids.numel() / (base * slot):.0%} load)"
            for o, ids in sorted(ks.items())
        ]
        print0(
            f"Engram backoff: mode={model_config.engram.backoff_mode} "
            f"p={model_config.engram.backoff_p} keep={model_config.engram.backoff_context_keep} | "
            + " | ".join(parts)
        )
model.to_empty(
    device=device
)  # 2) All tensors get storage on target device but with uninitialized (garbage) data
model.init_weights()  # 3) All tensors get initialized

# If we are resuming, overwrite the model parameters with those of the checkpoint
base_dir = get_base_dir()
output_dirname = args.model_tag if args.model_tag else f"d{args.depth}"  # e.g. d12
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
resuming = args.resume_from_step != -1
if resuming:
    print0(f"Resuming optimization from step {args.resume_from_step}")
    model_data, optimizer_data, meta_data = load_checkpoint(
        checkpoint_dir,
        args.resume_from_step,
        device,
        load_optimizer=True,
        rank=ddp_rank,
    )
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data  # free up this memory after the copy

# -----------------------------------------------------------------------------
# FP8 training initialization and management (this has to be done before torch.compile)

# Convert Linear layers to Float8Linear if --fp8 is set
if args.fp8:
    if device_type != "cuda":
        print0("Warning: FP8 training requires CUDA, ignoring --fp8 flag")
    else:
        # our custom fp8 is simpler than torchao, written for exact API compatibility
        from nano_scalemb.fp8 import Float8LinearConfig, convert_to_float8_training

        # from torchao.float8 import Float8LinearConfig, convert_to_float8_training
        import torch.nn as nn

        # Filter: dims must be divisible by 16 (FP8 hardware requirement) large enough
        def fp8_module_filter(mod: nn.Module, fqn: str) -> bool:
            if not isinstance(mod, nn.Linear):
                return False
            if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                return False
            if min(mod.in_features, mod.out_features) < 128:
                return False
            return True

        fp8_config = Float8LinearConfig.from_recipe_name(args.fp8_recipe)
        num_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
        convert_to_float8_training(
            model, config=fp8_config, module_filter_fn=fp8_module_filter
        )
        num_fp8 = sum(1 for m in model.modules() if "Float8" in type(m).__name__)
        num_skipped = num_linear - num_fp8
        print0(
            f"✓ FP8 training enabled ({args.fp8_recipe} scaling) - converted {num_fp8}/{num_linear} linear layers, skipped {num_skipped} (too small)"
        )


# Context manager to temporarily disable FP8 so that model evaluation remains in BF16
@contextmanager
def disable_fp8(model):
    """Temporarily swap Float8Linear modules with nn.Linear for BF16 evaluation.

    CastConfig is a frozen dataclass, so we can't mutate scaling_type. Instead,
    we swap out Float8Linear modules entirely and restore them after.
    """
    import torch.nn as nn

    # Find all Float8Linear modules and their locations
    fp8_locations = []  # list of (parent_module, attr_name, fp8_module)
    for name, module in model.named_modules():
        if "Float8" in type(module).__name__:
            if "." in name:
                parent_name, attr_name = name.rsplit(".", 1)
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name
            fp8_locations.append((parent, attr_name, module))

    if not fp8_locations:
        yield  # No FP8 modules, nothing to do
        return

    # Swap Float8Linear -> nn.Linear (shares the same weight tensor, no copy)
    for parent, attr_name, fp8_module in fp8_locations:
        linear = nn.Linear(
            fp8_module.in_features,
            fp8_module.out_features,
            bias=fp8_module.bias is not None,
            device=fp8_module.weight.device,
            dtype=fp8_module.weight.dtype,
        )
        linear.weight = fp8_module.weight  # share, don't copy
        if fp8_module.bias is not None:
            linear.bias = fp8_module.bias
        setattr(parent, attr_name, linear)

    try:
        yield
    finally:
        # Restore Float8Linear modules
        for parent, attr_name, fp8_module in fp8_locations:
            setattr(parent, attr_name, fp8_module)


# -----------------------------------------------------------------------------
# Compile the model

orig_model = model  # original, uncompiled model, for saving raw model state_dict and for inference/evaluation (because the shapes may change shape)
model = torch.compile(
    model, dynamic=False
)  # the inputs to model will never change shape so dynamic=False is safe

# -----------------------------------------------------------------------------
# Scaling laws and muP extrapolations to determine the optimal training horizon, batch size, learning rates, weight decay.

# Get the parameter counts of our model
param_counts = model.num_scaling_params()
print0(f"Parameter counts:")
for key, value in param_counts.items():
    print0(f"{key:24s}: {value:,}")
num_params = param_counts["total"]
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")


# 1) Use scaling laws to determine the optimal training horizon in tokens
# The compute-optimal models satisfy the Tokens:Params ratio of --target-param-data-ratio (derived experimentally via scaling laws analysis).
# We've already initialized the model so we have Params. Optimal Tokens is now simply target-param-data-ratio * Params
def get_scaling_params(m):
    # As for which params to use exactly, transformer matrices + lm_head gives cleanest scaling laws (see dev/LOG.md Jan 27, 2026)
    # For MoE runs we add the *active* expert params: a Chinchilla-style ratio on
    # total MoE params is meaningless, and dropping the experts entirely would
    # size an MoE run as if its extra capacity didn't exist.
    params_counts = m.num_scaling_params()
    scaling_params = params_counts["transformer_matrices"] + params_counts["lm_head"]
    scaling_params += params_counts["expert_active"]
    return scaling_params


num_scaling_params = get_scaling_params(model)
if args.target_param_data_ratio > 0:
    target_tokens = int(
        args.target_param_data_ratio * num_scaling_params
    )  # optimal tokens for the model we are about to train

    # Our reference model is d12, this is where a lot of hyperparameters are tuned and then transfered to higher depths (muP style)
    d12_ref = build_model_meta(12)  # creates the model on meta device
    D_REF = args.target_param_data_ratio * get_scaling_params(
        d12_ref
    )  # compute-optimal d12 training horizon in tokens (measured empirically)
elif args.target_flops > 0:
    target_tokens = int(args.target_flops / num_flops_per_token)
    D_REF = target_tokens
    print0(f"Using target FLOPs token estimate: {target_tokens:,} tokens")
else:
    assert args.num_iterations > 0 and args.total_batch_size > 0, (
        "--target-param-data-ratio=-1 requires explicit --num-iterations and --total-batch-size"
    )
    target_tokens = args.num_iterations * args.total_batch_size
    D_REF = target_tokens
    print0(
        f"Using fixed token budget: {target_tokens:,} tokens from {args.num_iterations:,} iterations x {args.total_batch_size:,} tokens"
    )
B_REF = 2**19  # optimal batch size at d12 ~= 524,288 tokens (measured empirically)

# 2) Now that we have the token horizon, we can calculate the optimal batch size
# We follow the Power Lines paper (Bopt ∝ D^0.383), ref: https://arxiv.org/abs/2505.13738
# The optimal batch size grows as approximately D^0.383, so e.g. if D doubles from d12 to d24, B should grow by 2^0.383 ≈ 1.3x.
total_batch_size = args.total_batch_size  # user-provided override is possible
if total_batch_size == -1:
    batch_size_ratio = target_tokens / D_REF
    predicted_batch_size = B_REF * batch_size_ratio**0.383
    total_batch_size = 2 ** round(
        math.log2(predicted_batch_size)
    )  # clamp to nearest power of 2 for efficiency
    print0(f"Auto-computed optimal batch size: {total_batch_size:,} tokens")

# 3) Knowing the batch size, we can now calculate a learning rate correction (bigger batch size allows higher learning rates)
batch_lr_scale = 1.0
batch_ratio = total_batch_size / B_REF  # B/B_ref
if batch_ratio != 1.0:
    # SGD: linear scaling with batch size is standard (not used in nano_scalemb)
    # AdamW: sqrt scaling is standard: η ∝ √(B/B_ref)
    # Muon: we will use the same scaling for Muon as for AdamW: η ∝ √(B/B_ref) (not studied carefully, assumption!)
    batch_lr_scale = batch_ratio**0.5  # η ∝ √(B/B_ref)
    print0(
        f"Scaling LRs by {batch_lr_scale:.4f} for batch size {total_batch_size:,} (reference: {B_REF:,})"
    )

# 4) Knowing the batch size and the token horizon, we can now calculate the appropriate weight decay scaling
# We adopt the T_epoch framework from https://arxiv.org/abs/2405.13698
# Central idea of the paper is that T_epoch = B/(η·λ·D) should remain constant.
# Above, we used learning rate scaling η ∝ √(B/B_ref). So it's a matter of ~10 lines of math to derive that to keep T_epoch constant, we need:
# λ = λ_ref · √(B/B_ref) · (D_ref/D)
# Note that these papers study AdamW, *not* Muon. We are blindly following AdamW theory for scaling hoping it ~works for Muon too.
weight_decay_scaled = (
    args.weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)
)
if weight_decay_scaled != args.weight_decay:
    print0(
        f"Scaling weight decay from {args.weight_decay:.6f} to {weight_decay_scaled:.6f} for depth {args.depth}"
    )

# -----------------------------------------------------------------------------
# Initialize the Optimizer (combined MuonAdamW: Muon for matrix params, AdamW for rest)
optimizer = model.setup_optimizer(
    # AdamW hyperparameters
    unembedding_lr=args.unembedding_lr * batch_lr_scale,
    embedding_lr=args.embedding_lr * batch_lr_scale,
    scalar_lr=args.scalar_lr * batch_lr_scale,
    adam_betas=(args.adam_beta1, args.adam_beta2),
    # Muon hyperparameters
    matrix_lr=args.matrix_lr * batch_lr_scale,
    weight_decay=weight_decay_scaled,
)

if resuming:
    optimizer.load_state_dict(optimizer_data)
    del optimizer_data

# -----------------------------------------------------------------------------
# Initialize the DataLoaders for train/val
dataloader_resume_state_dict = (
    None if not resuming else meta_data["dataloader_state_dict"]
)
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer,
    args.device_batch_size,
    args.max_seq_len,
    split="train",
    device=device,
    resume_state_dict=dataloader_resume_state_dict,
)
build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device
)
x, y, dataloader_state_dict = next(
    train_loader
)  # kick off load of the very first batch of data

# -----------------------------------------------------------------------------
# Calculate the number of iterations we will train for and set up the various schedulers

# num_iterations: either it is given, or from target flops, or from target data:param ratio (in that order)
assert (
    args.num_iterations > 0 or args.target_param_data_ratio > 0 or args.target_flops > 0
)
if args.num_iterations > 0:
    # Override num_iterations to a specific value if given
    num_iterations = args.num_iterations
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif args.target_flops > 0:
    # Calculate the number of iterations from the target flops (used in scaling laws analysis, e.g. runs/scaling_laws.sh)
    num_iterations = round(args.target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif args.target_param_data_ratio > 0:
    # Calculate the number of iterations from the target param data ratio (the most common use case)
    num_iterations = target_tokens // total_batch_size
    print0(
        f"Calculated number of iterations from target data:param ratio: {num_iterations:,}"
    )
else:
    raise ValueError("No training horizon specified")
total_tokens = (
    total_batch_size * num_iterations
)  # the actual number of tokens we will train for
print0(f"Total number of training tokens: {total_tokens:,}")
print0(
    f"Tokens : Scaling params ratio: {total_batch_size * num_iterations / num_scaling_params:.2f}"
)  # e.g. Chinchilla was ~20
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# Step at which the mid-training Engram row merge fires (0 = disabled). Computed
# from num_iterations so it tracks the actual horizon, like the LR schedule. The
# SEMANTIC merge is a separate, step-0 frozen path (see the pre-loop block below),
# so it does NOT use this mid-training schedule.
engram_semantic_merge = (
    engram_config is not None
    and engram_config.merge_frac
    and engram_config.merge_source == "semantic"
)
engram_merge_step = (
    round(engram_config.merge_at_frac * num_iterations)
    if engram_config is not None and engram_config.merge_frac
    and not engram_semantic_merge
    else 0
)
if engram_merge_step > 0:
    print0(
        f"Engram row merge scheduled at step {engram_merge_step:,} "
        f"(merge_frac={engram_config.merge_frac}, metric={engram_config.merge_metric})"
    )
if engram_semantic_merge:
    print0(
        f"Engram SEMANTIC row merge scheduled at step 0 "
        f"(merge_frac={engram_config.merge_frac}, metric={engram_config.merge_metric}, "
        f"prior={engram_config.semantic_prior_path})"
    )


# Learning rate schedule (linear warmup, constant, linear warmdown)
def get_lr_multiplier(it):
    warmup_iters = round(args.warmup_ratio * num_iterations)
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * args.final_lr_frac


# Momentum scheduler for Muon optimizer (warms up to 0.95 over the first 300 steps)
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum


# Weight decay scheduler for Muon optimizer (linearly decays to zero over the course of training)
def get_weight_decay(it):
    return weight_decay_scaled * (1 - it / num_iterations)


# -----------------------------------------------------------------------------
# Training loop

# Engram SEMANTIC row merge -- fires ONCE at step 0, before the training loop, so
# the first compiled forward already sees the merged alias (no mid-run recompile).
# Skipped on resume: the merged row_alias is already in the checkpoint and
# sync_merge_flags() restored it. The prior .pt is loaded on ALL ranks (identical
# file -> identical alias), matching merge_rows' no-communication DDP discipline.
if engram_semantic_merge and not resuming:
    prior = torch.load(engram_config.semantic_prior_path, map_location="cpu",
                       weights_only=False)
    feature = prior["feature"]          # [total_rows, proj_dim], fp16
    hit_mask = prior["hit_mask"]        # bool[total_rows]
    for table in orig_model._engram_memory_tables():
        if getattr(table, "_merged", None) is not None and not bool(table._merged):
            if feature.shape[0] != table.total_rows:
                raise ValueError(
                    f"semantic prior has {feature.shape[0]} rows but the engram "
                    f"table has {table.total_rows}; rebuild the prior with the "
                    f"matching --slot-multiplier"
                )
            dev = table.embedding.weight.device
            stats = table.merge_rows(
                engram_config.merge_frac,
                metric=engram_config.merge_metric,
                seed=engram_config.seed,
                feature_override=feature.to(dev).float(),
                keep_mask=hit_mask.to(dev),
                writeback=engram_config.merge_writeback,
                cluster=engram_config.merge_cluster,
            )
            n_part = int(hit_mask.sum())
            print0(
                f"Engram SEMANTIC row merge @ step 0: "
                f"{stats['total_rows']:,} -> {stats['effective_rows']:,} effective "
                f"rows ({stats['effective_rows']/stats['total_rows']:.1%}); "
                f"{n_part:,} participating (hit>0, "
                f"{n_part/stats['total_rows']:.1%} coverage), metric="
                f"{engram_config.merge_metric}"
            )

# Loop state (variables updated by the training loop)
if not resuming:
    step = 0
    val_bpb = None  # will be set if eval_every > 0
    min_val_bpb = float("inf")
    smooth_train_loss = 0  # EMA of training loss
    total_training_time = 0  # total wall-clock time of training
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_bpb = meta_data["val_bpb"]
    min_val_bpb = loop_state["min_val_bpb"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]

# Figure out the needed gradient accumulation micro-steps to reach the desired total batch size per step
tokens_per_fwdbwd = (
    args.device_batch_size * args.max_seq_len
)  # tokens per iteration for a single rank
world_tokens_per_fwdbwd = (
    tokens_per_fwdbwd * ddp_world_size
)  # total tokens per iteration for all ranks
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(
    f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}"
)
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(
    f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}"
)

# Only the "pq" address source emits a commitment loss; skip the model walk otherwise.
engram_has_aux = (
    model_config.engram is not None and model_config.engram.address_source == "pq"
)

# Routers whose per-step .item() diagnostics (regret/MaxVio/entropy) we gate to
# log steps only: an every-step host sync serializes the GPU pipeline and blocks
# torch.compile. Gathered from orig_model, which shares module objects with the
# compiled wrapper.
moe_routers = (
    [m for m in orig_model.modules() if hasattr(m, "collect_stats")]
    if model_config.moe is not None
    else []
)

# Go!
while True:
    last_step = (
        step == num_iterations
    )  # loop runs num_iterations+1 times so that we can eval/save at the end
    flops_so_far = num_flops_per_token * total_batch_size * step

    # once in a while: evaluate the val bpb (all ranks participate)
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (
            args.device_batch_size * args.max_seq_len * ddp_world_size
        )
        with disable_fp8(model), autocast_ctx:
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.6f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log(
            {
                "step": step,
                "total_training_flops": flops_so_far,
                "total_training_time": total_training_time,
                "val/bpb": val_bpb,
            }
        )
        model.train()

    # once in a while: estimate the CORE metric (all ranks participate)
    # use the original uncompiled model because the inputs keep changing shape
    # disable FP8 for evaluation to use BF16 for more consistent/accurate results
    results = {}
    if args.core_metric_every > 0 and (
        last_step or (step > 0 and step % args.core_metric_every == 0)
    ):
        model.eval()
        with disable_fp8(orig_model), autocast_ctx:
            results = evaluate_core(
                orig_model,
                tokenizer,
                device,
                max_per_task=args.core_metric_max_per_task,
            )
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log(
            {
                "step": step,
                "total_training_flops": flops_so_far,
                "core_metric": results["core_metric"],
                "centered_results": results["centered_results"],
            }
        )
        model.train()

    # once in a while: sample from the model (only on master process)
    # use the original uncompiled model because the inputs keep changing shape
    if (
        args.sample_every > 0
        and master_process
        and (last_step or (step > 0 and step % args.sample_every == 0))
    ):
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        engine = Engine(orig_model, tokenizer)  # use orig_model to avoid recompilation
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with disable_fp8(orig_model), autocast_ctx:
                sample, _ = engine.generate_batch(
                    tokens, num_samples=1, max_tokens=16, temperature=0
                )
            print0(tokenizer.decode(sample[0]))
        model.train()

    # save checkpoint: at the end of the run, or every save_every steps, except at the first step or the resume step
    if last_step or (
        step > 0
        and step != args.resume_from_step
        and args.save_every > 0
        and step % args.save_every == 0
    ):
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(),  # model parameters
            optimizer.state_dict(),  # optimizer state
            {  # metadata saved as json
                "step": step,
                "val_bpb": val_bpb,  # loss at last step
                "model_config": model_config_kwargs,
                "user_config": user_config,  # inputs to the training script
                "device_batch_size": args.device_batch_size,
                "max_seq_len": args.max_seq_len,
                "total_batch_size": total_batch_size,
                "dataloader_state_dict": dataloader_state_dict,
                "loop_state": {  # all loop state (other than step) so that we can resume training
                    "min_val_bpb": min_val_bpb,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                },
            },
            rank=ddp_rank,
        )

    # termination conditions (TODO: possibly also add loss explosions etc.)
    if last_step:
        break

    # mid-training value-aware row merge (idea 1B): once, at merge_at_frac of
    # training, cluster each Engram table's rows by learned value and redirect
    # merged rows onto a survivor for the rest of the run. Fires AFTER the
    # checkpoint block so a save at the same step captures the pre-merge state;
    # operates on orig_model (shared tables deduped). One recompile may follow.
    if engram_merge_step > 0 and step == engram_merge_step:
        for table in orig_model._engram_memory_tables():
            if getattr(table, "_merged", None) is not None and not bool(table._merged):
                stats = table.merge_rows(
                    engram_config.merge_frac,
                    metric=engram_config.merge_metric,
                    seed=engram_config.seed,
                    writeback=engram_config.merge_writeback,
                    cluster=engram_config.merge_cluster,
                )
                print0(
                    f"Engram row merge @ step {step}: "
                    f"{stats['total_rows']:,} -> {stats['effective_rows']:,} "
                    f"effective rows ({stats['effective_rows']/stats['total_rows']:.1%}), "
                    f"metric={engram_config.merge_metric} "
                    f"writeback={engram_config.merge_writeback} "
                    f"cluster={engram_config.merge_cluster}"
                )

    # -------------------------------------------------------------------------
    # single training step
    # evaluate the gradient
    synchronize()
    t0 = time.time()
    aux_loss_f = 0.0  # last micro-step's balance loss, for logging
    engram_aux_f = 0.0  # last micro-step's Engram VQ commitment loss
    # Collect router diagnostics only on steps we log them; off-steps skip the
    # .item() syncs in router_stats(), keeping the forward clean for compile.
    log_moe_step = (
        bool(moe_routers) and args.moe_log_every > 0 and step % args.moe_log_every == 0
    )
    for r in moe_routers:
        r.collect_stats = log_moe_step
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        # Snapshot the pure LM loss BEFORE adding the balance loss, so the logged
        # train loss and bpb stay comparable with the dense arms.
        train_loss = loss.detach()  # for logging
        if moe_routers:
            aux = collect_aux_loss(orig_model)  # graph-connected scalar, or None
            if aux is not None:
                loss = loss + aux
                aux_loss_f = aux.detach().item()
        if engram_has_aux:
            # VQ commitment loss from the continuous-Engram discretizers.
            eaux = collect_engram_aux_loss(orig_model)
            if eaux is not None:
                loss = loss + eaux
                engram_aux_f = eaux.detach().item()
        loss = (
            loss / grad_accum_steps
        )  # each .backward() is a grad sum => normalize loss here
        loss.backward()
        x, y, dataloader_state_dict = next(
            train_loader
        )  # prefetch the next batch while the GPU is busy with forward/backward
    # step the optimizer
    lrm = get_lr_multiplier(step)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group["kind"] == "muon":
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    optimizer.step()
    model.zero_grad(set_to_none=True)
    train_loss_f = train_loss.item()  # .item() is a CPU-GPU sync point
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # logging (CPU action only)
    ema_beta = 0.9  # EMA decay factor for some smoothing just for nicer logging
    smooth_train_loss = (
        ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    )  # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (
        1 - ema_beta ** (step + 1)
    )  # debias the EMA
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    if step > 10:
        total_training_time += dt  # only count the time after the first 10 steps
    # Calculate ETA based on average time per step (excluding first 10 steps)
    steps_done = step - 10
    if steps_done > 0:
        avg_time_per_step = total_training_time / steps_done
        remaining_steps = num_iterations - step
        eta_seconds = remaining_steps * avg_time_per_step
        eta_str = f" | eta: {eta_seconds / 60:.1f}m"
    else:
        eta_str = ""
    epoch = f"{dataloader_state_dict['epoch']} pq: {dataloader_state_dict['pq_idx']} rg: {dataloader_state_dict['rg_idx']}"
    print0(
        f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | bf16_mfu: {mfu:.2f} | epoch: {epoch} | total time: {total_training_time / 60:.2f}m{eta_str}"
    )
    # MoE instrumentation: per-layer + model-mean regret/MaxVio/entropy, read off
    # the last micro-batch forward.
    if log_moe_step:
        moe_stats = collect_moe_stats(orig_model)
        if moe_stats:
            print0(
                f"  MoE | regret_mean: {moe_stats['moe/regret_mean']:.5f} | "
                f"max_vio_mean: {moe_stats['moe/max_vio_mean']:.4f} | "
                f"entropy_mean: {moe_stats['moe/entropy_mean']:.4f} | "
                f"aux_loss: {aux_loss_f:.5f}"
            )
            wandb_run.log({"step": step, **moe_stats, "moe/aux_loss": aux_loss_f})

    if engram_has_aux and step % 100 == 0:
        print0(f"  Engram | vq_commitment: {engram_aux_f:.5f}")
        wandb_run.log({"step": step, "engram/vq_commitment": engram_aux_f})

    if step % 100 == 0:
        log_data = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "train/epoch": epoch,
        }
        wandb_run.log(log_data)

    # state update
    first_step_of_run = (step == 0) or (resuming and step == args.resume_from_step)
    step += 1

    # The garbage collector is sadly a little bit overactive and for some poorly understood reason,
    # it spends ~500ms scanning for cycles quite frequently, just to end up cleaning up very few tiny objects each time.
    # So we manually manage and help it out here
    if first_step_of_run:
        gc.collect()  # manually collect a lot of garbage from setup
        gc.freeze()  # immediately freeze all currently surviving objects and exclude them from GC
        gc.disable()  # nuclear intervention here: disable GC entirely except:
    elif step % 5000 == 0:  # every 5000 steps...
        gc.collect()  # manually collect, just to be safe for very, very long runs

# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time / 60:.2f}m")
if val_bpb is not None:
    print0(f"Minimum validation bpb: {min_val_bpb:.6f}")

# Log to report
from nano_scalemb.report import get_report

get_report().log(
    section="Base model training",
    data=[
        user_config,  # CLI args
        {  # stats about the training setup
            "Number of parameters": num_params,
            "Number of FLOPs per token": f"{num_flops_per_token:e}",
            "Calculated number of iterations": num_iterations,
            "Number of training tokens": total_tokens,
            "Tokens : Scaling params ratio": total_batch_size
            * num_iterations
            / num_scaling_params,
            "DDP world size": ddp_world_size,
            "warmup_ratio": args.warmup_ratio,
            "warmdown_ratio": args.warmdown_ratio,
            "final_lr_frac": args.final_lr_frac,
        },
        {  # stats about training outcomes
            "Minimum validation bpb": min_val_bpb if val_bpb is not None else None,
            "Final validation bpb": val_bpb,
            "CORE metric estimate": results.get("core_metric", None),
            "MFU %": f"{mfu:.2f}%",
            "Total training flops": f"{flops_so_far:e}",
            "Total training time": f"{total_training_time / 60:.2f}m",
            "Peak memory usage": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
        },
    ],
)

# cleanup
wandb_run.finish()  # wandb run finish
compute_cleanup()
