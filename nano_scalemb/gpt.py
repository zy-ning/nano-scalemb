"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration
"""

from functools import partial
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nano_scalemb.common import get_dist_info, print0
from nano_scalemb.optim import MuonAdamW, DistMuonAdamW

# Our custom Flash Attention module that automatically uses FA3 on Hopper+ and SDPA fallback elsewhere
from nano_scalemb.flash_attention import flash_attn

from nano_scalemb.engram import (
    EngramConfig,
    Engram,
    CompressedTokenizerProjection,
    MultiHeadEmbedding,
    NgramHasher,
    build_memory_table,
    init_memory_table,
)
from nano_scalemb.mhc import (
    MHCConfig,
    MHCHead,
    ManifoldConstrainedHyperConnections,
    StreamExpand,
)
from nano_scalemb.moe.block import MoEBlock, MoEPool
from nano_scalemb.moe.config import MoEConfig
from nano_scalemb.moe.experts import Experts


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6  # number of query heads
    n_kv_head: int = 6  # number of key/value heads (GQA)
    n_embd: int = 768
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (half context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "SSSL"
    # Optional Engram configuration. None disables Engram entirely (default, backward compatible).
    # Recommended layer placement for 12-layer models: layer_ids=(2, 6) (early layers per paper).
    # Example: engram=EngramConfig(layer_ids=(2, 6))
    engram: Optional[EngramConfig] = None
    # Backbone mHC is the true persistent residual-stream backbone. When set
    # alongside Engram, the Engram composes as a per-stream branch (the faithful
    # Engram+mHC path). Without it, Engram falls back to a dense single stream.
    mhc: Optional[MHCConfig] = None
    # Optional MoE configuration. None keeps the dense MLP everywhere (default).
    # moe.share_blocks selects the arm: 0 = per-layer MoE, N > 0 = Mobius (N
    # routed pools shared across depth). See nano_scalemb/moe/.
    moe: Optional[MoEConfig] = None


def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]  # split up last dim into two halves
    y1 = x1 * cos + x2 * sin  # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = (
            nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
            if has_ve(layer_idx, config.n_layer)
            else None
        )

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            ve_gate = self.ve_gate
            assert ve_gate is not None
            gate = 2 * torch.sigmoid(
                ve_gate(x[..., : self.ve_gate_channels])
            )  # (B, T, n_kv_head), range (0, 2)
            ve = ve.to(v.dtype)
            v = v + gate.unsqueeze(-1) * ve

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)  # QK norm
        if k.dtype != q.dtype:
            k = k.to(q.dtype)
        if v.dtype != q.dtype:
            v = v.to(q.dtype)

        # Flash Attention (FA3 on Hopper+, PyTorch SDPA fallback elsewhere)
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        if kv_cache is None:
            # Training: causal attention with optional sliding window
            y = flash_attn.flash_attn_func(
                q, k, v, causal=True, window_size=window_size
            )
        else:
            # Inference: use flash_attn_with_kvcache which handles cache management
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q,
                k_cache,
                v_cache,
                k=k,
                v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            # Advance position after last layer processes
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        # Re-assemble the heads and project back to residual stream
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx, engram=None, moe_pool=None):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        # MoE layers swap the dense MLP for an MoEBlock. MoEBlock.forward has the
        # same (x) -> y contract as MLP.forward, so nothing below changes: both
        # the dense path and the mHC lambda call self.mlp identically.
        self.is_moe = moe_pool is not None
        self.mlp = (
            MoEBlock(config.n_embd, config.moe, moe_pool) if self.is_moe else MLP(config)
        )
        # Optional Engram module (paper Figure 1, p.3: Engram fires before attention)
        self.engram: Optional[Engram] = engram
        self.mhc = config.mhc
        if config.mhc is not None:
            self.mhc_attn = ManifoldConstrainedHyperConnections(
                config.mhc.num_streams,
                dim=config.n_embd,
                layer_index=layer_idx,
                sinkhorn_iters=config.mhc.sinkhorn_iters,
            )
            self.mhc_mlp = ManifoldConstrainedHyperConnections(
                config.mhc.num_streams,
                dim=config.n_embd,
                layer_index=layer_idx,
                sinkhorn_iters=config.mhc.sinkhorn_iters,
            )
            if self.engram is not None:
                self.mhc_engram = ManifoldConstrainedHyperConnections(
                    config.mhc.num_streams,
                    dim=config.n_embd,
                    layer_index=layer_idx,
                    sinkhorn_iters=config.mhc.sinkhorn_iters,
                )

    def forward(
        self,
        x,
        ve,
        cos_sin,
        window_size,
        kv_cache,
        input_ids=None,
        compressed_input_ids=None,
    ):
        if self.mhc is None:
            if ve is not None:
                assert ve.size(0) == x.size(0), (
                    "ve batch size must match hidden batch"
                )
            if input_ids is not None:
                assert input_ids.size(0) == x.size(0), (
                    "input_ids batch size must match hidden batch"
                )
            if compressed_input_ids is not None:
                assert compressed_input_ids.size(0) == x.size(0), (
                    "compressed_input_ids batch size must match hidden batch"
                )
            # Engram fires before attention when present (paper Figure 1, p.3)
            if self.engram is not None and input_ids is not None:
                engram_input_ids = input_ids
                engram_compressed_ids = compressed_input_ids
                x = x + self.engram(
                    x,
                    engram_input_ids,
                    compressed_input_ids=engram_compressed_ids,
                ).to(x.dtype)
            x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
            x = x + self.mlp(norm(x))
            return x

        streams = self.mhc.num_streams
        assert x.size(0) % streams == 0, "mHC hidden batch must divide by num_streams"
        batch = x.size(0) // streams
        if ve is not None:
            assert ve.size(0) == batch, "ve batch size must match mHC base batch"
        if input_ids is not None:
            assert input_ids.size(0) == batch, (
                "input_ids batch size must match mHC base batch"
            )
        if compressed_input_ids is not None:
            assert compressed_input_ids.size(0) == batch, (
                "compressed_input_ids batch size must match mHC base batch"
            )

        if self.engram is not None and input_ids is not None:
            # Faithful Engram+mHC: the Engram emits a per-stream contribution and
            # the backbone mHC depth connection routes it into the residual streams.
            engram = self.engram
            mhc_engram = self.mhc_engram
            engram_input_ids = input_ids
            engram_compressed_ids = compressed_input_ids

            branch_input, residuals, beta = mhc_engram.width_connection(x)
            branch_outputs = engram.forward_mhc_branches(
                branch_input,
                engram_input_ids,
                compressed_input_ids=engram_compressed_ids,
            ).to(branch_input.dtype)
            x = mhc_engram.depth_connection_streams(
                branch_outputs, residuals, beta=beta
            )

        x = self.mhc_attn(
            x,
            lambda branch_input: self.attn(
                norm(branch_input), ve, cos_sin, window_size, kv_cache
            ),
        )
        x = self.mhc_mlp(x, lambda branch_input: self.mlp(norm(branch_input)))
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        if config.mhc is not None and config.engram is not None:
            assert config.engram.mhc_num_streams == config.mhc.num_streams, (
                "Backbone mHC with Engram requires matching stream counts"
            )
        self.stream_expand = None
        self.mhc_head = None
        if config.mhc is not None:
            # Backbone mHC over persistent residual streams; the Engram composes
            # with this as a per-stream branch (the faithful Engram+mHC path).
            self.stream_expand = StreamExpand(config.mhc.num_streams)
            self.mhc_head = MHCHead(
                config.mhc.num_streams,
                dim=config.n_embd,
            )
        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        self.window_sizes = self._compute_window_sizes(config)
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = (
            (config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to
        ) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(
                f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency"
            )
        # Build Engram modules if configured.
        # Recommended layer placement for 12-layer models: layer_ids=(2, 6) (early layers per paper).
        # Example: GPTConfig(engram=EngramConfig(layer_ids=(2, 6)))
        engram_modules: dict = {}
        compression = None
        self.engram_shared_memory: Optional[MultiHeadEmbedding] = None
        self.engram_shared_discretizer = None
        if config.engram is not None:
            with torch.device("cpu"):
                if config.engram.use_tokenizer_compression:
                    from nano_scalemb.tokenizer import get_tokenizer

                    _tokenizer = get_tokenizer()
                    compression = CompressedTokenizerProjection(
                        _tokenizer, pad_id=config.engram.pad_id
                    )
                self._ngram_hasher = NgramHasher(
                    config.engram, config.vocab_size, compression
                )
                # Shared-memory ablation: one table for every Engram layer,
                # owned here (registered exactly once) and handed to the Engrams
                # by reference. Every layer also hashes with the canonical
                # layer's primes, so the same n-gram hits the same row at every
                # depth. This is the Engram analogue of the Mobius shared pool.
                if config.engram.share_memory and config.engram.layer_ids:
                    self.engram_shared_memory = self._build_shared_engram_memory(
                        config, self._ngram_hasher
                    )
                    # Share the discretizer too, when addressing is contextual.
                    # Otherwise every layer gets its own seeded projection and
                    # the same hidden state lands on unrelated rows of the
                    # SHARED table at different depths -- a bigger table, not a
                    # shared memory (see EngramConfig.share_memory).
                    if config.engram.address_source in ("lsh", "pq", "hybrid"):
                        from nano_scalemb.discretize import build_discretizer

                        self.engram_shared_discretizer = build_discretizer(
                            config.engram,
                            config.n_embd,
                            min(config.engram.layer_ids),
                            num_hash_heads=self._ngram_hasher.num_hash_heads,
                        )
            for layer_idx in config.engram.layer_ids:
                engram_modules[str(layer_idx)] = Engram(
                    cfg=config.engram,
                    layer_id=layer_idx,
                    d_model=config.n_embd,
                    tokenizer_vocab_size=config.vocab_size,
                    compression=compression,
                    backbone_mhc=config.mhc is not None,
                    shared_embedding=self.engram_shared_memory,
                    shared_discretizer=self.engram_shared_discretizer,
                )
        # Register Engram modules so they are tracked by PyTorch
        self.engram_modules = nn.ModuleDict(engram_modules)
        self._engram_compression = compression

        # Build the routed MoE pools. Per-layer MoE gets one pool per MoE layer;
        # Mobius (share_blocks > 0) gets a smaller pool list that MoE layers index
        # into round-robin, so router weights, expert weights and balancer state
        # are shared across depth. The pools are registered here exactly once and
        # handed to Blocks by reference (see MoEBlock's docstring for why).
        self.moe_pools = nn.ModuleList()
        self._moe_pool_for_layer: dict = {}
        if config.moe is not None:
            num_pools = config.moe.num_pools(config.n_layer)
            self.moe_pools = nn.ModuleList(
                [MoEPool(config.n_embd, config.moe) for _ in range(num_pools)]
            )
            for layer_idx in config.moe.moe_layer_ids(config.n_layer):
                pool_idx = config.moe.pool_index(layer_idx, config.n_layer)
                self._moe_pool_for_layer[layer_idx] = self.moe_pools[pool_idx]

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(padded_vocab_size, config.n_embd),
                "h": nn.ModuleList(
                    [
                        Block(
                            config,
                            layer_idx,
                            engram=(
                                self.engram_modules[str(layer_idx)]
                                if str(layer_idx) in self.engram_modules
                                else None
                            ),
                            moe_pool=self._moe_pool_for_layer.get(layer_idx),
                        )
                        for layer_idx in range(config.n_layer)
                    ]
                ),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, padded_vocab_size, bias=False)
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        # Separate parameters so they can have different optimizer treatment
        self.resid_lambdas = nn.Parameter(
            torch.ones(config.n_layer)
        )  # fake init, real init in init_weights()
        self.x0_lambdas = nn.Parameter(
            torch.zeros(config.n_layer)
        )  # fake init, real init in init_weights()
        # Value embeddings (ResFormer-style): alternating layers, last layer always included
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict(
            {
                str(i): nn.Embedding(padded_vocab_size, kv_dim)
                for i in range(config.n_layer)
                if has_ve(i, config.n_layer)
            }
        )
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = (
            config.sequence_len * 10
        )  # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer(
            "cos", cos, persistent=False
        )  # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    @staticmethod
    def _build_shared_engram_memory(config, hasher):
        """One memory table for every Engram layer (the share_memory ablation).

        Sized from the canonical layer's primes, which under share_memory are the
        primes every layer hashes with.
        """
        engram_cfg = config.engram
        assert engram_cfg.address_source != "pkm", (
            "share_memory with address_source='pkm' is not supported: the "
            "product-key memory owns its own keys and values, so there is no "
            "MultiHeadEmbedding table to share."
        )
        num_heads = hasher.num_hash_heads
        head_dim = engram_cfg.memory_dim // num_heads
        canonical = hasher.resolve_layer_id(min(engram_cfg.layer_ids))
        head_vocab_sizes = [
            hasher.prime_table[canonical][ngram_idx][head_idx]
            # len(ngram_orders), not max_ngram_size - 1: at max_ngram_size=1 the
            # hasher keeps one unigram order, so the arithmetic version would
            # build a table with no heads at all.
            for ngram_idx in range(len(hasher.ngram_orders))
            for head_idx in range(engram_cfg.n_head_per_ngram)
        ]
        return build_memory_table(
            engram_cfg, head_vocab_sizes, head_dim, config.n_embd,
            vocab_size=hasher.vocab_size,
        )

    def _engram_memory_tables(self):
        """Every distinct memory table in the model, deduplicated.

        With share_memory a single table is read by several Engram layers, so
        callers that iterate tables (init, optimizer groups, param counting) must
        see it once, not once per layer.
        """
        tables = {}
        if getattr(self, "engram_shared_memory", None) is not None:
            tables[id(self.engram_shared_memory)] = self.engram_shared_memory
        if hasattr(self, "engram_modules"):
            for engram in self.engram_modules.values():
                table = engram.memory_table
                if table is not None:
                    tables[id(table)] = table
        return list(tables.values())

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """

        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = (
            3**0.5 * n_embd**-0.5
        )  # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        for block in self.transformer.h:
            torch.nn.init.uniform_(
                block.attn.c_q.weight, -s, s
            )  # weights use Uniform to avoid outliers
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)  # projections are zero
            if block.is_moe:
                # MoEBlock inits its per-layer parts only; the routed pool is
                # shared under Mobius so it is initialized once, below.
                block.mlp.reset_parameters()
            else:
                torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
                torch.nn.init.zeros_(block.mlp.c_proj.weight)
            if block.engram is not None:
                block.engram.init_weights()
            if self.config.mhc is not None:
                block.mhc_attn.init_weights()
                block.mhc_mlp.init_weights()
                if hasattr(block, "mhc_engram"):
                    block.mhc_engram.init_weights()
        if self.mhc_head is not None:
            self.mhc_head.init_weights()

        # Shared Engram memory: owned here, so initialized here exactly once
        # (Engram.init_weights skips a table it does not own).
        if self.engram_shared_memory is not None:
            init_memory_table(self.engram_shared_memory, self.config.engram.ablation_mode)
        if self.engram_shared_discretizer is not None:
            # Owned here, so initialized here exactly once (its buffers come back
            # as garbage from to_empty like every other buffer).
            self.engram_shared_discretizer.reset_parameters()

        # Routed MoE pools: initialized once each, since Mobius shares them.
        # This also zeroes the balancer buffers, which come back uninitialized
        # from to_empty().
        for pool in self.moe_pools:
            pool.reset_parameters()

        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)  # 1.0 => typical residual connections at init
        self.x0_lambdas.fill_(
            0.1
        )  # 0.1 => small initial weight for skip connection to input embedding

        # Value embeddings (init like c_v: uniform with same std)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # Gate weights init to zero so gates start at sigmoid(0) = 0.5, scaled by 2 -> 1.0 (neutral)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to bf16: optimizer can tolerate it and it saves memory
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)
            for ve in self.value_embeds.values():
                ve.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()  # keep them in bfloat16
        cos, sin = (
            cos[None, :, None, :],
            sin[None, :, None, :],
        )  # add batch and head dims for later broadcasting
        return cos, sin

    def _compute_window_sizes(self, config):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (half context)
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), (
            f"Invalid window_pattern: {pattern}. Use only S and L."
        )
        # Map characters to window sizes
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def _engram_embedding_params(self):
        return [p for p in self._all_engram_embedding_params() if p.requires_grad]

    def _all_engram_embedding_params(self):
        # Deduplicated: with EngramConfig.share_memory one table serves every
        # Engram layer and must appear in the optimizer/param counts once.
        #
        # content_parameters(), not all of them: a LowRankMemory also owns a dense
        # query projection, which belongs in the Muon group and in the generic
        # FLOP term like any other matrix, not with the addressed rows.
        return [
            p
            for table in self._engram_memory_tables()
            for p in table.content_parameters()
        ]

    def _expert_params(self):
        """The 3D routed-expert weight tensors (w_in / w_out).

        Muon is 2D-only, so these need their own AdamW group at expert_lr rather
        than falling into the generic >2D bucket at matrix_lr.
        """
        return [
            p
            for mod in self.modules()
            if isinstance(mod, Experts)
            for p in mod.parameters()
        ]

    def _split_transformer_block_params(self):
        engram_embed_params = self._engram_embedding_params()
        all_engram_embed_ids = {id(p) for p in self._all_engram_embedding_params()}
        expert_params = self._expert_params()
        expert_param_ids = {id(p) for p in expert_params}
        muon_params = []
        nonmatrix_decay_params = []
        nonmatrix_nodecay_params = []
        mhc_head_params = []
        if self.mhc_head is not None:
            mhc_head_params = list(self.mhc_head.parameters())
        # The routed MoE pools live on the model (not inside transformer.h) so
        # Mobius can share them across layers; scan them here alongside the
        # blocks, the same way mhc_head params are folded in.
        moe_pool_params = list(self.moe_pools.parameters())
        # Same for a shared Engram memory table: with share_memory it hangs off
        # the GPT, so a LowRankMemory's query projection would otherwise reach no
        # optimizer group at all and trip the param-count assert below.
        engram_table_params = [
            p for table in self._engram_memory_tables() for p in table.parameters()
        ]
        seen: set = set()
        for p in (
            list(self.transformer.h.parameters())
            + mhc_head_params
            + moe_pool_params
            + engram_table_params
        ):
            if id(p) in all_engram_embed_ids or id(p) in expert_param_ids:
                continue
            # Per-layer tables are reachable both ways; count each param once.
            if id(p) in seen:
                continue
            seen.add(id(p))
            if p.ndim == 2:
                muon_params.append(p)
            elif p.ndim > 2:
                nonmatrix_decay_params.append(p)
            else:
                nonmatrix_nodecay_params.append(p)
        return (
            muon_params,
            nonmatrix_decay_params,
            nonmatrix_nodecay_params,
            engram_embed_params,
            expert_params,
        )

    def _moe_layer_pools(self):
        """The routed pool each MoE layer reads, in layer order.

        With Mobius the same pool object appears several times: the weights are
        stored once but *read* once per layer, and FLOP/active-param accounting
        has to follow the reads, not the storage.
        """
        return [block.mlp.pool for block in self.transformer.h if block.is_moe]

    def _moe_flops_per_token(self):
        """Active MoE FLOPs per token, and the pool params to exclude from the
        generic param-based FLOP estimate.

        The generic ``6 * nparams`` term is wrong for MoE twice over: it counts
        all n_experts when only top_k fire per token (over-count), and with a
        shared Mobius pool it counts the weights once when several layers read
        them (under-count). So we exclude the pools entirely and add the real
        per-layer active cost back here.
        """
        pools = self._moe_layer_pools()
        if not pools:
            return 0, 0
        d_model = self.config.n_embd
        flops = 0
        for pool in pools:
            experts = pool.experts
            # router: d_model x n_experts matmul, every token
            flops += 6 * d_model * experts.n_experts
            # experts: top_k of them fire, each is w_in (d_ff x d_model) + w_out
            flops += 6 * experts.top_k * 2 * d_model * experts.d_ff
        # Deduplicate for the exclusion: shared pools are stored once.
        pool_params_numel = sum(
            p.numel() for p in {id(p): p for p in self.moe_pools.parameters()}.values()
        )
        return flops, pool_params_numel

    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
        With sliding windows, effective_seq_len varies per layer (capped by window size).
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        """
        nparams = sum(p.numel() for p in self.parameters())
        # Exclude non-matmul params: embeddings and per-layer scalars
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        engram_embeds_numel = sum(p.numel() for p in self._all_engram_embedding_params())
        # MoE pools are excluded from the generic term and accounted for exactly
        # (active experts only, once per reading layer) in moe_flops below.
        moe_flops, moe_pool_numel = self._moe_flops_per_token()
        # Product-key memory: the value bank is addressed (topk of n_keys**2 rows
        # per head), so the generic 6*nparams term would charge for the whole
        # bank. Exclude it and add the real cost back: scoring 2*n_keys sub-keys
        # plus reading topk value rows, per head per layer.
        pk_values_numel = 0
        pk_flops = 0
        # A constant-row read is a pure gather and costs nothing here. With
        # value_rank > 0 the gathered rows ARE multiplied (q through w_in, then
        # rank-space through w_out), so the excluded rows owe FLOPs back -- once
        # per *reading* layer, since a shared table is read at every depth.
        engram_read_flops = 0
        for engram in getattr(self, "engram_modules", {}).values():
            table = engram.memory_table
            if table is not None:
                engram_read_flops += table.read_flops_per_token()
            pk = engram.product_key
            if pk is not None:
                pk_values_numel += pk.values.numel()
                pk_flops += 6 * pk.num_heads * (
                    2 * pk.n_keys * pk.half_dim  # sub-key scoring
                    + pk.topk * pk.value_dim  # weighted value read
                )
        nparams_exclude = (
            self.transformer.wte.weight.numel()
            + value_embeds_numel
            + engram_embeds_numel
            + moe_pool_numel
            + pk_values_numel
            + self.resid_lambdas.numel()
            + self.x0_lambdas.numel()
        )
        h, q, t = (
            self.config.n_head,
            self.config.n_embd // self.config.n_head,
            self.config.sequence_len,
        )
        # Sum attention FLOPs per layer, accounting for sliding window
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]  # (left, right) tuple, we use left
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = (
            6 * (nparams - nparams_exclude)
            + attn_flops
            + moe_flops
            + pk_flops
            + engram_read_flops
        )
        return num_flops_per_token

    def num_scaling_params(self):
        """
        Return detailed parameter counts for scaling law analysis.
        Different papers use different conventions:
        - Kaplan et al. excluded embedding parameters
        - Chinchilla included all parameters
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper)

        Returns a dict with counts for each parameter group, so downstream analysis
        can experiment with which combination gives the cleanest scaling laws.

        For MoE runs, `expert_total` is the stored expert weight and
        `expert_active` is the share a single token actually reads, counted once
        per reading layer (so a shared Mobius pool read by L layers contributes L
        times to `expert_active` but once to `expert_total`). Compare MoE and
        dense arms on active, not total.
        """
        # Count each group separately (mirrors the grouping in setup_optimizers)
        (
            matrix_params,
            block_decay_params,
            block_nodecay_params,
            engram_embed_params,
            expert_params,
        ) = self._split_transformer_block_params()
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        engram_embeds = sum(p.numel() for p in engram_embed_params)
        engram_frozen_embeds = sum(
            p.numel()
            for p in self._all_engram_embedding_params()
            if not p.requires_grad
        )
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in matrix_params)
        transformer_nonmatmul = sum(p.numel() for p in block_decay_params) + sum(
            p.numel() for p in block_nodecay_params
        )
        # Engram memory is addressed, not dense: a token reads exactly
        # num_hash_heads rows of head_dim each, per Engram layer, however large
        # the table is. Counting the whole table as "active" would make Engram
        # look ~5 orders of magnitude more active than it is and wreck any
        # iso-active comparison against MoE or dense.
        engram_active = 0
        engram_stored = 0
        for engram in getattr(self, "engram_modules", {}).values():
            table = engram.memory_table
            if table is not None:
                # With value_rank > 0 a read pulls rank*(query_dim + head_dim)
                # weights per head instead of head_dim, so the table asks the
                # memory rather than assuming constant rows.
                engram_active += table.active_params_per_token()
            pk = engram.product_key
            if pk is not None:
                # Product-key memory is addressed too: a token reads topk of the
                # n_keys**2 value rows per head. The keys themselves ARE read
                # densely (every sub-key is scored), so they count as active in
                # full -- unlike the values.
                engram_stored += pk.values.numel()
                engram_active += pk.num_heads * pk.topk * pk.value_dim
        expert_total = sum(p.numel() for p in expert_params)
        # Active expert weight: top_k/n_experts of the bank, per reading layer.
        expert_active = 0
        for pool in self._moe_layer_pools():
            experts = pool.experts
            bank = experts.w_in.numel() + experts.w_out.numel()
            expert_active += round(bank * experts.top_k / experts.n_experts)
        # Router gates are dense but, like the experts, a shared Mobius pool's
        # gate is stored once and read once per layer. Tracked separately so
        # active accounting can count reads while `total` counts storage.
        router_total = sum(
            p.numel()
            for pool in {id(p): p for p in self.moe_pools}.values()
            for p in pool.router.parameters()
        )
        router_active = sum(
            sum(p.numel() for p in pool.router.parameters())
            for pool in self._moe_layer_pools()
        )
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = (
            wte
            + value_embeds
            + engram_embeds
            + engram_frozen_embeds
            + lm_head
            + transformer_matrices
            + transformer_nonmatmul
            + expert_total
            + scalars
        )
        assert total == sum(p.numel() for p in self.parameters()), (
            "Parameter count mismatch"
        )
        return {
            "wte": wte,
            "value_embeds": value_embeds,
            "engram_embeds": engram_embeds,
            "engram_frozen_embeds": engram_frozen_embeds,
            "engram_active": engram_active,
            "engram_pk_values": engram_stored,
            "lm_head": lm_head,
            "transformer_matrices": transformer_matrices,
            "transformer_nonmatmul": transformer_nonmatmul,
            "expert_total": expert_total,
            "expert_active": expert_active,
            "router_total": router_total,
            "router_active": router_active,
            "scalars": scalars,
            "total": total,
        }

    def setup_optimizer(
        self,
        unembedding_lr=0.004,
        embedding_lr=0.2,
        matrix_lr=0.02,
        weight_decay=0.0,
        adam_betas=(0.8, 0.95),
        scalar_lr=0.5,
    ):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Separate out all parameters into groups
        (
            matrix_params,
            block_decay_params,
            block_nodecay_params,
            engram_embed_params,
            expert_params,
        ) = self._split_transformer_block_params()
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        assert len(trainable_params) == len(matrix_params) + len(
            embedding_params
        ) + len(lm_head_params) + len(value_embeds_params) + len(
            block_decay_params
        ) + len(block_nodecay_params) + len(resid_params) + len(x0_params) + len(
            engram_embed_params
        ) + len(expert_params)

        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(
            f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}"
        )

        # Build param_groups with all required fields explicit
        param_groups = [
            # AdamW groups (embeddings, lm_head, scalars)
            dict(
                kind="adamw",
                params=lm_head_params,
                lr=unembedding_lr * dmodel_lr_scale,
                betas=adam_betas,
                eps=1e-10,
                weight_decay=0.0,
            ),
            dict(
                kind="adamw",
                params=embedding_params,
                lr=embedding_lr * dmodel_lr_scale,
                betas=adam_betas,
                eps=1e-10,
                weight_decay=0.0,
            ),
            dict(
                kind="adamw",
                params=value_embeds_params,
                lr=embedding_lr * dmodel_lr_scale,
                betas=adam_betas,
                eps=1e-10,
                weight_decay=0.0,
            ),
            dict(
                kind="adamw",
                params=block_decay_params,
                lr=matrix_lr,
                betas=adam_betas,
                eps=1e-10,
                weight_decay=weight_decay,
            ),
            dict(
                kind="adamw",
                params=block_nodecay_params,
                lr=matrix_lr,
                betas=adam_betas,
                eps=1e-10,
                weight_decay=0.0,
            ),
            dict(
                kind="adamw",
                params=resid_params,
                lr=scalar_lr * 0.01,
                betas=adam_betas,
                eps=1e-10,
                weight_decay=0.0,
            ),
            dict(
                kind="adamw",
                params=x0_params,
                lr=scalar_lr,
                betas=(0.96, 0.95),
                eps=1e-10,
                weight_decay=0.0,
            ),  # higher beta1 for x0
        ]
        # Routed expert weights are 3D, and Muon is 2D-only, so they get their own
        # AdamW group at the (typically much lower) expert_lr.
        if expert_params:
            param_groups.append(
                dict(
                    kind="adamw",
                    params=expert_params,
                    lr=self.config.moe.expert_lr * dmodel_lr_scale,
                    betas=adam_betas,
                    eps=1e-10,
                    weight_decay=weight_decay,
                )
            )
        # Add Engram embeddings group if engram is enabled
        if engram_embed_params:
            param_groups.insert(
                4,
                dict(
                    kind="adamw",
                    params=engram_embed_params,
                    lr=embedding_lr
                    * self.config.engram.embedding_lr_mult
                    * dmodel_lr_scale,
                    betas=adam_betas,
                    eps=1e-10,
                    weight_decay=0.0,
                ),
            )
        # Muon groups (matrix params, grouped by shape for stacking)
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(
                dict(
                    kind="muon",
                    params=group_params,
                    lr=matrix_lr,
                    momentum=0.95,
                    ns_steps=5,
                    beta2=0.95,
                    weight_decay=weight_decay,
                )
            )

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(
        self,
        idx,
        targets=None,
        kv_cache=None,
        loss_reduction="mean",
        engram_input_ids=None,
    ):
        _, T = idx.size()
        if engram_input_ids is None:
            engram_input_ids = idx
        assert engram_input_ids.size(0) == idx.size(0), (
            "engram_input_ids must match idx batch size"
        )
        assert engram_input_ids.size(1) >= T, (
            "engram_input_ids length must be >= idx length"
        )
        compressed_engram_input_ids = None
        if self._engram_compression is not None:
            compressed_engram_input_ids = self._engram_compression.compress(
                engram_input_ids
            )

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), (
            f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        )
        assert idx.device == self.cos.device, (
            f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        )
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = (
            self.cos[:, T0 : T0 + T],
            self.sin[:, T0 : T0 + T],
        )  # truncate cache to current sequence length

        # Forward the trunk of the Transformer
        x = self.transformer.wte(idx)  # embed current token
        x = norm(x)
        x0 = x  # save initial normalized embedding for x0 residual
        if self.stream_expand is not None:
            x = self.stream_expand(x)
            x0 = self.stream_expand(x0)
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            # Pass input_ids to blocks that have Engram; None for non-Engram layers (no-op)
            block_input_ids = engram_input_ids if block.engram is not None else None
            block_compressed_ids = (
                compressed_engram_input_ids if block.engram is not None else None
            )
            x = block(
                x,
                ve,
                cos_sin,
                self.window_sizes[i],
                kv_cache,
                input_ids=block_input_ids,
                compressed_input_ids=block_compressed_ids,
            )
        x = norm(x)
        if self.mhc_head is not None:
            x = self.mhc_head(x)

        # Forward the lm_head (compute logits)
        softcap = 20  # smoothly cap the logits to the range [-softcap, softcap]
        logits = self.lm_head(
            x
        )  # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
        logits = logits[..., : self.config.vocab_size]  # slice to remove padding
        logits = logits.float()  # switch to fp32 for logit softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap)  # squash the logits

        if targets is not None:
            # training: given the targets, compute and return the loss
            # TODO experiment with chunked cross-entropy?
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=loss_reduction,
            )
            return loss
        else:
            # inference: just return the logits directly
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)  # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids)  # (B, T, vocab_size)
            logits = logits[:, -1, :]  # (B, vocab_size)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
