"""
Engram core components for nano-engram:
  - CompressedTokenizerProjection: builds P: V -> V' lookup table
  - _is_prime / _next_prime: pure-Python primality utilities
  - NgramHasher: multi-head multiplicative-XOR hashing, on-device
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import normalizers, Regex

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class EngramConfig:
    """Configuration for the Engram module.

    The Engram has two execution paths, selected by whether the backbone runs
    with true mHC (multi-stream residual):

      - backbone mHC ON  -> faithful Engram+mHC: the Engram emits a per-stream
        contribution (forward_mhc_branches) that the backbone's
        ManifoldConstrainedHyperConnections (mhc_engram) routes. This is the
        paper-faithful reproduction.
      - backbone mHC OFF -> dense fallback: a single-stream gated contribution
        (Engram.forward).

    mhc_num_streams must match the backbone num_streams when backbone mHC is on.

    Note: an earlier local pseudo-mHC fusion (the Engram mixing its own streams
    via alpha/beta/branch_scales/post_fuse) was a non-faithful reproduction and
    has been removed; old checkpoints' params for it are stripped on load.
    """

    layer_ids: Tuple[int, ...] = ()
    max_ngram_size: int = 3
    n_head_per_ngram: int = 8
    memory_dim: int = 1280
    slot_multiplier: int = 18  # base table size ≈ V' * slot_multiplier
    kernel_size: int = 4
    use_tokenizer_compression: bool = True
    embedding_lr_mult: float = 5.0
    pad_id: int = 0
    seed: int = 0
    ablation_mode: str = "none"
    mhc_num_streams: int = 4


# ---------------------------------------------------------------------------
# Tokenizer compression
# ---------------------------------------------------------------------------


def _build_normalizer() -> normalizers.Normalizer:
    """NFKC -> NFD -> StripAccents -> Lowercase -> collapse whitespace -> strip."""
    SENTINEL = "\ue000"
    return normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), SENTINEL),
            normalizers.Strip(),
            normalizers.Replace(SENTINEL, " "),
        ]
    )


class CompressedTokenizerProjection:
    """
    Precomputes a surjective projection P: V -> V' that collapses tokens
    with the same normalized textual form to the same canonical ID.

    Attributes
    ----------
    lookup_table : torch.Tensor, shape [V], dtype int64
        raw_id -> canonical_id
    compressed_vocab_size : int
        |V'|, i.e. number of distinct canonical IDs
    compressed_pad_id : int
        compressed ID corresponding to the original pad_id
    """

    def __init__(self, tokenizer, pad_id: int = 0):
        """
        Parameters
        ----------
        tokenizer : RustBPETokenizer or HuggingFaceTokenizer
            Must expose get_vocab_size() and a decode([id]) -> str method.
        pad_id : int
            Token ID used for left-padding in N-gram shifts.
        """
        self._normalizer = _build_normalizer()
        vocab_size = tokenizer.get_vocab_size()

        old2new: Dict[int, int] = {}
        key2new: Dict[str, int] = {}
        new_tokens: List[str] = []

        for tid in range(vocab_size):
            text = tokenizer.decode([tid])

            # replacement character -> use raw token string so it gets its own ID
            if "\ufffd" in text:
                key = tokenizer.id_to_token(tid) or text
            else:
                norm = self._normalizer.normalize_str(text)
                key = norm if norm else text

            nid = key2new.get(key)
            if nid is None:
                nid = len(new_tokens)
                key2new[key] = nid
                new_tokens.append(key)
            old2new[tid] = nid

        # store as a torch int64 tensor for O(1) device-side indexing later
        table = torch.empty(vocab_size, dtype=torch.int64, device="cpu")
        for tid in range(vocab_size):
            table[tid] = old2new[tid]

        self.lookup_table: torch.Tensor = table  # [V]
        self.compressed_vocab_size: int = len(new_tokens)
        self.compressed_pad_id: int = old2new[pad_id]
        self._lookup_table_by_device: Dict[str, torch.Tensor] = {}

    def _lookup_table_for_device(self, device: torch.device) -> torch.Tensor:
        key = str(device)
        table = self._lookup_table_by_device.get(key)
        if table is None:
            table = self.lookup_table.to(device)
            self._lookup_table_by_device[key] = table
        return table

    def compress(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Map raw token IDs -> compressed IDs, purely on device.

        Parameters
        ----------
        input_ids : torch.Tensor, int64, any shape

        Returns
        -------
        torch.Tensor, same shape and device, int64
        """
        table = self._lookup_table_for_device(input_ids.device)
        return table[input_ids]


# ---------------------------------------------------------------------------
# Primality helpers (pure Python, no sympy)
# ---------------------------------------------------------------------------


def _is_prime(n: int) -> bool:
    """Return True if n is a prime number (pure Python)."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def _next_prime(start: int, seen: set) -> int:
    """Return the smallest prime > start that is not in *seen*."""
    candidate = start + 1
    while True:
        if _is_prime(candidate) and candidate not in seen:
            return candidate
        candidate += 1


# ---------------------------------------------------------------------------
# N-gram hasher
# ---------------------------------------------------------------------------


class NgramHasher:
    """
    Multi-head multiplicative-XOR N-gram hasher.

    For each layer in cfg.layer_ids and for each N-gram order n in
    [2, cfg.max_ngram_size] and for each of cfg.n_head_per_ngram heads,
    a globally unique prime p_{layer,n,head} is pre-allocated.

    The hash of a compressed token sequence is:

        mix = x_t * m_0 XOR x_{t-1} * m_1 XOR ... XOR x_{t-n+1} * m_{n-1}
        hash_{head} = mix mod p_{layer,n,head}

    All arithmetic is done on the same device as input_ids.

    Parameters
    ----------
    cfg : EngramConfig
    tokenizer_vocab_size : int
        The *raw* (pre-compression) vocabulary size V.
    compression : CompressedTokenizerProjection | None
        If cfg.use_tokenizer_compression is True, must be provided.
    """

    def __init__(
        self,
        cfg: EngramConfig,
        tokenizer_vocab_size: int,
        compression: "CompressedTokenizerProjection | None" = None,
    ):
        self.cfg = cfg
        self.compression = compression
        self.use_compression = cfg.use_tokenizer_compression and compression is not None

        if self.use_compression:
            assert compression is not None
            V_compressed = compression.compressed_vocab_size
            self.pad_id = compression.compressed_pad_id
        else:
            V_compressed = tokenizer_vocab_size
            self.pad_id = cfg.pad_id

        # ------------------------------------------------------------------
        # 1. Compute globally unique prime table sizes
        #    Iterate layer -> ngram order -> head to allocate primes in a
        #    deterministic order so cross-layer collisions are impossible.
        # ------------------------------------------------------------------
        seen_primes: set = set()
        # prime_table[layer_id][ngram_idx][head_idx] = prime
        self.prime_table: Dict[int, List[List[int]]] = {}

        for layer_id in sorted(cfg.layer_ids):
            layer_primes: List[List[int]] = []
            for ngram_order in range(2, cfg.max_ngram_size + 1):
                head_primes: List[int] = []
                # base size for this (layer, ngram order) slot
                base_size = V_compressed * cfg.slot_multiplier
                search_start = base_size - 1
                for _ in range(cfg.n_head_per_ngram):
                    p = _next_prime(search_start, seen_primes)
                    seen_primes.add(p)
                    head_primes.append(p)
                    search_start = p
                layer_primes.append(head_primes)
            self.prime_table[layer_id] = layer_primes

        # ------------------------------------------------------------------
        # 2. Compute per-layer odd multipliers from a seeded RNG
        #    (one multiplier per position in the N-gram window)
        # ------------------------------------------------------------------
        PRIME_1 = 10007
        import random

        self.layer_multipliers: Dict[int, List[int]] = {}

        # safe max for int64 multiply: use 2^30 range
        M_bound = 1 << 30

        for layer_id in sorted(cfg.layer_ids):
            rng = random.Random(cfg.seed + PRIME_1 * layer_id)
            mults: List[int] = []
            for _ in range(cfg.max_ngram_size):
                r = rng.randint(0, M_bound - 1)
                mults.append(r * 2 + 1)  # ensure odd
            self.layer_multipliers[layer_id] = mults

        # pre-register as tensors keyed by layer_id (stored as plain Python
        # dicts; moved to device lazily in hash())
        self._mult_tensors: Dict[int, torch.Tensor] = {
            lid: torch.tensor(mults, dtype=torch.int64, device="cpu")
            for lid, mults in self.layer_multipliers.items()
        }
        self._prime_tensors: Dict[int, List[torch.Tensor]] = {
            lid: [
                torch.tensor(head_ps, dtype=torch.int64, device="cpu")
                for head_ps in ngram_ps
            ]
            for lid, ngram_ps in self.prime_table.items()
        }
        self._mult_tensors_by_device: Dict[Tuple[int, str], torch.Tensor] = {}
        self._prime_tensors_by_device: Dict[Tuple[int, int, str], torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def num_hash_heads(self) -> int:
        """Total hash channels per token: (max_ngram_size - 1) * n_head_per_ngram."""
        return (self.cfg.max_ngram_size - 1) * self.cfg.n_head_per_ngram

    def _mults_for_device(self, layer_id: int, device: torch.device) -> torch.Tensor:
        key = (layer_id, str(device))
        mults = self._mult_tensors_by_device.get(key)
        if mults is None:
            mults = self._mult_tensors[layer_id].to(device)
            self._mult_tensors_by_device[key] = mults
        return mults

    def _primes_for_device(
        self, layer_id: int, ngram_idx: int, device: torch.device
    ) -> torch.Tensor:
        key = (layer_id, ngram_idx, str(device))
        primes = self._prime_tensors_by_device.get(key)
        if primes is None:
            primes = self._prime_tensors[layer_id][ngram_idx].to(device)
            self._prime_tensors_by_device[key] = primes
        return primes

    def hash(
        self,
        input_ids: torch.Tensor,
        layer_id: int,
        compressed_input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute N-gram hashes for a batch of token sequences.

        Parameters
        ----------
        input_ids : torch.Tensor, shape [B, T], int64
            Raw (pre-compression) token IDs.
        layer_id : int
            Must be one of cfg.layer_ids.

        Returns
        -------
        torch.Tensor, shape [B, T, (max_ngram_size-1) * n_head_per_ngram], int64
            Hash indices on the same device as input_ids.
        """
        assert layer_id in self.prime_table, f"Unknown layer_id={layer_id}"
        device = input_ids.device

        # 1. Optionally compress token IDs
        if compressed_input_ids is not None:
            x = compressed_input_ids
        elif self.use_compression:
            assert self.compression is not None
            x = self.compression.compress(input_ids)  # still on device
        else:
            x = input_ids  # [B, T], int64

        B, T = x.shape

        # 2. Multipliers for this layer (on device)
        mults = self._mults_for_device(layer_id, device)  # [max_ngram_size]

        # 3. Precompute shifted token matrices
        #    shift_k[k] = tokens at position t-k (left-padded with pad_id)
        pad_id = self.pad_id
        shifts: List[torch.Tensor] = [x]  # shift 0 = current token
        for k in range(1, self.cfg.max_ngram_size):
            # left-pad k positions, keep same T
            padded = torch.full((B, T), pad_id, dtype=torch.int64, device=device)
            padded[:, k:] = x[:, : T - k]
            shifts.append(padded)

        # 4. Compute hash for each ngram order and each head
        all_hashes: List[torch.Tensor] = []
        for ngram_idx, n in enumerate(range(2, self.cfg.max_ngram_size + 1)):
            # mix = XOR of token_k * multiplier_k  (pure int64, no overflow in Python)
            mix = shifts[0] * mults[0]  # [B, T]
            for k in range(1, n):
                mix = torch.bitwise_xor(mix, shifts[k] * mults[k])

            head_primes = self._primes_for_device(layer_id, ngram_idx, device)
            all_hashes.append(torch.remainder(mix.unsqueeze(-1), head_primes))

        # 5. Stack: [B, T, (N-1)*K]
        return torch.cat(all_hashes, dim=2)


# ---------------------------------------------------------------------------
# MultiHeadEmbedding
# ---------------------------------------------------------------------------


class MultiHeadEmbedding(nn.Module):
    """
    Single flat embedding table with per-head offsets.

    Parameters
    ----------
    list_of_N : List[int]
        Vocabulary size for each head; the flat table has size sum(list_of_N).
    D : int
        Embedding dimension per head.
    """

    def __init__(self, list_of_N: List[int], D: int) -> None:
        super().__init__()
        self.num_heads = len(list_of_N)
        self.embedding_dim = D

        # cumulative offsets: [0, N_0, N_0+N_1, ...]
        offsets = [0]
        for n in list_of_N[:-1]:
            offsets.append(offsets[-1] + n)
        self.offsets = torch.tensor(offsets, dtype=torch.long, device="cpu")
        self._offsets_by_device: Dict[str, torch.Tensor] = {}

        total_N = sum(list_of_N)
        self.embedding = nn.Embedding(num_embeddings=total_N, embedding_dim=D)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ids : torch.Tensor, shape [..., H], int64
            Hash indices for each head (last dimension = num_heads).

        Returns
        -------
        torch.Tensor, shape [..., H, D]
        """
        key = str(input_ids.device)
        if key not in self._offsets_by_device:
            self._offsets_by_device[key] = self.offsets.to(input_ids.device)
        offsets = self._offsets_by_device[key]
        shifted = input_ids + offsets  # broadcast offsets over batch dims
        return self.embedding(shifted)


# ---------------------------------------------------------------------------
# ShortConv  (paper Equation 5)
# ---------------------------------------------------------------------------


class ShortConv(nn.Module):
    """
    Depthwise causal 1-D convolution implementing paper Eq. 5:

        Y = SiLU(Conv1D(RMSNorm(V_tilde))) + V_tilde

    Parameters
    ----------
    d_model : int
        Channel dimension (D).
    kernel_size : int
        Convolution kernel width (default 4, per EngramConfig).
    dilation : int
        Dilation factor; set to max_ngram_size to match paper.
    norm_eps : float
        Epsilon for RMSNorm.
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int = 4,
        dilation: int = 1,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(d_model, eps=norm_eps)
        # depthwise causal Conv1d: causal via left-padding, trimmed in forward
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            groups=d_model,
            bias=False,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape [B, T, D]
            Gated value tensor V_tilde.

        Returns
        -------
        torch.Tensor, shape [B, T, D]
            SiLU(Conv1D(RMSNorm(x))) + x
        """
        _, T, _ = x.shape
        normed = self.norm(x)  # [B, T, D]
        # Conv1d expects [B, C, L]
        y = self.conv(normed.transpose(1, 2))  # [B, D, T + left-pad]
        y = y[..., :T]  # causal trim — discard right-padded future positions
        y = self.act(y)
        y = y.transpose(1, 2)  # [B, T, D]
        return y + x  # residual (paper Eq. 5: + V_tilde)


# ---------------------------------------------------------------------------
# Engram  (paper Equations 4 & 5)
# ---------------------------------------------------------------------------


class Engram(nn.Module):
    """
    Conditional memory module following the Engram paper.

    Implements:
      - N-gram hashing (NgramHasher)
      - Multi-head embedding lookup (MultiHeadEmbedding)
      - Gating via paper Eq. 4:
            alpha_t = sigmoid(RMSNorm(h_t)^T RMSNorm(k_t) / sqrt(d))
      - Output via paper Eq. 5 (ShortConv):
            Y = SiLU(Conv1D(RMSNorm(gate * value_proj(e)))) + gate * value_proj(e)

    Parameters
    ----------
    cfg : EngramConfig
    layer_id : int
        Which transformer layer this Engram is attached to (must be in cfg.layer_ids).
    d_model : int
        Hidden dimension of the backbone model.
    tokenizer_vocab_size : int
        Raw (pre-compression) vocabulary size V.
    compression : CompressedTokenizerProjection or None
        Required when cfg.use_tokenizer_compression is True.
    """

    def __init__(
        self,
        cfg: EngramConfig,
        layer_id: int,
        d_model: int,
        tokenizer_vocab_size: int,
        compression: "CompressedTokenizerProjection | None" = None,
        backbone_mhc: bool = False,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_id = layer_id
        self.d_model = d_model
        # backbone_mhc selects the path/parameters:
        #   True  -> faithful per-stream branch path (forward_mhc_branches)
        #   False -> dense single-stream fallback (forward)
        self.backbone_mhc = backbone_mhc

        # --- Hasher ---
        self.hasher = NgramHasher(cfg, tokenizer_vocab_size, compression)
        num_heads = (
            self.hasher.num_hash_heads
        )  # (max_ngram_size - 1) * n_head_per_ngram

        # --- Per-head vocabulary sizes (primes) ---
        head_vocab_sizes: List[int] = []
        for ngram_idx in range(cfg.max_ngram_size - 1):
            for head_idx in range(cfg.n_head_per_ngram):
                head_vocab_sizes.append(
                    self.hasher.prime_table[layer_id][ngram_idx][head_idx]
                )

        # --- Embedding dim per head ---
        assert cfg.memory_dim % num_heads == 0, (
            f"memory_dim ({cfg.memory_dim}) must be divisible by num_heads ({num_heads})"
        )
        head_dim = cfg.memory_dim // num_heads

        self.multi_head_embedding = MultiHeadEmbedding(
            list_of_N=head_vocab_sizes,
            D=head_dim,
        )

        engram_hidden_size = cfg.memory_dim  # = num_heads * head_dim

        # --- Shared projection + output conv (both paths) ---
        self.value_proj = nn.Linear(engram_hidden_size, d_model, bias=False)

        # --- Short conv for output (paper Eq. 5) ---
        self.short_conv = ShortConv(
            d_model=d_model,
            kernel_size=cfg.kernel_size,
            dilation=cfg.max_ngram_size,
        )

        self.mhc_num_streams = cfg.mhc_num_streams
        if self.backbone_mhc:
            # Faithful path: per-stream gated reads routed by the backbone's
            # mHC depth connection (see GPT Block.forward / mhc_engram).
            self.stream_query_norm = nn.RMSNorm(d_model)
            self.stream_key_proj = nn.Linear(
                engram_hidden_size, self.mhc_num_streams * d_model, bias=False
            )
        else:
            # Dense fallback: single-stream gate (paper Eq. 4) used when the
            # backbone does not run mHC.
            self.key_proj = nn.Linear(engram_hidden_size, d_model, bias=False)
            self.norm_query = nn.RMSNorm(d_model)
            self.norm_key = nn.RMSNorm(d_model)

    def _compute_embeddings(
        self,
        input_ids: torch.Tensor,
        compressed_input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hashes = self.hasher.hash(
            input_ids,
            self.layer_id,
            compressed_input_ids=compressed_input_ids,
        )
        return self.multi_head_embedding(hashes).flatten(start_dim=-2)

    def _align_embeddings_to_hidden(
        self, x: torch.Tensor, embeddings: torch.Tensor
    ) -> torch.Tensor:
        assert embeddings.size(0) == x.size(0), (
            "Engram embeddings batch size must match hidden batch"
        )
        assert embeddings.size(1) >= x.size(1), (
            "Engram token context length must be >= hidden sequence length"
        )
        if embeddings.size(1) != x.size(1):
            embeddings = embeddings[:, -x.size(1) :]
        return embeddings

    def _dense_forward(
        self, x: torch.Tensor, embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Dense single-stream gate (paper Eq. 4/5), used without backbone mHC."""
        key = self.key_proj(embeddings)
        normed_query = self.norm_query(x)
        normed_key = self.norm_key(key)
        gate = (normed_query * normed_key).sum(dim=-1, keepdim=True)
        gate = gate / math.sqrt(self.d_model)
        gate = torch.sigmoid(gate)
        value_tilde = gate * self.value_proj(embeddings)
        return self.short_conv(value_tilde)

    def _mhc_gated_values(
        self, x: torch.Tensor, embeddings: torch.Tensor
    ) -> torch.Tensor:
        query = self.stream_query_norm(x).unsqueeze(2)  # [B, T, 1, D]
        keys = self.stream_key_proj(embeddings).view(
            embeddings.size(0), embeddings.size(1), self.mhc_num_streams, self.d_model
        )
        keys = F.rms_norm(keys, (self.d_model,))
        gates = torch.sigmoid(
            (query * keys).sum(dim=-1, keepdim=True) / math.sqrt(self.d_model)
        )
        shared_value = self.value_proj(embeddings).unsqueeze(2)  # [B, T, 1, D]
        return gates * shared_value

    def _short_conv_streams(self, x: torch.Tensor) -> torch.Tensor:
        B, T, S, D = x.shape
        x = x.transpose(1, 2).reshape(B * S, T, D)
        y = self.short_conv(x)
        return y.reshape(B, S, T, D).transpose(1, 2)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor,
        compressed_input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Dense single-stream Engram contribution (used without backbone mHC).

        Parameters
        ----------
        x : torch.Tensor, shape [B, T, D]
            Hidden states from the backbone at this layer.
        input_ids : torch.Tensor, shape [B, T], int64
            Raw token IDs (pre-compression).

        Returns
        -------
        torch.Tensor, shape [B, T, D]
            Residual contribution to add to the hidden states.
        """
        assert not self.backbone_mhc, (
            "Engram.forward is the dense (no-mHC) path; with backbone mHC use "
            "forward_mhc_branches()"
        )
        embeddings = self._compute_embeddings(
            input_ids, compressed_input_ids=compressed_input_ids
        )
        embeddings = self._align_embeddings_to_hidden(x, embeddings)
        return self._dense_forward(x, embeddings)

    def forward_mhc_branches(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor,
        compressed_input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return branch-specific Engram contributions for true backbone mHC.

        x is the mHC branch input / H_pre tensor with base batch B. The output
        keeps the branch axis so the backbone mHC depth connection can attach
        each contribution to its matching residual stream.
        """
        assert self.backbone_mhc, (
            "forward_mhc_branches is only valid with backbone mHC enabled"
        )
        embeddings = self._compute_embeddings(
            input_ids, compressed_input_ids=compressed_input_ids
        )
        embeddings = self._align_embeddings_to_hidden(x, embeddings)
        gated = self._mhc_gated_values(x, embeddings)
        return self._short_conv_streams(gated)

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def init_weights(self) -> None:
        init_bound = math.sqrt(3.0) * (self.d_model**-0.5)
        embedding_weight = self.multi_head_embedding.embedding.weight
        ablation_mode = self.cfg.ablation_mode
        if ablation_mode == "none":
            nn.init.normal_(embedding_weight, mean=0.0, std=1.0)
            embedding_weight.requires_grad_(True)
        elif ablation_mode == "randomize":
            nn.init.normal_(embedding_weight, mean=0.0, std=1.0)
            embedding_weight.requires_grad_(False)
        elif ablation_mode == "uniform":
            shared_row = torch.empty(
                embedding_weight.shape[1],
                device=embedding_weight.device,
                dtype=embedding_weight.dtype,
            )
            nn.init.normal_(shared_row, mean=0.0, std=1.0)
            embedding_weight.copy_(shared_row.unsqueeze(0).expand_as(embedding_weight))
            embedding_weight.requires_grad_(False)
        else:
            raise ValueError(f"Unknown Engram ablation_mode={ablation_mode!r}")
        nn.init.zeros_(self.value_proj.weight)
        nn.init.ones_(self.short_conv.norm.weight)
        nn.init.zeros_(self.short_conv.conv.weight)
        if self.backbone_mhc:
            nn.init.uniform_(self.stream_key_proj.weight, -init_bound, init_bound)
            nn.init.ones_(self.stream_query_norm.weight)
        else:
            nn.init.uniform_(self.key_proj.weight, -init_bound, init_bound)
            nn.init.ones_(self.norm_query.weight)
            nn.init.ones_(self.norm_key.weight)
