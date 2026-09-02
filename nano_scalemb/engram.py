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
    # Cross-layer memory sharing ablation. False (default) gives every Engram
    # layer its own table AND its own hash primes/multipliers, so the same n-gram
    # lands on unrelated rows at different depths. True collapses all Engram
    # layers onto ONE table with ONE addressing scheme: the same n-gram reads the
    # same slot everywhere, and the memory has to serve every depth at once.
    # This asks whether the per-layer memories learn genuinely different things,
    # and is the Engram analogue of the Mobius shared expert pool (see
    # nano_scalemb/moe/). Each layer keeps its own gate, value_proj and conv, so
    # only the memory contents are tied.
    share_memory: bool = False
    # Sharing the memory TABLE and sharing the ADDRESSING SCHEME are separable, and
    # the study's largest surviving effect (-0.0037 to -0.0039, 9-12x, and NOT a
    # capacity effect, §9.2b) has never been decomposed into the two. This knob
    # opens the 2x2:
    #
    #                    per-layer hash            shared hash
    #   per-layer table  `engram`  0.767555        same INDEX, different content
    #   shared table     one POOL, different rows  `engram_shared` 0.763884
    #
    # None (default) means "follow share_memory", which reproduces every earlier
    # run exactly. Set True/False to move independently of the table.
    #
    # Mechanics: the per-head PRIME sizes fix the table geometry, so they must be
    # canonical whenever the table is shared *or* the addressing is; the per-layer
    # MULTIPLIERS decide which row inside that geometry an n-gram lands on, so they
    # are canonical only when the addressing is shared.
    share_hash: "bool | None" = None
    # --- Continuous Engram: what the memory is addressed BY -------------------
    # "tokens" (default) is the paper's design: hash raw token IDs, so the
    # memory can only key on surface form ("New York" and "NYC" land on
    # unrelated rows). The other sources derive the address from the layer's
    # hidden state, so the memory is keyed by what the model currently
    # represents:
    #   "lsh" - frozen random projection -> sign bits. No learned addressing at
    #           all, so it isolates "address by hidden state" from "learn the
    #           addressing"; the cleanest control against the token baseline.
    #   "pq"  - learned product-quantization codebooks (EMA + straight-through).
    #   "pkm" - product-key memory: learned keys, soft top-k read. Replaces the
    #           hash+lookup entirely and is fully differentiable, but changes
    #           the read as well as the address (see nano_scalemb/product_key).
    # With a contextual source, max_ngram_size=1 becomes meaningful (address on
    # the current position only) since the hidden state already carries context.
    address_source: str = "tokens"
    discretize_normalize: bool = True  # RMSNorm before discretizing (scale invariance)
    lsh_bits: int = 20  # LSH code width; alphabet is 2**lsh_bits
    pq_subspaces: int = 4  # PQ sub-spaces; alphabet is codebook_size**subspaces
    pq_codebook_size: int = 64  # keep codebook_size**subspaces < 2**31
    pq_decay: float = 0.99  # EMA decay for codebook updates
    pq_commitment_weight: float = 0.25
    pkm_n_keys: int = 512  # rows per head = pkm_n_keys**2
    pkm_topk: int = 32
    pkm_query_dim: int = 256
    # One independent latent per hash head, instead of one code broadcast to all
    # of them. The single-code design conditions the whole read on ~lsh_bits of
    # the hidden state however many heads there are; per-head latents multiply
    # that by num_hash_heads. It also removes the 31-bit packing limit that
    # forced PQ to use very few, very wide sub-spaces.
    address_latents_per_head: bool = False
    # Address off a stop-gradient hidden state and emit no commitment loss.
    # The commitment term pulls the residual stream toward its centroid and was
    # measured collapsing it to ~1 effective dimension.
    pq_detach_encoder: bool = False
    # Re-centre each LSH bit on an EMA of its own projection mean, so every bit
    # is ~balanced and the realized code entropy approaches the nominal width.
    lsh_balance: bool = False
    # Fraction of each n-gram order's heads that keep TOKEN addressing under
    # address_source="hybrid"; the rest address by hidden state. The token
    # Engram injects information h has discarded (exact recent token identity),
    # while contextual addressing only re-reads what is already in h -- so the
    # hybrid asks whether semantics ADD anything on top of exact match, rather
    # than replacing it.
    hybrid_token_head_frac: float = 0.5
    # --- Readout whitening (diagonal) -----------------------------------------
    # Scale each retrieved row by 1/(its hit rate), so a row that is addressed
    # 1000x more often than average does not dominate the summed read.
    #
    # Why: an Engram read is a Hebbian kernel memory (Garcia et al. 2026,
    # arXiv:2607.10034, Thm 3.1), whose decoding margin is signal minus
    # cross-talk, and whose cross-talk scale is set by the *key crowding*
    # statistic E_K = max_i ||K_i||^2. Their Lemma B.3 shows that among all PSD
    # preconditioners with a fixed average self-kernel, whitening by the
    # empirical feature covariance minimizes an upper bound on E_K -- and it is
    # what turns their construction from "near-optimal" into the first
    # closed-form one to actually hit W = Theta(F log F). For a one-hot feature
    # map like ours that covariance is diagonal and its entries are exactly the
    # per-row hit frequencies, so the whole fix is Algorithm 1's `diag` mode:
    # divide row r's contribution by its frequency.
    #
    # Two deliberate divergences from the paper, both because their fact sets hit
    # every key exactly once while token n-grams are Zipfian:
    #   - readout_whiten_max caps the weight (default 1.0 = downweight-only).
    #     Upweighting a cold row does nothing for crowding, since a row that is
    #     never read cannot crowd anything, but it does amplify that row's random
    #     init straight into the read.
    #   - readout_whiten_power < 1 softens the correction. Full 1/frequency on a
    #     power law effectively deletes the most frequent n-grams from the read.
    #
    # Caveat worth knowing before reading a result: the memory rows are trained
    # with AdamW, which normalizes gradient scale, so a row can partly re-absorb
    # its weight by growing. This is a weaker intervention here than in the
    # paper's setting, where the readout is solved in closed form and cannot
    # adapt. It bites hardest early, and the hit rates keep moving, so it does
    # not cancel exactly.
    readout_whiten: bool = False
    readout_whiten_power: float = 1.0
    readout_whiten_decay: float = 0.99  # EMA over per-row hit rates
    readout_whiten_max: float = 1.0  # cap; >1 admits upweighting of cold rows
    # --- What a row CONTAINS: a constant, or a function of the hidden state ----
    # value_rank=0 (default) is the paper's Engram: row r is a constant vector,
    # so a read depends on the hidden state only through *which* row it selects.
    # value_rank=k>0 makes row r a rank-k linear map applied to a shared
    # projection of h (see LowRankMemory), so the value varies continuously
    # within a cell.
    #
    # This is the axis the study never varied, and 8.1 of
    # docs/conditional_capacity_study.md argues it is the one that mattered all
    # along: MoE and Mobius (values are functions of h) beat dense by 0.0079,
    # while all 10 pure-contextual Engram arms (values are constants, addresses
    # are noisy) sat within 0.0024 of dense. 0.0066 between the two families, at
    # 7.5x the contextual error bar -- bigger than every effect 4 and 5.1 argue
    # over. rank 0 is the Engram end of that ladder, large rank with few rows is
    # the MoE end.
    #
    # The prediction being tested is *differential*, not "bigger is better":
    # contextual addressing should gain a lot from rank > 0, and token addressing
    # should gain little or lose, because a token n-gram is an exact discrete key
    # for which a constant is the right payload -- and rank > 0 costs addresses
    # (rank 1 has ~2.6x fewer rows at fixed bytes, rank 4 ~10x fewer). If both
    # gain equally then this is merely "expressive values help" and the account
    # in 8.1 is wrong.
    value_rank: int = 0
    value_query_dim: int = 128  # width of the shared projection of h
    value_activation: str = "gelu"  # "gelu" | "relu" | "identity"
    # --- Separate key and value per row ---------------------------------------
    # Today one stored vector serves BOTH roles: the read is projected once by
    # stream_key_proj to compute the relevance gate and once by value_proj to
    # produce the payload. Since both projections are shared across every row,
    # a row's key is a fixed linear function of its value -- two rows with
    # similar content are forced to have similar relevance, and a row cannot be
    # sharply selective about context while carrying arbitrary content.
    #
    # key_dim > 0 stores an extra per-head key vector that feeds ONLY the gate,
    # leaving the value dims to carry only payload. This is the k/v split that
    # attention, MoE routers and product-key memories all already have and the
    # Engram does not.
    #
    # Prediction from the row-count result (docs/conditional_capacity_study.md
    # §8.4): row bytes go from head_dim to head_dim + key_dim, so at iso-total
    # this costs rows, and rows are worth ~0.00195 per doubling. key_dim=16 on
    # head_dim=80 costs 17% of rows ~= 0.0005, so the split has to beat that to
    # show. Keep key_dim small; the gate output is a scalar per stream, so it
    # does not need much width.
    key_dim: int = 0
    # --- Count-gated backoff across n-gram orders -----------------------------
    # Weight each head's contribution by a learned function of how often its row
    # is addressed: w = 2*sigmoid(a*log1p(rel_hit_rate) + b), per head, with
    # a=b=0 at init so it starts exactly neutral.
    #
    # This is Kneser-Ney backoff, which was the single largest win in classical
    # n-gram LMing and which the Engram does not have: it reads every order in
    # parallel and concatenates them with fixed weights, so a 3-gram seen twice
    # is trusted like one seen ten thousand times. A rare row is also the one
    # most likely to be a collision victim, and collisions are the binding
    # constraint we measured (row count is worth 0.00195 per doubling at 12x its
    # error bar, while row *content* expressiveness measured 0.01x-0.09x).
    #
    # Unlike readout_whiten, which imposes a fixed monotone 1/frequency rule,
    # this LEARNS the count-to-trust mapping and learns it per head, so it can
    # differ by n-gram order. The two overlap; both default off.
    count_gate: bool = False
    count_gate_decay: float = 0.99  # EMA for the hit-rate estimate it reads
    # Fourier features on log-count, ON TOP of the linear term:
    #   logit = a*u + b + sum_k [ c_sin,k sin(w_k u) + c_cos,k cos(w_k u) ]
    # with u = log1p(rel_hit_rate) and w_k geometric over the realized range.
    #
    # Why: the linear gate is strictly MONOTONE in count, and classical smoothing
    # is not. Katz discounting distrusts count-1 rows because they are unreliable
    # *estimates*, while a count-10000 row is reliable but generic -- so the
    # optimal trust curve is plausibly unimodal, peaking at middling counts. The
    # 2-parameter gate cannot express that shape at all. A Fourier basis on
    # log-count can, at 2K extra scalars per head.
    #
    # Additive, not a replacement: the measured solution was strongly linear
    # (a ~ -0.3 on all 64 heads across 4 gates), and reconstructing a linear trend
    # out of sinusoids is what Fourier bases are worst at. Keeping a*u + b makes
    # this a strict SUPERSET of count_gate -- zero the Fourier coefficients and it
    # is bit-identical -- so it can only lose through optimization or variance,
    # not expressiveness.
    count_gate_fourier: int = 0  # number of frequency bands K (0 = linear only)
    count_gate_fourier_max_u: float = 7.0  # log1p(1100); covers the Zipfian tail
    # Split a row's hit rate into the two things it conflates:
    #   f_o = min over the heads of n-gram order o   -> that n-gram's OWN frequency
    #   c_h = rate_h - f_o                            -> head h's collision excess
    # rate_h = f_o + c_h exactly, so this is a decomposition, not an approximation.
    #
    # Why it is estimable for free: every head of an order hashes the SAME n-gram
    # with a DIFFERENT prime. If a row is hot because its n-gram is frequent, all
    # of that order's heads are hot together; if it is hot from collisions, the
    # other heads are hot for unrelated reasons. So the min across an order's
    # heads is the tightest available upper bound on the n-gram's own frequency,
    # and each head's excess over it is that head's collision load. No cardinality
    # sketch, no extra buffers -- all H rates are already gathered per position.
    #
    # This matters because the Fourier gate came back MONOTONE (§9.2): given a
    # basis that can fit a unimodal trust curve, the model declined. The likely
    # reason is that hit rate conflates "frequent" with "crowded", and both argue
    # for downweighting, so no non-monotonicity can appear. Katz discounting is
    # about the reliability of a count for a SPECIFIC n-gram -- f_o, not rate_h.
    # With the channels separated the hypothesis can finally be tested: is trust
    # non-monotone in f_o once collision load is held in its own term?
    #
    # Strict superset: the raw-rate term is kept, so zeroing the new coefficients
    # recovers count_gate / count_gate_fourier bit-for-bit.
    count_gate_decouple: bool = False
    # --- Hash backoff (token addressing only) ---------------------------------
    # The tables are 100% saturated (study §9.2d), so a frequent, useful n-gram
    # shares its row with a cloud of rare n-grams whose gradients corrupt its
    # learned value; the count gate can only discount the WHOLE blended row.
    # Backoff routes every n-gram NOT in a frozen, precomputed kept-set to its
    # own per-token key (pad,...,t) -- a single well-trained backoff row per
    # ending token -- so the kept, frequent rows are left uncontaminated. This
    # is Katz/stupid-backoff done in the hash address. It adds NO learnable
    # params (the kept-set is a frozen buffer), so it is iso to the same recipe
    # without it. See scripts/build_engram_backoff_keyset.py for the kept-set.
    #
    # "" (default) disables backoff, reproducing every earlier run bit-for-bit.
    backoff_keyset_path: str = ""
    backoff_mode: str = "none"       # "topp" | "fw_topp"; recorded, keyset is frozen
    backoff_p: float = 0.0           # nucleus threshold used to build the keyset (provenance)
    backoff_context_keep: int = 1    # tokens kept on backoff: 1 -> (*,*,t), 2 -> (*,b,t)
    # --- Mid-training value-aware row merge (token addressing only) -----------
    # §9.2d/e + the round-19 backoff scout all say collisions in a LEARNED table
    # are tolerated (absorbed superposition), not corruption. This is the dual
    # question: if collided rows are fine, are DISTINCT rows that ended up with
    # similar learned values redundant? At merge_at_frac of training we cluster
    # rows by their learned value and redirect every member's hash address onto
    # one survivor row (an int64 `row_alias` gather before the embedding lookup),
    # shrinking the EFFECTIVE table by merge_frac for the rest of training.
    # Pre-registered as a COMPRESSION test (does bpb stay flat at 2x/4x fewer
    # effective rows?), not a bpb win. Adds NO learnable params -- row_alias is a
    # buffer, merged rows stay allocated but go dead -- so it is iso to the same
    # recipe without it.
    #
    # 0.0 (default) disables the merge: row_alias is the identity, reproducing
    # every earlier run bit-for-bit.
    merge_frac: float = 0.0          # fraction of rows/head merged away (0 -> off)
    merge_at_frac: float = 0.5       # fraction of training when the one-shot merge fires
    merge_metric: str = "cosine"     # "cosine" | "lsh" -- how rows are clustered
    # merge_source picks WHAT the clustering sees. "value" (default) clusters on
    # the LEARNED value table mid-training -- the schedule with a 2x over-provision
    # sweet spot. "semantic" clusters on a STATIC external feature (a frozen
    # token-embedding centroid per row, built offline from a trained wte by
    # scripts/build_engram_semantic_prior.py) and fires at step 0, so the table
    # trains already-collapsed onto a semantic partition. The learned-value merge
    # is capped by half-trained value quality; the semantic prior sidesteps that by
    # importing a fully-trained embedding's geometry. Still TOKEN-indexed (rows are
    # hash(token-window) buckets) -- NOT the closed hidden-state idea-2.
    merge_source: str = "value"      # "value" | "semantic"
    semantic_prior_path: str = ""    # feature/hit_mask .pt for merge_source="semantic"


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


def ngram_orders(max_ngram_size: int) -> List[int]:
    """N-gram orders the hasher allocates primes and heads for.

    Normally 2..N: with token addressing a unigram "memory" is just a second
    token embedding, so the paper starts at bigrams. Contextual addressing
    (address_source != "tokens") makes n=1 meaningful -- the code already
    summarizes context -- so max_ngram_size=1 is allowed and means "address on
    the current position only".

    Single definition because five call sites depend on this agreeing exactly:
    prime allocation, multiplier count, num_hash_heads, the hash loop, and the
    Engram's per-head vocab sizes.
    """
    assert max_ngram_size >= 1, f"max_ngram_size must be >= 1, got {max_ngram_size}"
    return [1] if max_ngram_size == 1 else list(range(2, max_ngram_size + 1))


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
        # Under the shared-memory ablation every layer hashes with the FIRST
        # layer's primes and multipliers, so a given n-gram addresses the same
        # slot at every depth. Without that the layers would share a table but
        # not an addressing scheme, which is just a bigger table, not sharing.
        self.share_memory = cfg.share_memory
        self.share_hash = (
            cfg.share_hash if cfg.share_hash is not None else cfg.share_memory
        )
        self.canonical_layer_id = min(cfg.layer_ids) if cfg.layer_ids else None

        if cfg.address_source == "pkm":
            # Product-key memory bypasses hashing entirely; the hasher is built
            # only so num_hash_heads still reports the head count. Table sizing
            # is irrelevant, so keep the token defaults.
            V_compressed = tokenizer_vocab_size
            self.pad_id = cfg.pad_id
            self.use_compression = False
        elif cfg.address_source == "hybrid":
            # Half the heads address by token n-gram, half by hidden state, so
            # the tables must cover whichever alphabet is larger. Tokenizer
            # compression stays ON: the token heads still use it.
            from nano_scalemb.discretize import code_alphabet_size

            tok_alphabet = (
                compression.compressed_vocab_size
                if (cfg.use_tokenizer_compression and compression is not None)
                else tokenizer_vocab_size
            )
            V_compressed = max(tok_alphabet, code_alphabet_size(cfg))
            self.pad_id = (
                compression.compressed_pad_id
                if (cfg.use_tokenizer_compression and compression is not None)
                else cfg.pad_id
            )
        elif cfg.address_source != "tokens":
            # Continuous addressing: the "vocabulary" is the discretizer's code
            # alphabet, not the tokenizer's. Table sizes still scale as
            # alphabet * slot_multiplier, but the alphabet can be far larger
            # than V, so cap it -- with 2^20 LSH codes and slot_multiplier=18 a
            # naive table would be ~19M rows per head.
            from nano_scalemb.discretize import code_alphabet_size

            V_compressed = min(code_alphabet_size(cfg), tokenizer_vocab_size)
            # Codes are unsigned and dense from 0; reserve the top code as the
            # left-padding sentinel so shifted-in positions cannot collide with
            # a real code.
            self.pad_id = V_compressed - 1
            self.use_compression = False
        elif self.use_compression:
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

        for layer_id in self._prime_layer_ids():
            layer_primes: List[List[int]] = []
            for ngram_order in ngram_orders(cfg.max_ngram_size):
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

        for layer_id in self._mult_layer_ids():
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

        # --- Hash backoff: load the frozen kept-set (token addressing only) ---
        # self._hash_base is the radix used to encode an n-gram window as one
        # int64 id; it MUST match the base the precompute used (V_compressed).
        self._hash_base = V_compressed
        self.backoff_context_keep = int(getattr(cfg, "backoff_context_keep", 1))
        # order -> sorted int64 kept window-ids (CPU); device copies cached lazily.
        self._backoff_keysets: Dict[int, torch.Tensor] = {}
        self._backoff_keysets_by_device: Dict[Tuple[int, str], torch.Tensor] = {}
        path = getattr(cfg, "backoff_keyset_path", "") or ""
        if path:
            assert cfg.address_source == "tokens", (
                "hash backoff (backoff_keyset_path) only applies to "
                f"address_source='tokens', got {cfg.address_source!r}"
            )
            blob = torch.load(path, map_location="cpu", weights_only=False)
            keysets = blob["keysets"] if isinstance(blob, dict) and "keysets" in blob else blob
            for order, ids in keysets.items():
                order = int(order)
                # window-id = Σ_k token_k * base^k must stay inside int64.
                assert V_compressed ** order < 2 ** 62, (
                    f"backoff order {order} with base {V_compressed} overflows the "
                    "int64 window-id encoding; use a hashed id for high orders"
                )
                t = torch.as_tensor(ids, dtype=torch.int64, device="cpu")
                # sorted + unique so searchsorted membership is exact
                self._backoff_keysets[order] = torch.unique(t)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _prime_layer_ids(self) -> List[int]:
        """Layer ids that get their own per-head prime sizes (table geometry).

        Canonical only when the table or the addressing is shared: one table
        demands one geometry, and identical indices demand identical moduli.
        """
        if (self.share_memory or self.share_hash) and self.canonical_layer_id is not None:
            return [self.canonical_layer_id]
        return sorted(self.cfg.layer_ids)

    def _mult_layer_ids(self) -> List[int]:
        """Layer ids that get their own hash multipliers (which row, not how many)."""
        if self.share_hash and self.canonical_layer_id is not None:
            return [self.canonical_layer_id]
        return sorted(self.cfg.layer_ids)

    def resolve_prime_layer_id(self, layer_id: int) -> int:
        """Map a layer to the layer whose prime table (geometry) it uses."""
        if (self.share_memory or self.share_hash) and self.canonical_layer_id is not None:
            return self.canonical_layer_id
        return layer_id

    def resolve_mult_layer_id(self, layer_id: int) -> int:
        """Map a layer to the layer whose hash multipliers it uses."""
        if self.share_hash and self.canonical_layer_id is not None:
            return self.canonical_layer_id
        return layer_id

    # Table sizing asks about geometry, so it resolves to the prime layer.
    resolve_layer_id = resolve_prime_layer_id

    @property
    def ngram_orders(self) -> List[int]:
        return ngram_orders(self.cfg.max_ngram_size)

    @property
    def num_hash_heads(self) -> int:
        """Total hash channels per token: len(ngram_orders) * n_head_per_ngram."""
        return len(self.ngram_orders) * self.cfg.n_head_per_ngram

    def _mults_for_device(self, layer_id: int, device: torch.device) -> torch.Tensor:
        key = (layer_id, str(device))
        mults = self._mult_tensors_by_device.get(key)
        if mults is None:
            mults = self._mult_tensors[layer_id].to(device)
            self._mult_tensors_by_device[key] = mults
        return mults

    def _keyset_for_device(self, order: int, device: torch.device) -> torch.Tensor | None:
        """Sorted kept window-ids for an n-gram order, on device (None if no backoff)."""
        ids = self._backoff_keysets.get(order)
        if ids is None:
            return None
        key = (order, str(device))
        dev_ids = self._backoff_keysets_by_device.get(key)
        if dev_ids is None:
            dev_ids = ids.to(device)
            self._backoff_keysets_by_device[key] = dev_ids
        return dev_ids

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
        # Geometry and row-choice resolve separately (EngramConfig.share_hash).
        prime_layer = self.resolve_prime_layer_id(layer_id)
        mult_layer = self.resolve_mult_layer_id(layer_id)
        assert prime_layer in self.prime_table, f"Unknown layer_id={layer_id}"
        assert mult_layer in self.layer_multipliers, f"Unknown layer_id={layer_id}"
        device = input_ids.device

        # 1. Optionally compress token IDs
        if compressed_input_ids is not None:
            x = compressed_input_ids
        elif self.use_compression:
            assert self.compression is not None
            x = self.compression.compress(input_ids)  # still on device
        else:
            x = input_ids  # [B, T], int64

        # Per-head codes: [B, T, num_hash_heads] means every head addresses off
        # its OWN latent, so the read is conditioned on num_heads x code_bits
        # rather than the same code_bits broadcast to every head. A [B, T] code
        # (tokens, or a single-latent discretizer) takes the original path.
        n_orders = len(self.ngram_orders)
        K = self.cfg.n_head_per_ngram
        per_head = x.ndim == 3
        if per_head:
            assert x.shape[-1] == self.num_hash_heads, (
                f"per-head codes need last dim == num_hash_heads "
                f"({self.num_hash_heads}), got {x.shape[-1]}"
            )
            B, T, _ = x.shape
            # [B, T, n_orders, K] so order `idx` owns heads [idx*K, (idx+1)*K)
            x = x.view(B, T, n_orders, K)
        else:
            B, T = x.shape

        # 2. Multipliers for this layer (on device)
        mults = self._mults_for_device(mult_layer, device)  # [max_ngram_size]

        # 3. Precompute shifted token matrices
        #    shift_k[k] = tokens at position t-k (left-padded with pad_id)
        pad_id = self.pad_id
        shifts: List[torch.Tensor] = [x]  # shift 0 = current token
        for k in range(1, self.cfg.max_ngram_size):
            # left-pad k positions along time, keep same T
            padded = torch.full_like(x, pad_id)
            padded[:, k:] = x[:, : T - k]
            shifts.append(padded)

        # 4. Compute hash for each ngram order and each head
        all_hashes: List[torch.Tensor] = []
        for ngram_idx, n in enumerate(self.ngram_orders):
            # mix = XOR of code_k * multiplier_k (int64; codes are < 2^31 so the
            # product cannot overflow -- see discretize.MAX_CODE_BITS)
            def _slice(t):
                return t[:, :, ngram_idx] if per_head else t

            # Hash backoff (token addressing only): windows NOT in the frozen
            # kept-set drop their context to pad, so they land on a single per-
            # token (pad,...,t) backoff row instead of a random shared row. The
            # kept, frequent windows hash exactly as without backoff.
            ord_shifts = shifts
            keyset = None if per_head else self._keyset_for_device(n, device)
            if keyset is not None:
                radix, wid = 1, shifts[0].clone()   # id = Σ_k shift_k * base^k
                for k in range(1, n):
                    radix *= self._hash_base
                    wid = wid + shifts[k] * radix
                idx = torch.searchsorted(keyset, wid)
                idx_cl = idx.clamp(max=keyset.numel() - 1)
                keep = (idx < keyset.numel()) & (keyset[idx_cl] == wid)  # [B, T]
                ord_shifts = list(shifts[:n])
                pad_full = torch.full_like(shifts[0], pad_id)
                for k in range(self.backoff_context_keep, n):
                    ord_shifts[k] = torch.where(keep, shifts[k], pad_full)

            mix = _slice(ord_shifts[0]) * mults[0]  # [B, T] or [B, T, K]
            for k in range(1, n):
                mix = torch.bitwise_xor(mix, _slice(ord_shifts[k]) * mults[k])

            head_primes = self._primes_for_device(prime_layer, ngram_idx, device)
            if per_head:
                # mix is already [B, T, K]: head k mods by its own prime.
                all_hashes.append(torch.remainder(mix, head_primes))
            else:
                all_hashes.append(torch.remainder(mix.unsqueeze(-1), head_primes))

        # 5. Stack: [B, T, n_orders * K]
        return torch.cat(all_hashes, dim=2)


# ---------------------------------------------------------------------------
# MultiHeadEmbedding
# ---------------------------------------------------------------------------


# Guards the whitening divide and pins the weight at exactly 1.0 for a row of
# average hit rate. Small enough that it does not shape the hot tail; cold rows
# are governed by readout_whiten_max, not by this.
_WHITEN_EPS = 1e-3

# "identity" makes a row a plain linear map, isolating the nonlinearity from the
# rank increase; "gelu" matches the MoE arms this ladder runs toward.
_VALUE_ACTIVATIONS = {
    "gelu": F.gelu,
    "relu": F.relu,
    "identity": lambda x: x,
}


class AddressedMemory(nn.Module):
    """Shared machinery for a flat hash-addressed table: per-head offsets and
    optional readout whitening. Subclasses decide what a row *contains*.

    Parameters
    ----------
    list_of_N : List[int]
        Vocabulary size for each head; the flat table has size sum(list_of_N).
    D : int
        Output dimension per head.
    whiten : bool
        Scale each read row by a function of its own hit rate. See
        EngramConfig.readout_whiten for the derivation and the caveats.
    whiten_power, whiten_decay, whiten_max : float
        Exponent on the correction, EMA decay for the hit-rate estimate, and cap
        on the resulting weight.
    """

    def __init__(
        self,
        list_of_N: List[int],
        D: int,
        whiten: bool = False,
        whiten_power: float = 1.0,
        whiten_decay: float = 0.99,
        whiten_max: float = 1.0,
        count_gate: bool = False,
        count_gate_decay: float = 0.99,
        count_gate_fourier: int = 0,
        count_gate_fourier_max_u: float = 7.0,
        count_gate_decouple: bool = False,
        heads_per_group: int = 0,
    ) -> None:
        super().__init__()
        self.num_heads = len(list_of_N)
        self.embedding_dim = D
        self.head_sizes = list(list_of_N)
        self.total_rows = sum(list_of_N)

        # cumulative offsets: [0, N_0, N_0+N_1, ...]
        offsets = [0]
        for n in list_of_N[:-1]:
            offsets.append(offsets[-1] + n)
        self.offsets = torch.tensor(offsets, dtype=torch.long, device="cpu")
        self._offsets_by_device: Dict[str, torch.Tensor] = {}
        self._alias_by_device: Dict[str, torch.Tensor] = {}

        # Row alias: flat-index -> flat-index redirection applied before the
        # gather. Identity (arange) is an exact no-op reproducing every run
        # byte-for-byte; a mid-training merge (merge_rows) rewrites it so several
        # hash addresses read one survivor row, shrinking the EFFECTIVE table.
        # Persistent: it IS the addressing after a merge, so a resumed run that
        # restarted it at identity would revert the merge. Re-filled to identity
        # in init_memory_table (the to_empty() trap -- construction values do not
        # survive the GPT's meta -> to_empty()).
        self.register_buffer(
            "row_alias", torch.arange(self.total_rows, dtype=torch.long)
        )
        # Persistent bool: whether a merge has fired. Round-trips so a resumed
        # merged run keeps applying the alias (a plain attr would default to
        # False on rebuild and silently revert the merge).
        self.register_buffer("_merged", torch.zeros((), dtype=torch.bool))
        # Python-bool mirror of the _merged buffer, used to gate the alias gather
        # in _shift. This MUST be a plain bool, not `if self._merged:` on the
        # tensor buffer: branching on a live 0-dim tensor inside the compiled
        # forward makes torch.compile emit a different graph (guard + altered
        # reduction order) than code with no such branch, which perturbs the
        # forward NUMERICALLY even on the identity path (measured: step-0 loss
        # 10.400112 vs the clean 10.400093, diverging within one optimizer step).
        # A Python bool is constant-folded by dynamo, so the unmerged graph is
        # byte-identical to the pre-merge code. Kept in sync with the buffer in
        # merge_rows, init_memory_table, and after checkpoint load
        # (sync_merge_flags); load_state_dict(assign=True) replaces the buffer
        # without touching this attr, so the sync call is mandatory on resume.
        self._merged_flag = False

        self.whiten = whiten
        self.whiten_power = whiten_power
        self.whiten_decay = whiten_decay
        self.whiten_max = whiten_max
        self.count_gate = count_gate
        # Both features read the same statistic; whichever is on pays for it.
        self.track_hit_rate = whiten or count_gate
        self.hit_rate_decay = count_gate_decay if count_gate else whiten_decay
        if self.track_hit_rate:
            # Stored in units where 1.0 == "addressed as often as a uniform row of
            # this head", which folds the per-head alphabet size into the update
            # and keeps the read path one gather with no per-head bookkeeping.
            #
            # Persistent: the hit rates ARE part of the read, so a resumed run
            # that restarted them at uniform would step-change every read.
            self.register_buffer("hit_rate", torch.ones(self.total_rows))
        if count_gate:
            # Per-head (a, b) for w = 2*sigmoid(a*log1p(rel) + b). Zero-init makes
            # w exactly 1.0 at every row, so the feature starts as a no-op and
            # cannot confound the arm with an init-time change in read magnitude.
            self.count_gate_scale = nn.Parameter(torch.zeros(self.num_heads))
            self.count_gate_bias = nn.Parameter(torch.zeros(self.num_heads))
        self.count_gate_fourier = count_gate_fourier if count_gate else 0
        self.count_gate_fourier_max_u = count_gate_fourier_max_u
        # Heads of one n-gram order hash the same n-gram, so they are the group the
        # frequency estimate is minimised over. 0 -> treat all heads as one group.
        self.heads_per_group = heads_per_group or self.num_heads
        self.count_gate_decouple = bool(count_gate and count_gate_decouple)
        if self.count_gate_decouple:
            assert self.num_heads % self.heads_per_group == 0, (
                f"num_heads {self.num_heads} must be divisible by heads_per_group "
                f"{self.heads_per_group} to group heads by n-gram order"
            )
            # Separate coefficients for the two decoupled channels. Zero-init keeps
            # the strict-superset property.
            self.count_gate_freq_scale = nn.Parameter(torch.zeros(self.num_heads))
            self.count_gate_collide_scale = nn.Parameter(torch.zeros(self.num_heads))
        if self.count_gate_fourier > 0:
            # [H, 2K]: sin and cos coefficient per head per band. Zero-init keeps
            # the strict-superset property -- at init this IS the linear gate.
            self.count_gate_fourier_coef = nn.Parameter(
                torch.zeros(self.num_heads, 2 * self.count_gate_fourier)
            )
            # Geometric frequencies: band 0 spans half a period over [0, max_u],
            # band k halves the wavelength. Non-persistent and regenerated in
            # init, same pattern as the LSH projection in discretize.py -- they
            # are a fixed basis, not learned state, so they need not round-trip.
            self.register_buffer(
                "count_gate_freqs",
                torch.zeros(self.count_gate_fourier),
                persistent=False,
            )

    # -- subclass contract -------------------------------------------------
    def content_parameters(self) -> List[nn.Parameter]:
        """The addressed rows themselves, as opposed to any dense projection.

        These get the Engram embedding LR multiplier and are excluded from the
        6*nparams FLOP term, because they are gathered rather than multiplied.
        """
        raise NotImplementedError

    def active_params_per_token(self) -> int:
        """Row weight a single token actually reads, per reading layer."""
        raise NotImplementedError

    def read_flops_per_token(self) -> int:
        """FLOPs the read itself costs, beyond the gather. Zero for a plain
        lookup; nonzero once a row is a linear map instead of a constant."""
        return 0

    def reset_content(self, ablation_mode: str) -> None:
        raise NotImplementedError

    def _value_matrix(self) -> torch.Tensor:
        """[total_rows, Dv] learned values that row clustering reads. Subclass hook."""
        raise NotImplementedError

    def _merge_writeback(self, alias: torch.Tensor) -> None:
        """Copy survivor rows into their merged-away members. Subclass hook."""
        raise NotImplementedError

    @torch.no_grad()
    def _update_hit_rate(self, shifted: torch.Tensor) -> None:
        """EMA the per-row hit rate from this batch's addresses."""
        flat = shifted.reshape(-1)
        # Every position contributes exactly one hit to every head, so the most
        # hits any single head can have seen is flat.numel() / num_heads.
        n_positions = flat.numel() / self.num_heads
        hits = torch.zeros_like(self.hit_rate)
        hits.index_add_(0, flat, torch.ones_like(flat, dtype=hits.dtype))
        # Ranks see different data and the read scale must not depend on which
        # rank computed it, so average the counts (same pattern as the LSH bit
        # thresholds in nano_scalemb/discretize.py).
        import torch.distributed as _dist

        if _dist.is_available() and _dist.is_initialized():
            _dist.all_reduce(hits, op=_dist.ReduceOp.AVG)
        decay = self.hit_rate_decay
        self.hit_rate.mul_(decay)
        offset = 0
        for n in self.head_sizes:
            # rel = hits * N_h / n_positions, so a uniformly addressed row sits at
            # 1.0 whatever its head's alphabet size.
            self.hit_rate[offset : offset + n].add_(
                hits[offset : offset + n], alpha=(1.0 - decay) * n / n_positions
            )
            offset += n

    def _whiten_weight(self, shifted: torch.Tensor) -> torch.Tensor:
        """Per-read scale, shape [..., H]. Exactly 1.0 at average hit rate."""
        rel = self.hit_rate[shifted]
        weight = (1.0 + _WHITEN_EPS) / (rel + _WHITEN_EPS)
        if self.whiten_power != 1.0:
            weight = weight.pow(self.whiten_power)
        return weight.clamp(max=self.whiten_max)

    def _count_gate_weight(self, shifted: torch.Tensor) -> torch.Tensor:
        """Learned per-head trust in a row, from how often it is addressed.

        Shape [..., H]. Exactly 1.0 everywhere at init (all coefficients 0).
        """
        rel = self.hit_rate[shifted]
        u = torch.log1p(rel)
        logit = self.count_gate_scale * u + self.count_gate_bias
        # Channel the Fourier basis is applied to: the raw rate normally, but the
        # decoupled n-gram frequency when available, since that is the quantity
        # Katz discounting is actually about (EngramConfig.count_gate_decouple).
        basis_u = u
        if self.count_gate_decouple:
            freq, collide = self.decouple_hit_rate(rel)
            logit = (
                logit
                + self.count_gate_freq_scale * torch.log1p(freq)
                + self.count_gate_collide_scale * torch.log1p(collide)
            )
            basis_u = torch.log1p(freq)
        if self.count_gate_fourier > 0:
            # [..., H, K] phases, then contract sin/cos against the per-head coefs
            phase = basis_u.unsqueeze(-1) * self.count_gate_freqs
            basis = torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1)
            logit = logit + (basis * self.count_gate_fourier_coef).sum(-1)
        return 2.0 * torch.sigmoid(logit)

    def decouple_hit_rate(self, rel: torch.Tensor):
        """Split per-head hit rates into (own-ngram frequency, collision excess).

        rel: [..., H] -> (freq [..., H], collide [..., H]), broadcast so each head
        carries its order's frequency estimate and its own excess.

        Every head of an n-gram order hashes the SAME n-gram under a different
        prime, so a row that is hot because its n-gram is frequent is hot in ALL
        of that order's heads, while collision-driven heat is independent across
        them. The per-order min is therefore the tightest upper bound available on
        the n-gram's own rate, and the excess over it is that head's crowding.
        Exact decomposition: rel == freq + collide.
        """
        g = self.heads_per_group
        grouped = rel.unflatten(-1, (rel.shape[-1] // g, g))
        freq = grouped.amin(dim=-1, keepdim=True).expand_as(grouped)
        return freq.flatten(-2), (grouped - freq).flatten(-2)

    def _shift(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Per-head indices -> indices into the flat table (after any merge).

        The row_alias redirection is applied here, so it flows to the gather, the
        count gate AND the hit-rate update together: a merged-away row is read by
        nobody, credited to nobody, and receives no gradient -- it simply goes
        dead, while its survivor absorbs its traffic. Identity alias == no-op.
        """
        key = str(input_ids.device)
        if key not in self._offsets_by_device:
            self._offsets_by_device[key] = self.offsets.to(input_ids.device)
        flat = input_ids + self._offsets_by_device[key]  # offsets broadcast
        if self._merged_flag:  # Python bool: dynamo constant-folds the no-op path
            if key not in self._alias_by_device:
                self._alias_by_device[key] = self.row_alias.to(input_ids.device)
            flat = self._alias_by_device[key][flat]
        return flat

    def _apply_row_weights(
        self, out: torch.Tensor, shifted: torch.Tensor
    ) -> torch.Tensor:
        """Whitening and/or count gating, then fold this batch into the stats."""
        if not self.track_hit_rate:
            return out
        # Weights from the hit rates BEFORE folding in this batch, so a read is
        # never scaled by its own occurrence.
        weight = None
        if self.whiten:
            weight = self._whiten_weight(shifted)
        if self.count_gate:
            gate = self._count_gate_weight(shifted)
            weight = gate if weight is None else weight * gate
        if self.training:
            self._update_hit_rate(shifted)
        return out * weight.unsqueeze(-1).to(out.dtype)

    # -- value-aware row merge (idea 1B) -----------------------------------
    _COSINE_MAX = 8192  # above this, exact pairwise cosine is infeasible -> LSH

    def sync_merge_flags(self) -> None:
        """Refresh the Python-bool _merged_flag from the _merged buffer.

        load_state_dict(assign=True) on resume swaps the _merged buffer in but
        leaves the plain attr stale, so a resumed MERGED run would keep the
        no-op graph and silently revert the merge. Call this after any external
        write to the buffer (checkpoint load) so _shift routes through the alias.
        """
        self._merged_flag = bool(self._merged.item())
        if not self._merged_flag:
            self._alias_by_device.clear()

    @torch.no_grad()
    def merge_rows(self, frac: float, metric: str = "cosine", seed: int = 0,
                   feature_override: torch.Tensor | None = None,
                   keep_mask: torch.Tensor | None = None) -> dict:
        """Cluster each head's rows by learned value and alias members onto one
        survivor, shrinking the EFFECTIVE table by ~frac. Returns row stats.

        Deterministic and identical across DDP ranks WITHOUT communication: the
        weights are DDP-synced, hit_rate is all-reduced in _update_hit_rate, and
        the projection is seeded -- so every rank computes the same alias. Merged
        rows keep their allocation (iso params) but go dead: never addressed,
        never gradient-updated; their stale optimizer state is irrelevant.

        feature_override : [total_rows, d] or None
            When given, cluster on THIS matrix instead of the learned value table
            (merge_source="semantic": a static token-embedding centroid per row).
            Must be row-aligned with the flat table (same total_rows).
        keep_mask : bool[total_rows] or None
            When given, only rows with keep_mask=True PARTICIPATE in clustering;
            rows with keep_mask=False keep their identity alias (never merged, never
            a survivor). Used by the semantic prior so featureless rows -- which no
            sampled n-gram reached, so their centroid is 0 -- are not all collapsed
            into one survivor. `keep` (survivors per head) is taken over the
            participating rows only.
        """
        if not 0.0 < frac < 1.0:
            raise ValueError(f"merge_frac must be in (0, 1), got {frac}")
        if metric not in ("cosine", "lsh"):
            raise ValueError(f"merge_metric must be 'cosine' | 'lsh', got {metric!r}")
        values = feature_override if feature_override is not None else self._value_matrix()
        if values.shape[0] != self.total_rows:
            raise ValueError(
                f"feature_override has {values.shape[0]} rows, expected total_rows="
                f"{self.total_rows}"
            )
        device = values.device
        alias = torch.arange(self.total_rows, device=device)
        hit = self.hit_rate.to(device) if self.track_hit_rate else None
        gen = torch.Generator(device="cpu").manual_seed(seed)
        for h in range(self.num_heads):
            start = int(self.offsets[h])
            N = self.head_sizes[h]
            part = None
            if keep_mask is not None:
                part = keep_mask[start : start + N].to(device)
                n_part = int(part.sum())
            else:
                n_part = N
            keep = max(1, round((1.0 - frac) * n_part))
            if keep >= n_part:
                continue
            Vh = values[start : start + N].float()
            hh = hit[start : start + N] if hit is not None else None
            survivor_local = self._cluster_head(Vh, hh, keep, metric, gen, part)
            alias[start : start + N] = survivor_local.to(device) + start
        # Copy each survivor's full row into its merged-away members so a save
        # taken before the next optimizer step is self-consistent. Survivors
        # alias to themselves, so their rows are unchanged.
        self._merge_writeback(alias)
        self.row_alias.copy_(alias)
        self._merged.fill_(True)
        self._merged_flag = True
        self._alias_by_device.clear()
        return dict(total_rows=self.total_rows,
                    effective_rows=int(torch.unique(alias).numel()))

    @torch.no_grad()
    def _cluster_head(self, V, hit, keep, metric, gen, participate=None) -> torch.Tensor:
        """Assign each of V's N rows to one of `keep` survivor rows (local idx).

        Survivor per cluster is the highest-hit-rate member (protect the
        well-trained row); ties break to the lowest index. Returns LongTensor[N]
        of survivor local indices (survivors map to themselves).

        participate : bool[N] or None
            When given, only True rows may cluster/be survivors; False rows map to
            themselves (identity). `keep` is the survivor budget over the True rows."""
        N = V.shape[0]
        if participate is not None:
            idx = torch.nonzero(participate, as_tuple=False).flatten()
            out = torch.arange(N, device=V.device)              # False rows: identity
            if idx.numel() == 0:
                return out
            sub_hit = hit[idx] if hit is not None else None
            sub_surv = self._cluster_head(V[idx], sub_hit, min(keep, idx.numel()),
                                          metric, gen, None)     # local to the subset
            out[idx] = idx[sub_surv]                             # map subset survivors back
            return out
        Vn = F.normalize(V, dim=-1)
        score = hit if hit is not None else torch.zeros(N, device=V.device)
        if metric == "cosine" and N <= self._COSINE_MAX:
            # Exact: survivors = the `keep` highest-hit rows; every row joins its
            # nearest survivor by cosine (a survivor's self-similarity 1.0 wins).
            surv = torch.topk(score, keep).indices
            sim = Vn @ Vn[surv].T                       # [N, keep]
            nearest = sim.argmax(dim=-1)                # index into surv
            return surv[nearest]
        # Scalable path: a b-bit sign-LSH code puts value-similar rows on nearby
        # integer codes; sort by code and cut into `keep` contiguous groups.
        b = max(1, min(30, int(math.ceil(math.log2(keep))) + 4))
        R = torch.randn(V.shape[1], b, generator=gen).to(V.device, V.dtype)
        bits = (Vn @ R > 0).long()                      # [N, b]
        pw = (2 ** torch.arange(b, device=V.device)).long()
        code = (bits * pw).sum(-1)
        order = torch.argsort(code)                     # similar codes adjacent
        grp_of_sorted = (torch.arange(N, device=V.device) * keep) // N
        group = torch.empty(N, dtype=torch.long, device=V.device)
        group[order] = grp_of_sorted                    # keep groups, ~N/keep each
        # survivor per group = argmax score, tie -> lowest index (vectorized)
        best = torch.full((keep,), float("-inf"), device=V.device)
        best.scatter_reduce_(0, group, score.float(), reduce="amax", include_self=True)
        is_best = score.float() >= best[group]
        cand = torch.where(is_best, torch.arange(N, device=V.device),
                           torch.full((N,), N, device=V.device))
        surv_of_group = torch.full((keep,), N, dtype=torch.long, device=V.device)
        surv_of_group.scatter_reduce_(0, group, cand, reduce="amin", include_self=True)
        return surv_of_group[group]


class MultiHeadEmbedding(AddressedMemory):
    """A row is a constant vector: the paper's Engram memory.

    This is the ``value_rank=0`` end of the ladder in EngramConfig.value_rank.

    With ``key_dim > 0`` a row stores an extra key vector that feeds only the
    relevance gate, so the value dims carry only payload -- see
    EngramConfig.key_dim. The key and value live in ONE table of width
    ``key_dim + D``, not two tables, so a row is still a single gather.
    """

    def __init__(
        self, list_of_N: List[int], D: int, key_dim: int = 0, **weight_kwargs
    ) -> None:
        super().__init__(list_of_N, D, **weight_kwargs)
        self.key_dim = key_dim
        self.embedding = nn.Embedding(
            num_embeddings=self.total_rows, embedding_dim=key_dim + D
        )

    def content_parameters(self) -> List[nn.Parameter]:
        return list(self.embedding.parameters())

    def active_params_per_token(self) -> int:
        return self.num_heads * (self.key_dim + self.embedding_dim)

    def reset_content(self, ablation_mode: str) -> None:
        _init_addressed_rows(self.embedding.weight, ablation_mode, std=1.0)

    def _value_matrix(self) -> torch.Tensor:
        # Cluster on the VALUE part only; the key dims feed only the gate.
        return self.embedding.weight.data[:, self.key_dim :]

    def _merge_writeback(self, alias: torch.Tensor) -> None:
        self.embedding.weight.data.copy_(self.embedding.weight.data[alias])

    def forward(self, input_ids: torch.Tensor, query: torch.Tensor | None = None):
        """
        Parameters
        ----------
        input_ids : torch.Tensor, shape [..., H], int64
            Hash indices for each head (last dimension = num_heads).
        query : unused
            Accepted so both memory kinds share a call signature.

        Returns
        -------
        torch.Tensor, shape [..., H, key_dim + D]. Callers split it with
        ``split_key_value``; with key_dim=0 the key part is empty.
        """
        shifted = self._shift(input_ids)
        return self._apply_row_weights(self.embedding(shifted), shifted)

    def split_key_value(self, rows: torch.Tensor):
        """[..., H, key_dim + D] -> (values [..., H, D], keys or None).

        Row weights (whitening, count gate) are applied to the whole row before
        this, so the key is scaled with its value -- they are one row and a
        relevance score computed from a differently-scaled key than the payload
        it gates would be incoherent.
        """
        if self.key_dim == 0:
            return rows, None
        return rows[..., self.key_dim :], rows[..., : self.key_dim]


class LowRankMemory(AddressedMemory):
    """A row is a rank-k linear map, applied to a shared projection of h.

        read_h = W_out[r] @ act(W_in[r] @ q),   q = query_proj(rms_norm(h))

    Why this exists (docs/conditional_capacity_study.md 8.1): a constant-valued
    memory over a *continuous* key space discards every bit of within-cell
    variation, and the cell is the only resolution a noisy address has. Token
    n-grams do not have that problem -- the address is exact, and a constant is
    the right thing to store -- which is why the 10 pure-contextual arms all
    landed near dense while MoE, whose "rows" are functions of the input, beat
    dense by 0.0079. This module is the missing rung between them: rank 0 is the
    Engram, large rank with few rows is an MoE expert bank.

    Expressiveness per address is paid for in address count. At d20 with
    memory_dim=1280 (16 heads x 80) and query_dim=128, one row costs
    ``rank * (128 + 80)`` against 80 for a constant, so at fixed table bytes
    rank 1 has ~2.6x fewer rows and rank 4 has ~10x fewer.

    ``query_proj`` lives here, not on the Engram, so that share_memory shares the
    featurization along with the table. Per-layer projections into one shared
    table would send the same hidden state to unrelated maps at different depths
    -- a bigger table, not a shared memory (the same argument the discretizer
    gets in EngramConfig.share_memory).
    """

    def __init__(
        self,
        list_of_N: List[int],
        D: int,
        d_model: int,
        rank: int,
        query_dim: int,
        activation: str = "gelu",
        **weight_kwargs,
    ) -> None:
        super().__init__(list_of_N, D, **weight_kwargs)
        assert rank >= 1, "LowRankMemory needs rank >= 1; rank 0 is MultiHeadEmbedding"
        if activation not in _VALUE_ACTIVATIONS:
            raise ValueError(
                f"value_activation must be one of {sorted(_VALUE_ACTIVATIONS)}, "
                f"got {activation!r}"
            )
        self.rank = rank
        self.query_dim = query_dim
        self.activation = activation
        self.w_in = nn.Embedding(self.total_rows, rank * query_dim)
        self.w_out = nn.Embedding(self.total_rows, rank * D)
        self.query_proj = nn.Linear(d_model, query_dim, bias=False)

    def content_parameters(self) -> List[nn.Parameter]:
        # query_proj deliberately excluded: it is a dense matrix, so it belongs in
        # the Muon group and in the 6*nparams FLOP term like any other projection.
        return [self.w_in.weight, self.w_out.weight]

    def active_params_per_token(self) -> int:
        return self.num_heads * self.rank * (self.query_dim + self.embedding_dim)

    def read_flops_per_token(self) -> int:
        # Unlike a constant lookup, the gathered rows are multiplied: q through
        # w_in, then the rank-space activations through w_out.
        return 6 * self.active_params_per_token()

    def reset_content(self, ablation_mode: str) -> None:
        # Scales chosen so a read has ~unit per-element scale, matching the
        # normal(0, 1) rows of the constant table: rms_norm(h) is O(1) per
        # element, query_proj keeps q at O(1), then each matmul is normalized by
        # its fan-in.
        _init_addressed_rows(
            self.w_in.weight, ablation_mode, std=self.query_dim**-0.5
        )
        _init_addressed_rows(self.w_out.weight, ablation_mode, std=self.rank**-0.5)

    def split_key_value(self, rows: torch.Tensor):
        """No separate key: rank-k rows carry payload only."""
        return rows, None

    def forward(self, input_ids: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ids : torch.Tensor, shape [B, T, H], int64
        query : torch.Tensor, shape [B, T, d_model]
            Hidden state at this layer. Must already be aligned to input_ids.

        Returns
        -------
        torch.Tensor, shape [B, T, H, D]
        """
        assert query is not None, "LowRankMemory needs the hidden state"
        assert query.shape[:2] == input_ids.shape[:2], (
            f"query {tuple(query.shape[:2])} must be aligned to addresses "
            f"{tuple(input_ids.shape[:2])}"
        )
        shifted = self._shift(input_ids)
        # Scale-invariant: h's magnitude drifts a lot over training, and the read
        # should key on direction (same reasoning as discretize_normalize).
        q = self.query_proj(F.rms_norm(query, (query.size(-1),)))
        B, T, H = input_ids.shape
        w_in = self.w_in(shifted).view(B, T, H, self.rank, self.query_dim)
        hidden = torch.einsum("bthrq,btq->bthr", w_in, q.to(w_in.dtype))
        hidden = _VALUE_ACTIVATIONS[self.activation](hidden)
        w_out = self.w_out(shifted).view(B, T, H, self.rank, self.embedding_dim)
        out = torch.einsum("bthrd,bthr->bthd", w_out, hidden)
        return self._apply_row_weights(out, shifted)


def collect_engram_aux_loss(model):
    """Sum the per-layer discretizer auxiliary loss from the last forward.

    Only the VQ ("pq") source produces one -- a commitment term pulling the
    hidden state toward the codebook it selected. Returns a graph-connected
    scalar, or None for every other address source. Call once per forward,
    before backward. Mirrors moe.block.collect_aux_loss.
    """
    total = None
    for module in model.modules():
        if isinstance(module, Engram):
            aux = module.aux_loss()
            if aux is not None:
                total = aux if total is None else total + aux
    return total


@torch.no_grad()
def _init_addressed_rows(
    weight: torch.Tensor, ablation_mode: str, std: float
) -> None:
    """Initialize one table of addressed rows per the payload ablation."""
    if ablation_mode == "none":
        nn.init.normal_(weight, mean=0.0, std=std)
        weight.requires_grad_(True)
    elif ablation_mode == "randomize":
        nn.init.normal_(weight, mean=0.0, std=std)
        weight.requires_grad_(False)
    elif ablation_mode == "uniform":
        shared_row = torch.empty(
            weight.shape[1], device=weight.device, dtype=weight.dtype
        )
        nn.init.normal_(shared_row, mean=0.0, std=std)
        weight.copy_(shared_row.unsqueeze(0).expand_as(weight))
        weight.requires_grad_(False)
    else:
        raise ValueError(f"Unknown Engram ablation_mode={ablation_mode!r}")


def build_memory_table(
    cfg: EngramConfig, head_vocab_sizes: List[int], head_dim: int, d_model: int
) -> AddressedMemory:
    """The Engram memory table for this config: constant rows, or rank-k maps.

    One factory so the per-layer table (Engram.__init__) and the shared table
    (GPT._build_shared_engram_memory) cannot drift apart.
    """
    weight_kwargs = dict(
        whiten=cfg.readout_whiten,
        whiten_power=cfg.readout_whiten_power,
        whiten_decay=cfg.readout_whiten_decay,
        whiten_max=cfg.readout_whiten_max,
        count_gate=cfg.count_gate,
        count_gate_decay=cfg.count_gate_decay,
        count_gate_fourier=cfg.count_gate_fourier,
        count_gate_fourier_max_u=cfg.count_gate_fourier_max_u,
        count_gate_decouple=cfg.count_gate_decouple,
        # Heads are laid out order-major (all of order 2, then all of order 3),
        # so one n-gram order is a contiguous run of n_head_per_ngram heads.
        heads_per_group=cfg.n_head_per_ngram,
    )
    if cfg.value_rank <= 0:
        return MultiHeadEmbedding(
            list_of_N=head_vocab_sizes,
            D=head_dim,
            key_dim=cfg.key_dim,
            **weight_kwargs,
        )
    # key_dim is not wired into LowRankMemory: the rank axis measured 0.01x-0.09x
    # of its error bar (§8.4), so combining the two would spend an arm on a dead
    # variable. base_train rejects the combination rather than silently ignoring it.
    assert cfg.key_dim == 0, "key_dim with value_rank > 0 is not supported"
    return LowRankMemory(
        list_of_N=head_vocab_sizes,
        D=head_dim,
        d_model=d_model,
        rank=cfg.value_rank,
        query_dim=cfg.value_query_dim,
        activation=cfg.value_activation,
        **weight_kwargs,
    )


@torch.no_grad()
def init_memory_table(table: AddressedMemory, ablation_mode: str) -> None:
    """Initialize an Engram memory table according to the payload ablation.

    Lives outside Engram because under the shared-memory ablation one table
    serves several Engram layers and must be initialized exactly once.
    """
    if table.track_hit_rate:
        # Buffers come back as uninitialized garbage from the GPT's meta ->
        # to_empty(); starting at uniform also makes whitening a no-op on step 0
        # and ramp in over ~1/(1-decay) steps as the statistics accumulate.
        table.hit_rate.fill_(1.0)
    # Same to_empty() trap: row_alias must be re-set to identity so a fresh run
    # is byte-identical. A resumed MERGED run overwrites both via load_state_dict
    # (assign=True) after this, restoring the alias and _merged=True.
    table.row_alias.copy_(torch.arange(table.total_rows, device=table.row_alias.device))
    table._merged.fill_(False)
    table._merged_flag = False
    table._alias_by_device.clear()
    if table.count_gate:
        # Same trap: torch.zeros at construction does NOT survive to_empty(), and
        # zero is what makes the count gate start at exactly w = 1.0.
        table.count_gate_scale.zero_()
        table.count_gate_bias.zero_()
    if table.count_gate_decouple:
        # Same to_empty() trap: zero is what keeps the decoupled terms neutral.
        table.count_gate_freq_scale.zero_()
        table.count_gate_collide_scale.zero_()
    if table.count_gate_fourier > 0:
        table.count_gate_fourier_coef.zero_()
        # w_k = pi * 2^k / max_u: band 0 is half a period across the log-count
        # range, each band halves the wavelength.
        k = torch.arange(table.count_gate_fourier, dtype=torch.float32)
        table.count_gate_freqs.copy_(
            (math.pi * (2.0**k) / table.count_gate_fourier_max_u).to(
                table.count_gate_freqs.device
            )
        )
    if isinstance(table, LowRankMemory):
        init_bound = math.sqrt(3.0) * (table.query_proj.in_features**-0.5)
        nn.init.uniform_(table.query_proj.weight, -init_bound, init_bound)
    table.reset_content(ablation_mode)


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
        shared_embedding: "MultiHeadEmbedding | None" = None,
        shared_discretizer=None,
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
        # Under the shared-memory ablation this resolves to the canonical layer's
        # primes, so every layer's table has identical geometry and can be the
        # same tensor.
        prime_layer_id = self.hasher.resolve_layer_id(layer_id)
        head_vocab_sizes: List[int] = []
        for ngram_idx in range(len(self.hasher.ngram_orders)):
            for head_idx in range(cfg.n_head_per_ngram):
                head_vocab_sizes.append(
                    self.hasher.prime_table[prime_layer_id][ngram_idx][head_idx]
                )

        # --- Embedding dim per head ---
        assert cfg.memory_dim % num_heads == 0, (
            f"memory_dim ({cfg.memory_dim}) must be divisible by num_heads ({num_heads})"
        )
        head_dim = cfg.memory_dim // num_heads

        engram_hidden_size = cfg.memory_dim  # = num_heads * head_dim
        # With key_dim > 0 the gate is computed from the rows' own keys, which are
        # a narrower tensor than the payload, so its projection is sized to that.
        # key_dim=0 keeps the paper's shape: gate and payload from the same read.
        gate_input_size = (
            num_heads * cfg.key_dim if cfg.key_dim > 0 else engram_hidden_size
        )

        # ablation_mode="mlp" is the capacity / FLOP-matched control: it keeps the
        # full Engram branch machinery (value_proj, gate, short conv, mHC routing)
        # but removes all n-gram addressing. The payload is a learned projection of
        # the hidden state instead of a memory lookup, so there is no memory table
        # at all. Comparing it to real (content+addressing) and randomize/uniform
        # (addressing only) isolates how much of the lift is just the added gated
        # branch capacity vs the n-gram lookup itself.
        # --- Address source (continuous Engram) ---
        # "tokens": hash token IDs (paper default, unchanged).
        # "lsh"/"pq": discretize the hidden state to a code, then hash exactly as
        #   before -- the n-gram mixing, primes and table are all reused.
        # "pkm": replace hash+lookup with a differentiable product-key read.
        self.address_source = cfg.address_source
        self.is_pkm = cfg.address_source == "pkm"
        self._own_discretizer = None
        self._shared_discretizer: list = []
        self.product_key = None
        if self.is_pkm:
            from nano_scalemb.product_key import ProductKeyMemory

            self.product_key = ProductKeyMemory(
                d_model,
                num_heads=num_heads,
                value_dim=cfg.memory_dim // num_heads,
                n_keys=cfg.pkm_n_keys,
                topk=cfg.pkm_topk,
                query_dim=cfg.pkm_query_dim,
                seed=cfg.seed + 1009 * layer_id,
                normalize=cfg.discretize_normalize,
            )
        elif cfg.address_source != "tokens":
            if shared_discretizer is not None:
                # Owned by the GPT (share_memory): held by reference so its
                # buffers are registered exactly once, same reasoning as the
                # shared memory table.
                assert cfg.share_memory, "shared_discretizer requires share_memory"
                self._shared_discretizer = [shared_discretizer]
            else:
                from nano_scalemb.discretize import build_discretizer

                self._own_discretizer = build_discretizer(
                    cfg, d_model, layer_id, num_hash_heads=num_heads
                )

        self.is_mlp_control = cfg.ablation_mode == "mlp"
        # Under the shared-memory ablation the table is owned by the GPT and held
        # here by reference in a plain list, NOT registered as a submodule.
        # Registering it on every Engram would give the state_dict one key per
        # layer, and load_state_dict(assign=True) (see checkpoint_manager) would
        # then hand each key its own tensor and silently unshare the memory.
        self._shared_embedding: List[MultiHeadEmbedding] = []
        if self.is_mlp_control:
            self.multi_head_embedding = None
            self.payload_proj = nn.Linear(d_model, engram_hidden_size, bias=False)
        elif self.is_pkm:
            # The product-key memory owns its own value rows; a hash table here
            # would be dead weight that never receives gradient (and would trip
            # the optimizer's param-count assert).
            self.multi_head_embedding = None
        elif shared_embedding is not None:
            assert cfg.share_memory, "shared_embedding requires share_memory"
            assert shared_embedding.embedding_dim == head_dim
            self.multi_head_embedding = None
            self._shared_embedding = [shared_embedding]
        else:
            self.multi_head_embedding = build_memory_table(
                cfg, head_vocab_sizes, head_dim, d_model
            )

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
                gate_input_size, self.mhc_num_streams * d_model, bias=False
            )
        else:
            # Dense fallback: single-stream gate (paper Eq. 4) used when the
            # backbone does not run mHC.
            self.key_proj = nn.Linear(gate_input_size, d_model, bias=False)
            self.norm_query = nn.RMSNorm(d_model)
            self.norm_key = nn.RMSNorm(d_model)

    @property
    def discretizer(self):
        """The discretizer this layer addresses through: its own, or the GPT's
        shared one (share_memory), or None for token/pkm addressing."""
        if self._own_discretizer is not None:
            return self._own_discretizer
        return self._shared_discretizer[0] if self._shared_discretizer else None

    @property
    def owns_discretizer(self) -> bool:
        return self._own_discretizer is not None

    @property
    def memory_table(self) -> "MultiHeadEmbedding | None":
        """The table this layer reads: its own, or the shared one, or None
        (mlp control). Use this instead of ``multi_head_embedding``, which is
        None whenever the table is shared."""
        if self.multi_head_embedding is not None:
            return self.multi_head_embedding
        return self._shared_embedding[0] if self._shared_embedding else None

    @property
    def owns_memory_table(self) -> bool:
        """True when this layer's table is its own to initialize and optimize."""
        return self.multi_head_embedding is not None

    def _compute_embeddings(
        self,
        input_ids: torch.Tensor,
        compressed_input_ids: torch.Tensor | None = None,
        hidden: torch.Tensor | None = None,
    ) -> "Tuple[torch.Tensor, torch.Tensor | None]":
        """Returns (values, keys). keys is None unless EngramConfig.key_dim > 0."""
        if self.address_source == "hybrid":
            assert hidden is not None, "hybrid addressing needs the hidden state"
            tok = compressed_input_ids if compressed_input_ids is not None else input_ids
            if tok.size(1) != hidden.size(1):
                tok = tok[:, -hidden.size(1):]
            lsh = self.discretizer(hidden)
            if lsh.ndim == 3:
                lsh = lsh[..., 0]
            n_orders = len(self.hasher.ngram_orders)
            K = self.cfg.n_head_per_ngram
            n_tok = max(1, min(K - 1, round(self.cfg.hybrid_token_head_frac * K)))
            codes = lsh.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, n_orders, K).clone()
            codes[..., :n_tok] = tok.unsqueeze(-1).unsqueeze(-1)
            codes = codes.reshape(codes.size(0), codes.size(1), n_orders * K)
            hashes = self.hasher.hash(
                codes, self.layer_id, compressed_input_ids=codes
            )
        elif self.discretizer is not None:
            # Continuous addressing: discretize the hidden state to a code and
            # feed it through the SAME hash path the token codes use, so the
            # n-gram mixing, primes and table are all shared with the baseline.
            assert hidden is not None, "contextual addressing needs the hidden state"
            codes = self.discretizer(hidden)
            hashes = self.hasher.hash(
                codes, self.layer_id, compressed_input_ids=codes
            )
        else:
            hashes = self.hasher.hash(
                input_ids,
                self.layer_id,
                compressed_input_ids=compressed_input_ids,
            )
        table = self.memory_table
        if isinstance(table, LowRankMemory):
            # The token path can hash a longer context than the hidden state
            # covers (the caller slices afterwards), but a rank-k read has to
            # multiply each row by the *matching* position's query, so align the
            # addresses first and let _align_embeddings_to_hidden no-op.
            assert hidden is not None, "value_rank > 0 needs the hidden state"
            if hashes.size(1) != hidden.size(1):
                hashes = hashes[:, -hidden.size(1) :]
            rows = table(hashes, hidden)
        else:
            rows = table(hashes)
        values, keys = table.split_key_value(rows)
        return (
            values.flatten(start_dim=-2),
            None if keys is None else keys.flatten(start_dim=-2),
        )

    def aux_loss(self):
        """Discretizer auxiliary loss (VQ commitment), or None."""
        return self.discretizer.aux_loss() if self.discretizer is not None else None

    def _align_embeddings_to_hidden(
        self, x: torch.Tensor, embeddings: torch.Tensor
    ) -> torch.Tensor:
        if embeddings is None:
            return None
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
        self, x: torch.Tensor, embeddings: torch.Tensor, keys: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Dense single-stream gate (paper Eq. 4/5), used without backbone mHC."""
        # With key_dim > 0 the gate reads a row's OWN key instead of a shared
        # linear function of its value (EngramConfig.key_dim).
        key = self.key_proj(embeddings if keys is None else keys)
        normed_query = self.norm_query(x)
        normed_key = self.norm_key(key)
        gate = (normed_query * normed_key).sum(dim=-1, keepdim=True)
        gate = gate / math.sqrt(self.d_model)
        gate = torch.sigmoid(gate)
        value_tilde = gate * self.value_proj(embeddings)
        return self.short_conv(value_tilde)

    def _mhc_gated_values(
        self, x: torch.Tensor, embeddings: torch.Tensor, keys: torch.Tensor | None = None
    ) -> torch.Tensor:
        query = self.stream_query_norm(x).unsqueeze(2)  # [B, T, 1, D]
        # With key_dim > 0 the gate reads a row's OWN key instead of a shared
        # linear function of its value (EngramConfig.key_dim).
        gate_src = embeddings if keys is None else keys
        keys = self.stream_key_proj(gate_src).view(
            gate_src.size(0), gate_src.size(1), self.mhc_num_streams, self.d_model
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
        row_keys = None
        if self.is_mlp_control:
            embeddings = self.payload_proj(x)
        elif self.is_pkm:
            # Product-key memory reads directly from the hidden state; there is
            # no hash and no alignment step (it is already per-position).
            embeddings = self.product_key(x)
        else:
            embeddings, row_keys = self._compute_embeddings(
                input_ids, compressed_input_ids=compressed_input_ids, hidden=x
            )
            embeddings = self._align_embeddings_to_hidden(x, embeddings)
            row_keys = self._align_embeddings_to_hidden(x, row_keys)
        return self._dense_forward(x, embeddings, row_keys)

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
        row_keys = None
        if self.is_mlp_control:
            embeddings = self.payload_proj(x)
        elif self.is_pkm:
            # Product-key memory reads directly from the hidden state; there is
            # no hash and no alignment step (it is already per-position).
            embeddings = self.product_key(x)
        else:
            embeddings, row_keys = self._compute_embeddings(
                input_ids, compressed_input_ids=compressed_input_ids, hidden=x
            )
            embeddings = self._align_embeddings_to_hidden(x, embeddings)
            row_keys = self._align_embeddings_to_hidden(x, row_keys)
        gated = self._mhc_gated_values(x, embeddings, row_keys)
        return self._short_conv_streams(gated)

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def init_weights(self) -> None:
        init_bound = math.sqrt(3.0) * (self.d_model**-0.5)
        # Discretizer/PKM state lives in buffers, which come back as
        # uninitialized garbage from the GPT's meta -> to_empty(). Without this
        # the PQ codebook starts at ~1e38 and the commitment loss explodes.
        if self.owns_discretizer:
            # A shared discretizer is initialized once by the GPT instead.
            self._own_discretizer.reset_parameters()
        if self.product_key is not None:
            self.product_key.reset_parameters()
        if self.is_mlp_control:
            # Capacity-only control: learned hidden-state projection, no table.
            nn.init.uniform_(self.payload_proj.weight, -init_bound, init_bound)
        elif self.owns_memory_table:
            # A shared table is initialized once by the GPT instead.
            init_memory_table(self.multi_head_embedding, self.cfg.ablation_mode)
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
