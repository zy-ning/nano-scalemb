"""Hidden-state discretizers: continuous Engram addressing.

The token Engram addresses its memory by hashing raw token IDs, so it can only
remember *surface form*: "New York" and "NYC" land on unrelated rows. These
discretizers replace the address source with a code derived from the layer's
hidden state, so the memory is addressed by what the model currently
*represents* rather than by which token it read.

Every discretizer maps [B, T, D] float -> [B, T] int64 codes, which drop
straight into ``NgramHasher.hash(..., compressed_input_ids=codes)``. That seam
already exists for tokenizer compression, so the n-gram mixing, prime tables,
multi-head embedding, gate and short conv are all reused untouched, and the
token path stays byte-identical.

Two invariants every implementation must hold:

1. **Codes fit in 32 bits.** The hasher computes ``code * multiplier`` in int64
   with multipliers up to 2^31; a 33-bit code silently wraps negative and
   corrupts every address. Enforced by ``_assert_code_width``.
2. **Codes are non-negative**, since they are used with ``torch.remainder`` and
   as a pad sentinel.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# The hasher multiplies codes by odd multipliers < 2^31 in int64. Leave a bit of
# headroom below the 63-bit signed limit.
MAX_CODE_BITS = 31


def _assert_code_width(num_codes: int, label: str) -> None:
    if num_codes > (1 << MAX_CODE_BITS):
        raise ValueError(
            f"{label} produces {num_codes} distinct codes, which needs more than "
            f"{MAX_CODE_BITS} bits. NgramHasher multiplies codes by ~2^31 in "
            f"int64, so wider codes overflow and silently corrupt addressing. "
            f"Reduce the code width."
        )


class Discretizer(nn.Module):
    """Base class: hidden states -> integer codes.

    ``num_codes`` is the size of the code alphabet, i.e. the effective
    "vocabulary" the hasher sees in place of the tokenizer's.
    """

    num_codes: int

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D] -> [B, T] int64 in [0, num_codes)."""
        raise NotImplementedError

    def aux_loss(self):
        """Optional loss term (e.g. VQ commitment). None if unused."""
        return None


class LSHDiscretizer(Discretizer):
    """Random-projection LSH: sign bits of a frozen random projection.

    ``code = sum_i 1[<w_i, x> > 0] << i`` over ``n_bits`` random hyperplanes.

    Deliberately has **no learned parameters**: this is the control that isolates
    "address by hidden state" from "learn the addressing". Compared against the
    token Engram it changes only the address *source*, holding fixed-addressing /
    learned-contents constant -- so a win here is evidence about semantics vs
    surface form, not about learnable addressing.

    The projection is a non-persistent buffer regenerated from ``seed`` in
    ``reset_parameters``, so it survives meta-device init and is identical across
    DDP ranks without being carried in the checkpoint.
    """

    def __init__(
        self,
        d_model: int,
        n_bits: int = 20,
        seed: int = 0,
        normalize: bool = True,
        n_latents: int = 1,
        balance: bool = False,
        balance_decay: float = 0.99,
    ):
        super().__init__()
        if not 1 <= n_bits <= MAX_CODE_BITS:
            raise ValueError(f"n_bits must be in [1, {MAX_CODE_BITS}], got {n_bits}")
        self.d_model = d_model
        self.n_bits = n_bits
        self.seed = seed
        # Scale invariance: the code should depend on the direction of the hidden
        # state, not its magnitude, which drifts a lot over training.
        self.normalize = normalize
        # n_latents > 1 emits one INDEPENDENT code per hash head, so the memory
        # read is conditioned on n_latents * n_bits of the hidden state instead
        # of the same n_bits broadcast to every head.
        self.n_latents = n_latents
        # balance=True: threshold each bit at an EMA of its own mean projection
        # instead of at 0. sign(<w,h>) is only an unbiased bit if <w,h> has zero
        # median; on real (shifted, correlated) hidden states many bits are
        # almost always 0 or 1, which throws away most of the nominal code width.
        # Measured: LSH realized only 9.2-9.8 of 15 bits. Re-centering each bit
        # maximizes its marginal entropy for free.
        self.balance = balance
        self.balance_decay = balance_decay
        self.num_codes = 1 << n_bits
        _assert_code_width(self.num_codes, f"LSH(n_bits={n_bits})")
        self.register_buffer(
            "projection", torch.zeros(n_latents, d_model, n_bits), persistent=False
        )
        self.register_buffer(
            "bit_weights", (1 << torch.arange(n_bits, dtype=torch.int64)), persistent=False
        )
        # Persistent: the thresholds ARE the addressing scheme, so they must
        # round-trip or a resumed run would re-address every slot.
        self.register_buffer("bit_threshold", torch.zeros(n_latents, n_bits))

    @torch.no_grad()
    def reset_parameters(self):
        self.bit_threshold.zero_()
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        w = torch.randn(self.n_latents, self.d_model, self.n_bits, generator=gen)
        self.projection.copy_(w.to(self.projection.device))
        self.bit_weights.copy_(
            (1 << torch.arange(self.n_bits, dtype=torch.int64)).to(self.bit_weights.device)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.rms_norm(x, (x.size(-1),)) if self.normalize else x
        # [B, T, d] x [L, d, bits] -> [B, T, L, bits]
        proj = torch.einsum("btd,ldk->btlk", h.float(), self.projection.float())
        if self.balance:
            if self.training:
                with torch.no_grad():
                    mean = proj.mean(dim=(0, 1))  # [L, bits]
                    import torch.distributed as _d
                    if _d.is_available() and _d.is_initialized():
                        _d.all_reduce(mean, op=_d.ReduceOp.AVG)
                    self.bit_threshold.mul_(self.balance_decay).add_(
                        mean, alpha=1 - self.balance_decay
                    )
            proj = proj - self.bit_threshold
        bits = (proj > 0).to(torch.int64)
        codes = (bits * self.bit_weights).sum(dim=-1)  # [B, T, L]
        return codes.squeeze(-1) if self.n_latents == 1 else codes


class PQDiscretizer(Discretizer):
    """Product-quantization VQ: learned codebooks over disjoint sub-spaces.

    The hidden state is split into ``n_subspaces`` chunks; each is assigned to
    its nearest of ``codebook_size`` learned centroids, and the sub-codes are
    packed into one integer (mixed radix).

    Product quantization rather than plain VQ because the Engram's tables have
    ~10^5-10^6 rows: a flat codebook that large would make nearest-neighbour
    search cost more than the rest of the model. PQ gets an alphabet of
    ``codebook_size ** n_subspaces`` from only ``n_subspaces * codebook_size``
    distance computations.

    Codebooks are updated by EMA of the assigned inputs (van den Oord et al.),
    not by gradient descent -- EMA is markedly more stable when the input
    distribution is still moving, which it very much is early in training. The
    encoder gets gradient through a straight-through estimator plus a commitment
    loss.
    """

    def __init__(
        self,
        d_model: int,
        n_subspaces: int = 4,
        codebook_size: int = 256,
        seed: int = 0,
        decay: float = 0.99,
        commitment_weight: float = 0.25,
        eps: float = 1e-5,
        normalize: bool = True,
        per_head: bool = False,
        detach_encoder: bool = False,
    ):
        super().__init__()
        if d_model % n_subspaces != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_subspaces ({n_subspaces})"
            )
        self.d_model = d_model
        self.n_subspaces = n_subspaces
        self.codebook_size = codebook_size
        self.sub_dim = d_model // n_subspaces
        self.seed = seed
        self.decay = decay
        self.commitment_weight = commitment_weight
        self.eps = eps
        self.normalize = normalize
        # per_head=True: each sub-space's code addresses its own hash head, so
        # the codes are never combined and the 31-bit ceiling does not apply.
        # That is what makes many small sub-spaces affordable -- and many small
        # sub-spaces is the regime where k-means is actually meaningful. Packing
        # sub-codes into one integer (per_head=False) caps n_subspaces at
        # log(2^31)/log(codebook_size), which forced a very coarse quantizer.
        self.per_head = per_head
        # detach_encoder: address off a stop-gradient copy of the hidden state
        # and emit no commitment loss. The commitment term pulls the hidden
        # state toward its centroid, and applied to a live residual stream it
        # collapsed the representation to ~1 effective dimension (measured:
        # eff_dim 1/1280 vs 232/1280 for LSH on the same layers). Detaching
        # makes quantization purely an addressing device.
        self.detach_encoder = detach_encoder
        self.num_codes = codebook_size if per_head else codebook_size**n_subspaces
        if not per_head:
            _assert_code_width(
                self.num_codes,
                f"PQ(n_subspaces={n_subspaces}, codebook_size={codebook_size})",
            )

        # Codebooks and EMA state are buffers, not Parameters: updated by EMA in
        # no_grad, so handing them to the optimizer would double-update them.
        # Persistent so they round-trip through checkpoints.
        self.register_buffer("codebook", torch.zeros(n_subspaces, codebook_size, self.sub_dim))
        self.register_buffer("ema_cluster_size", torch.zeros(n_subspaces, codebook_size))
        self.register_buffer("ema_weight", torch.zeros(n_subspaces, codebook_size, self.sub_dim))
        # Mixed-radix place values: sub-code s contributes code_s * B^s.
        self.register_buffer(
            "radix",
            codebook_size ** torch.arange(n_subspaces, dtype=torch.int64),
            persistent=False,
        )
        # Persistent: whether the codebook has been seeded from real data. Must
        # round-trip, or resuming a checkpoint would re-init from the first
        # post-resume batch and throw away the learned codebook.
        self.register_buffer("_initialized", torch.zeros((), dtype=torch.bool))
        self._commitment_loss = None

    @torch.no_grad()
    def reset_parameters(self):
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        init = torch.randn(
            self.n_subspaces, self.codebook_size, self.sub_dim, generator=gen
        )
        self.codebook.copy_(init.to(self.codebook.device))
        self.ema_cluster_size.zero_()
        self.ema_weight.copy_(self.codebook)
        self._initialized.fill_(False)
        self.radix.copy_(
            (self.codebook_size ** torch.arange(self.n_subspaces, dtype=torch.int64)).to(
                self.radix.device
            )
        )
        self._commitment_loss = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        # Clear first: a stale loss from a previous forward could otherwise be
        # collected and added to the backward loss a second time.
        self._commitment_loss = None
        if self.detach_encoder:
            x = x.detach()
        h = F.rms_norm(x, (x.size(-1),)) if self.normalize else x
        # [B, T, S, sub_dim] -> [S, B*T, sub_dim]
        flat = h.reshape(B * T, self.n_subspaces, self.sub_dim).transpose(0, 1).float()
        cb = self.codebook.float()  # [S, K, sub_dim]

        # ||f - c||^2 = ||f||^2 - 2 f.c + ||c||^2; the ||f||^2 term is constant
        # per row, so it does not affect the argmin and is dropped.
        dist = (
            flat.pow(2).sum(-1, keepdim=True)
            - 2 * torch.bmm(flat, cb.transpose(1, 2))
            + cb.pow(2).sum(-1).unsqueeze(1)
        )  # [S, B*T, K]
        assign = dist.argmin(dim=-1)  # [S, B*T]

        if self.training and not self.detach_encoder:
            if not bool(self._initialized):
                # Data-dependent init: seed the codebook from real hidden states
                # on the first training batch. Gaussian-init centroids sit far
                # from the (RMS-normed, structured) activations, which makes the
                # commitment loss enormous and strands most centroids unused.
                self._init_from_data(flat)
                dist_ = (
                    flat.pow(2).sum(-1, keepdim=True)
                    - 2 * torch.bmm(flat, self.codebook.float().transpose(1, 2))
                    + self.codebook.float().pow(2).sum(-1).unsqueeze(1)
                )
                assign = dist_.argmin(dim=-1)
                cb = self.codebook.float()
            self._ema_update(flat, assign)
            quantized = torch.gather(
                cb, 1, assign.unsqueeze(-1).expand(-1, -1, self.sub_dim)
            )
            # Commitment: pull the encoder output toward the codebook it chose.
            # Only this direction -- the codebook side is handled by EMA.
            self._commitment_loss = self.commitment_weight * F.mse_loss(
                flat, quantized.detach()
            )
        elif self.training:
            # Detached: still adapt the codebook to the (moving) hidden states,
            # but never push back on the representation.
            if not bool(self._initialized):
                self._init_from_data(flat)
                cbf = self.codebook.float()
                assign = (
                    flat.pow(2).sum(-1, keepdim=True)
                    - 2 * torch.bmm(flat, cbf.transpose(1, 2))
                    + cbf.pow(2).sum(-1).unsqueeze(1)
                ).argmin(dim=-1)
            self._ema_update(flat, assign)

        if self.per_head:
            # [S, B*T] -> [B, T, S]: one independent code per hash head.
            return assign.transpose(0, 1).reshape(B, T, self.n_subspaces)
        # [S, B*T] -> [B, T] via mixed radix.
        codes = (assign.transpose(0, 1) * self.radix.unsqueeze(0)).sum(dim=-1)
        return codes.reshape(B, T)

    @torch.no_grad()
    def _init_from_data(self, flat: torch.Tensor) -> None:
        """Seed each sub-codebook from randomly chosen real sub-vectors.

        Sampling actual activations (rather than k-means) keeps this cheap and
        distribution-matched, which is what matters -- the EMA updates refine
        the centroids from there. Under DDP only rank 0 samples and broadcasts,
        so every rank starts from the same codebook.
        """
        import torch.distributed as dist

        n_rows = flat.size(1)
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        for s in range(self.n_subspaces):
            idx = torch.randint(0, n_rows, (self.codebook_size,), generator=gen)
            self.codebook[s].copy_(flat[s, idx.to(flat.device)].to(self.codebook.dtype))
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(self.codebook, src=0)
        self.ema_weight.copy_(self.codebook)
        self.ema_cluster_size.fill_(1.0)
        self._initialized.fill_(True)

    @torch.no_grad()
    def _ema_update(self, flat: torch.Tensor, assign: torch.Tensor) -> None:
        """EMA codebook update (van den Oord et al. Appendix A.1).

        Under DDP the counts and sums are all-reduced first, so every rank holds
        an identical codebook -- the same reasoning as the MoE loss-free
        balancer's bias buffer. Without this the ranks would silently drift into
        different addressing schemes.
        """
        import torch.distributed as dist

        onehot = F.one_hot(assign, self.codebook_size).to(flat.dtype)  # [S, B*T, K]
        counts = onehot.sum(dim=1)  # [S, K]
        sums = torch.bmm(onehot.transpose(1, 2), flat)  # [S, K, sub_dim]

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            dist.all_reduce(sums, op=dist.ReduceOp.SUM)

        self.ema_cluster_size.mul_(self.decay).add_(counts, alpha=1 - self.decay)
        self.ema_weight.mul_(self.decay).add_(sums, alpha=1 - self.decay)

        # Laplace smoothing so a momentarily unused centroid does not divide by 0
        # and vanish permanently.
        n = self.ema_cluster_size.sum(dim=-1, keepdim=True)
        smoothed = (
            (self.ema_cluster_size + self.eps)
            / (n + self.codebook_size * self.eps)
            * n
        )
        self.codebook.copy_(self.ema_weight / smoothed.unsqueeze(-1))

    def aux_loss(self):
        return self._commitment_loss

    @torch.no_grad()
    def codebook_usage(self) -> float:
        """Fraction of centroids with non-negligible EMA mass.

        The headline diagnostic for VQ: collapse (usage -> 1/K) means the memory
        is effectively addressed by a handful of codes no matter how big the
        table is.
        """
        used = (self.ema_cluster_size > self.eps).float().sum()
        return (used / self.ema_cluster_size.numel()).item()


def code_alphabet_size(cfg) -> int:
    """Number of distinct codes the configured discretizer can emit.

    Used by NgramHasher to size the hash tables when addressing is contextual.
    """
    if cfg.address_source in ("lsh", "hybrid"):
        return 1 << cfg.lsh_bits
    if cfg.address_source == "pq":
        # per-head codes are never combined, so the alphabet is one codebook
        return (
            cfg.pq_codebook_size
            if cfg.address_latents_per_head
            else cfg.pq_codebook_size**cfg.pq_subspaces
        )
    raise ValueError(f"address_source={cfg.address_source!r} has no code alphabet")


def build_discretizer(
    cfg, d_model: int, layer_id: int, num_hash_heads: int = 1
) -> Discretizer | None:
    """Construct the discretizer named by ``cfg.address_source``.

    The per-layer seed offset gives each Engram layer a different projection /
    codebook init, matching how the token path gives each layer different hash
    primes (unless memory is shared, in which case the caller passes a shared
    instance instead).
    """
    source = cfg.address_source
    if source == "tokens":
        return None
    seed = cfg.seed + 1009 * layer_id
    # address_latents_per_head: emit one independent code per hash head, so the
    # read is conditioned on num_heads x code_bits instead of the same bits
    # broadcast to every head. num_heads is passed in by the Engram.
    n_latents = num_hash_heads if cfg.address_latents_per_head else 1
    if source in ("lsh", "hybrid"):
        return LSHDiscretizer(
            d_model,
            n_bits=cfg.lsh_bits,
            seed=seed,
            normalize=cfg.discretize_normalize,
            n_latents=n_latents,
            balance=cfg.lsh_balance,
        )
    if source == "pq":
        # With per-head codes the sub-space count IS the head count, so each
        # sub-space is d_model/num_heads wide -- small enough for k-means to mean
        # something, which the packed variant could never afford.
        return PQDiscretizer(
            d_model,
            n_subspaces=num_hash_heads if cfg.address_latents_per_head else cfg.pq_subspaces,
            codebook_size=cfg.pq_codebook_size,
            seed=seed,
            decay=cfg.pq_decay,
            commitment_weight=cfg.pq_commitment_weight,
            normalize=cfg.discretize_normalize,
            per_head=cfg.address_latents_per_head,
            detach_encoder=cfg.pq_detach_encoder,
        )
    raise ValueError(f"address_source={source!r} has no discretizer")
