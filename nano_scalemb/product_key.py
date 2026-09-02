"""Product-key memory: differentiable nearest-key addressing.

Lample et al., "Large Memory Layers with Product Keys" (arXiv:1907.05242).

The third point on the addressing ladder. LSH and PQ both quantize the hidden
state to a hard integer code and then look the row up -- addressing is not
differentiable, and gradient reaches it only through a straight-through
estimator (PQ) or not at all (LSH). Product-key memory instead keeps a set of
learned keys, takes the top-k nearest, and reads a *softmax-weighted mixture* of
their values. The whole address path is differentiable.

The product trick makes that affordable. A flat memory of N rows needs N
key-dot-products per token. Factor the key set as a cartesian product of two
half-key sets of size sqrt(N), and the top-k over the full product is
recoverable from the top-k of each half: 2*sqrt(N) dot products instead of N.
For N = 10^6 that is 2000 instead of a million.

Note this changes the *read* as well as the address -- a soft mixture of k rows
rather than one hard row per head -- so it is the furthest departure from the
Engram design of the three. Comparisons against LSH/PQ therefore confound
"learned addressing" with "soft read"; the honest control for that is the LSH
arm, which changes only the address source.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProductKeyMemory(nn.Module):
    """Multi-head product-key memory.

    Emits ``[B, T, num_heads * value_dim]``, matching the flattened layout of
    ``MultiHeadEmbedding``, so it is a drop-in replacement for the
    hash-and-lookup path inside the Engram.

    Parameters
    ----------
    d_model : hidden width of the backbone.
    num_heads : independent memory heads (each with its own query and read).
    value_dim : width of each head's value rows; num_heads * value_dim must
        equal the Engram's memory_dim.
    n_keys : sub-keys per half. The memory holds n_keys**2 rows per head.
    topk : rows mixed per head per token.
    query_dim : width of the query/key space; split in half for the two
        sub-key sets, so it must be even.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        value_dim: int,
        n_keys: int = 512,
        topk: int = 32,
        query_dim: int = 256,
        seed: int = 0,
        normalize: bool = True,
    ):
        super().__init__()
        if query_dim % 2 != 0:
            raise ValueError(f"query_dim must be even, got {query_dim}")
        if topk > n_keys * n_keys:
            raise ValueError(f"topk ({topk}) exceeds memory size ({n_keys**2})")
        self.d_model = d_model
        self.num_heads = num_heads
        self.value_dim = value_dim
        self.n_keys = n_keys
        self.topk = topk
        self.query_dim = query_dim
        self.half_dim = query_dim // 2
        self.n_values = n_keys * n_keys
        self.seed = seed
        self.normalize = normalize

        # One query projection per head, produced in a single matmul.
        self.query_proj = nn.Linear(d_model, num_heads * query_dim, bias=False)
        # Two half-key sets per head: [H, 2, n_keys, half_dim].
        self.keys = nn.Parameter(torch.zeros(num_heads, 2, n_keys, self.half_dim))
        # Value rows per head. This is the memory proper.
        self.values = nn.Parameter(torch.zeros(num_heads, self.n_values, value_dim))

    @torch.no_grad()
    def reset_parameters(self):
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        bound = self.d_model**-0.5
        w = torch.empty(self.query_proj.weight.shape).uniform_(-bound, bound, generator=gen)
        self.query_proj.weight.copy_(w.to(self.query_proj.weight.device))
        # Keys on the unit-ish scale of the half query space so initial
        # dot-products are O(1) and the top-k is not dominated by norm.
        kb = self.half_dim**-0.5
        k = torch.empty(self.keys.shape).uniform_(-kb, kb, generator=gen)
        self.keys.copy_(k.to(self.keys.device))
        # Zero values: like the Engram's zero-init value_proj and the MoE's
        # zero-init w_out, the branch contributes nothing at init.
        self.values.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, d_model] -> [B, T, num_heads * value_dim]."""
        B, T, _ = x.shape
        H, K, R = self.num_heads, self.topk, self.n_keys

        h = F.rms_norm(x, (x.size(-1),)) if self.normalize else x
        q = self.query_proj(h).view(B * T, H, 2, self.half_dim).float()
        keys = self.keys.float()  # [H, 2, R, half_dim]

        # Scores against each half-key set: [B*T, H, 2, R]
        scores = torch.einsum("nhsd,hsrd->nhsr", q, keys) / math.sqrt(self.half_dim)
        # Top-k within each half independently.
        s1, i1 = scores[:, :, 0].topk(K, dim=-1)  # [B*T, H, K]
        s2, i2 = scores[:, :, 1].topk(K, dim=-1)

        # The K^2 cartesian candidates; the true global top-k over the full R^2
        # product is guaranteed to lie inside this set (Lample et al. §3.1).
        cand = s1.unsqueeze(-1) + s2.unsqueeze(-2)  # [B*T, H, K, K]
        cand_idx = i1.unsqueeze(-1) * R + i2.unsqueeze(-2)  # row id in [0, R^2)
        best, flat_pos = cand.view(B * T, H, -1).topk(K, dim=-1)  # [B*T, H, K]
        rows = torch.gather(cand_idx.view(B * T, H, -1), -1, flat_pos)

        weights = F.softmax(best, dim=-1).to(self.values.dtype)  # [B*T, H, K]

        # Read the selected rows via flat indexing into [H*n_values, value_dim].
        #
        # Deliberately NOT gather() on `values.unsqueeze(1).expand(H, B*T, ...)`:
        # that relies on gather never materializing a stride-0 expanded source,
        # and if it ever did the tensor would be H*B*T*n_values*value_dim --
        # tens of terabytes at real sizes. Flat indexing only ever touches the
        # K rows actually selected.
        flat_values = self.values.reshape(H * self.n_values, self.value_dim)
        head_offset = (
            torch.arange(H, device=rows.device).view(1, H, 1) * self.n_values
        )
        gathered = flat_values[rows + head_offset]  # [B*T, H, K, value_dim]
        out = (gathered * weights.unsqueeze(-1)).sum(dim=2)  # [B*T, H, value_dim]
        return out.reshape(B, T, H * self.value_dim)
