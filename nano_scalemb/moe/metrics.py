"""Router instrumentation: regret, load violation, entropy.

Adapted from the sibling harness ``nanochat-moe`` (``moe/metrics.py``).

- regret  = (free_topk_score - chosen_score) / T   [quality cost of balancing]
            computed on the RAW gate score s.
- MaxVio  : f_j = count_j / total * n - 1; MaxVio = max_j f_j. 0 == balanced.

Every function here syncs to host, so the router only calls them on log steps
(``Router.collect_stats``); an every-step ``.item()`` serializes the GPU pipeline
and breaks torch.compile tracing.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RouterStats:
    regret: float
    max_vio: float
    min_vio: float
    avg_vio: float
    entropy: float  # load-distribution entropy (nats)
    counts: torch.Tensor  # (n_experts,) long, on cpu
    active_experts: float = 0.0  # mean #active experts/token (== top_k for hard top-k)

    def as_dict(self, prefix: str = "") -> dict:
        return {
            f"{prefix}regret": self.regret,
            f"{prefix}max_vio": self.max_vio,
            f"{prefix}min_vio": self.min_vio,
            f"{prefix}avg_vio": self.avg_vio,
            f"{prefix}entropy": self.entropy,
            f"{prefix}active_experts": self.active_experts,
        }


@torch.no_grad()
def compute_regret(s: torch.Tensor, k_idx: torch.Tensor) -> float:
    """s: (T, n) raw gate score; k_idx: (T, k) selected experts. >=0; 0 == free top-k."""
    k = k_idx.shape[1]
    free = s.topk(k, dim=-1).values.sum()
    chosen = s.gather(-1, k_idx).sum()
    return ((free - chosen) / s.shape[0]).item()


@torch.no_grad()
def expert_counts(k_idx: torch.Tensor, n_experts: int) -> torch.Tensor:
    return torch.bincount(k_idx.reshape(-1), minlength=n_experts)


@torch.no_grad()
def vio_from_counts(counts: torch.Tensor):
    """f_j = count_j/total * n - 1. Returns (max, min, avg|.|). 0 == perfect balance."""
    c = counts.float()
    tot = c.sum()
    n = c.numel()
    if tot == 0:
        return float("inf"), -1.0, float("inf")
    fn = c / tot * n - 1.0
    return fn.max().item(), fn.min().item(), fn.abs().mean().item()


@torch.no_grad()
def load_entropy(counts: torch.Tensor) -> float:
    c = counts.float()
    tot = c.sum()
    if tot == 0:
        return 0.0
    p = c / tot
    p = p[p > 0]
    return (-(p * p.log()).sum()).item()


@torch.no_grad()
def router_stats(s: torch.Tensor, k_idx: torch.Tensor, n_experts: int) -> RouterStats:
    counts = expert_counts(k_idx, n_experts)
    mx, mn, av = vio_from_counts(counts)
    return RouterStats(
        regret=compute_regret(s, k_idx),
        max_vio=mx,
        min_vio=mn,
        avg_vio=av,
        entropy=load_entropy(counts),
        counts=counts.cpu(),
        active_experts=float(k_idx.shape[1]),
    )
