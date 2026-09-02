"""Balancer interface.

Adapted from the sibling harness ``nanochat-moe`` (``moe/balancers/base.py``).

Convention (additive bias): the router selects top-k of ``s + bias``, where bias
is a per-expert vector held as buffer STATE (not an optimizer parameter, no
autograd). Loss-free family: the bias affects selection ONLY -- the weights
multiplied into the expert outputs use the unbiased score s, so task-loss
gradients stay uncontaminated.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Balancer(nn.Module):
    """Base class. Stateless no-op by default; subclasses register buffers for
    bias state and implement update()."""

    def __init__(self, n_experts: int, top_k: int | None = None):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k

    def bias(self, s: torch.Tensor) -> torch.Tensor | float:
        """Per-expert additive bias added to s BEFORE top-k. no_grad. s: (T, n)."""
        return 0.0

    def aux_loss(self, s: torch.Tensor, k_idx: torch.Tensor):
        """Optional differentiable balance loss (GShard/Switch family). None if unused."""
        return None

    @torch.no_grad()
    def update(self, counts: torch.Tensor, s: torch.Tensor, k_idx: torch.Tensor) -> None:
        """Post-step state update (bias step). counts: (n,) load."""
        return None

    @torch.no_grad()
    def reset_state(self) -> None:
        """Re-initialize state buffers. Called from Router.reset_parameters AFTER
        the GPT's meta -> to_empty (buffers come back uninitialized, so stateful
        balancers MUST zero them here)."""
        return None


class NoopBalancer(Balancer):
    """No balancing -- pure free top-k. Reference lower bound on regret (== 0)."""
