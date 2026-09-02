"""Auxiliary-loss balancing (Switch Transformer, Fedus et al. 2021; GShard;
DeepSeekMoE expert-level balance loss).

Adapted from the sibling harness ``nanochat-moe`` (``moe/balancers/aux.py``).

Unlike loss-free, this does NOT bias selection (``bias() == 0``); it adds a
differentiable penalty to the training loss whose gradient flows through the
router probabilities, pushing mass away from overloaded experts:

    f_i = (n / (k * T)) * sum_t 1[expert i in top-k(token t)]   # hard load, mean_i(f)=1, NO grad
    P_i = (1 / T) * sum_t p_{t,i}                               # mean router prob, sum_i(P)=1, has grad
    L_aux = alpha * sum_i f_i * P_i

For the softmax gate p == s; for the sigmoid gate s does not sum to 1, so we
normalize per token to keep P a proper distribution. At uniform routing
``L_aux == alpha``.

Distributed: computed per-rank on local tokens (like the LM loss); the optimizer
averages gradients across ranks, so no explicit reduction here.
"""

from __future__ import annotations

import torch.nn.functional as F

from nano_scalemb.moe.balancers.base import Balancer


class AuxLossBalancer(Balancer):
    def __init__(self, n_experts: int, top_k: int | None = None, alpha: float = 0.01):
        super().__init__(n_experts, top_k)
        self.alpha = float(alpha)

    # bias() inherited == 0.0 -> selection on the unbiased score

    def aux_loss(self, s, k_idx):
        """s: (T, n) differentiable gate score; k_idx: (T, k). Returns a scalar tensor."""
        _, n = s.shape
        k = k_idx.shape[1]
        p = s / s.sum(dim=-1, keepdim=True).clamp_min(1e-9)  # per-token prob (T, n)
        P = p.mean(dim=0)  # (n,) has grad, sums to 1
        onehot = F.one_hot(k_idx, n).sum(dim=1).to(s.dtype)  # (T, n) in {0..k}
        f = onehot.mean(dim=0) * (n / k)  # (n,) mean_i == 1
        return self.alpha * (f.detach() * P).sum()  # == alpha at uniform
