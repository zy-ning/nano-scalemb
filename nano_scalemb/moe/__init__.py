"""Sparse Mixture-of-Experts for nano-scalemb.

Two arms share this machinery, selected by ``MoEConfig.share_blocks``:

- ``share_blocks == 0`` -> a standard **per-layer MoE**: every MoE layer owns a
  private router + expert pool. This is the conditional-capacity baseline.
- ``share_blocks == N`` -> **Mobius**: N routed expert pools live on the model and
  layer ``i`` reads pool ``i % N``, so the routed experts (and the router, and the
  balancer state) are shared *across depth*. Each MoE layer keeps a private,
  sigmoid-gated dense "shared expert" on top. This is the namesake loop of
  Intern-S2-Mobius.

Ported from the sibling harness ``nanochat-moe`` (``moe/``), trimmed to the core:
linear gate, two balancers (loss-free bias + Switch aux loss), and the scattermoe
/ torch expert backends. The router-variant research (STAR / SoftMoE / ReMoE /
ProbMoE / DirMoE / QB balancers) is deliberately not ported.
"""

from nano_scalemb.moe.block import (
    MoEBlock,
    MoEPool,
    SharedExpert,
    collect_aux_loss,
    collect_moe_stats,
)
from nano_scalemb.moe.config import MoEConfig
from nano_scalemb.moe.experts import Experts
from nano_scalemb.moe.router import Router

__all__ = [
    "MoEBlock",
    "MoEConfig",
    "MoEPool",
    "Router",
    "Experts",
    "SharedExpert",
    "collect_aux_loss",
    "collect_moe_stats",
]
