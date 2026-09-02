"""Balancer registry.

Adapted from the sibling harness ``nanochat-moe`` (``moe/balancers/``); the QB and
sequence-level balancer families are not ported.
"""

from __future__ import annotations

from nano_scalemb.moe.balancers.aux import AuxLossBalancer
from nano_scalemb.moe.balancers.base import Balancer, NoopBalancer
from nano_scalemb.moe.balancers.loss_free import LossFreeBalancer

_REGISTRY = {
    "noop": NoopBalancer,
    "loss_free": LossFreeBalancer,
    "aux": AuxLossBalancer,
}


def make_balancer(name: str, n_experts: int, top_k: int | None = None, **kwargs) -> Balancer:
    if name not in _REGISTRY:
        raise ValueError(f"unknown balancer {name!r}; have {sorted(_REGISTRY)}")
    return _REGISTRY[name](n_experts, top_k=top_k, **kwargs)


__all__ = [
    "AuxLossBalancer",
    "Balancer",
    "LossFreeBalancer",
    "NoopBalancer",
    "make_balancer",
]
