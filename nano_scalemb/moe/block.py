"""MoE / Mobius block: routed pool + optional per-layer dense expert.

Adapted from the sibling harness ``nanochat-moe`` (``moe/block.py``), split into
``MoEPool`` (the routed part, which Mobius shares across depth) and ``MoEBlock``
(the per-layer drop-in for the dense ``MLP``).

``MoEBlock.forward(x) -> y`` has the same signature and shape contract as
``gpt.MLP.forward``, so ``gpt.Block.forward`` needs no changes on either the dense
path or the mHC path. Router stats are stashed on ``self.last_stats`` rather than
returned; the train loop reads them via ``collect_moe_stats(model)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nano_scalemb.moe.balancers import make_balancer
from nano_scalemb.moe.config import MoEConfig
from nano_scalemb.moe.experts import Experts
from nano_scalemb.moe.router import Router

_ACT = {"gelu": F.gelu, "relu": F.relu, "silu": F.silu}


class SharedExpert(nn.Module):
    """Always-on dense FFN added to the routed output (DeepSeek-style).

    Off by default for the per-layer MoE arm; Mobius uses it as each layer's
    *private* capacity on top of the shared routed pool, which is what makes the
    shared-pool arm more than plain weight tying.
    """

    def __init__(self, d_model, d_ff, activation="gelu"):
        super().__init__()
        self.c_fc = nn.Linear(d_model, d_ff, bias=False)
        self.c_proj = nn.Linear(d_ff, d_model, bias=False)
        # Upstream Mobius gates the shared expert by a sigmoid of a learned
        # projection to a scalar. Held as a 1-D Parameter rather than an
        # nn.Linear(d_model, 1): a (1, d_model) weight is a vector wearing a
        # matrix costume, and gpt.py routes 2-D params to Muon, whose
        # Newton-Schulz orthogonalization is meaningless (and numerically
        # degenerate) on a single-row matrix. 1-D keeps it on AdamW.
        self.gate = nn.Parameter(torch.zeros(d_model))
        self.act = _ACT[activation]

    @torch.no_grad()
    def reset_parameters(self):
        s = 3**0.5 * self.c_fc.in_features**-0.5
        nn.init.uniform_(self.c_fc.weight, -s, s)
        nn.init.zeros_(self.c_proj.weight)  # zero init -> stable residual at init
        nn.init.zeros_(self.gate)  # gate starts at sigmoid(0) = 0.5

    def forward(self, x):
        h = F.linear(x, self.c_fc.weight.to(x.dtype))
        h = self.act(h)
        y = F.linear(h, self.c_proj.weight.to(x.dtype))
        g = torch.sigmoid((x * self.gate.to(x.dtype)).sum(-1, keepdim=True))
        return g * y


class MoEPool(nn.Module):
    """Router + expert bank: the unit Mobius shares across layers.

    Sharing a pool shares the router gate, the expert weights, AND the balancer
    state -- which is what upstream Mobius does (one ``meta_mlp`` block serving
    every layer that indexes into it).
    """

    def __init__(self, d_model: int, cfg: MoEConfig):
        super().__init__()
        self.cfg = cfg
        balancer = make_balancer(
            cfg.balancer, cfg.n_experts, top_k=cfg.top_k, **cfg.balancer_kwargs
        )
        self.router = Router(d_model, cfg.n_experts, cfg.top_k, balancer, gate=cfg.gate)
        self.experts = Experts(
            cfg.n_experts,
            d_model,
            cfg.d_ff_expert,
            cfg.top_k,
            activation=cfg.activation,
            backend=cfg.backend,
        )
        # Per-forward scratch. With a shared pool these are overwritten by each
        # layer that uses it, so the collectors below read them off MoEBlock (one
        # per layer) rather than off the pool.
        self.last_stats = None
        self.last_aux_loss = None

    @torch.no_grad()
    def reset_parameters(self):
        self.router.reset_parameters()  # also zeroes the balancer buffers
        self.experts.reset_parameters()

    def forward(self, xf: torch.Tensor):
        """xf: (T, d_model) already flattened. Returns (y, stats, aux_loss)."""
        k_w, k_idx, s, stats = self.router(xf)
        y = self.experts(xf, k_w.to(xf.dtype), k_idx)
        # The aux-loss family returns a graph-connected scalar (None for the bias
        # methods); the train loop sums it across layers into the backward loss.
        aux = self.router.balancer.aux_loss(s, k_idx)
        self.last_stats = stats
        self.last_aux_loss = aux
        return y, stats, aux


class MoEBlock(nn.Module):
    """Per-layer drop-in for the dense MLP.

    ``pool`` is passed in by the GPT. For the Mobius arm the same ``MoEPool``
    object is handed to several blocks; it is held in a plain list so it is NOT
    re-registered as a submodule here. Registering it twice would make
    ``model.parameters()`` and the ``state_dict`` disagree about how many tensors
    exist, and -- worse -- ``meta -> to_empty()`` would hand each registration its
    own fresh storage and silently break the sharing.
    """

    def __init__(self, d_model: int, cfg: MoEConfig, pool: MoEPool):
        super().__init__()
        self.cfg = cfg
        self._pool = [pool]  # deliberately not a submodule; see the docstring
        self.shared = (
            SharedExpert(d_model, cfg.shared_d_ff, cfg.activation)
            if cfg.shared_d_ff > 0
            else None
        )
        self.last_stats = None
        self.last_aux_loss = None

    @property
    def pool(self) -> MoEPool:
        return self._pool[0]

    @torch.no_grad()
    def reset_parameters(self):
        """Init the per-layer parts only. The pool is owned (and initialized) by
        the GPT, since with Mobius it is shared and must be initialized once."""
        if self.shared is not None:
            self.shared.reset_parameters()

    def forward(self, x):
        """x: (B, T, d) or (T, d). Flattened for routing, original shape restored."""
        shape = x.shape
        xf = x.reshape(-1, shape[-1])
        y, stats, aux = self.pool(xf)
        if self.shared is not None:
            y = y + self.shared(xf)
        self.last_stats = stats
        self.last_aux_loss = aux
        return y.view(shape)


def collect_moe_stats(model) -> dict:
    """Gather per-layer + model-mean router stats after a forward.

    Returns ``{'moe_layer/<i>/<metric>': v, 'moe/<metric>_mean': v}``; empty if
    there are no MoE layers or stats collection was off this step.
    """
    per_layer = [
        mod
        for mod in model.modules()
        if isinstance(mod, MoEBlock) and mod.last_stats is not None
    ]
    out: dict = {}
    if not per_layer:
        return out
    keys = ("regret", "max_vio", "min_vio", "avg_vio", "entropy", "active_experts")
    for i, mod in enumerate(per_layer):
        for kk in keys:
            out[f"moe_layer/{i}/{kk}"] = getattr(mod.last_stats, kk)
    for kk in keys:
        out[f"moe/{kk}_mean"] = sum(getattr(m.last_stats, kk) for m in per_layer) / len(
            per_layer
        )
    return out


def collect_aux_loss(model):
    """Sum the per-layer differentiable balance loss from the last forward.

    Returns a graph-connected scalar tensor, or None if no MoE layer produced one
    (bias-based balancers / noop). Call once per forward, before backward.
    """
    total = None
    for mod in model.modules():
        if isinstance(mod, MoEBlock) and mod.last_aux_loss is not None:
            al = mod.last_aux_loss
            total = al if total is None else total + al
    return total
