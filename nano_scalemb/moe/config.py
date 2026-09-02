"""MoE / Mobius configuration.

Adapted from the sibling harness ``nanochat-moe`` (``moe/config.py``), trimmed to
the core knobs and extended with ``share_blocks`` for the Mobius (cross-layer
shared expert pool) arm.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MoEConfig:
    """Configuration for the MoE / Mobius block.

    ``share_blocks`` selects the arm:

      - ``0`` -> per-layer MoE. Every MoE layer builds its own router + experts.
      - ``N > 0`` -> Mobius. The model owns N routed pools and MoE layer ``i``
        reads pool ``i % N``, sharing router weights, expert weights, and
        balancer state across depth. Pair with ``shared_d_ff > 0`` to give each
        layer a private gated dense expert (upstream Mobius always does).

    ``moe_every`` is orthogonal: it decides *which* layers are MoE at all, and
    the pool index is counted over MoE layers only (see ``pool_index``), so
    ``moe_every=2, share_blocks=2`` gives the MoE layers a strict A/B/A/B pool
    pattern rather than aliasing them all onto pool 0.
    """

    n_experts: int = 8
    top_k: int = 2
    d_ff_expert: int = 256  # fine-grained: typically < d_model
    moe_every: int = 1  # MoE replaces the dense MLP where layer_idx % moe_every == 0
    share_blocks: int = 0  # 0 = per-layer MoE; N > 0 = Mobius with N shared pools
    shared_d_ff: int = 0  # per-layer private dense expert (0 = off)
    activation: str = "gelu"  # expert activation
    gate: str = "sigmoid"  # gate score fn: "sigmoid" (DeepSeek-V3 style) | "softmax"
    backend: str = "scattermoe"  # "scattermoe" (Triton) | "fallback" (torch reference)
    balancer: str = "loss_free"  # see nano_scalemb.moe.balancers
    balancer_kwargs: dict = field(default_factory=dict)
    expert_lr: float = 0.004  # AdamW lr for the 3D expert weights (Muon is 2D-only)

    def is_moe_layer(self, layer_idx: int) -> bool:
        return layer_idx % self.moe_every == 0

    def moe_layer_ids(self, n_layer: int) -> tuple[int, ...]:
        """Layer indices that get an MoE block, in order."""
        return tuple(i for i in range(n_layer) if self.is_moe_layer(i))

    def num_pools(self, n_layer: int) -> int:
        """Number of distinct routed pools the model needs.

        Per-layer MoE needs one pool per MoE layer; Mobius needs ``share_blocks``
        of them (clamped down if there are fewer MoE layers than pools).
        """
        n_moe_layers = len(self.moe_layer_ids(n_layer))
        if self.share_blocks <= 0:
            return n_moe_layers
        return min(self.share_blocks, n_moe_layers)

    def pool_index(self, layer_idx: int, n_layer: int) -> int:
        """Which pool MoE layer ``layer_idx`` reads.

        Counted over MoE layers (not raw depth) so ``moe_every`` and
        ``share_blocks`` compose without aliasing.
        """
        assert self.is_moe_layer(layer_idx), f"layer {layer_idx} is not an MoE layer"
        rank = self.moe_layer_ids(n_layer).index(layer_idx)
        num_pools = self.num_pools(n_layer)
        return rank % num_pools if num_pools > 0 else 0
