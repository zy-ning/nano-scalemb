"""MoE scaling ladder and param accounting.

Adapted from the sibling harness ``nanochat-moe`` (``moe/ladder.py``).

Three independent axes over a base where the *active* expert FLOPs equal one dense
FFN (``ff_mult * d_model``, matching ``gpt.MLP``). Given model dim ``md`` and
``base_ff = ff_mult * md``:

    d_ff_expert = round(base_ff / G)     # granularity G  (finer = more, smaller experts)
    top_k       = round(A * G)           # active_mult A  (A=1 => active FLOPs == one dense FFN)
    n_experts   = round(X * G)           # expansion X    (total capacity / total params)
    => sparsity = n_experts / top_k = X / A   (independent of granularity)

Moving G at fixed (A, X) holds active AND total params constant -> pure
granularity. Moving X at fixed (A, G) holds active constant, grows total -> pure
sparsity.

Unlike upstream, ``gpt.estimate_flops`` and ``gpt.num_scaling_params`` here are
active-aware, so the ordinary ``--target-param-data-ratio`` horizon works for MoE
runs; there is no separate MoE token-budget path.
"""

from __future__ import annotations

from nano_scalemb.moe.config import MoEConfig

HEAD_DIM_DEFAULT = 128
ASPECT_RATIO_DEFAULT = 64


def model_dim_for_depth(
    depth: int,
    aspect_ratio: int = ASPECT_RATIO_DEFAULT,
    head_dim: int = HEAD_DIM_DEFAULT,
) -> int:
    """The harness rule (see ``scripts/base_train.build_model_meta``):
    dim = depth * aspect_ratio, rounded up to a multiple of head_dim."""
    base = depth * aspect_ratio
    return ((base + head_dim - 1) // head_dim) * head_dim


def moe_config_from_knobs(
    model_dim: int,
    granularity: float,
    expansion: float,
    active_mult: float = 1.0,
    ff_mult: int = 4,
    moe_every: int = 2,
    round_experts_to: int = 4,
    **moe_kw,
) -> MoEConfig:
    """Resolve (G, X, A) -> MoEConfig.

    ``n_experts`` is rounded UP to a multiple of ``round_experts_to`` so the
    expert-dim reduce_scatter in the distributed optimizer divides evenly across
    world sizes; pass 0 to disable.
    """
    base_ff = ff_mult * model_dim
    d_ff_expert = max(1, round(base_ff / granularity))
    top_k = max(1, round(active_mult * granularity))
    n_experts = max(top_k, round(expansion * granularity))
    if round_experts_to:
        rem = n_experts % round_experts_to
        if rem:
            n_experts += round_experts_to - rem
    return MoEConfig(
        n_experts=n_experts,
        top_k=top_k,
        d_ff_expert=d_ff_expert,
        moe_every=moe_every,
        **moe_kw,
    )


def describe(model_dim: int, cfg: MoEConfig, ff_mult: int = 4) -> dict:
    """Recover the (G, A, X, sparsity) a config corresponds to, for logging."""
    base_ff = ff_mult * model_dim
    G = base_ff / cfg.d_ff_expert
    A = cfg.top_k * cfg.d_ff_expert / base_ff
    X = cfg.n_experts * cfg.d_ff_expert / base_ff
    return {
        "G": G,
        "A": A,
        "X": X,
        "sparsity": cfg.n_experts / cfg.top_k,
        "n_experts": cfg.n_experts,
        "top_k": cfg.top_k,
        "d_ff_expert": cfg.d_ff_expert,
    }


def total_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def active_params(model) -> int:
    """Params a single token's forward actually touches.

    Conditional weight (routed experts, addressed memory) is swapped out of the
    stored total and replaced by the amount actually *read*, counted once per
    reading layer. The read view is the one that matters here: it is what drives
    FLOPs, and it is the only view under which a shared pool and per-layer pools
    of the same active width come out equal — which is exactly the claim an
    iso-active comparison rests on.

    Deliberately NOT ``total - inactive_stored``: for a shared Mobius pool those
    differ (one bank, many readers), and mixing the two views made the same
    config report two different active counts.
    """
    counts = model.num_scaling_params()
    # Both Engram memory forms are addressed rather than dense: the hash table
    # (engram_embeds) and the product-key value bank (engram_pk_values). Both
    # are swapped out for engram_active, the amount a token actually reads.
    engram_stored = (
        counts["engram_embeds"]
        + counts["engram_frozen_embeds"]
        + counts["engram_pk_values"]
    )
    return (
        total_params(model)
        - counts["expert_total"]
        + counts["expert_active"]
        - counts["router_total"]
        + counts["router_active"]
        - engram_stored
        + counts["engram_active"]
    )
