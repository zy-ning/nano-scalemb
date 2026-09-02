import math

import pytest
import torch

import nano_scalemb.flash_attention as fa_module
from nano_scalemb.gpt import GPT, GPTConfig
from nano_scalemb.mhc import MHCConfig
from nano_scalemb.moe.balancers import make_balancer
from nano_scalemb.moe.block import MoEBlock, MoEPool, collect_aux_loss, collect_moe_stats
from nano_scalemb.moe.config import MoEConfig
from nano_scalemb.moe.experts import Experts
from nano_scalemb.moe.ladder import (
    active_params,
    describe,
    moe_config_from_knobs,
    total_params,
)
from nano_scalemb.moe.router import Router


def tiny_moe_config(**overrides):
    """Small, CPU-friendly MoE config. Always the pure-torch expert backend."""
    kwargs = dict(
        n_experts=8,
        top_k=2,
        d_ff_expert=16,
        moe_every=2,
        backend="fallback",
        balancer="noop",
    )
    kwargs.update(overrides)
    return MoEConfig(**kwargs)


def build_gpt(moe=None, mhc=None, n_layer=4, n_embd=32):
    """Build a tiny GPT through the real meta -> to_empty -> init_weights path."""
    config = GPTConfig(
        sequence_len=16,
        vocab_size=64,
        n_layer=n_layer,
        n_head=2,
        n_kv_head=2,
        n_embd=n_embd,
        window_pattern="L",
        moe=moe,
        mhc=mhc,
    )
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device="cpu")
    model.init_weights()
    return model


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


def test_router_weights_are_topk_normalized():
    torch.manual_seed(0)
    router = Router(16, n_experts=8, top_k=3, balancer=make_balancer("noop", 8))
    router.reset_parameters()

    k_w, k_idx, s, stats = router(torch.randn(12, 16))

    assert k_w.shape == (12, 3)
    assert k_idx.shape == (12, 3)
    assert s.shape == (12, 8)
    assert torch.allclose(k_w.sum(-1), torch.ones(12), atol=1e-5)
    assert (k_idx >= 0).all() and (k_idx < 8).all()


def test_router_selects_on_biased_score_but_weights_use_unbiased_score():
    """The defining property of loss-free balancing: the bias steers which
    experts fire, never the value multiplied into their output."""
    torch.manual_seed(0)
    n, k = 8, 2
    balancer = make_balancer("loss_free", n, top_k=k)
    router = Router(16, n_experts=n, top_k=k, balancer=balancer)
    router.reset_parameters()
    x = torch.randn(6, 16)

    # Force expert 0 to always be selected via a large bias, and expert 1 never.
    with torch.no_grad():
        balancer.bias_buf.zero_()
        balancer.bias_buf[0] = 100.0
        balancer.bias_buf[1] = -100.0
    k_w, k_idx, s, _ = router(x)

    assert (k_idx == 0).any(dim=-1).all(), "biased-up expert should always be selected"
    assert not (k_idx == 1).any(), "biased-down expert should never be selected"
    # Weights must come from the raw score, not score+bias: if the bias leaked in,
    # expert 0's normalized weight would be ~1.0 everywhere.
    raw = s.gather(-1, k_idx)
    assert torch.allclose(k_w, raw / raw.sum(-1, keepdim=True), atol=1e-6)


def test_router_noop_balancer_has_zero_regret():
    torch.manual_seed(0)
    router = Router(16, n_experts=8, top_k=2, balancer=make_balancer("noop", 8))
    router.reset_parameters()

    _, _, _, stats = router(torch.randn(32, 16))

    assert stats is not None
    assert stats.regret == pytest.approx(0.0, abs=1e-6)


def test_router_collect_stats_gates_the_host_sync():
    torch.manual_seed(0)
    router = Router(16, n_experts=8, top_k=2, balancer=make_balancer("noop", 8))
    router.reset_parameters()

    router.collect_stats = False
    assert router(torch.randn(8, 16))[3] is None
    router.collect_stats = True
    assert router(torch.randn(8, 16))[3] is not None


# ---------------------------------------------------------------------------
# Balancers
# ---------------------------------------------------------------------------


def test_loss_free_bias_is_zero_mean_and_penalizes_overload():
    balancer = make_balancer("loss_free", 4, top_k=1, gamma=0.1)
    balancer.reset_state()
    s = torch.full((8, 4), 0.5)
    k_idx = torch.zeros(8, 1, dtype=torch.long)

    counts = torch.tensor([8, 0, 0, 0])  # expert 0 wildly overloaded
    balancer.update(counts, s, k_idx)

    assert balancer.bias_buf.mean().abs() < 1e-6, "constant-shift DoF must be removed"
    assert balancer.bias_buf[0] < 0, "overloaded expert gets pushed down"
    assert (balancer.bias_buf[1:] > 0).all(), "underloaded experts get pushed up"


def test_loss_free_reset_state_zeroes_the_bias():
    balancer = make_balancer("loss_free", 4, top_k=1)
    balancer.bias_buf.fill_(3.0)

    balancer.reset_state()

    assert torch.equal(balancer.bias_buf, torch.zeros(4))


def test_aux_loss_equals_alpha_at_uniform_routing():
    n, k, alpha = 4, 1, 0.01
    balancer = make_balancer("aux", n, top_k=k, alpha=alpha)
    # Uniform scores, and every expert chosen an equal number of times.
    s = torch.full((8, n), 0.25)
    k_idx = torch.arange(n).repeat(2).view(8, 1)

    assert balancer.aux_loss(s, k_idx).item() == pytest.approx(alpha, abs=1e-6)


def test_aux_loss_gradient_pushes_mass_off_the_overloaded_expert():
    n, alpha = 4, 1.0
    balancer = make_balancer("aux", n, top_k=1, alpha=alpha)
    s = torch.full((8, n), 0.25, requires_grad=True)
    k_idx = torch.zeros(8, 1, dtype=torch.long)  # expert 0 takes every token

    balancer.aux_loss(s, k_idx).backward()

    grad = s.grad.sum(dim=0)
    assert grad[0] > 0, "overloaded expert's prob should be pushed down"
    assert (grad[1:] < grad[0]).all()


def test_noop_balancer_contributes_no_bias_and_no_aux_loss():
    balancer = make_balancer("noop", 4, top_k=1)

    assert balancer.bias(torch.zeros(2, 4)) == 0.0
    assert balancer.aux_loss(torch.zeros(2, 4), torch.zeros(2, 1, dtype=torch.long)) is None


def test_unknown_balancer_raises():
    with pytest.raises(ValueError, match="unknown balancer"):
        make_balancer("nope", 4)


# ---------------------------------------------------------------------------
# Experts
# ---------------------------------------------------------------------------


def test_experts_are_zero_at_init_and_gradients_flow():
    torch.manual_seed(0)
    experts = Experts(4, d_model=8, d_ff=6, top_k=2, backend="fallback")
    experts.reset_parameters()
    x = torch.randn(5, 8, requires_grad=True)
    k_idx = torch.randint(0, 4, (5, 2))
    k_w = torch.full((5, 2), 0.5)

    y = experts(x, k_w, k_idx)

    assert y.shape == (5, 8)
    # w_out is zero-init (the dense c_proj convention), so the block is a no-op
    # at init and the residual stream starts unperturbed.
    assert torch.equal(y, torch.zeros_like(y))

    experts.w_out.data.normal_(std=0.1)
    y = experts(x, k_w, k_idx)
    y.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert torch.isfinite(experts.w_in.grad).all()


def test_experts_reject_unknown_backend_and_activation():
    with pytest.raises(ValueError, match="unknown backend"):
        Experts(2, 4, 4, 1, backend="nope")
    with pytest.raises(ValueError, match="unknown activation"):
        Experts(2, 4, 4, 1, activation="nope", backend="fallback")


@pytest.mark.parametrize("backend", ["fallback", "scattermoe"])
def test_experts_backward_under_autocast(backend):
    """scattermoe's autograd.Function is not autocast-aware: its backward mixes
    the autocast-dtype grouped GEMM output with fp32 `gates` and raises
    "expected scalar type Float but found BFloat16". Experts disables autocast
    for the kernel region and casts by hand; both backends must survive a real
    autocast training step, with fp32 grads on the fp32 masters."""
    if backend == "scattermoe":
        pytest.importorskip("scattermoe")
    if not torch.cuda.is_available():
        pytest.skip("autocast(bf16) + scattermoe need CUDA")
    torch.manual_seed(0)
    experts = Experts(8, d_model=64, d_ff=32, top_k=2, backend=backend).cuda()
    experts.reset_parameters()
    experts.w_out.data.normal_(std=0.02)
    x = torch.randn(64, 64, device="cuda", requires_grad=True)
    k_idx = torch.randint(0, 8, (64, 2), device="cuda")
    k_w = torch.rand(64, 2, device="cuda")
    k_w = k_w / k_w.sum(-1, keepdim=True)

    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        y = experts(x, k_w, k_idx)
    assert y.dtype == torch.bfloat16
    y.float().sum().backward()

    # Master weights stay fp32, so their grads must too.
    assert experts.w_in.grad is not None and experts.w_in.grad.dtype == torch.float32
    assert experts.w_out.grad.dtype == torch.float32
    assert torch.isfinite(experts.w_in.grad).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_scattermoe_matches_the_torch_fallback():
    """The Triton kernel and the reference loop must agree. Requires the gpu
    extra (scattermoe + a CUDA device), skipped otherwise."""
    pytest.importorskip("scattermoe")
    if not torch.cuda.is_available():
        pytest.skip("scattermoe is a CUDA/Triton kernel")
    torch.manual_seed(0)
    n_experts, d_model, d_ff, top_k, T = 8, 32, 16, 2, 64

    ref = Experts(n_experts, d_model, d_ff, top_k, backend="fallback").cuda()
    ref.reset_parameters()
    ref.w_out.data.normal_(std=0.1)
    fast = Experts(n_experts, d_model, d_ff, top_k, backend="scattermoe").cuda()
    fast.load_state_dict(ref.state_dict())

    x = torch.randn(T, d_model, device="cuda")
    k_idx = torch.randint(0, n_experts, (T, top_k), device="cuda")
    k_w = torch.rand(T, top_k, device="cuda")
    k_w = k_w / k_w.sum(-1, keepdim=True)

    # Loose tolerance on purpose: scattermoe's Triton GEMMs accumulate in tf32
    # (~1e-3 relative), so this checks the kernel computes the same *function*,
    # not bit-identical fp32 arithmetic.
    assert torch.allclose(ref(x, k_w, k_idx), fast(x, k_w, k_idx), atol=1e-3, rtol=2e-2)


# ---------------------------------------------------------------------------
# Block / pool plumbing
# ---------------------------------------------------------------------------


def test_moe_block_does_not_register_the_pool_as_a_submodule():
    """The pool must be registered exactly once (on the GPT), or meta -> to_empty
    hands each registration its own storage and silently breaks sharing."""
    cfg = tiny_moe_config()
    pool = MoEPool(32, cfg)
    block = MoEBlock(32, cfg, pool)

    assert block.pool is pool
    assert [n for n, _ in block.named_parameters()] == []
    assert not any(m is pool for m in block.modules())


def test_moe_block_shared_expert_is_gated_and_optional():
    torch.manual_seed(0)
    cfg = tiny_moe_config(shared_d_ff=16)
    pool = MoEPool(32, cfg)
    pool.reset_parameters()
    block = MoEBlock(32, cfg, pool)
    block.reset_parameters()

    assert block.shared is not None
    assert tiny_moe_config(shared_d_ff=0).shared_d_ff == 0

    x = torch.randn(2, 3, 32)
    assert block(x).shape == x.shape

    # gate weight is zero-init -> sigmoid(0) = 0.5, and c_proj is zero, so the
    # shared expert also starts as a no-op.
    assert torch.equal(block(x), torch.zeros_like(x))


def test_collect_helpers_are_empty_without_moe_layers():
    model = build_gpt(moe=None)

    assert collect_moe_stats(model) == {}
    assert collect_aux_loss(model) is None


# ---------------------------------------------------------------------------
# Cross-layer sharing (Mobius) vs per-layer MoE
# ---------------------------------------------------------------------------


def test_per_layer_moe_gives_each_layer_its_own_pool():
    model = build_gpt(moe=tiny_moe_config(moe_every=2), n_layer=4)

    moe_blocks = [b.mlp for b in model.transformer.h if b.is_moe]
    assert len(moe_blocks) == 2
    assert len(model.moe_pools) == 2
    assert moe_blocks[0].pool is not moe_blocks[1].pool
    assert moe_blocks[0].pool.experts.w_in is not moe_blocks[1].pool.experts.w_in


def test_mobius_shares_one_pool_across_layers_through_meta_init():
    model = build_gpt(moe=tiny_moe_config(moe_every=2, share_blocks=1), n_layer=4)

    moe_blocks = [b.mlp for b in model.transformer.h if b.is_moe]
    assert len(moe_blocks) == 2
    assert len(model.moe_pools) == 1
    # Identity must survive meta -> to_empty -> init_weights, not just __init__.
    assert moe_blocks[0].pool is moe_blocks[1].pool
    assert moe_blocks[0].pool.experts.w_in is moe_blocks[1].pool.experts.w_in
    assert moe_blocks[0].pool.router.Wg.weight is moe_blocks[1].pool.router.Wg.weight
    # Sharing the pool shares the balancer state too, as upstream Mobius does.
    assert moe_blocks[0].pool.router.balancer is moe_blocks[1].pool.router.balancer


def test_mobius_round_robins_pools_over_moe_layers():
    model = build_gpt(moe=tiny_moe_config(moe_every=1, share_blocks=2), n_layer=4)

    pools = [b.mlp.pool for b in model.transformer.h]
    assert len(model.moe_pools) == 2
    assert pools[0] is pools[2] and pools[1] is pools[3]
    assert pools[0] is not pools[1]


def test_mobius_clamps_pool_count_to_the_number_of_moe_layers():
    model = build_gpt(moe=tiny_moe_config(moe_every=2, share_blocks=8), n_layer=4)

    assert len(model.moe_pools) == 2  # only 2 MoE layers exist to serve


def test_sharing_saves_expert_params_but_keeps_active_params_equal():
    """The whole point of the Mobius arm: same per-token compute, fewer weights."""
    per_layer = build_gpt(moe=tiny_moe_config(moe_every=2), n_layer=4)
    mobius = build_gpt(moe=tiny_moe_config(moe_every=2, share_blocks=1), n_layer=4)

    pl, mb = per_layer.num_scaling_params(), mobius.num_scaling_params()
    assert mb["expert_total"] == pl["expert_total"] // 2
    assert mb["expert_active"] == pl["expert_active"]
    assert mb["router_total"] == pl["router_total"] // 2
    assert mb["router_active"] == pl["router_active"]
    assert total_params(mobius) < total_params(per_layer)
    # Storage drops; reads do not. See
    # test_active_params_is_identical_for_shared_and_per_layer_pools.
    assert active_params(mobius) == active_params(per_layer)


# ---------------------------------------------------------------------------
# Full-model forward / backward
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "moe_kwargs",
    [
        dict(),
        dict(share_blocks=1, shared_d_ff=16),
        dict(balancer="loss_free"),
        dict(balancer="aux", balancer_kwargs={"alpha": 0.01}),
        dict(gate="softmax"),
        dict(moe_every=1),
    ],
    ids=["per-layer", "mobius", "loss_free", "aux", "softmax", "every-layer"],
)
def test_gpt_forward_backward_is_finite(moe_kwargs):
    torch.manual_seed(0)
    model = build_gpt(moe=tiny_moe_config(**moe_kwargs), n_layer=4)
    idx = torch.randint(0, 64, (2, 8))

    prev_impl = fa_module._override_impl
    fa_module._override_impl = "sdpa"
    try:
        loss = model(idx, idx)
        aux = collect_aux_loss(model)
        if aux is not None:
            loss = loss + aux
        loss.backward()
    finally:
        fa_module._override_impl = prev_impl

    assert torch.isfinite(loss)
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )


def test_gpt_composes_with_mhc():
    torch.manual_seed(0)
    model = build_gpt(
        moe=tiny_moe_config(share_blocks=1, shared_d_ff=16),
        mhc=MHCConfig(num_streams=4, sinkhorn_iters=3),
        n_layer=4,
    )
    idx = torch.randint(0, 64, (2, 8))

    prev_impl = fa_module._override_impl
    fa_module._override_impl = "sdpa"
    try:
        logits = model(idx)
    finally:
        fa_module._override_impl = prev_impl

    assert logits.shape == (2, 8, 64)
    assert torch.isfinite(logits).all()


def test_gpt_moe_loss_decreases_over_a_few_optimizer_steps():
    torch.manual_seed(0)
    model = build_gpt(moe=tiny_moe_config(balancer="loss_free"), n_layer=4)
    optimizer = model.setup_optimizer()
    idx = torch.randint(0, 64, (2, 8))

    prev_impl = fa_module._override_impl
    fa_module._override_impl = "sdpa"
    try:
        losses = []
        for _ in range(8):
            loss = model(idx, idx)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            model.zero_grad(set_to_none=True)
    finally:
        fa_module._override_impl = prev_impl

    assert losses[-1] < losses[0]


def test_collect_moe_stats_reports_per_layer_and_mean():
    torch.manual_seed(0)
    model = build_gpt(moe=tiny_moe_config(moe_every=2), n_layer=4)
    idx = torch.randint(0, 64, (2, 8))

    prev_impl = fa_module._override_impl
    fa_module._override_impl = "sdpa"
    try:
        model(idx, idx)
    finally:
        fa_module._override_impl = prev_impl
    stats = collect_moe_stats(model)

    assert "moe/regret_mean" in stats
    assert "moe_layer/0/max_vio" in stats and "moe_layer/1/max_vio" in stats
    assert "moe_layer/2/max_vio" not in stats  # only 2 of the 4 layers are MoE
    assert stats["moe/active_experts_mean"] == 2.0


# ---------------------------------------------------------------------------
# FLOP / parameter accounting
# ---------------------------------------------------------------------------


def test_num_scaling_params_total_still_matches_the_model():
    for moe in (tiny_moe_config(), tiny_moe_config(share_blocks=1, shared_d_ff=16)):
        model = build_gpt(moe=moe, n_layer=4)
        counts = model.num_scaling_params()
        assert counts["total"] == sum(p.numel() for p in model.parameters())


def test_dense_model_reports_no_expert_params():
    counts = build_gpt(moe=None).num_scaling_params()

    assert counts["expert_total"] == 0
    assert counts["expert_active"] == 0


def test_estimate_flops_counts_active_experts_not_all_of_them():
    """A 64-expert top-2 model must not be charged 16x the FLOPs of a 4-expert
    top-2 model: only top_k experts fire per token. That over-count is the bug
    this accounting fixes.

    The two are not exactly equal -- the wider model has a wider router gate --
    so we assert the gap is *only* that gate, not the expert bank.
    """
    narrow = build_gpt(moe=tiny_moe_config(n_experts=4, top_k=2), n_layer=4)
    wide = build_gpt(moe=tiny_moe_config(n_experts=64, top_k=2), n_layer=4)

    n_pools = len(wide.moe_pools)
    router_gap = 6 * wide.config.n_embd * (64 - 4) * n_pools
    assert wide.estimate_flops() - narrow.estimate_flops() == router_gap
    # ...while the stored expert weight really did grow 16x.
    assert (
        wide.num_scaling_params()["expert_total"]
        == 16 * narrow.num_scaling_params()["expert_total"]
    )
    assert wide.num_scaling_params()["expert_active"] == (
        narrow.num_scaling_params()["expert_active"]
    )
    assert total_params(wide) > total_params(narrow)


def test_estimate_flops_scales_with_top_k():
    k2 = build_gpt(moe=tiny_moe_config(n_experts=8, top_k=2), n_layer=4)
    k4 = build_gpt(moe=tiny_moe_config(n_experts=8, top_k=4), n_layer=4)

    assert k4.estimate_flops() > k2.estimate_flops()


def test_mobius_flops_count_every_reading_layer():
    """Shared weights are stored once but read once per layer, so a shared pool
    must not look cheaper in FLOPs than a per-layer one."""
    per_layer = build_gpt(moe=tiny_moe_config(moe_every=2), n_layer=4)
    mobius = build_gpt(moe=tiny_moe_config(moe_every=2, share_blocks=1), n_layer=4)

    assert mobius.estimate_flops() == per_layer.estimate_flops()


def test_active_params_is_identical_for_shared_and_per_layer_pools():
    """The claim an iso-active sweep rests on: sharing a pool changes what is
    STORED, never what a token READS. active_params must therefore be equal for
    per-layer and Mobius at the same expert width, even though totals differ.

    Regression guard: an earlier version computed active as
    `total - inactive_stored`, which double-counts a shared bank's savings and
    reported a *lower* active count for Mobius than for per-layer MoE.
    """
    per_layer = build_gpt(moe=tiny_moe_config(moe_every=2), n_layer=4)
    mobius = build_gpt(moe=tiny_moe_config(moe_every=2, share_blocks=1), n_layer=4)

    assert active_params(mobius) == active_params(per_layer)
    assert total_params(mobius) < total_params(per_layer)


def test_active_params_discounts_the_engram_table():
    """Engram memory is addressed, not dense: a token reads num_hash_heads rows
    however big the table is. Growing the table must not move active_params."""
    from nano_scalemb.engram import EngramConfig

    def build_engram(slot_multiplier):
        engram = EngramConfig(
            layer_ids=(0, 2),
            max_ngram_size=3,
            n_head_per_ngram=2,
            memory_dim=32,
            slot_multiplier=slot_multiplier,
            use_tokenizer_compression=False,
        )
        config = GPTConfig(
            sequence_len=16,
            vocab_size=64,
            n_layer=4,
            n_head=2,
            n_kv_head=2,
            n_embd=32,
            window_pattern="L",
            engram=engram,
        )
        with torch.device("meta"):
            model = GPT(config)
        model.to_empty(device="cpu")
        model.init_weights()
        return model

    small, big = build_engram(2), build_engram(8)

    assert total_params(big) > total_params(small)
    assert active_params(big) == active_params(small)
    # A token reads num_hash_heads * head_dim per Engram layer: 2 layers x
    # (max_ngram_size-1=2 orders * 2 heads) * (memory_dim=32 / 4 heads) = 64.
    assert small.num_scaling_params()["engram_active"] == 64


def test_active_params_matches_dense_at_iso_active_width():
    """At A=1 with top_k*d_ff_expert == the dense FFN width, the MoE arm's active
    params equal the dense model's (modulo the small router gate)."""
    n_embd, n_layer = 128, 2
    dense = build_gpt(moe=None, n_layer=n_layer, n_embd=n_embd)
    # dense MLP is 4*n_embd wide; 8 experts of width 4*n_embd/8, top_k 1 each...
    # use top_k=8 of width 4*n_embd/8 to match the full dense width.
    moe = build_gpt(
        moe=tiny_moe_config(
            n_experts=8, top_k=8, d_ff_expert=4 * n_embd // 8, moe_every=1
        ),
        n_layer=n_layer,
        n_embd=n_embd,
    )

    router_params = sum(p.numel() for p in moe.moe_pools.parameters() if p.ndim == 2)
    assert active_params(moe) == active_params(dense) + router_params


# ---------------------------------------------------------------------------
# Ladder
# ---------------------------------------------------------------------------


def test_ladder_granularity_axis_holds_active_and_total_constant():
    md = 512
    cfgs = [moe_config_from_knobs(md, G, expansion=8, active_mult=1) for G in (2, 4, 8)]

    actives = {c.top_k * c.d_ff_expert for c in cfgs}
    totals = {c.n_experts * c.d_ff_expert for c in cfgs}
    assert len(actives) == 1, "moving G at fixed (A, X) must hold active constant"
    assert len(totals) == 1, "moving G at fixed (A, X) must hold total constant"


def test_ladder_sparsity_axis_holds_active_and_grows_total():
    md = 512
    lo = moe_config_from_knobs(md, 4, expansion=4, active_mult=1)
    hi = moe_config_from_knobs(md, 4, expansion=16, active_mult=1)

    assert lo.top_k * lo.d_ff_expert == hi.top_k * hi.d_ff_expert
    assert hi.n_experts * hi.d_ff_expert > lo.n_experts * lo.d_ff_expert


def test_ladder_rounds_expert_count_for_even_sharding():
    cfg = moe_config_from_knobs(512, granularity=3, expansion=3, round_experts_to=4)

    assert cfg.n_experts % 4 == 0


def test_describe_recovers_the_knobs():
    md, G, X, A = 512, 8.0, 8.0, 1.0
    cfg = moe_config_from_knobs(md, G, X, active_mult=A, round_experts_to=0)

    d = describe(md, cfg)

    assert d["G"] == pytest.approx(G)
    assert d["A"] == pytest.approx(A)
    assert d["X"] == pytest.approx(X)
    assert d["sparsity"] == pytest.approx(X / A)


# ---------------------------------------------------------------------------
# Config, optimizer, checkpoint compatibility
# ---------------------------------------------------------------------------


def test_pool_index_counts_over_moe_layers_not_raw_depth():
    cfg = tiny_moe_config(moe_every=2, share_blocks=2)

    assert cfg.moe_layer_ids(8) == (0, 2, 4, 6)
    # Naive layer_idx % share_blocks would map every MoE layer onto pool 0.
    assert [cfg.pool_index(i, 8) for i in (0, 2, 4, 6)] == [0, 1, 0, 1]


def test_router_rejects_top_k_larger_than_experts():
    with pytest.raises(AssertionError):
        Router(8, n_experts=4, top_k=8, balancer=make_balancer("noop", 4))


def test_expert_params_get_their_own_adamw_group_at_expert_lr():
    model = build_gpt(moe=tiny_moe_config(expert_lr=0.004), n_layer=4)
    dmodel_scale = (model.config.n_embd / 768) ** -0.5

    optimizer = model.setup_optimizer(matrix_lr=0.02)

    expert_ids = {id(p) for m in model.modules() if isinstance(m, Experts) for p in m.parameters()}
    groups = [
        g
        for g in optimizer.param_groups
        if g["params"] and all(id(p) in expert_ids for p in g["params"])
    ]
    assert len(groups) == 1, "3D expert tensors need exactly one dedicated group"
    assert groups[0]["kind"] == "adamw", "Muon is 2D-only"
    assert groups[0]["lr"] == pytest.approx(0.004 * dmodel_scale)
    # ...and they must not have leaked into a Muon group.
    for g in optimizer.param_groups:
        if g["kind"] == "muon":
            assert not any(id(p) in expert_ids for p in g["params"])


def test_shared_expert_gate_is_not_a_muon_matrix():
    """A (1, d_model) gate weight is a vector wearing a matrix costume. Left as
    an nn.Linear it lands in a Muon group, where Newton-Schulz on a single-row
    matrix is meaningless and crashes Inductor codegen during optimizer compile.
    """
    model = build_gpt(moe=tiny_moe_config(shared_d_ff=16), n_layer=4)

    gates = [
        b.mlp.shared.gate for b in model.transformer.h if b.is_moe and b.mlp.shared
    ]
    assert gates, "expected shared experts on the MoE layers"
    assert all(g.ndim == 1 for g in gates)

    optimizer = model.setup_optimizer()
    gate_ids = {id(g) for g in gates}
    for group in optimizer.param_groups:
        if group["kind"] == "muon":
            assert not any(id(p) in gate_ids for p in group["params"])


def test_checkpoint_config_without_moe_patches_to_none():
    from nano_scalemb.checkpoint_manager import _patch_missing_config_keys

    config_dict = {"window_pattern": "L", "engram": None, "mhc": None}

    _patch_missing_config_keys(config_dict)

    assert config_dict["moe"] is None
    assert GPTConfig(n_layer=2, **config_dict).moe is None


def test_checkpoint_config_rehydrates_moe_dict():
    from dataclasses import asdict

    from nano_scalemb.checkpoint_manager import _patch_missing_config_keys

    cfg = tiny_moe_config(share_blocks=2)
    config_dict = {"window_pattern": "L", "engram": None, "mhc": None, "moe": asdict(cfg)}

    _patch_missing_config_keys(config_dict)

    assert isinstance(config_dict["moe"], MoEConfig)
    assert config_dict["moe"] == cfg
    # is_moe_layer() would crash on a plain dict, which is the bug this guards.
    assert config_dict["moe"].is_moe_layer(0)
