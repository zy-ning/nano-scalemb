import torch

from nano_scalemb.mhc import MHCHead, StreamExpand, sinkhorn_knopps


def test_sinkhorn_knopps_produces_doubly_stochastic_matrix():
    torch.manual_seed(0)
    log_alpha = torch.randn(2, 4, 4)

    alpha = sinkhorn_knopps(log_alpha, iters=25)

    row_sums = alpha.sum(dim=-1)
    col_sums = alpha.sum(dim=-2)
    ones = torch.ones_like(row_sums)

    assert torch.allclose(row_sums, ones, atol=1e-3, rtol=1e-3)
    assert torch.allclose(col_sums, ones, atol=1e-3, rtol=1e-3)
    assert torch.isfinite(alpha).all()


def test_stream_expand_shape_and_repeat():
    torch.manual_seed(0)
    B, S, T, D = 2, 4, 3, 5
    x = torch.arange(B * T * D, dtype=torch.float32).view(B, T, D)

    expanded = StreamExpand(S)(x)

    assert expanded.shape == (B * S, T, D)
    # repeat_interleave: each base row appears S consecutive times
    assert torch.equal(expanded.view(B, S, T, D)[:, 0], x)
    assert torch.equal(expanded.view(B, S, T, D)[:, S - 1], x)


def test_mhc_head_learned_shape_finite_and_gradients_flow():
    torch.manual_seed(0)
    B, S, T, D = 2, 4, 3, 5
    residuals = torch.randn(B * S, T, D, requires_grad=True)

    head = MHCHead(num_residual_streams=S, dim=D)
    with torch.no_grad():
        assert head.dynamic_head_fn is not None
        assert head.head_scale is not None
        assert head.head_base is not None
        assert head.norm is not None
        head.dynamic_head_fn.fill_(0.25)
        head.head_scale.fill_(0.5)
        head.head_base.fill_(0.1)
        head.norm.weight.fill_(1.0)

    out = head(residuals)

    assert out.shape == (B, T, D)
    assert torch.isfinite(out).all()

    out.sum().backward()

    assert residuals.grad is not None and torch.isfinite(residuals.grad).all()
    assert head.dynamic_head_fn is not None
    assert head.head_scale is not None
    assert head.head_base is not None
    assert head.dynamic_head_fn.grad is not None and torch.isfinite(head.dynamic_head_fn.grad).all()
    assert head.head_scale.grad is not None and torch.isfinite(head.head_scale.grad).all()
    assert head.head_base.grad is not None and torch.isfinite(head.head_base.grad).all()


def test_mhc_head_single_stream_is_identity():
    torch.manual_seed(0)
    residuals = torch.randn(2, 3, 5)
    head = MHCHead(num_residual_streams=1, dim=5)
    assert torch.equal(head(residuals), residuals)
