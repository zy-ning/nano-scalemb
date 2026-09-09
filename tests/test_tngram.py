"""Unit sanity for the tensorized n-gram (CP) value table. No training.

Asserts:
  (a) forward returns [B, T, n_orders*K, head_dim] and flattens to memory_dim;
  (b) split_key_value returns (values, None);
  (c) the order-2 block differs from the order-3 block (they are not the same read);
  (d) param count == N*V*(K*R) + K*d*R + n_orders*K*R + n_orders*K, and matches the
      param-matched anchors within 0.2% for R in {10, 20};
  (e) active_params_per_token() == K*N*R;
  (f) Theorem 4.1 recovery: zeroing w kills ONLY the lower-order (order-2) block,
      the top-order (order-3) block is untouched;
  (g) seed determinism (same init -> same forward);
  (h) merge hooks raise (a factorized table has no rows to merge).

Run with:
    python -m pytest tests/test_tngram.py -v
"""
import torch

from nano_scalemb.engram import TensorizedNgramMemory


V = 23686          # compressed alphabet (matches the d20 shared-table config)
K = 8              # n_head_per_ngram
HEAD_DIM = 80      # memory_dim(1280) / (n_orders(2) * K(8))
MAX_N = 3
ORDERS = [2, 3]    # ngram_orders(max_ngram_size=3)
N_ORDERS = len(ORDERS)


def make(rank, vocab=V, share_factors=True):
    m = TensorizedNgramMemory(
        vocab_size=vocab,
        head_dim=HEAD_DIM,
        d_model=1280,
        rank=rank,
        n_head_per_ngram=K,
        ngram_orders=list(ORDERS),
        max_ngram_size=MAX_N,
        share_factors=share_factors,
    )
    m.reset_content("none")
    return m


def rand_windows(B, T, vocab=V, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (B, T, MAX_N), generator=g)


def test_forward_shape_and_flatten():
    m = make(10)
    B, T = 2, 7
    out = m(rand_windows(B, T))
    assert out.shape == (B, T, N_ORDERS * K, HEAD_DIM), out.shape
    flat = out.flatten(start_dim=-2)
    assert flat.shape == (B, T, N_ORDERS * K * HEAD_DIM), flat.shape
    assert flat.shape[-1] == 1280
    print("(a) forward [B,T,16,80] -> flatten 1280: OK")


def test_split_key_value():
    m = make(10)
    out = m(rand_windows(1, 4))
    values, keys = m.split_key_value(out)
    assert keys is None
    assert values is out
    print("(b) split_key_value -> (values, None): OK")


def test_order_blocks_differ():
    m = make(12)
    out = m(rand_windows(3, 5, seed=1))
    order2 = out[:, :, :K, :]
    order3 = out[:, :, K:, :]
    assert not torch.allclose(order2, order3), "order-2 and order-3 reads coincide"
    print("(c) order-2 block != order-3 block: OK")


def _param_count(m):
    return sum(p.numel() for p in m.parameters())


def test_param_count_formula_and_anchors():
    for rank, anchor in ((10, 5_685_536), (20, 11_370_359)):
        m = make(rank)
        expected = (
            MAX_N * V * (K * rank)          # A factors
            + K * HEAD_DIM * rank           # F
            + N_ORDERS * K * rank           # w
            + N_ORDERS * K                  # log_scale
        )
        got = _param_count(m)
        assert got == expected, (rank, got, expected)
        rel = abs(got - anchor) / anchor
        assert rel < 0.002, (rank, got, anchor, rel)
        print(f"(d) R={rank}: {got:,} params, {rel*100:.2f}% off anchor {anchor:,}: OK")


def test_active_params_per_token():
    m = make(10)
    assert m.active_params_per_token() == K * MAX_N * 10
    print("(e) active_params_per_token == K*N*R: OK")


def test_theorem_recovery_w_zero_kills_lower_order_only():
    m = make(10)
    w = rand_windows(2, 6, seed=3)
    before = m(w)
    with torch.no_grad():
        m.w.zero_()
    after = m(w)
    # order-3 (top order) uses no w -> unchanged; order-2 absorbs one w -> zeroed
    # (RMSNorm of a zero vector is zero, so the whole lower-order read collapses).
    assert torch.allclose(before[:, :, K:, :], after[:, :, K:, :]), "top order moved"
    assert torch.count_nonzero(after[:, :, :K, :]) == 0, "lower order not zeroed by w=0"
    print("(f) w=0 zeroes only the lower-order block: OK")


def test_seed_determinism():
    torch.manual_seed(0)
    m1 = make(10)
    torch.manual_seed(0)
    m2 = make(10)
    w = rand_windows(2, 5, seed=9)
    assert torch.equal(m1(w), m2(w)), "same seed gave different reads"
    print("(g) seed determinism: OK")


def test_merge_hooks_raise():
    m = make(10)
    for fn in (
        lambda: m._value_matrix(),
        lambda: m._merge_writeback(torch.arange(4)),
        lambda: m._merge_writeback_mean(torch.arange(4), None),
    ):
        try:
            fn()
        except NotImplementedError:
            continue
        raise AssertionError("merge hook should have raised NotImplementedError")
    print("(h) merge hooks raise NotImplementedError: OK")


def test_independent_factors_shape_and_isoparam():
    # Independent A: each order owns its own factors, no w. Iso-param with shared
    # R=10 at R_ind=6, since sum(orders)=5 slots * 6 == max_ngram(3) * 10 for A.
    m = make(6, share_factors=False)
    assert m.w is None
    B, T = 2, 5
    out = m(rand_windows(B, T, seed=4))
    assert out.shape == (B, T, N_ORDERS * K, HEAD_DIM), out.shape
    # A-param count: (2+3) orders * V * K * R_ind
    a_params = sum(w.numel() for w in m._a_weights())
    assert a_params == sum(ORDERS) * V * K * 6, a_params
    shared10_a = MAX_N * V * K * 10
    assert a_params == shared10_a, (a_params, shared10_a)
    print(f"(i) independent A: shape OK, A-params {a_params:,} == shared R=10 A: OK")


def test_independent_orders_are_untied():
    # Zeroing one order's factors must not touch the other order's read.
    m = make(6, share_factors=False)
    w = rand_windows(2, 5, seed=7)
    before = m(w)
    with torch.no_grad():
        for emb in m.A_orders[0]:  # order-2 factors
            emb.weight.zero_()
    after = m(w)
    assert torch.allclose(before[:, :, K:, :], after[:, :, K:, :]), "order-3 moved"
    assert torch.count_nonzero(after[:, :, :K, :]) == 0, "order-2 not zeroed"
    print("(j) independent orders untied (zero order-2 leaves order-3): OK")


def test_independent_active_params():
    m = make(6, share_factors=False)
    # sum(orders) gathers per head (one per used slot), not MAX_N.
    assert m.active_params_per_token() == K * sum(ORDERS) * 6
    print("(k) independent active_params_per_token == K*sum(orders)*R: OK")


if __name__ == "__main__":
    test_forward_shape_and_flatten()
    test_split_key_value()
    test_order_blocks_differ()
    test_param_count_formula_and_anchors()
    test_active_params_per_token()
    test_theorem_recovery_w_zero_kills_lower_order_only()
    test_seed_determinism()
    test_merge_hooks_raise()
    test_independent_factors_shape_and_isoparam()
    test_independent_orders_are_untied()
    test_independent_active_params()
    print("\nALL TN-GRAM UNIT CHECKS PASSED")
