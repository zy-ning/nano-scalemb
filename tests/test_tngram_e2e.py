"""End-to-end smoke test: a small GPT with a TN-gram (CP-factorized) Engram
builds, runs forward+backward on CPU, and reports the expected iso-param counts.

Compression is OFF so the test is hermetic (V_compressed == vocab_size, no
tokenizer artifacts needed). Run with the project venv:

    PYTHONPATH=. ../nanochat-moe/.venv/bin/python tests/test_tngram_e2e.py
"""
import torch

from nano_scalemb.engram import EngramConfig, TensorizedNgramMemory
from nano_scalemb.gpt import GPT, GPTConfig


def build(cp_rank, vocab=512, n_embd=128, n_layer=4, seq=32, share_factors=True):
    # memory_dim = n_orders(2) * n_head_per_ngram(K) * head_dim; pick K/head_dim
    # so it divides n_embd cleanly for the value_proj path.
    engram = EngramConfig(
        layer_ids=(1, 2),
        max_ngram_size=3,
        n_head_per_ngram=4,
        memory_dim=n_embd,          # 2*4*16 = 128
        slot_multiplier=1,
        share_memory=True,
        address_source="tokens",
        use_tokenizer_compression=False,
        value_table="tngram",
        cp_rank=cp_rank,
        tngram_share_factors=share_factors,
        pad_id=0,
    )
    cfg = GPTConfig(
        sequence_len=seq,
        vocab_size=vocab,
        n_layer=n_layer,
        n_head=n_embd // 16,
        n_kv_head=n_embd // 16,
        n_embd=n_embd,
        window_pattern="SSSL",
        engram=engram,
    )
    with torch.device("meta"):
        model = GPT(cfg)
    model.to_empty(device="cpu")
    model.init_weights()
    return model, cfg


def test_shared_table_is_tngram():
    model, _ = build(cp_rank=8)
    table = model.engram_shared_memory
    assert isinstance(table, TensorizedNgramMemory), type(table)
    print("(a) shared Engram table is TensorizedNgramMemory: OK")


def test_forward_backward():
    torch.manual_seed(0)
    model, cfg = build(cp_rank=8)
    # Engram value_proj is zero-initialized (the branch starts as a no-op), so
    # at step 0 no gradient reaches the table. Perturb the read projections so
    # the CP factors are actually in the loss graph.
    with torch.no_grad():
        for e in model.engram_modules.values():
            e.value_proj.weight.normal_(std=0.02)
    B, T = 2, cfg.sequence_len
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    targets = torch.randint(0, cfg.vocab_size, (B, T))
    out = model(idx, targets=targets)
    loss = out if torch.is_tensor(out) else out[0]
    assert torch.isfinite(loss), loss
    loss.backward()
    # A factors and F both receive gradient (the read touches both).
    table = model.engram_shared_memory
    assert table.A[0].weight.grad is not None
    assert torch.count_nonzero(table.A[0].weight.grad) > 0, "A got no gradient"
    assert table.F.grad is not None and torch.count_nonzero(table.F.grad) > 0, "F got no gradient"
    assert table.w.grad is not None and torch.count_nonzero(table.w.grad) > 0, "w got no gradient"
    print(f"(b) forward+backward on CPU, loss={loss.item():.4f}, grads flow to A/F/w: OK")


def test_param_count_reported():
    # The model-level assert in the param breakdown (total == sum(params))
    # exercises that F/w/log_scale all land in a counted optimizer bucket.
    for cp_rank in (8, 16):
        model, _ = build(cp_rank=cp_rank)
        model._split_transformer_block_params()  # trips the bucket-coverage assert
        n = sum(p.numel() for p in model.parameters())
        assert n > 0
        print(f"(c) cp_rank={cp_rank}: {n:,} total params, optimizer split covers all: OK")


def test_independent_factors_forward_backward():
    # Independent-A path: grads must flow to the per-order factors and F, and the
    # optimizer split must cover A_orders (no w in this mode).
    torch.manual_seed(0)
    model, cfg = build(cp_rank=6, share_factors=False)
    with torch.no_grad():
        for e in model.engram_modules.values():
            e.value_proj.weight.normal_(std=0.02)
    B, T = 2, cfg.sequence_len
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    targets = torch.randint(0, cfg.vocab_size, (B, T))
    out = model(idx, targets=targets)
    loss = out if torch.is_tensor(out) else out[0]
    assert torch.isfinite(loss), loss
    loss.backward()
    table = model.engram_shared_memory
    assert table.w is None
    a0 = table.A_orders[0][0].weight
    assert a0.grad is not None and torch.count_nonzero(a0.grad) > 0, "A_orders no grad"
    assert table.F.grad is not None and torch.count_nonzero(table.F.grad) > 0, "F no grad"
    model._split_transformer_block_params()  # bucket-coverage assert covers A_orders
    print(f"(d) independent-A forward+backward, loss={loss.item():.4f}, split covers all: OK")


if __name__ == "__main__":
    test_shared_table_is_tngram()
    test_forward_backward()
    test_param_count_reported()
    test_independent_factors_forward_backward()
    print("\nALL TN-GRAM E2E SMOKE CHECKS PASSED")
