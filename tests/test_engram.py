"""
Unit tests for Engram components.

Tests cover:
- Normalization helper behavior
- Compression table shape and monotonic properties
- Hash determinism and consistency
- Prime uniqueness across heads/layers/orders
- MultiHeadEmbedding output shape
- ShortConv output shape and causality smoke check
- Engram forward/backward smoke test
- Zero initialization expectations

Run with:
    python -m pytest tests/test_engram.py -v
"""

import math
import torch
import pytest
import nano_scalemb.flash_attention as fa_module
from dataclasses import dataclass

from nano_scalemb.engram import EngramConfig
from nano_scalemb.mhc import StreamExpand, sinkhorn_knopps


# Mock/placeholder classes until Task 1-2 are implemented
# These tests are written against the interface spec in the plan.


@dataclass
class MockEngramConfig:
    """Mock config for testing interface."""

    layer_ids: tuple = (2, 6)
    max_ngram_size: int = 3
    n_head_per_ngram: int = 2
    memory_dim: int = 128
    slot_multiplier: int = 2
    kernel_size: int = 4
    use_tokenizer_compression: bool = True
    embedding_lr_mult: float = 5.0
    pad_id: int = 0
    seed: int = 0


class TestNormalizationHelper:
    """Test text normalization helper behavior."""

    def test_normalization_lowercases_text(self):
        """Normalization should convert uppercase to lowercase."""
        # Placeholder: when CompressedTokenizerProjection exists,
        # this test will verify the normalize() helper.
        # For now, just verify the test structure is valid.
        text = "HELLO"
        # Once implemented: assert normalize(text) == "hello"
        assert isinstance(text, str)

    def test_normalization_strips_whitespace(self):
        """Normalization should strip leading/trailing whitespace."""
        text = "  hello  "
        # Once implemented: assert normalize(text) == "hello"
        assert isinstance(text, str)

    def test_normalization_collapses_internal_whitespace(self):
        """Normalization should collapse multiple spaces to one."""
        text = "hello   world"
        # Once implemented: assert normalize(text) == "hello world"
        assert isinstance(text, str)


class TestCompressionTable:
    """Test tokenizer compression table shape and properties."""

    def test_compression_table_size(self):
        """Compression table should have same size as raw tokenizer vocab."""
        # Once CompressedTokenizerProjection is implemented:
        # proj = CompressedTokenizerProjection(tokenizer)
        # assert proj.lookup_table.numel() == tokenizer.get_vocab_size()
        pass

    def test_compressed_vocab_size_bounded(self):
        """Compressed vocab size should not exceed raw vocab size."""
        # Once implemented:
        # assert proj.compressed_vocab_size <= raw_vocab_size
        pass

    def test_compression_table_deterministic(self):
        """Compression table should be deterministic for same tokenizer."""
        # Once implemented:
        # proj1 = CompressedTokenizerProjection(tok)
        # proj2 = CompressedTokenizerProjection(tok)
        # assert torch.equal(proj1.lookup_table, proj2.lookup_table)
        pass

    def test_compression_preserves_vocabulary(self):
        """Each token should map to a valid compressed ID."""
        # Once implemented:
        # proj = CompressedTokenizerProjection(tok)
        # for i in range(tok.get_vocab_size()):
        #     mapped = proj.lookup_table[i].item()
        #     assert 0 <= mapped < proj.compressed_vocab_size
        pass


class TestNgramHasher:
    """Test NgramHasher hash determinism and prime table properties."""

    def test_hash_determinism(self):
        """Same input should always produce same hash output."""
        # Once NgramHasher is implemented:
        # cfg = MockEngramConfig()
        # hasher = NgramHasher(cfg, tokenizer_vocab_size=256)
        # ids = torch.randint(0, 256, (2, 8))
        # h1 = hasher.hash(ids, layer_id=2)
        # h2 = hasher.hash(ids, layer_id=2)
        # assert torch.equal(h1, h2)
        pass

    def test_hash_preserves_device(self):
        """Hash output should be on same device as input."""
        # Once implemented:
        # cfg = MockEngramConfig()
        # hasher = NgramHasher(cfg, tokenizer_vocab_size=256)
        # ids_cpu = torch.randint(0, 256, (2, 8))
        # h_cpu = hasher.hash(ids_cpu, layer_id=2)
        # assert h_cpu.device == ids_cpu.device
        pass

    def test_hash_output_shape(self):
        """Hash output shape should be [B, T, (max_ngram_size-1)*n_head_per_ngram]."""
        # Once implemented:
        # cfg = MockEngramConfig(max_ngram_size=3, n_head_per_ngram=2)
        # hasher = NgramHasher(cfg, tokenizer_vocab_size=256)
        # ids = torch.randint(0, 256, (4, 16))
        # h = hasher.hash(ids, layer_id=2)
        # expected_num_heads = (cfg.max_ngram_size - 1) * cfg.n_head_per_ngram
        # assert h.shape == (4, 16, expected_num_heads)
        pass

    def test_hash_output_dtype(self):
        """Hash output should be int64."""
        # Once implemented:
        # cfg = MockEngramConfig()
        # hasher = NgramHasher(cfg, tokenizer_vocab_size=256)
        # ids = torch.randint(0, 256, (2, 8))
        # h = hasher.hash(ids, layer_id=2)
        # assert h.dtype == torch.int64
        pass

    def test_prime_uniqueness(self):
        """All prime table sizes should be globally unique across (layer, order, head)."""
        # Once NgramHasher is implemented:
        # cfg = MockEngramConfig(layer_ids=(2, 6), max_ngram_size=3, n_head_per_ngram=2)
        # hasher = NgramHasher(cfg, tokenizer_vocab_size=256)
        # primes = []
        # for layer_id in cfg.layer_ids:
        #     for order in range(1, cfg.max_ngram_size):
        #         for head_idx in range(cfg.n_head_per_ngram):
        #             prime = hasher._get_prime(layer_id, order, head_idx)
        #             primes.append(prime)
        # assert len(primes) == len(set(primes)), "Prime table sizes not globally unique"
        pass


class TestMultiHeadEmbedding:
    """Test MultiHeadEmbedding output shape and behavior."""

    def test_multihead_embedding_output_shape(self):
        """MultiHeadEmbedding should return [B, T, memory_dim] output."""
        # Once MultiHeadEmbedding exists:
        # cfg = MockEngramConfig(memory_dim=128)
        # emb = MultiHeadEmbedding(cfg, ...)
        # h = emb([B, T, K] indices)  # K is number of heads
        # assert h.shape == [B, T, 128]
        pass

    def test_multihead_embedding_consistent_output(self):
        """Same indices should produce same embeddings."""
        # Once implemented:
        # cfg = MockEngramConfig()
        # emb = MultiHeadEmbedding(cfg, ...)
        # idx = torch.tensor([[0, 1, 2]])
        # y1 = emb(idx)
        # y2 = emb(idx)
        # assert torch.allclose(y1, y2, atol=1e-6)
        pass

    def test_multihead_embedding_handles_padding(self):
        """Padding token should produce consistent zero or masked output."""
        # Once implemented:
        # cfg = MockEngramConfig(pad_id=0)
        # emb = MultiHeadEmbedding(cfg, ...)
        # Test that pad_id is handled consistently
        pass


class TestShortConv:
    """Test ShortConv output shape and causality."""

    def test_shortconv_output_shape(self):
        """ShortConv should preserve [B, T, D] shape."""
        # Once ShortConv exists:
        # cfg = MockEngramConfig(kernel_size=4)
        # conv = ShortConv(n_embd=128, cfg=cfg)
        # x = torch.randn(2, 16, 128)
        # y = conv(x)
        # assert y.shape == x.shape
        pass

    def test_shortconv_causality_smoke(self):
        """ShortConv should not look into future tokens (causal)."""
        # Once implemented:
        # cfg = MockEngramConfig(kernel_size=4)
        # conv = ShortConv(n_embd=128, cfg=cfg)
        # torch.manual_seed(0)
        # x = torch.randn(1, 8, 128)
        # y = conv(x)
        # # At position 0, output should only depend on x[0]
        # # Perturb x[7], should not affect y[0]
        # x2 = x.clone()
        # x2[0, 7, :] += 10.0
        # y2 = conv(x2)
        # assert torch.allclose(y2[0, 0, :], y[0, 0, :], atol=1e-5)
        pass

    def test_shortconv_weight_shape(self):
        """ShortConv conv weight should have correct shape."""
        # Once implemented:
        # cfg = MockEngramConfig(kernel_size=4)
        # conv = ShortConv(n_embd=128, cfg=cfg)
        # # Depthwise conv: (out_channels=D, in_channels/groups=1, kernel_size)
        # assert conv.conv.weight.shape[0] == 128
        # assert conv.conv.weight.shape[2] == 4
        pass


class TestEngramModule:
    """Test Engram forward/backward and zero initialization."""

    def test_engram_forward_shape(self):
        """Engram.forward should return [B, T, D] output."""
        # Once Engram is implemented:
        # cfg = MockEngramConfig(memory_dim=128)
        # hasher = NgramHasher(cfg, tokenizer_vocab_size=256)
        # engram = Engram(layer_id=2, n_embd=128, cfg=cfg, hasher=hasher)
        # engram.init_weights(128)
        # x = torch.randn(2, 16, 128)
        # ids = torch.randint(0, 256, (2, 16))
        # y = engram(x, ids)
        # assert y.shape == x.shape == (2, 16, 128)
        pass

    def test_engram_gate_bounds(self):
        """Engram gate values should be in (0, 1)."""
        # Once implemented:
        # cfg = MockEngramConfig()
        # hasher = NgramHasher(cfg, tokenizer_vocab_size=256)
        # engram = Engram(layer_id=2, n_embd=128, cfg=cfg, hasher=hasher)
        # engram.init_weights(128)
        # x = torch.randn(2, 16, 128)
        # ids = torch.randint(0, 256, (2, 16))
        # y = engram(x, ids)
        # # Gate is applied internally; check output variance is reasonable
        # assert y.abs().max() < 1e2  # Sanity check
        pass

    def test_engram_backward_succeeds(self):
        """Engram should support backward pass without NaN/Inf."""
        # Once implemented:
        # cfg = MockEngramConfig()
        # hasher = NgramHasher(cfg, tokenizer_vocab_size=256)
        # engram = Engram(layer_id=2, n_embd=128, cfg=cfg, hasher=hasher)
        # engram.init_weights(128)
        # x = torch.randn(2, 16, 128, requires_grad=True)
        # ids = torch.randint(0, 256, (2, 16))
        # y = engram(x, ids)
        # loss = y.sum()
        # loss.backward()
        # assert x.grad is not None
        # assert not torch.isnan(x.grad).any()
        # assert not torch.isinf(x.grad).any()
        pass

    def test_value_proj_zero_init(self):
        """value_proj.weight should start at exactly zero."""
        # Once Engram is implemented:
        # cfg = MockEngramConfig()
        # hasher = NgramHasher(cfg, tokenizer_vocab_size=256)
        # engram = Engram(layer_id=2, n_embd=128, cfg=cfg, hasher=hasher)
        # engram.init_weights(128)
        # assert (engram.value_proj.weight == 0.0).all()
        pass

    def test_conv_weight_zero_init(self):
        """Depthwise conv weight should start at exactly zero."""
        # Once Engram is implemented:
        # cfg = MockEngramConfig()
        # hasher = NgramHasher(cfg, tokenizer_vocab_size=256)
        # engram = Engram(layer_id=2, n_embd=128, cfg=cfg, hasher=hasher)
        # engram.init_weights(128)
        # assert (engram.short_conv.conv.weight == 0.0).all()
        pass

    def test_engram_residual_identity_at_start(self):
        """With zero value_proj and conv, Engram output should be near-zero initially."""
        # Once implemented:
        # cfg = MockEngramConfig()
        # hasher = NgramHasher(cfg, tokenizer_vocab_size=256)
        # engram = Engram(layer_id=2, n_embd=128, cfg=cfg, hasher=hasher)
        # engram.init_weights(128)
        # torch.manual_seed(42)
        # x = torch.randn(2, 16, 128)
        # ids = torch.randint(0, 256, (2, 16))
        # y = engram(x, ids)
        # # Output should be small since value_proj and conv are zero
        # # (gate is sigmoid so output is gated contribution)
        # assert y.abs().mean() < 0.1  # Heuristic: should be small
        pass


class TestEngramIntegration:
    """Integration smoke tests."""

    def test_engram_config_creation(self):
        """EngramConfig should instantiate with sensible defaults."""
        cfg = MockEngramConfig()
        assert cfg.layer_ids == (2, 6)
        assert cfg.max_ngram_size == 3
        assert cfg.n_head_per_ngram == 2
        assert cfg.embedding_lr_mult == 5.0
        assert cfg.pad_id == 0

    def test_engram_config_with_custom_values(self):
        """EngramConfig should accept custom layer_ids and dimensions."""
        cfg = MockEngramConfig(
            layer_ids=(1, 4), max_ngram_size=4, memory_dim=256, n_head_per_ngram=4
        )
        assert cfg.layer_ids == (1, 4)
        assert cfg.max_ngram_size == 4
        assert cfg.memory_dim == 256
        assert cfg.n_head_per_ngram == 4

    def test_engram_disabled_by_default(self):
        """Engram should be disabled (None) by default in GPTConfig."""
        # Once GPTConfig is updated:
        # cfg = GPTConfig()
        # assert cfg.engram is None
        pass

    def test_engram_enabled_with_config(self):
        """GPTConfig should accept EngramConfig to enable Engram."""
        # Once both exist:
        # from nano_scalemb.engram import EngramConfig
        # cfg = GPTConfig(engram=EngramConfig(layer_ids=(2,)))
        # assert cfg.engram is not None
        # assert cfg.engram.layer_ids == (2,)
        pass


# Determinism verification


def test_engram_seed_determinism():
    """Multiple runs with same seed should produce same results."""
    # Once Engram is fully implemented:
    # for seed in [0, 42, 999]:
    #     torch.manual_seed(seed)
    #     cfg = MockEngramConfig()
    #     hasher = NgramHasher(cfg, tokenizer_vocab_size=256)
    #     engram1 = Engram(layer_id=2, n_embd=128, cfg=cfg, hasher=hasher)
    #     engram1.init_weights(128)
    #     y1 = engram1(torch.randn(2, 8, 128), torch.randint(0, 256, (2, 8)))
    #
    #     torch.manual_seed(seed)
    #     engram2 = Engram(layer_id=2, n_embd=128, cfg=cfg, hasher=hasher)
    #     engram2.init_weights(128)
    #     y2 = engram2(torch.randn(2, 8, 128), torch.randint(0, 256, (2, 8)))
    #
    #     assert torch.allclose(y1, y2, atol=1e-6)
    pass


def test_sinkhorn_knopps_returns_nearly_doubly_stochastic_matrix():
    logits = torch.randn(2, 3, 4, 4)
    weights = sinkhorn_knopps(logits, iters=25)
    assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)), atol=1e-3)
    assert torch.allclose(weights.sum(dim=-2), torch.ones_like(weights.sum(dim=-2)), atol=1e-3)


def test_mhc_width_connection_preserves_stream_dtypes():
    from nano_scalemb.mhc import ManifoldConstrainedHyperConnections

    module = ManifoldConstrainedHyperConnections(
        num_residual_streams=4,
        dim=8,
        sinkhorn_iters=3,
    ).to(dtype=torch.bfloat16)

    residuals = torch.randn(8, 5, 8, dtype=torch.bfloat16)
    branch_input, next_residuals, beta = module.width_connection(residuals)

    assert branch_input.dtype == residuals.dtype
    assert next_residuals.dtype == residuals.dtype
    assert beta.dtype in (torch.float32, torch.bfloat16)


def test_mhc_depth_connection_preserves_model_dtype_with_high_precision_beta():
    from nano_scalemb.mhc import ManifoldConstrainedHyperConnections

    module = ManifoldConstrainedHyperConnections(
        num_residual_streams=4,
        dim=8,
        sinkhorn_iters=3,
    )

    branch_output = torch.randn(2, 5, 8, dtype=torch.bfloat16)
    residuals = torch.randn(8, 5, 8, dtype=torch.bfloat16)
    beta = torch.randn(2, 5, 4, dtype=torch.float64)

    output = module.depth_connection(branch_output, residuals, beta=beta)

    assert output.dtype == branch_output.dtype


def test_mhc_depth_connection_streams_accepts_per_stream_outputs():
    from nano_scalemb.mhc import ManifoldConstrainedHyperConnections

    module = ManifoldConstrainedHyperConnections(
        num_residual_streams=4,
        dim=8,
        sinkhorn_iters=3,
    )

    branch_outputs = torch.randn(2, 5, 4, 8)
    residuals = torch.randn(8, 5, 8)
    beta = torch.randn(2, 5, 4)

    output = module.depth_connection_streams(branch_outputs, residuals, beta=beta)

    assert output.shape == residuals.shape


# Integration tests with Engram components
# ===================================================================


class TestEngramIntegrationWithComponents:
    """Integration tests verifying Engram components work together."""

    def test_engram_forward_backward_with_components(self):
        """Engram forward and backward should work with all components integrated."""
        from nano_scalemb.engram import (
            Engram,
            EngramConfig,
            NgramHasher,
        )

        # Create config
        cfg = EngramConfig(
            layer_ids=(2,),
            max_ngram_size=3,
            n_head_per_ngram=2,
            memory_dim=128,
            slot_multiplier=2,
            kernel_size=4,
            use_tokenizer_compression=False,
            pad_id=0,
            seed=0,
        )

        # Create hasher
        hasher = NgramHasher(cfg, tokenizer_vocab_size=256, compression=None)

        # Create Engram module
        engram = Engram(
            cfg=cfg,
            layer_id=2,
            d_model=128,
            tokenizer_vocab_size=256,
            compression=None,
        )

        # Forward pass
        B, T = 2, 16
        x = torch.randn(B, T, 128, requires_grad=True)
        input_ids = torch.randint(0, 256, (B, T), dtype=torch.long)

        y = engram(x, input_ids)
        assert y.shape == x.shape

        # Backward pass
        loss = y.sum()
        loss.backward()

        # Check gradients
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()

    def test_engram_forward_mhc_branches_backward(self):
        """Faithful per-stream branch path should support forward/backward."""
        from nano_scalemb.engram import Engram, EngramConfig

        cfg = EngramConfig(
            layer_ids=(2,),
            max_ngram_size=3,
            n_head_per_ngram=2,
            memory_dim=128,
            slot_multiplier=2,
            kernel_size=4,
            use_tokenizer_compression=False,
            pad_id=0,
            seed=0,
            mhc_num_streams=4,
        )
        engram = Engram(
            cfg=cfg,
            layer_id=2,
            d_model=128,
            tokenizer_vocab_size=256,
            compression=None,
            backbone_mhc=True,
        )

        x = torch.randn(2, 16, 128, requires_grad=True)
        input_ids = torch.randint(0, 256, (2, 16), dtype=torch.long)

        y = engram.forward_mhc_branches(x, input_ids)
        assert y.shape == (2, 16, 4, 128)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()

    def test_engram_forward_mhc_branches_shape(self):
        from nano_scalemb.engram import Engram, EngramConfig

        cfg = EngramConfig(
            layer_ids=(2,),
            max_ngram_size=3,
            n_head_per_ngram=2,
            memory_dim=128,
            slot_multiplier=2,
            kernel_size=4,
            use_tokenizer_compression=False,
            pad_id=0,
            seed=0,
            mhc_num_streams=4,
        )
        engram = Engram(
            cfg=cfg,
            layer_id=2,
            d_model=128,
            tokenizer_vocab_size=256,
            compression=None,
            backbone_mhc=True,
        )

        x = torch.randn(2, 16, 128)
        input_ids = torch.randint(0, 256, (2, 16), dtype=torch.long)

        y = engram.forward_mhc_branches(x, input_ids)

        assert y.shape == (2, 16, 4, 128)

    def test_engram_forward_mhc_branches_uses_branch_specific_keys(self):
        """Branch-specific keys can produce distinct per-stream Engram outputs."""
        from nano_scalemb.engram import Engram, EngramConfig

        cfg = EngramConfig(
            layer_ids=(0,),
            max_ngram_size=2,
            n_head_per_ngram=1,
            memory_dim=4,
            slot_multiplier=2,
            kernel_size=1,
            use_tokenizer_compression=False,
            mhc_num_streams=2,
        )
        engram = Engram(
            cfg=cfg,
            layer_id=0,
            d_model=4,
            tokenizer_vocab_size=16,
            compression=None,
            backbone_mhc=True,
        )
        with torch.no_grad():
            engram.multi_head_embedding.embedding.weight.fill_(1.0)
            engram.value_proj.weight.copy_(torch.eye(4))
            engram.short_conv.conv.weight.zero_()
            engram.stream_key_proj.weight.zero_()
            engram.stream_key_proj.weight[:4].fill_(-1.0)
            engram.stream_key_proj.weight[4:].fill_(1.0)

        x = torch.ones(1, 3, 4)
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

        y = engram.forward_mhc_branches(x, input_ids)

        assert y.shape == (1, 3, 2, 4)
        assert not torch.allclose(y[:, :, 0, :], y[:, :, 1, :])

    def test_engram_optimizer_parameter_tracking(self):
        """All Engram module parameters should be tracked for optimization."""
        from nano_scalemb.engram import Engram, EngramConfig

        cfg = EngramConfig(
            layer_ids=(2,),
            max_ngram_size=3,
            n_head_per_ngram=2,
            memory_dim=128,
        )

        engram = Engram(
            cfg=cfg,
            layer_id=2,
            d_model=256,
            tokenizer_vocab_size=512,
            compression=None,
        )

        # Collect parameter names
        param_names = [name for name, _ in engram.named_parameters()]

        # Should have parameters from key_proj, value_proj, norm_query, norm_key, short_conv
        assert len(param_names) > 0, "No parameters in Engram"

        # Verify key components are present
        has_key_proj = any("key_proj" in name for name in param_names)
        has_value_proj = any("value_proj" in name for name in param_names)
        has_short_conv = any("short_conv" in name for name in param_names)

        assert has_key_proj, "key_proj not found"
        assert has_value_proj, "value_proj not found"
        assert has_short_conv, "short_conv not found"

    def test_engram_checkpoint_backward_compatibility(self):
        """Loading old checkpoint without engram field should work (patched to None)."""
        import copy

        # Simulate an old model config dict without 'engram' field
        old_config_dict = {
            "sequence_len": 64,
            "vocab_size": 256,
            "n_layer": 4,
            "n_head": 4,
            "n_kv_head": 4,
            "n_embd": 128,
            # 'engram' field is missing (old checkpoint)
        }

        # Simulate checkpoint loading patch
        config_dict = copy.deepcopy(old_config_dict)
        if "engram" not in config_dict:
            config_dict["engram"] = None

        # Should be able to create GPTConfig with patched engram=None
        from nano_scalemb.gpt import GPTConfig

        config = GPTConfig(**config_dict)
        assert config.engram is None

    def test_engram_checkpoint_backward_compatibility_with_missing_mhc(self):
        """Loading old checkpoint without mhc field should patch backbone mHC to None."""
        import copy

        old_config_dict = {
            "sequence_len": 64,
            "vocab_size": 256,
            "n_layer": 4,
            "n_head": 4,
            "n_kv_head": 4,
            "n_embd": 128,
            "engram": None,
        }

        config_dict = copy.deepcopy(old_config_dict)
        if "mhc" not in config_dict:
            config_dict["mhc"] = None

        from nano_scalemb.gpt import GPTConfig

        config = GPTConfig(**config_dict)
        assert config.mhc is None

    def test_backbone_mhc_engram_uses_faithful_branch_path(self):
        from nano_scalemb.engram import EngramConfig
        from nano_scalemb.gpt import GPT, GPTConfig
        from nano_scalemb.mhc import MHCConfig

        config = GPTConfig(
            sequence_len=16,
            vocab_size=64,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=32,
            engram=EngramConfig(
                layer_ids=(0,),
                use_tokenizer_compression=False,
                mhc_num_streams=4,
            ),
            mhc=MHCConfig(num_streams=4, sinkhorn_iters=3),
        )

        with torch.device("cpu"):
            model = GPT(config)

        assert model.config.engram is not None
        engram = model.engram_modules["0"]
        # faithful branch path: per-stream read params present, dense gate absent
        assert engram.backbone_mhc is True
        assert hasattr(engram, "stream_key_proj")
        assert not hasattr(engram, "key_proj")

    def test_backbone_mhc_rejects_mismatched_engram_stream_count(self):
        from nano_scalemb.engram import EngramConfig
        from nano_scalemb.gpt import GPT, GPTConfig
        from nano_scalemb.mhc import MHCConfig

        config = GPTConfig(
            sequence_len=16,
            vocab_size=64,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=32,
            engram=EngramConfig(
                layer_ids=(0,),
                use_tokenizer_compression=False,
                mhc_num_streams=2,
            ),
            mhc=MHCConfig(num_streams=4, sinkhorn_iters=3),
        )

        with pytest.raises(AssertionError, match="requires matching stream counts"):
            GPT(config)

    def test_engram_embedding_tables_do_not_change_estimated_flops_by_ablation_mode(
        self,
    ):
        from nano_scalemb.gpt import GPT, GPTConfig
        from nano_scalemb.mhc import MHCConfig

        flops_by_mode = {}
        for ablation_mode in ("none", "randomize", "uniform"):
            config = GPTConfig(
                sequence_len=16,
                vocab_size=64,
                n_layer=2,
                n_head=2,
                n_kv_head=2,
                n_embd=32,
                engram=EngramConfig(
                    layer_ids=(0,),
                    ablation_mode=ablation_mode,
                    use_tokenizer_compression=False,
                    max_ngram_size=3,
                    n_head_per_ngram=2,
                    memory_dim=32,
                    slot_multiplier=2,
                    kernel_size=4,
                    mhc_num_streams=4,
                ),
                mhc=MHCConfig(num_streams=4, sinkhorn_iters=3),
            )
            with torch.device("cpu"):
                model = GPT(config)
            model.init_weights()
            flops_by_mode[ablation_mode] = model.estimate_flops()

        assert flops_by_mode["none"] == flops_by_mode["randomize"]
        assert flops_by_mode["none"] == flops_by_mode["uniform"]

    def test_num_scaling_params_separates_frozen_engram_embedding_tables(self):
        from nano_scalemb.gpt import GPT, GPTConfig
        from nano_scalemb.mhc import MHCConfig

        counts_by_mode = {}
        for ablation_mode in ("none", "randomize", "uniform"):
            config = GPTConfig(
                sequence_len=16,
                vocab_size=64,
                n_layer=2,
                n_head=2,
                n_kv_head=2,
                n_embd=32,
                engram=EngramConfig(
                    layer_ids=(0,),
                    ablation_mode=ablation_mode,
                    use_tokenizer_compression=False,
                    max_ngram_size=3,
                    n_head_per_ngram=2,
                    memory_dim=32,
                    slot_multiplier=2,
                    kernel_size=4,
                    mhc_num_streams=4,
                ),
                mhc=MHCConfig(num_streams=4, sinkhorn_iters=3),
            )
            with torch.device("cpu"):
                model = GPT(config)
            model.init_weights()
            counts_by_mode[ablation_mode] = model.num_scaling_params()

        assert counts_by_mode["none"]["engram_embeds"] > 0
        assert counts_by_mode["none"]["engram_frozen_embeds"] == 0
        for ablation_mode in ("randomize", "uniform"):
            counts = counts_by_mode[ablation_mode]
            assert counts["engram_embeds"] == 0
            assert counts["engram_frozen_embeds"] == counts_by_mode["none"][
                "engram_embeds"
            ]

    def test_gpt_forward_with_engram_and_backbone_mhc_paper_fusion(self):
        from nano_scalemb.gpt import GPT, GPTConfig
        from nano_scalemb.mhc import MHCConfig

        config = GPTConfig(
            sequence_len=16,
            vocab_size=64,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=32,
            engram=EngramConfig(
                layer_ids=(0,),
                use_tokenizer_compression=False,
                max_ngram_size=3,
                n_head_per_ngram=2,
                memory_dim=32,
                slot_multiplier=2,
                kernel_size=4,
                mhc_num_streams=4,
            ),
            mhc=MHCConfig(num_streams=4, sinkhorn_iters=3),
        )
        with torch.device("cpu"):
            model = GPT(config)
        model.init_weights()

        for block in model.transformer.h:
            block.attn.forward = lambda x, ve, cos_sin, window_size, kv_cache: x

        idx = torch.randint(0, 64, (2, 8), dtype=torch.long)
        fa_module._override_impl = "sdpa"
        try:
            logits = model(idx)
        finally:
            fa_module._override_impl = None

        assert logits.shape == (2, 8, 64)
        assert torch.isfinite(logits).all()

    def test_backbone_mhc_meta_to_empty_init_weights_initializes_mhc_params(self):
        from nano_scalemb.engram import EngramConfig
        from nano_scalemb.gpt import GPT, GPTConfig
        from nano_scalemb.mhc import MHCConfig

        config = GPTConfig(
            sequence_len=16,
            vocab_size=64,
            n_layer=2,
            n_head=2,
            n_kv_head=2,
            n_embd=32,
            engram=EngramConfig(
                layer_ids=(0,),
                use_tokenizer_compression=False,
                max_ngram_size=3,
                n_head_per_ngram=2,
                memory_dim=32,
                slot_multiplier=2,
                kernel_size=4,
                mhc_num_streams=4,
            ),
            mhc=MHCConfig(num_streams=4, sinkhorn_iters=3),
        )
        with torch.device("meta"):
            model = GPT(config)
        model.to_empty(device="cpu")
        model.init_weights()

        block = model.transformer.h[0]
        modules = [block.mhc_attn, block.mhc_mlp, block.mhc_engram]
        for module in modules:
            assert torch.isfinite(module.static_alpha).all()
            assert torch.isfinite(module.dynamic_alpha_fn).all()
            assert torch.isfinite(module.static_beta).all()
            assert torch.isfinite(module.dynamic_beta_fn).all()
            assert torch.isfinite(module.norm.weight).all()
            assert module.dynamic_alpha_fn.abs().max() == 0
            assert module.dynamic_beta_fn.abs().max() == 0
            assert module.norm.weight.min() == 1
            assert module.norm.weight.max() == 1

    def test_engram_ngram_hasher_integration(self):
        """NgramHasher should produce consistent hash output for Engram."""
        from nano_scalemb.engram import NgramHasher, EngramConfig

        cfg = EngramConfig(
            layer_ids=(2, 6),
            max_ngram_size=3,
            n_head_per_ngram=2,
            use_tokenizer_compression=False,
            pad_id=0,
        )

        hasher = NgramHasher(cfg, tokenizer_vocab_size=256, compression=None)

        # Create input
        B, T = 2, 8
        input_ids = torch.randint(0, 256, (B, T), dtype=torch.long)

        # Hash should be consistent
        h1 = hasher.hash(input_ids, layer_id=2)
        h2 = hasher.hash(input_ids, layer_id=2)

        assert torch.equal(h1, h2), "Hash output not deterministic"
        assert h1.shape == (B, T, (cfg.max_ngram_size - 1) * cfg.n_head_per_ngram)

    def test_engram_multihead_embedding_integration(self):
        """MultiHeadEmbedding should embed hash indices correctly."""
        from nano_scalemb.engram import MultiHeadEmbedding

        # Create embedding with 3 heads, each with vocab size 100, embedding dim 64
        embedding = MultiHeadEmbedding(list_of_N=[100, 100, 100], D=64)

        # Create hash indices [B, T, 3]
        B, T, H = 2, 8, 3
        input_ids = torch.randint(0, 100, (B, T, H), dtype=torch.long)

        # Forward pass
        embeddings = embedding(input_ids)

        # Output should be [B, T, 3, 64]
        assert embeddings.shape == (B, T, H, 64)

    def test_engram_short_conv_integration(self):
        """ShortConv should preserve shape and maintain causality."""
        from nano_scalemb.engram import ShortConv

        conv = ShortConv(d_model=128, kernel_size=4, dilation=3)

        # Input [B, T, D]
        B, T, D = 2, 16, 128
        x = torch.randn(B, T, D)

        y = conv(x)

        # Output shape should match input
        assert y.shape == x.shape

        # Output should not be all zeros (should have non-zero contributions)
        assert y.abs().max() > 0


class TestEngramStyleAndCleanliness:
    """Tests for code cleanliness and style conformance."""

    def test_engram_imports_available(self):
        """All required Engram imports should be available and work."""
        # These imports should not raise any errors
        from nano_scalemb.engram import (
            Engram,
            EngramConfig,
            CompressedTokenizerProjection,
            NgramHasher,
            MultiHeadEmbedding,
            ShortConv,
        )

        # Verify classes are callable
        assert callable(Engram)
        assert callable(EngramConfig)
        assert callable(CompressedTokenizerProjection)
        assert callable(NgramHasher)
        assert callable(MultiHeadEmbedding)
        assert callable(ShortConv)

    def test_engram_config_defaults_sensible(self):
        """EngramConfig should have sensible default values."""
        from nano_scalemb.engram import EngramConfig

        cfg = EngramConfig()
        assert cfg.max_ngram_size >= 2
        assert cfg.n_head_per_ngram >= 1
        assert cfg.memory_dim >= 64
        assert cfg.slot_multiplier >= 1
        assert cfg.kernel_size >= 1
        assert isinstance(cfg.use_tokenizer_compression, bool)
        assert cfg.embedding_lr_mult > 0
        assert cfg.pad_id >= 0
        assert cfg.seed >= 0

    def test_engram_config_layer_ids_tuple(self):
        """EngramConfig layer_ids should be a tuple."""
        from nano_scalemb.engram import EngramConfig

        cfg = EngramConfig(layer_ids=(0, 2, 6))
        assert isinstance(cfg.layer_ids, tuple)
        assert cfg.layer_ids == (0, 2, 6)

        # Empty default
        cfg2 = EngramConfig()
        assert isinstance(cfg2.layer_ids, tuple)

    def test_engram_config_custom_values(self):
        """EngramConfig should accept custom parameter values."""
        from nano_scalemb.engram import EngramConfig

        cfg = EngramConfig(
            layer_ids=(1, 4),
            max_ngram_size=4,
            n_head_per_ngram=4,
            memory_dim=512,
            embedding_lr_mult=10.0,
        )
        assert cfg.layer_ids == (1, 4)
        assert cfg.max_ngram_size == 4
        assert cfg.n_head_per_ngram == 4
        assert cfg.memory_dim == 512
        assert cfg.embedding_lr_mult == 10.0


class TestEngramSharedMemory:
    """EngramConfig.share_memory: one memory table, one addressing scheme, for
    every Engram layer. The Engram analogue of the Mobius shared expert pool
    (see nano_scalemb/moe/), asking whether the per-layer memories learn
    genuinely different things or whether one memory would do."""

    @staticmethod
    def _build(share_memory, n_layer=4, mhc=None):
        from nano_scalemb.engram import EngramConfig
        from nano_scalemb.gpt import GPT, GPTConfig

        engram = EngramConfig(
            layer_ids=(0, 2),
            max_ngram_size=3,
            n_head_per_ngram=2,
            memory_dim=32,
            slot_multiplier=2,
            use_tokenizer_compression=False,
            share_memory=share_memory,
        )
        config = GPTConfig(
            sequence_len=16,
            vocab_size=64,
            n_layer=n_layer,
            n_head=2,
            n_kv_head=2,
            n_embd=32,
            window_pattern="L",
            engram=engram,
            mhc=mhc,
        )
        with torch.device("meta"):
            model = GPT(config)
        model.to_empty(device="cpu")
        model.init_weights()
        return model

    def test_defaults_to_off(self):
        from nano_scalemb.engram import EngramConfig

        assert EngramConfig().share_memory is False

    def test_layers_keep_separate_tables_by_default(self):
        model = self._build(share_memory=False)

        t0 = model.engram_modules["0"].memory_table
        t2 = model.engram_modules["2"].memory_table
        assert t0 is not t2
        assert t0.embedding.weight is not t2.embedding.weight
        assert model.engram_shared_memory is None

    def test_sharing_ties_the_table_through_meta_init(self):
        model = self._build(share_memory=True)

        t0 = model.engram_modules["0"].memory_table
        t2 = model.engram_modules["2"].memory_table
        # Identity must survive meta -> to_empty -> init_weights, not just __init__.
        assert t0 is t2
        assert t0.embedding.weight is t2.embedding.weight
        assert model.engram_shared_memory is t0

    def test_sharing_ties_the_addressing_scheme_too(self):
        """A shared table with per-layer primes is just a bigger table, not
        sharing: the same n-gram has to land on the same row at every depth."""
        shared = self._build(share_memory=True)
        separate = self._build(share_memory=False)
        ids = torch.randint(0, 64, (2, 6))

        s0 = shared.engram_modules["0"].hasher.hash(ids, 0)
        s2 = shared.engram_modules["2"].hasher.hash(ids, 2)
        assert torch.equal(s0, s2)

        p0 = separate.engram_modules["0"].hasher.hash(ids, 0)
        p2 = separate.engram_modules["2"].hasher.hash(ids, 2)
        assert not torch.equal(p0, p2)

    def test_shared_table_is_stored_and_counted_once(self):
        shared = self._build(share_memory=True)
        separate = self._build(share_memory=False)

        keys = [k for k in shared.state_dict() if k.endswith("embedding.weight")]
        assert keys == ["engram_shared_memory.embedding.weight"]

        # One table instead of two, in the param count and the optimizer group.
        # Not exactly half: primes are allocated sequentially, so the second
        # layer's own table would get slightly larger ones.
        shared_embed = sum(p.numel() for p in shared._all_engram_embedding_params())
        separate_embed = sum(p.numel() for p in separate._all_engram_embedding_params())
        assert shared_embed < separate_embed
        assert shared_embed == pytest.approx(separate_embed / 2, rel=0.1)
        assert shared.num_scaling_params()["engram_embeds"] == shared_embed
        assert shared.num_scaling_params()["total"] == sum(
            p.numel() for p in shared.parameters()
        )

    def test_per_layer_gate_and_projection_stay_private(self):
        """Only the memory contents are tied; each layer still reads it its own
        way, so the ablation isolates the memory rather than the whole branch."""
        model = self._build(share_memory=True)

        e0, e2 = model.engram_modules["0"], model.engram_modules["2"]
        assert e0.value_proj.weight is not e2.value_proj.weight
        assert e0.short_conv.conv.weight is not e2.short_conv.conv.weight

    def test_forward_backward_is_finite_and_optimizer_builds(self):
        import nano_scalemb.flash_attention as fa_module

        torch.manual_seed(0)
        model = self._build(share_memory=True)
        idx = torch.randint(0, 64, (2, 8))

        prev_impl = fa_module._override_impl
        fa_module._override_impl = "sdpa"
        try:
            loss = model(idx, idx)
            loss.backward()
        finally:
            fa_module._override_impl = prev_impl

        assert torch.isfinite(loss)
        grad = model.engram_shared_memory.embedding.weight.grad
        assert grad is not None and torch.isfinite(grad).all()
        model.setup_optimizer()  # the param-count assert is the real check

    def test_composes_with_backbone_mhc(self):
        import nano_scalemb.flash_attention as fa_module

        from nano_scalemb.mhc import MHCConfig

        torch.manual_seed(0)
        model = self._build(
            share_memory=True, mhc=MHCConfig(num_streams=4, sinkhorn_iters=3)
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

    def test_old_checkpoint_config_defaults_share_memory_off(self):
        from nano_scalemb.checkpoint_manager import _patch_missing_config_keys

        config_dict = {
            "window_pattern": "L",
            "mhc": None,
            "engram": {
                "layer_ids": [2, 6],
                "max_ngram_size": 3,
                "n_head_per_ngram": 8,
                "memory_dim": 1280,
                "slot_multiplier": 18,
                "kernel_size": 4,
                "use_tokenizer_compression": True,
                "embedding_lr_mult": 5.0,
                "pad_id": 0,
                "seed": 0,
                "ablation_mode": "none",
                "mhc_num_streams": 4,
            },
        }

        _patch_missing_config_keys(config_dict)

        assert config_dict["engram"].share_memory is False


class TestReadoutWhitening:
    """EngramConfig.readout_whiten: divide each read row by its own hit rate.

    The Hebbian-memory reading of an Engram (Garcia et al. 2026,
    arXiv:2607.10034) makes the cross-talk scale a function of *key crowding*,
    and their Lemma B.3 shows whitening by the empirical feature covariance
    minimizes an upper bound on it. For a one-hot feature map that covariance is
    diagonal with the per-row hit frequencies on it, so whitening reduces to
    downweighting rows the addressing hits often.
    """

    @staticmethod
    def _table(sizes=(5, 7), dim=4, **kw):
        from nano_scalemb.engram import MultiHeadEmbedding

        table = MultiHeadEmbedding(list_of_N=list(sizes), D=dim, **kw)
        torch.nn.init.normal_(table.embedding.weight)
        return table

    def test_defaults_to_off(self):
        from nano_scalemb.engram import EngramConfig

        cfg = EngramConfig()
        assert cfg.readout_whiten is False
        assert self._table().whiten is False

    def test_off_is_a_bit_exact_no_op(self):
        table = self._table()
        assert not hasattr(table, "hit_rate")
        ids = torch.tensor([[[0, 1], [2, 3]]])
        assert torch.equal(table(ids), table.embedding(ids + table.offsets))

    def test_identity_at_uniform_hit_rate(self):
        """hit_rate == 1 everywhere means every row is averagely addressed, so
        there is nothing to correct and the read must be unchanged."""
        plain = self._table()
        whitened = self._table(whiten=True)
        whitened.embedding.weight.data.copy_(plain.embedding.weight.data)
        whitened.hit_rate.fill_(1.0)
        whitened.eval()  # freeze the statistics so the read is the only effect
        ids = torch.tensor([[[0, 1], [4, 6]]])
        torch.testing.assert_close(whitened(ids), plain(ids))

    def test_downweights_hot_rows_relative_to_cold(self):
        table = self._table(whiten=True)
        table.eval()
        table.hit_rate.fill_(1.0)
        table.hit_rate[0] = 50.0  # head-0 row 0 is addressed 50x average
        ids = torch.tensor([[[0, 0]]])
        out = table(ids)
        hot = out[0, 0, 0].norm()
        cold = out[0, 0, 1].norm()
        row_hot = table.embedding.weight[0].norm()
        row_cold = table.embedding.weight[table.offsets[1]].norm()
        # The cold row passes through untouched; the hot one is scaled by ~1/50.
        torch.testing.assert_close(cold, row_cold)
        assert hot < row_hot / 40

    def test_max_caps_cold_row_upweighting(self):
        """Default whiten_max=1.0 must not amplify a never-addressed row: it sits
        at random init, so boosting it injects noise without reducing crowding."""
        table = self._table(whiten=True)
        table.eval()
        table.hit_rate.fill_(0.0)  # nothing has ever been addressed
        ids = torch.tensor([[[0, 0]]])
        torch.testing.assert_close(
            table(ids), table.embedding(ids + table.offsets)
        )
        # ... but the faithful (uncapped) form is reachable and does boost.
        boosted = self._table(whiten=True, whiten_max=float("inf"))
        boosted.eval()
        boosted.hit_rate.fill_(0.0)
        assert boosted(ids).abs().max() > 100 * boosted.embedding.weight.abs().max() / 2

    def test_power_softens_the_correction(self):
        ids = torch.tensor([[[0, 0]]])
        full = self._table(whiten=True, whiten_power=1.0)
        half = self._table(whiten=True, whiten_power=0.5)
        half.embedding.weight.data.copy_(full.embedding.weight.data)
        for t in (full, half):
            t.eval()
            t.hit_rate.fill_(1.0)
            t.hit_rate[0] = 100.0
        w_full = full(ids)[0, 0, 0].norm()
        w_half = half(ids)[0, 0, 0].norm()
        # sqrt(1/100) = 1/10 vs 1/100, so halving the power leaves ~10x more.
        assert w_half > 5 * w_full

    def test_hit_rate_tracks_addressing_frequency(self):
        """One row addressed at every position should land near its relative
        rate (= alphabet size, since it takes all of that head's mass)."""
        table = self._table(sizes=(5, 7), whiten=True, whiten_decay=0.0)
        table.train()
        ids = torch.zeros(4, 8, 2, dtype=torch.long)  # always row 0 of each head
        table(ids)
        # decay=0 means the EMA is just this batch: head 0 has 5 rows, head 1 has
        # 7, and each gave all its hits to row 0.
        assert table.hit_rate[0].item() == pytest.approx(5.0)
        assert table.hit_rate[table.offsets[1]].item() == pytest.approx(7.0)
        assert table.hit_rate[1].item() == pytest.approx(0.0)

    def test_hit_rate_is_uniform_under_uniform_addressing(self):
        """The whole point of the per-head normalization: an alphabet addressed
        uniformly must sit at 1.0 regardless of how big that alphabet is."""
        table = self._table(sizes=(5, 7), whiten=True, whiten_decay=0.0)
        table.train()
        # head 0 cycles 0..4, head 1 cycles 0..6, over lcm(5,7)=35 positions
        pos = torch.arange(35)
        ids = torch.stack([pos % 5, pos % 7], dim=-1).unsqueeze(0)
        table(ids)
        torch.testing.assert_close(
            table.hit_rate, torch.ones_like(table.hit_rate), atol=1e-5, rtol=1e-5
        )

    def test_eval_mode_does_not_update_statistics(self):
        table = self._table(whiten=True, whiten_decay=0.5)
        table.eval()
        before = table.hit_rate.clone()
        table(torch.zeros(2, 4, 2, dtype=torch.long))
        assert torch.equal(table.hit_rate, before)

    def test_read_ignores_its_own_occurrence(self):
        """The weight is taken before the update, so a row's first read is not
        already discounted by the fact that it is being read."""
        table = self._table(whiten=True, whiten_decay=0.0)
        table.train()
        ids = torch.zeros(1, 1, 2, dtype=torch.long)
        out = table(ids)
        torch.testing.assert_close(out, table.embedding(ids + table.offsets))
        assert table.hit_rate[0].item() > 1.0  # but the update did land

    def test_gradient_flows_scaled_by_the_weight(self):
        table = self._table(whiten=True)
        table.eval()
        table.hit_rate.fill_(1.0)
        table.hit_rate[0] = 10.0
        ids = torch.tensor([[[0, 0]]])
        table(ids).sum().backward()
        grad = table.embedding.weight.grad
        hot = grad[0].abs().sum()
        cold = grad[table.offsets[1]].abs().sum()
        assert hot < cold / 5  # the preconditioner scales the gradient too

    def test_survives_meta_to_empty_and_init(self):
        """The exact failure mode that made the PQ codebook start at 1e38: a
        buffer that to_empty() fills with garbage and nothing re-initializes."""
        table = self._build_whitened(share_memory=False).engram_modules["0"].memory_table
        assert table.whiten
        torch.testing.assert_close(table.hit_rate, torch.ones_like(table.hit_rate))

    @staticmethod
    def _build_whitened(share_memory, n_layer=4, **kw):
        from nano_scalemb.engram import EngramConfig
        from nano_scalemb.gpt import GPT, GPTConfig

        engram = EngramConfig(
            layer_ids=(0, 2),
            max_ngram_size=3,
            n_head_per_ngram=2,
            memory_dim=32,
            slot_multiplier=2,
            use_tokenizer_compression=False,
            share_memory=share_memory,
            readout_whiten=True,
            **kw,
        )
        config = GPTConfig(
            sequence_len=16,
            vocab_size=64,
            n_layer=n_layer,
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

    def test_shared_memory_shares_one_set_of_statistics(self):
        """With share_memory the layers read one table, so they must also share
        its hit rates -- two copies would whiten the same row differently at
        different depths."""
        model = self._build_whitened(share_memory=True)
        t0 = model.engram_modules["0"].memory_table
        t2 = model.engram_modules["2"].memory_table
        assert t0 is t2
        assert t0.hit_rate.data_ptr() == t2.hit_rate.data_ptr()
        keys = [k for k in model.state_dict() if "hit_rate" in k]
        assert len(keys) == 1, keys

    def test_per_layer_memory_keeps_separate_statistics(self):
        model = self._build_whitened(share_memory=False)
        t0 = model.engram_modules["0"].memory_table
        t2 = model.engram_modules["2"].memory_table
        assert t0.hit_rate.data_ptr() != t2.hit_rate.data_ptr()

    def test_statistics_round_trip_through_state_dict(self):
        model = self._build_whitened(share_memory=False)
        table = model.engram_modules["0"].memory_table
        table.hit_rate.fill_(3.5)
        reloaded = self._build_whitened(share_memory=False)
        reloaded.load_state_dict(model.state_dict())
        assert reloaded.engram_modules["0"].memory_table.hit_rate[0].item() == 3.5

    def test_forward_is_finite_end_to_end(self):
        model = self._build_whitened(share_memory=False)
        model.train()
        idx = torch.randint(0, 64, (2, 16))
        logits = model(idx)
        assert logits.shape == (2, 16, 64)
        assert torch.isfinite(logits).all()

    def test_whitening_adds_no_parameters(self):
        plain = TestEngramSharedMemory._build(share_memory=False)
        whitened = self._build_whitened(share_memory=False)
        assert sum(p.numel() for p in plain.parameters()) == sum(
            p.numel() for p in whitened.parameters()
        )
        # and the scaling-param accounting still reconciles
        counts = whitened.num_scaling_params()
        assert counts["total"] == sum(p.numel() for p in whitened.parameters())

    def test_old_checkpoint_config_defaults_whitening_off(self):
        from nano_scalemb.checkpoint_manager import _patch_missing_config_keys

        config_dict = {
            "window_pattern": "L",
            "mhc": None,
            "engram": {"layer_ids": [2, 6], "memory_dim": 1280},
        }
        _patch_missing_config_keys(config_dict)
        assert config_dict["engram"].readout_whiten is False

    def test_per_head_mean_hit_rate_is_invariant(self):
        """Whatever the addressing does, each head's mean relative rate stays 1:
        the head's hits always sum to the position count. This is what makes
        "weight == 1" mean "averagely addressed" rather than something that
        drifts with the batch, so a drift here would silently rescale every read.
        """
        table = self._table(sizes=(5, 7), whiten=True, whiten_decay=0.9)
        table.train()
        gen = torch.Generator().manual_seed(0)
        for _ in range(5):
            ids = torch.stack(
                [
                    torch.randint(0, 5, (3, 9), generator=gen),
                    torch.randint(0, 7, (3, 9), generator=gen),
                ],
                dim=-1,
            )
            table(ids)
            assert table.hit_rate[:5].mean().item() == pytest.approx(1.0, abs=1e-5)
            assert table.hit_rate[5:].mean().item() == pytest.approx(1.0, abs=1e-5)


class TestLowRankValues:
    """EngramConfig.value_rank: a memory row is a rank-k linear map of the hidden
    state instead of a constant vector.

    The axis the 23-arm study never varied (docs/conditional_capacity_study.md
    §8.1). rank 0 is the paper's Engram; large rank with few rows is an MoE
    expert bank. Expressiveness per address is paid for in address count, so the
    interesting prediction is differential: contextual addressing should gain
    from rank > 0 and token addressing should not.
    """

    @staticmethod
    def _memory(rank=1, sizes=(5, 7), D=4, d_model=8, query_dim=6, **kw):
        from nano_scalemb.engram import LowRankMemory

        mem = LowRankMemory(
            list_of_N=list(sizes), D=D, d_model=d_model, rank=rank,
            query_dim=query_dim, **kw
        )
        from nano_scalemb.engram import init_memory_table

        init_memory_table(mem, "none")
        return mem

    @staticmethod
    def _build(rank, share_memory=False, n_layer=4, **kw):
        from nano_scalemb.engram import EngramConfig
        from nano_scalemb.gpt import GPT, GPTConfig

        engram = EngramConfig(
            layer_ids=(0, 2),
            max_ngram_size=3,
            n_head_per_ngram=2,
            memory_dim=32,
            slot_multiplier=2,
            use_tokenizer_compression=False,
            share_memory=share_memory,
            value_rank=rank,
            value_query_dim=8,
            **kw,
        )
        config = GPTConfig(
            sequence_len=16, vocab_size=64, n_layer=n_layer, n_head=2,
            n_kv_head=2, n_embd=32, window_pattern="L", engram=engram,
        )
        with torch.device("meta"):
            model = GPT(config)
        model.to_empty(device="cpu")
        model.init_weights()
        return model

    def test_defaults_to_rank_zero_constant_rows(self):
        from nano_scalemb.engram import EngramConfig, MultiHeadEmbedding

        assert EngramConfig().value_rank == 0
        table = self._build(rank=0).engram_modules["0"].memory_table
        assert isinstance(table, MultiHeadEmbedding)

    def test_rank_selects_the_low_rank_table(self):
        from nano_scalemb.engram import LowRankMemory

        table = self._build(rank=1).engram_modules["0"].memory_table
        assert isinstance(table, LowRankMemory)

    def test_read_depends_on_the_hidden_state_at_a_fixed_address(self):
        """The defining property. A constant-row memory returns the same vector
        for the same address no matter what h is; that is exactly what discards
        within-cell variation when the address is a quantized hidden state."""
        mem = self._memory()
        ids = torch.zeros(1, 1, 2, dtype=torch.long)
        gen = torch.Generator().manual_seed(0)
        a = mem(ids, torch.randn(1, 1, 8, generator=gen))
        b = mem(ids, torch.randn(1, 1, 8, generator=gen))
        assert not torch.allclose(a, b), "rank-k read must vary with h"

    def test_constant_rows_do_not_depend_on_the_hidden_state(self):
        from nano_scalemb.engram import MultiHeadEmbedding

        table = MultiHeadEmbedding(list_of_N=[5, 7], D=4)
        torch.nn.init.normal_(table.embedding.weight)
        ids = torch.zeros(1, 1, 2, dtype=torch.long)
        gen = torch.Generator().manual_seed(0)
        torch.testing.assert_close(
            table(ids, torch.randn(1, 1, 8, generator=gen)),
            table(ids, torch.randn(1, 1, 8, generator=gen)),
        )

    def test_read_is_scale_invariant_in_the_hidden_state(self):
        """rms_norm on the query, same reasoning as discretize_normalize: h's
        magnitude drifts a lot over training and the read should key on
        direction."""
        mem = self._memory()
        ids = torch.zeros(1, 1, 2, dtype=torch.long)
        h = torch.randn(1, 1, 8, generator=torch.Generator().manual_seed(0))
        torch.testing.assert_close(mem(ids, h), mem(ids, 7.5 * h))

    def test_identity_activation_is_linear_in_the_query(self):
        mem = self._memory(activation="identity")
        ids = torch.zeros(1, 1, 2, dtype=torch.long)
        gen = torch.Generator().manual_seed(0)
        h1 = torch.randn(1, 1, 8, generator=gen)
        h2 = torch.randn(1, 1, 8, generator=gen)
        # rms_norm is not linear, so compare against a query built by hand
        q1, q2 = mem.query_proj(h1), mem.query_proj(h2)
        shifted = mem._shift(ids)
        def read(q):
            w_in = mem.w_in(shifted).view(1, 1, 2, mem.rank, mem.query_dim)
            hid = torch.einsum("bthrq,btq->bthr", w_in, q)
            w_out = mem.w_out(shifted).view(1, 1, 2, mem.rank, mem.embedding_dim)
            return torch.einsum("bthrd,bthr->bthd", w_out, hid)
        torch.testing.assert_close(read(q1 + q2), read(q1) + read(q2))

    def test_gelu_activation_is_not_linear(self):
        mem = self._memory(activation="gelu")
        ids = torch.zeros(1, 1, 2, dtype=torch.long)
        gen = torch.Generator().manual_seed(0)
        h1 = torch.randn(1, 1, 8, generator=gen)
        h2 = torch.randn(1, 1, 8, generator=gen)
        assert not torch.allclose(mem(ids, h1 + h2), mem(ids, h1) + mem(ids, h2))

    def test_rejects_unknown_activation(self):
        with pytest.raises(ValueError, match="value_activation"):
            self._memory(activation="swiglu")

    def test_rejects_misaligned_query(self):
        mem = self._memory()
        with pytest.raises(AssertionError, match="aligned"):
            mem(torch.zeros(1, 4, 2, dtype=torch.long), torch.randn(1, 3, 8))

    def test_read_scale_matches_the_constant_table_at_init(self):
        """The init stds are chosen so a rank-k read has ~unit per-element scale,
        like the normal(0, 1) rows it replaces. Otherwise the ladder would
        confound expressiveness with a change in branch magnitude.

        Measured ~0.6 against the constant table's ~1.0, and flat in rank (0.58
        at rank 1, 0.66 at rank 8). The gap is gelu halving the variance, which
        MoE experts share; what matters is that it does not drift with rank.
        """
        stds = []
        for rank in (1, 8):
            mem = self._memory(
                rank=rank, D=16, d_model=64, query_dim=32, sizes=(64, 64)
            )
            h = torch.randn(4, 32, 64, generator=torch.Generator().manual_seed(0))
            ids = torch.randint(
                0, 64, (4, 32, 2), generator=torch.Generator().manual_seed(1)
            )
            stds.append(mem(ids, h).std().item())
        assert all(0.4 < s < 1.6 for s in stds), stds
        assert max(stds) / min(stds) < 2.0, stds

    def test_active_params_and_flops_follow_the_rank(self):
        rank, query_dim = 3, 6
        mem = self._memory(rank=rank, D=4, query_dim=query_dim)
        assert mem.active_params_per_token() == 2 * rank * (query_dim + 4)
        assert mem.read_flops_per_token() == 6 * mem.active_params_per_token()

    def test_constant_rows_cost_no_read_flops(self):
        from nano_scalemb.engram import MultiHeadEmbedding

        assert MultiHeadEmbedding(list_of_N=[5], D=4).read_flops_per_token() == 0

    def test_content_params_scale_with_rank_and_exclude_the_projection(self):
        base = self._memory(rank=1, D=4, query_dim=6)
        wide = self._memory(rank=4, D=4, query_dim=6)
        n = lambda m: sum(p.numel() for p in m.content_parameters())
        assert n(wide) == 4 * n(base)
        proj_ids = {id(base.query_proj.weight)}
        assert not any(id(p) in proj_ids for p in base.content_parameters())

    def test_accounting_reconciles_across_the_ladder(self):
        for rank in (0, 1, 4):
            for share in (False, True):
                model = self._build(rank, share_memory=share)
                counts = model.num_scaling_params()
                assert counts["total"] == sum(p.numel() for p in model.parameters())
                # a read pulls rank*(query_dim + head_dim) per head per layer
                expected = 0
                for engram in model.engram_modules.values():
                    expected += engram.memory_table.active_params_per_token()
                assert counts["engram_active"] == expected

    def test_flops_grow_with_rank(self):
        flops = [self._build(r).estimate_flops() for r in (0, 1, 4)]
        assert flops[0] < flops[1] < flops[2]

    def test_optimizer_places_rows_and_projection_in_different_groups(self):
        """w_in/w_out are addressed memory and get the Engram LR multiplier;
        query_proj is a dense matrix and belongs to Muon. If the projection fell
        into the embedding group it would silently train at 5x the matrix LR."""
        model = self._build(rank=1)
        opt = model.setup_optimizer()
        table = model.engram_modules["0"].memory_table
        row_ids = {id(table.w_in.weight), id(table.w_out.weight)}
        proj_id = id(table.query_proj.weight)
        kinds = {}
        for group in opt.param_groups:
            for p in group["params"]:
                if id(p) in row_ids or id(p) == proj_id:
                    kinds[id(p)] = group["kind"]
        assert all(kinds[i] == "adamw" for i in row_ids), kinds
        assert kinds[proj_id] == "muon", kinds

    def test_shared_memory_shares_the_projection_too(self):
        """Sharing the table but not the featurization would send the same hidden
        state to unrelated maps at different depths -- a bigger table, not a
        shared memory (the argument share_memory already makes for the
        discretizer)."""
        model = self._build(rank=1, share_memory=True)
        t0 = model.engram_modules["0"].memory_table
        t2 = model.engram_modules["2"].memory_table
        assert t0 is t2
        keys = [k for k in model.state_dict() if "query_proj" in k and "engram" in k]
        assert len({model.state_dict()[k].data_ptr() for k in keys}) == 1, keys

    def test_per_layer_memory_keeps_separate_projections(self):
        model = self._build(rank=1, share_memory=False)
        t0 = model.engram_modules["0"].memory_table
        t2 = model.engram_modules["2"].memory_table
        assert t0.query_proj.weight.data_ptr() != t2.query_proj.weight.data_ptr()

    def test_forward_and_backward_are_finite(self):
        model = self._build(rank=2)
        model.train()
        idx = torch.randint(0, 64, (2, 16))
        loss = model(idx, idx).sum()
        loss.backward()
        table = model.engram_modules["0"].memory_table
        for name in ("w_in", "w_out", "query_proj"):
            grad = getattr(table, name).weight.grad
            assert grad is not None, name
            assert torch.isfinite(grad).all(), name

    def test_survives_meta_to_empty_and_init(self):
        table = self._build(rank=2).engram_modules["0"].memory_table
        for w in (table.w_in.weight, table.w_out.weight, table.query_proj.weight):
            assert torch.isfinite(w).all()
            assert w.abs().max() < 100

    def test_composes_with_contextual_addressing(self):
        """The arm the ladder exists to run: hidden-state addressing whose values
        are also functions of the hidden state."""
        model = self._build(rank=1, address_source="lsh", lsh_bits=6)
        model.train()
        out = model(torch.randint(0, 64, (2, 16)))
        assert torch.isfinite(out).all()

    def test_composes_with_readout_whitening(self):
        model = self._build(rank=1, readout_whiten=True)
        table = model.engram_modules["0"].memory_table
        torch.testing.assert_close(table.hit_rate, torch.ones_like(table.hit_rate))
        model.train()
        assert torch.isfinite(model(torch.randint(0, 64, (2, 16)))).all()
        assert not torch.equal(table.hit_rate, torch.ones_like(table.hit_rate))

    def test_frozen_ablations_freeze_both_row_tables(self):
        for mode in ("randomize", "uniform"):
            table = self._build(rank=1, ablation_mode=mode).engram_modules[
                "0"
            ].memory_table
            assert not table.w_in.weight.requires_grad, mode
            assert not table.w_out.weight.requires_grad, mode

    def test_uniform_ablation_makes_every_row_identical(self):
        table = self._build(rank=1, ablation_mode="uniform").engram_modules[
            "0"
        ].memory_table
        for w in (table.w_in.weight, table.w_out.weight):
            torch.testing.assert_close(w[0].expand_as(w), w)

    def test_state_dict_round_trip(self):
        model = self._build(rank=2)
        reloaded = self._build(rank=2)
        reloaded.load_state_dict(model.state_dict())
        a = model.engram_modules["0"].memory_table
        b = reloaded.engram_modules["0"].memory_table
        torch.testing.assert_close(a.w_in.weight, b.w_in.weight)
        torch.testing.assert_close(a.query_proj.weight, b.query_proj.weight)

    def test_old_checkpoint_config_defaults_rank_zero(self):
        from nano_scalemb.checkpoint_manager import _patch_missing_config_keys

        config_dict = {
            "window_pattern": "L",
            "mhc": None,
            "engram": {"layer_ids": [2, 6], "memory_dim": 1280},
        }
        _patch_missing_config_keys(config_dict)
        assert config_dict["engram"].value_rank == 0

    def test_composes_with_hybrid_addressing(self):
        """hybrid is the likely follow-up arm: some heads addressed by token
        n-gram, some by hidden state, all with rank-k values."""
        model = self._build(
            rank=1, address_source="hybrid", lsh_bits=6, hybrid_token_head_frac=0.5
        )
        model.train()
        assert torch.isfinite(model(torch.randint(0, 64, (2, 16)))).all()

    def test_rejects_rank_without_a_table(self):
        """pkm and the mlp control have no MultiHeadEmbedding, so value_rank
        would be silently inert. base_train rejects those combinations; this pins
        that the table really is absent, which is why."""
        for kw in (dict(address_source="pkm"), dict(ablation_mode="mlp")):
            model = self._build(rank=0, **kw)
            assert model.engram_modules["0"].memory_table is None, kw


class TestKeyValueSplit:
    """EngramConfig.key_dim: a row stores its own key for the gate, separate from
    its value.

    Without it, one stored vector serves both roles and both projections are
    shared across every row, so a row's key is a fixed linear function of its
    value: two rows with similar content are forced into similar relevance. This
    is the k/v split attention, MoE routers and product-key memories all have.
    """

    @staticmethod
    def _build(key_dim=0, mhc=True, n_layer=4, **kw):
        from nano_scalemb.engram import EngramConfig
        from nano_scalemb.mhc import MHCConfig
        from nano_scalemb.gpt import GPT, GPTConfig

        engram = EngramConfig(
            layer_ids=(0, 2), max_ngram_size=3, n_head_per_ngram=2, memory_dim=32,
            slot_multiplier=2, use_tokenizer_compression=False, mhc_num_streams=4,
            key_dim=key_dim, **kw,
        )
        config = GPTConfig(
            sequence_len=16, vocab_size=64, n_layer=n_layer, n_head=2, n_kv_head=2,
            n_embd=32, window_pattern="L", engram=engram,
            mhc=MHCConfig(num_streams=4) if mhc else None,
        )
        with torch.device("meta"):
            model = GPT(config)
        model.to_empty(device="cpu")
        model.init_weights()
        return model

    def test_defaults_to_shared_vector(self):
        from nano_scalemb.engram import EngramConfig

        assert EngramConfig().key_dim == 0
        table = self._build(key_dim=0).engram_modules["0"].memory_table
        assert table.key_dim == 0
        values, keys = table.split_key_value(torch.zeros(1, 1, 4, 8))
        assert keys is None

    def test_row_width_grows_by_key_dim(self):
        plain = self._build(key_dim=0).engram_modules["0"].memory_table
        split = self._build(key_dim=3).engram_modules["0"].memory_table
        assert plain.embedding.weight.shape[1] == 8  # memory_dim 32 / 4 heads
        assert split.embedding.weight.shape[1] == 11  # + key_dim

    def test_split_returns_value_then_key_halves(self):
        table = self._build(key_dim=3).engram_modules["0"].memory_table
        rows = torch.arange(11, dtype=torch.float32).view(1, 1, 1, 11)
        values, keys = table.split_key_value(rows)
        torch.testing.assert_close(keys, rows[..., :3])
        torch.testing.assert_close(values, rows[..., 3:])
        assert values.shape[-1] == table.embedding_dim

    def test_gate_projection_is_sized_to_the_keys(self):
        """The gate reads keys, so its projection must take num_heads*key_dim, not
        the payload width. Getting this wrong would silently feed it value dims."""
        for mhc, attr in ((True, "stream_key_proj"), (False, "key_proj")):
            engram = self._build(key_dim=3, mhc=mhc).engram_modules["0"]
            assert getattr(engram, attr).in_features == 4 * 3, attr
            plain = self._build(key_dim=0, mhc=mhc).engram_modules["0"]
            assert getattr(plain, attr).in_features == 32, attr  # memory_dim

    def test_value_projection_is_unaffected(self):
        for key_dim in (0, 3):
            engram = self._build(key_dim=key_dim).engram_modules["0"]
            assert engram.value_proj.in_features == 32

    def test_active_params_count_the_key_dims(self):
        model = self._build(key_dim=3)
        counts = model.num_scaling_params()
        # 4 heads x (3 key + 8 value) x 2 engram layers
        assert counts["engram_active"] == 4 * 11 * 2
        assert counts["total"] == sum(p.numel() for p in model.parameters())

    def test_forward_and_backward_finite_both_paths(self):
        for mhc in (True, False):
            model = self._build(key_dim=3, mhc=mhc)
            model.train()
            idx = torch.randint(0, 64, (2, 16))
            model(idx, idx).sum().backward()
            grad = model.engram_modules["0"].memory_table.embedding.weight.grad
            assert grad is not None and torch.isfinite(grad).all(), mhc

    def test_key_dims_receive_gradient(self):
        """If the gate did not actually read them, the key columns would sit at
        init forever and the whole feature would be a silent no-op.

        Checked with value_proj perturbed off zero. At init value_proj IS zero, so
        `gate * value` gives the entire gate branch zero gradient — pre-existing
        behaviour, identical at key_dim=0, and it clears after one step.
        """
        model = self._build(key_dim=3)
        model.train()
        engram = model.engram_modules["0"]
        torch.nn.init.normal_(engram.value_proj.weight, std=0.02)
        idx = torch.randint(0, 64, (2, 16))
        model(idx, idx).sum().backward()
        grad = engram.memory_table.embedding.weight.grad
        assert grad[:, :3].abs().sum() > 0, "key columns got no gradient"
        assert grad[:, 3:].abs().sum() > 0, "value columns got no gradient"

    def test_gate_branch_is_gradient_free_only_at_init(self):
        """Documents the above: zero-init value_proj gates off the gate branch on
        step 0 whether or not key_dim is set, so it is not a key_dim artifact."""
        for key_dim in (0, 3):
            model = self._build(key_dim=key_dim)
            model.train()
            idx = torch.randint(0, 64, (2, 16))
            model(idx, idx).sum().backward()
            engram = model.engram_modules["0"]
            assert engram.stream_key_proj.weight.grad.abs().max() == 0.0, key_dim
            assert engram.value_proj.weight.grad.abs().max() > 0.0, key_dim

    def test_optimizer_accepts_the_split(self):
        model = self._build(key_dim=3)
        model.setup_optimizer()  # the param-count assert is the test

    def test_composes_with_shared_memory_and_contextual_addressing(self):
        for kw in (dict(share_memory=True), dict(address_source="lsh", lsh_bits=6)):
            model = self._build(key_dim=3, **kw)
            model.train()
            assert torch.isfinite(model(torch.randint(0, 64, (2, 16)))).all(), kw

    def test_rejected_with_value_rank(self):
        with pytest.raises(AssertionError, match="key_dim"):
            self._build(key_dim=3, value_rank=1)

    def test_old_checkpoint_config_defaults_key_dim_zero(self):
        from nano_scalemb.checkpoint_manager import _patch_missing_config_keys

        d = {"window_pattern": "L", "mhc": None,
             "engram": {"layer_ids": [2, 6], "memory_dim": 1280}}
        _patch_missing_config_keys(d)
        assert d["engram"].key_dim == 0


class TestCountGate:
    """EngramConfig.count_gate: weight each head by a learned function of how
    often its row is addressed.

    Kneser-Ney backoff, which the Engram lacks -- it reads every n-gram order in
    parallel with fixed weights, so a 3-gram seen twice is trusted like one seen
    ten thousand times. Rare rows are also the likeliest collision victims, and
    collisions are the binding constraint (row count is worth 0.00195 per
    doubling at 12x its bar; row content expressiveness measured 0.01x-0.09x).
    """

    @staticmethod
    def _table(sizes=(5, 7), dim=4, **kw):
        from nano_scalemb.engram import MultiHeadEmbedding

        t = MultiHeadEmbedding(list_of_N=list(sizes), D=dim, count_gate=True, **kw)
        torch.nn.init.normal_(t.embedding.weight)
        t.count_gate_scale.data.zero_()
        t.count_gate_bias.data.zero_()
        return t

    def test_defaults_to_off(self):
        from nano_scalemb.engram import EngramConfig, MultiHeadEmbedding

        assert EngramConfig().count_gate is False
        t = MultiHeadEmbedding(list_of_N=[5], D=4)
        assert t.count_gate is False and t.track_hit_rate is False
        assert not hasattr(t, "count_gate_scale")

    def test_is_exactly_neutral_at_init(self):
        """Zero-init must give w == 1.0 at every row, so enabling the flag cannot
        confound an arm with a change in read magnitude."""
        from nano_scalemb.engram import MultiHeadEmbedding

        plain = MultiHeadEmbedding(list_of_N=[5, 7], D=4)
        gated = self._table()
        gated.embedding.weight.data.copy_(plain.embedding.weight.data)
        gated.eval()
        gated.hit_rate.copy_(torch.rand(gated.total_rows) * 50)  # any statistics
        ids = torch.tensor([[[0, 1], [3, 5]]])
        torch.testing.assert_close(gated(ids), plain(ids))

    def test_positive_scale_upweights_frequent_rows(self):
        t = self._table()
        t.eval()
        t.count_gate_scale.data.fill_(1.0)
        t.hit_rate.fill_(1.0)
        t.hit_rate[0] = 100.0
        ids = torch.tensor([[[0, 0]]])
        out = t(ids)
        hot = (out[0, 0, 0] / t.embedding.weight[0]).mean()
        cold = (out[0, 0, 1] / t.embedding.weight[t.offsets[1]]).mean()
        assert hot > cold, (hot.item(), cold.item())

    def test_negative_scale_downweights_frequent_rows(self):
        """The backoff direction: trust a very frequent row less."""
        t = self._table()
        t.eval()
        t.count_gate_scale.data.fill_(-1.0)
        t.hit_rate.fill_(1.0)
        t.hit_rate[0] = 100.0
        ids = torch.tensor([[[0, 0]]])
        out = t(ids)
        hot = (out[0, 0, 0] / t.embedding.weight[0]).mean()
        cold = (out[0, 0, 1] / t.embedding.weight[t.offsets[1]]).mean()
        assert hot < cold

    def test_weight_is_learned_per_head(self):
        t = self._table()
        assert t.count_gate_scale.shape == (2,)
        t.eval()
        t.count_gate_bias.data = torch.tensor([1.0, -1.0])
        t.hit_rate.fill_(1.0)
        ids = torch.tensor([[[0, 0]]])
        out = t(ids)
        r0 = (out[0, 0, 0] / t.embedding.weight[0]).mean()
        r1 = (out[0, 0, 1] / t.embedding.weight[t.offsets[1]]).mean()
        assert r0 > 1.0 > r1, (r0.item(), r1.item())

    def test_tracks_hit_rate_without_whitening(self):
        t = self._table(whiten=False)
        assert t.track_hit_rate and not t.whiten
        t.train()
        t(torch.zeros(2, 4, 2, dtype=torch.long))
        assert t.hit_rate[0].item() > 1.0

    def test_gate_params_receive_gradient(self):
        t = self._table()
        t.eval()
        t.hit_rate.fill_(2.0)
        t(torch.tensor([[[0, 1]]])).sum().backward()
        assert t.count_gate_scale.grad.abs().sum() > 0
        assert t.count_gate_bias.grad.abs().sum() > 0

    def test_composes_with_whitening(self):
        t = self._table(whiten=True)
        assert t.whiten and t.count_gate
        t.eval()
        t.hit_rate.fill_(1.0)
        ids = torch.tensor([[[0, 1]]])
        assert torch.isfinite(t(ids)).all()

    def test_survives_meta_to_empty_and_init(self):
        """torch.zeros at construction does NOT survive to_empty(); zero is what
        makes the gate neutral, so a missing reset would start it at garbage."""
        model = TestKeyValueSplit._build(key_dim=0, count_gate=True)
        table = model.engram_modules["0"].memory_table
        assert table.count_gate_scale.abs().max().item() == 0.0
        assert table.count_gate_bias.abs().max().item() == 0.0
        torch.testing.assert_close(table.hit_rate, torch.ones_like(table.hit_rate))

    def test_optimizer_places_gate_params_outside_the_embedding_group(self):
        """They are 1-D scalars, not addressed rows, so they must not inherit the
        5x Engram embedding LR multiplier."""
        model = TestKeyValueSplit._build(key_dim=0, count_gate=True)
        table = model.engram_modules["0"].memory_table
        embed_ids = {id(p) for p in table.content_parameters()}
        assert id(table.count_gate_scale) not in embed_ids
        opt = model.setup_optimizer()
        found = [g["kind"] for g in opt.param_groups
                 if any(id(p) == id(table.count_gate_scale) for p in g["params"])]
        assert found and "adamw" in found[0], found

    def test_end_to_end_finite(self):
        model = TestKeyValueSplit._build(key_dim=0, count_gate=True)
        model.train()
        idx = torch.randint(0, 64, (2, 16))
        model(idx, idx).sum().backward()
        assert torch.isfinite(model(idx)).all()

    def test_old_checkpoint_config_defaults_count_gate_off(self):
        from nano_scalemb.checkpoint_manager import _patch_missing_config_keys

        d = {"window_pattern": "L", "mhc": None,
             "engram": {"layer_ids": [2, 6], "memory_dim": 1280}}
        _patch_missing_config_keys(d)
        assert d["engram"].count_gate is False


class TestCountGateFourier:
    """EngramConfig.count_gate_fourier: Fourier features on log-count, added to
    the count gate's linear term.

    The linear gate is strictly monotone in count. Katz discounting is not: a
    count-1 row is an unreliable *estimate* while a count-10000 row is reliable
    but generic, so the optimal trust curve is plausibly unimodal. This basis can
    express that; a*u+b cannot.
    """

    @staticmethod
    def _table(K=4, sizes=(5, 7), dim=4):
        from nano_scalemb.engram import MultiHeadEmbedding, init_memory_table

        t = MultiHeadEmbedding(
            list_of_N=list(sizes), D=dim, count_gate=True, count_gate_fourier=K
        )
        init_memory_table(t, "none")
        return t

    def test_defaults_to_off(self):
        from nano_scalemb.engram import EngramConfig, MultiHeadEmbedding

        assert EngramConfig().count_gate_fourier == 0
        t = MultiHeadEmbedding(list_of_N=[5], D=4, count_gate=True)
        assert t.count_gate_fourier == 0
        assert not hasattr(t, "count_gate_fourier_coef")

    def test_requires_the_count_gate(self):
        """The Fourier terms are added to that gate's logit, so without it they
        would have nothing to attach to."""
        from nano_scalemb.engram import MultiHeadEmbedding

        t = MultiHeadEmbedding(list_of_N=[5], D=4, count_gate=False,
                               count_gate_fourier=4)
        assert t.count_gate_fourier == 0

    def test_is_a_strict_superset_of_the_linear_gate(self):
        """Zero Fourier coefficients must reproduce the linear gate bit-for-bit,
        so the arm can only lose through optimization or variance, never through
        expressiveness."""
        from nano_scalemb.engram import MultiHeadEmbedding, init_memory_table

        lin = MultiHeadEmbedding(list_of_N=[5, 7], D=4, count_gate=True)
        init_memory_table(lin, "none")
        fou = self._table()
        fou.embedding.weight.data.copy_(lin.embedding.weight.data)
        for t in (lin, fou):
            t.eval()
            t.hit_rate.copy_(torch.linspace(0, 40, t.total_rows))
        for m in (lin, fou):
            m.count_gate_scale.data.fill_(-0.3)
            m.count_gate_bias.data.fill_(-1.4)
        ids = torch.tensor([[[0, 1], [3, 6]]])
        torch.testing.assert_close(fou(ids), lin(ids))

    def test_exactly_neutral_at_init(self):
        t = self._table()
        t.eval()
        t.hit_rate.copy_(torch.rand(t.total_rows) * 50)
        ids = torch.tensor([[[0, 1]]])
        rows = t.embedding(t._shift(ids))
        torch.testing.assert_close(t(ids), rows)

    def test_can_express_a_non_monotone_trust_curve(self):
        """The whole point. Fit the gate to a unimodal target in log-count and
        check it gets there -- the linear gate provably cannot."""
        t = self._table(K=6)
        t.train()
        u = torch.linspace(0.0, 7.0, 64)
        target = torch.exp(-0.5 * ((u - 3.0) / 0.8) ** 2)  # bump at u=3
        params = [t.count_gate_scale, t.count_gate_bias, t.count_gate_fourier_coef]
        opt = torch.optim.Adam(params, lr=0.05)
        for _ in range(600):
            phase = u.unsqueeze(-1) * t.count_gate_freqs
            basis = torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1)
            logit = (t.count_gate_scale[0] * u + t.count_gate_bias[0]
                     + (basis * t.count_gate_fourier_coef[0]).sum(-1))
            loss = ((2 * torch.sigmoid(logit) - target) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        assert loss.item() < 5e-3, loss.item()

    def test_linear_gate_cannot_express_it(self):
        """Control for the above: same fit with the Fourier terms removed must
        fail, which is what makes the previous test meaningful."""
        from nano_scalemb.engram import MultiHeadEmbedding

        t = MultiHeadEmbedding(list_of_N=[5, 7], D=4, count_gate=True)
        u = torch.linspace(0.0, 7.0, 64)
        target = torch.exp(-0.5 * ((u - 3.0) / 0.8) ** 2)
        opt = torch.optim.Adam([t.count_gate_scale, t.count_gate_bias], lr=0.05)
        for _ in range(600):
            logit = t.count_gate_scale[0] * u + t.count_gate_bias[0]
            loss = ((2 * torch.sigmoid(logit) - target) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        assert loss.item() > 5e-2, loss.item()

    def test_frequencies_are_geometric_and_span_the_range(self):
        t = self._table(K=4)
        f = t.count_gate_freqs
        assert f.shape == (4,)
        ratios = (f[1:] / f[:-1]).tolist()
        assert all(abs(r - 2.0) < 1e-5 for r in ratios), ratios
        # band 0 = half a period across [0, max_u]
        assert abs(f[0].item() * t.count_gate_fourier_max_u - math.pi) < 1e-4

    def test_coefficients_receive_gradient(self):
        t = self._table()
        t.eval()
        t.hit_rate.fill_(5.0)
        t(torch.tensor([[[0, 1]]])).sum().backward()
        assert t.count_gate_fourier_coef.grad.abs().sum() > 0

    def test_frequency_buffer_is_non_persistent(self):
        """A fixed basis, not learned state: regenerated in init, so it must not
        bloat the checkpoint or fail to load on an old one."""
        t = self._table()
        assert not any("count_gate_freqs" in k for k in t.state_dict())
        assert any("count_gate_fourier_coef" in k for k in t.state_dict())

    def test_survives_meta_to_empty_and_init(self):
        model = TestKeyValueSplit._build(
            key_dim=0, count_gate=True, count_gate_fourier=4
        )
        table = model.engram_modules["0"].memory_table
        assert table.count_gate_fourier_coef.abs().max().item() == 0.0
        assert table.count_gate_freqs.abs().max().item() > 0.0  # regenerated
        model.train()
        assert torch.isfinite(model(torch.randint(0, 64, (2, 16)))).all()

    def test_adds_only_scalars(self):
        plain = TestKeyValueSplit._build(key_dim=0, count_gate=True)
        fou = TestKeyValueSplit._build(key_dim=0, count_gate=True,
                                       count_gate_fourier=4)
        extra = (sum(p.numel() for p in fou.parameters())
                 - sum(p.numel() for p in plain.parameters()))
        assert extra == 2 * 4 * 4 * 2, extra  # 2*K per head, 4 heads, 2 layers
        counts = fou.num_scaling_params()
        assert counts["total"] == sum(p.numel() for p in fou.parameters())

    def test_optimizer_accepts_it(self):
        model = TestKeyValueSplit._build(key_dim=0, count_gate=True,
                                          count_gate_fourier=4)
        model.setup_optimizer()

    def test_old_checkpoint_config_defaults_off(self):
        from nano_scalemb.checkpoint_manager import _patch_missing_config_keys

        d = {"window_pattern": "L", "mhc": None,
             "engram": {"layer_ids": [2, 6], "memory_dim": 1280}}
        _patch_missing_config_keys(d)
        assert d["engram"].count_gate_fourier == 0


class TestCountGateDecouple:
    """EngramConfig.count_gate_decouple: split a row's hit rate into the n-gram's
    own frequency and that head's collision excess.

    A row hit 1000x by ONE n-gram is a reliable frequent fact; a row hit 1000x by
    500 colliding n-grams is mush. The plain gate cannot tell them apart, which is
    the likeliest reason the Fourier gate came back monotone (§9.2) -- both causes
    argue for downweighting, so no non-monotonicity can appear.

    Estimable for free: every head of an n-gram order hashes the SAME n-gram under
    a different prime, so frequency-driven heat is common across an order's heads
    while collision-driven heat is not.
    """

    @staticmethod
    def _table(heads=4, per_group=2, dim=3, K=0, sizes=None):
        from nano_scalemb.engram import MultiHeadEmbedding, init_memory_table

        t = MultiHeadEmbedding(
            list_of_N=(sizes or [8] * heads), D=dim, count_gate=True,
            count_gate_decouple=True, heads_per_group=per_group,
            count_gate_fourier=K,
        )
        init_memory_table(t, "none")
        return t

    def test_defaults_to_off(self):
        from nano_scalemb.engram import EngramConfig, MultiHeadEmbedding

        assert EngramConfig().count_gate_decouple is False
        t = MultiHeadEmbedding(list_of_N=[8, 8], D=3, count_gate=True)
        assert t.count_gate_decouple is False
        assert not hasattr(t, "count_gate_freq_scale")

    def test_requires_the_count_gate(self):
        from nano_scalemb.engram import MultiHeadEmbedding

        t = MultiHeadEmbedding(list_of_N=[8], D=3, count_gate=False,
                               count_gate_decouple=True)
        assert t.count_gate_decouple is False

    def test_decomposition_is_exact(self):
        """rate == freq + collide by construction, not approximately."""
        t = self._table(heads=4, per_group=2)
        rel = torch.tensor([[[3.0, 7.0, 1.0, 1.5]]])
        f, c = t.decouple_hit_rate(rel)
        torch.testing.assert_close(f + c, rel)

    def test_frequency_is_the_per_order_minimum(self):
        t = self._table(heads=4, per_group=2)
        rel = torch.tensor([[[3.0, 7.0, 1.0, 1.5]]])
        f, c = t.decouple_hit_rate(rel)
        # group 0 = heads {0,1} -> min 3.0 ; group 1 = heads {2,3} -> min 1.0
        torch.testing.assert_close(f, torch.tensor([[[3.0, 3.0, 1.0, 1.0]]]))
        torch.testing.assert_close(c, torch.tensor([[[0.0, 4.0, 0.0, 0.5]]]))

    def test_collision_excess_is_non_negative(self):
        t = self._table(heads=6, per_group=3)
        gen = torch.Generator().manual_seed(0)
        rel = torch.rand(4, 5, 6, generator=gen) * 100
        f, c = t.decouple_hit_rate(rel)
        assert (c >= 0).all()
        assert (f <= rel).all()

    def test_separates_frequent_from_crowded(self):
        """The whole point: two heads with the SAME hit rate but different
        collision structure must be gated differently."""
        t = self._table(heads=4, per_group=2)
        t.eval()
        t.count_gate_collide_scale.data.fill_(-1.0)  # distrust crowded rows
        # head 0: rate 10 all from its own n-gram (its group-mate also 10)
        # head 2: rate 10 but its group-mate is 1, so 9 of it is collision
        rel = torch.tensor([[[10.0, 10.0, 10.0, 1.0]]])
        f, c = t.decouple_hit_rate(rel)
        assert c[0, 0, 0].item() == 0.0 and c[0, 0, 2].item() == 9.0
        logit = (t.count_gate_scale * torch.log1p(rel) + t.count_gate_bias
                 + t.count_gate_freq_scale * torch.log1p(f)
                 + t.count_gate_collide_scale * torch.log1p(c))
        w = 2 * torch.sigmoid(logit)
        assert w[0, 0, 0] > w[0, 0, 2], (w[0, 0, 0].item(), w[0, 0, 2].item())
        # ... and the plain gate, seeing only rate, cannot distinguish them
        plain = t.count_gate_scale * torch.log1p(rel) + t.count_gate_bias
        assert plain[0, 0, 0] == plain[0, 0, 2]

    def test_is_a_strict_superset_of_the_plain_gate(self):
        from nano_scalemb.engram import MultiHeadEmbedding, init_memory_table

        plain = MultiHeadEmbedding(list_of_N=[8]*4, D=3, count_gate=True)
        init_memory_table(plain, "none")
        dec = self._table(heads=4, per_group=2)
        dec.embedding.weight.data.copy_(plain.embedding.weight.data)
        for m in (plain, dec):
            m.eval()
            m.hit_rate.copy_(torch.linspace(0, 30, m.total_rows))
            m.count_gate_scale.data.fill_(-0.3)
            m.count_gate_bias.data.fill_(-1.4)
        ids = torch.tensor([[[0, 3, 5, 7]]])
        torch.testing.assert_close(dec(ids), plain(ids))

    def test_exactly_neutral_at_init(self):
        t = self._table(heads=4, per_group=2, K=3)
        t.eval()
        t.hit_rate.copy_(torch.rand(t.total_rows) * 40)
        ids = torch.tensor([[[0, 2, 4, 6]]])
        torch.testing.assert_close(t(ids), t.embedding(t._shift(ids)))

    def test_fourier_basis_moves_to_the_frequency_channel(self):
        """The Katz hypothesis is about a specific n-gram's count, so the basis
        must see freq, not the rate that conflates freq with crowding."""
        t = self._table(heads=4, per_group=2, K=2)
        t.eval()
        t.count_gate_fourier_coef.data.fill_(0.5)
        # two configs with identical rate per head but different freq/collide split
        a = torch.tensor([[[6.0, 6.0]]]).expand(1, 1, 2).clone()
        t.hit_rate.zero_()
        w_of = {}
        for name, rates in (("own", [6.0, 6.0, 6.0, 6.0]),
                            ("crowded", [6.0, 0.0, 6.0, 0.0])):
            rel = torch.tensor([rates])
            f, _ = t.decouple_hit_rate(rel)
            phase = torch.log1p(f).unsqueeze(-1) * t.count_gate_freqs
            basis = torch.cat((torch.sin(phase), torch.cos(phase)), -1)
            w_of[name] = (basis * t.count_gate_fourier_coef).sum(-1)[0, 0].item()
        assert w_of["own"] != w_of["crowded"], w_of

    def test_gradients_reach_both_new_channels(self):
        t = self._table(heads=4, per_group=2)
        t.eval()
        t.hit_rate.copy_(torch.linspace(1, 20, t.total_rows))
        t(torch.tensor([[[0, 3, 5, 7]]])).sum().backward()
        assert t.count_gate_freq_scale.grad.abs().sum() > 0
        assert t.count_gate_collide_scale.grad.abs().sum() > 0

    def test_rejects_head_count_not_divisible_by_group(self):
        from nano_scalemb.engram import MultiHeadEmbedding

        with pytest.raises(AssertionError, match="heads_per_group"):
            MultiHeadEmbedding(list_of_N=[8]*5, D=3, count_gate=True,
                               count_gate_decouple=True, heads_per_group=2)

    def test_end_to_end_and_grouping_matches_ngram_orders(self):
        model = TestKeyValueSplit._build(
            key_dim=0, count_gate=True, count_gate_decouple=True,
            count_gate_fourier=3,
        )
        table = model.engram_modules["0"].memory_table
        # the test model is max_ngram_size=3, n_head_per_ngram=2 -> 4 heads, 2 orders
        assert table.num_heads == 4 and table.heads_per_group == 2
        assert table.count_gate_freq_scale.abs().max().item() == 0.0
        model.train()
        idx = torch.randint(0, 64, (2, 16))
        model(idx, idx).sum().backward()
        assert torch.isfinite(model(idx)).all()

    def test_adds_only_scalars(self):
        plain = TestKeyValueSplit._build(key_dim=0, count_gate=True)
        dec = TestKeyValueSplit._build(key_dim=0, count_gate=True,
                                        count_gate_decouple=True)
        extra = (sum(p.numel() for p in dec.parameters())
                 - sum(p.numel() for p in plain.parameters()))
        assert extra == 2 * 4 * 2, extra  # freq+collide scale, 4 heads, 2 layers
        assert dec.num_scaling_params()["total"] == sum(
            p.numel() for p in dec.parameters())
        dec.setup_optimizer()

    def test_old_checkpoint_config_defaults_off(self):
        from nano_scalemb.checkpoint_manager import _patch_missing_config_keys

        d = {"window_pattern": "L", "mhc": None,
             "engram": {"layer_ids": [2, 6], "memory_dim": 1280}}
        _patch_missing_config_keys(d)
        assert d["engram"].count_gate_decouple is False


class TestShareHashDecoupled:
    """EngramConfig.share_hash: separate "one shared table" from "one shared
    addressing scheme".

    Cross-layer sharing is the study's largest surviving effect and is NOT a
    capacity effect (§9.2b), but it bundles two things: all layers drawing rows
    from one pool, and the same n-gram landing on the SAME row at every depth.
    This knob opens the 2x2 that separates them.

    Mechanics: per-head prime sizes fix the table geometry, so they must be
    canonical whenever the table or the addressing is shared; the multipliers
    decide which row inside that geometry an n-gram hits, so they are canonical
    only when the addressing is shared.
    """

    @staticmethod
    def _hasher(share_memory, share_hash):
        from nano_scalemb.engram import EngramConfig, NgramHasher

        cfg = EngramConfig(
            layer_ids=(2, 6), max_ngram_size=3, n_head_per_ngram=2,
            slot_multiplier=2, use_tokenizer_compression=False,
            share_memory=share_memory, share_hash=share_hash,
        )
        return NgramHasher(cfg, tokenizer_vocab_size=64)

    @staticmethod
    def _hashes(h, ids):
        return {lid: h.hash(ids, lid) for lid in (2, 6)}

    def test_default_follows_share_memory(self):
        """None must reproduce every earlier run bit-for-bit."""
        from nano_scalemb.engram import EngramConfig

        assert EngramConfig().share_hash is None
        ids = torch.randint(0, 64, (2, 8), generator=torch.Generator().manual_seed(0))
        for share_memory in (False, True):
            auto = self._hashes(self._hasher(share_memory, None), ids)
            explicit = self._hashes(self._hasher(share_memory, share_memory), ids)
            for lid in (2, 6):
                torch.testing.assert_close(auto[lid], explicit[lid])

    def test_shared_hash_gives_identical_indices_across_layers(self):
        ids = torch.randint(0, 64, (2, 8), generator=torch.Generator().manual_seed(1))
        h = self._hashes(self._hasher(share_memory=True, share_hash=True), ids)
        torch.testing.assert_close(h[2], h[6])

    def test_per_layer_hash_gives_different_indices_across_layers(self):
        """The 'one pool, different rows' cell: same table geometry, so indices
        stay in range, but an n-gram lands somewhere else at each depth."""
        ids = torch.randint(0, 64, (2, 8), generator=torch.Generator().manual_seed(2))
        hasher = self._hasher(share_memory=True, share_hash=False)
        h = self._hashes(hasher, ids)
        assert not torch.equal(h[2], h[6])
        # geometry must still be shared, or the indices could overflow one table
        assert hasher.resolve_prime_layer_id(6) == hasher.resolve_prime_layer_id(2)
        assert hasher.resolve_mult_layer_id(6) != hasher.resolve_mult_layer_id(2)

    def test_indices_stay_in_range_of_the_shared_geometry(self):
        hasher = self._hasher(share_memory=True, share_hash=False)
        ids = torch.randint(0, 64, (4, 16), generator=torch.Generator().manual_seed(3))
        canonical = hasher.resolve_prime_layer_id(2)
        sizes = [hasher.prime_table[canonical][o][k]
                 for o in range(len(hasher.ngram_orders))
                 for k in range(hasher.cfg.n_head_per_ngram)]
        for lid in (2, 6):
            hh = hasher.hash(ids, lid)
            for head, n in enumerate(sizes):
                assert int(hh[..., head].max()) < n, (lid, head)
                assert int(hh[..., head].min()) >= 0

    def test_shared_hash_without_shared_table_shares_geometry(self):
        """The 'same index, different content' cell: separate tables, but an
        n-gram addresses the same row number in each."""
        ids = torch.randint(0, 64, (2, 8), generator=torch.Generator().manual_seed(4))
        hasher = self._hasher(share_memory=False, share_hash=True)
        h = self._hashes(hasher, ids)
        torch.testing.assert_close(h[2], h[6])

    def test_all_four_cells_are_distinguishable(self):
        ids = torch.randint(0, 64, (2, 8), generator=torch.Generator().manual_seed(5))
        same = {}
        for sm in (False, True):
            for sh in (False, True):
                h = self._hashes(self._hasher(sm, sh), ids)
                same[(sm, sh)] = torch.equal(h[2], h[6])
        # index identity across layers tracks share_hash, not share_memory
        assert same[(True, True)] and same[(False, True)]
        assert not same[(True, False)] and not same[(False, False)]

    def test_full_model_builds_all_four_cells(self):
        from nano_scalemb.engram import EngramConfig
        from nano_scalemb.gpt import GPT, GPTConfig

        for sm in (False, True):
            for sh in (False, True):
                engram = EngramConfig(
                    layer_ids=(0, 2), max_ngram_size=3, n_head_per_ngram=2,
                    memory_dim=32, slot_multiplier=2,
                    use_tokenizer_compression=False,
                    share_memory=sm, share_hash=sh,
                )
                config = GPTConfig(
                    sequence_len=16, vocab_size=64, n_layer=4, n_head=2,
                    n_kv_head=2, n_embd=32, window_pattern="L", engram=engram,
                )
                with torch.device("meta"):
                    model = GPT(config)
                model.to_empty(device="cpu")
                model.init_weights()
                model.train()
                out = model(torch.randint(0, 64, (2, 16)))
                assert torch.isfinite(out).all(), (sm, sh)
                t0 = model.engram_modules["0"].memory_table
                t2 = model.engram_modules["2"].memory_table
                shared = t0 is t2
                assert shared == sm, (sm, sh, shared)
                model.setup_optimizer()

    def test_old_checkpoint_config_defaults_to_auto(self):
        from nano_scalemb.checkpoint_manager import _patch_missing_config_keys

        d = {"window_pattern": "L", "mhc": None,
             "engram": {"layer_ids": [2, 6], "memory_dim": 1280,
                        "share_memory": True}}
        _patch_missing_config_keys(d)
        cfg = d["engram"]
        assert cfg.share_hash is None and cfg.share_memory is True


class TestEngramHashBackoff:
    """Hash backoff: rare windows route to their per-token (*,*,t) key."""

    def _hasher(self, tmp_path, keysets=None, context_keep=1, seed=0):
        from nano_scalemb.engram import NgramHasher, EngramConfig
        import torch as _t
        path = ""
        if keysets is not None:
            path = str(tmp_path / "ks.pt")
            _t.save({"keysets": keysets}, path)
        cfg = EngramConfig(
            layer_ids=(0,), max_ngram_size=3, n_head_per_ngram=2,
            use_tokenizer_compression=False, address_source="tokens", seed=seed,
            slot_multiplier=4, backoff_keyset_path=path,
            backoff_context_keep=context_keep,
        )
        return NgramHasher(cfg, tokenizer_vocab_size=64, compression=None), cfg

    def test_backoff_off_is_bit_identical(self, tmp_path):
        # Empty keyset path must reproduce the current hash exactly.
        h_off, _ = self._hasher(tmp_path, keysets=None)
        h_ref, _ = self._hasher(tmp_path, keysets=None)
        x = torch.randint(1, 64, (3, 20), dtype=torch.long)
        assert torch.equal(h_off.hash(x, 0), h_ref.hash(x, 0))

    def test_membership_matches_bruteforce(self, tmp_path):
        base = 64  # no compression -> V_compressed == vocab
        x = torch.randint(1, 64, (2, 16), dtype=torch.long)
        # order-3 window id at each pos: cur + prev*base + prev2*base^2
        cur = x
        prev = torch.full_like(x, 0); prev[:, 1:] = x[:, :-1]
        prev2 = torch.full_like(x, 0); prev2[:, 2:] = x[:, :-2]
        ids = (cur + prev * base + prev2 * base * base).reshape(-1)
        keep_ids = torch.unique(ids)[::3]  # keep every third distinct window
        h, _ = self._hasher(tmp_path, keysets={3: keep_ids})
        h0, _ = self._hasher(tmp_path, keysets=None)
        rb, r0 = h.hash(x, 0), h0.hash(x, 0)
        # order-3 columns are heads [2,3]; kept positions unchanged, others changed-or-equal
        keep_set = set(keep_ids.tolist())
        flat_ids = ids.reshape(x.shape)
        for b in range(2):
            for t in range(16):
                kept = int(flat_ids[b, t]) in keep_set
                same = torch.equal(rb[b, t, 2:4], r0[b, t, 2:4])
                if kept:
                    assert same, (b, t, "kept window must be unchanged")

    def test_nonkept_routes_to_backoff_row(self, tmp_path):
        # A non-kept trigram must hash to the row of a synthetic (pad,pad,t) input.
        base, pad = 64, 0
        h, _ = self._hasher(tmp_path, keysets={3: torch.tensor([1])})  # keep only id=1
        x = torch.randint(2, 64, (1, 8), dtype=torch.long)  # avoid id collisions with 1
        rb = h.hash(x, 0)
        # reference: no-backoff hash of context-padded x at a full-context position
        h0, _ = self._hasher(tmp_path, keysets=None)
        pos = 5
        xb = x.clone(); xb[0, pos - 1] = pad; xb[0, pos - 2] = pad
        ref = h0.hash(xb, 0)
        assert torch.equal(rb[0, pos, 2:4], ref[0, pos, 2:4])

    def test_backoff_adds_no_parameters(self, tmp_path):
        # Frozen buffer, not an nn.Parameter -> iso to the same recipe without it.
        from nano_scalemb.gpt import GPT, GPTConfig
        from nano_scalemb.mhc import MHCConfig
        base = 64
        keep = torch.tensor([1, 2, 3], dtype=torch.long)
        path = str(tmp_path / "ks.pt"); torch.save({"keysets": {3: keep}}, path)

        def build(bo_path):
            cfg = GPTConfig(
                sequence_len=16, vocab_size=64, n_layer=2, n_head=2, n_kv_head=2, n_embd=32,
                engram=EngramConfig(
                    layer_ids=(0,), use_tokenizer_compression=False, max_ngram_size=3,
                    n_head_per_ngram=2, memory_dim=32, slot_multiplier=2, kernel_size=4,
                    mhc_num_streams=4, backoff_keyset_path=bo_path),
                mhc=MHCConfig(num_streams=4, sinkhorn_iters=3),
            )
            with torch.device("cpu"):
                m = GPT(cfg)
            return sum(p.numel() for p in m.parameters())

        assert build("") == build(path)

    def test_compose_with_count_gate_and_mhc_forward_finite(self, tmp_path):
        from nano_scalemb.gpt import GPT, GPTConfig
        from nano_scalemb.mhc import MHCConfig
        keep = torch.tensor([1, 2, 3, 5, 8], dtype=torch.long)
        path = str(tmp_path / "ks.pt"); torch.save({"keysets": {3: keep}}, path)
        config = GPTConfig(
            sequence_len=16, vocab_size=64, n_layer=2, n_head=2, n_kv_head=2, n_embd=32,
            engram=EngramConfig(
                layer_ids=(0,), use_tokenizer_compression=False, max_ngram_size=3,
                n_head_per_ngram=2, memory_dim=32, slot_multiplier=2, kernel_size=4,
                mhc_num_streams=4, count_gate=True, count_gate_decouple=True,
                backoff_keyset_path=path, backoff_mode="topp"),
            mhc=MHCConfig(num_streams=4, sinkhorn_iters=3),
        )
        with torch.device("cpu"):
            model = GPT(config)
        model.init_weights()
        for block in model.transformer.h:
            block.attn.forward = lambda x, ve, cos_sin, window_size, kv_cache: x
        idx = torch.randint(0, 64, (2, 8), dtype=torch.long)
        fa_module._override_impl = "sdpa"
        try:
            logits = model(idx)
        finally:
            fa_module._override_impl = None
        assert logits.shape == (2, 8, 64)
        assert torch.isfinite(logits).all()


class TestEngramRowMerge:
    """Idea 1B: mid-training value-aware row merge. An alias buffer redirects
    several hash addresses onto one survivor row, shrinking the EFFECTIVE table
    while adding no learnable params (iso). Pre-registered as a COMPRESSION test.
    """

    def _table(self, list_of_N, D=8, key_dim=0, **kw):
        """A MultiHeadEmbedding with identity alias, initialized like a real run."""
        from nano_scalemb.engram import MultiHeadEmbedding, init_memory_table
        t = MultiHeadEmbedding(list_of_N, D, key_dim=key_dim, **kw)
        init_memory_table(t, ablation_mode="none")
        return t

    def test_merge_off_is_identity_and_iso(self):
        # merge_frac=0 path: alias is arange, _merged is False, forward reads the
        # raw embedding rows -- bit-identical to a table that never grew the
        # feature. And row_alias/_merged are buffers, so param count is unchanged.
        from nano_scalemb.engram import MultiHeadEmbedding
        t = self._table([5, 5], D=8)
        assert torch.equal(t.row_alias, torch.arange(10))
        assert not bool(t._merged)
        idx = torch.randint(0, 5, (3, 7, 2), dtype=torch.long)  # per-head indices
        out = t(idx)
        # gather by hand: head h reads embedding row (idx + offset_h)
        flat = idx + t.offsets  # [3,7,2]
        ref = t.embedding(flat)
        assert torch.equal(out, ref)
        # buffers, not params
        names = {n for n, _ in t.named_parameters()}
        assert not any("row_alias" in n or "_merged" in n for n in names)

    def test_alias_routing(self):
        # Hand-set an alias entry and confirm forward reads the survivor's value.
        t = self._table([6], D=8)
        # make row 4 point at row 1
        with torch.no_grad():
            t.row_alias[4] = 1
            t._merged.fill_(True)
        t.sync_merge_flags()  # external buffer write -> refresh the Python-bool gate
        idx = torch.tensor([[4]], dtype=torch.long)  # [B=1, H=1], head 0 idx 4
        out = t(idx)  # [B=1, H=1, D]
        assert torch.equal(out[0, 0], t.embedding.weight.data[1])

    def test_merge_hits_target_and_clusters_by_value(self):
        # Build a single head whose rows form clear value clusters, merge to 50%,
        # and check: exactly `keep` distinct survivors, and every row aliases to a
        # survivor that is its nearest (highest-cosine) among survivors.
        import torch.nn.functional as F
        t = self._table([8], D=4, count_gate=True)  # count_gate -> hit_rate exists
        with torch.no_grad():
            # 4 tight pairs: rows {0,1},{2,3},{4,5},{6,7} nearly identical
            base = torch.tensor([[3., 0, 0, 0], [0, 3., 0, 0],
                                 [0, 0, 3., 0], [0, 0, 0, 3.]])
            W = torch.empty(8, 4)
            for p in range(4):
                W[2 * p] = base[p]
                W[2 * p + 1] = base[p] + 0.01
            t.embedding.weight.data.copy_(W)
            t.hit_rate.copy_(torch.arange(8, dtype=torch.float) + 1.0)  # distinct
        stats = t.merge_rows(frac=0.5, metric="cosine", seed=0)
        assert stats["total_rows"] == 8
        assert stats["effective_rows"] == 4  # keep = round(0.5*8) = 4
        alias = t.row_alias
        survivors = torch.unique(alias)
        assert survivors.numel() == 4
        # every survivor maps to itself
        assert torch.all(alias[survivors] == survivors)
        # every row's survivor is its argmax-cosine among survivors
        Vn = F.normalize(t.embedding.weight.data.float(), dim=-1)
        sim = Vn @ Vn[survivors].T
        want = survivors[sim.argmax(dim=-1)]
        assert torch.equal(alias, want)

    def test_merge_never_crosses_head_partition(self):
        # A row in head h must only ever alias to a flat index inside head h's
        # [offset_h, offset_h + N_h) block -- heads are disjoint address spaces.
        t = self._table([5, 7, 4], D=6, count_gate=True)
        with torch.no_grad():
            t.embedding.weight.data.normal_()
            t.hit_rate.normal_().abs_()
        t.merge_rows(frac=0.6, metric="cosine", seed=1)
        offsets = t.offsets.tolist() + [t.total_rows]
        for h, N in enumerate(t.head_sizes):
            lo, hi = offsets[h], offsets[h] + N
            block = t.row_alias[lo:hi]
            assert torch.all((block >= lo) & (block < hi)), (
                f"head {h} aliased outside its partition")

    def test_lsh_metric_scales_and_hits_target(self):
        # The LSH path (used above _COSINE_MAX) also produces exactly `keep`
        # survivors per head and stays inside each partition.
        t = self._table([9], D=5, count_gate=True)
        with torch.no_grad():
            t.embedding.weight.data.normal_()
            t.hit_rate.normal_().abs_()
        stats = t.merge_rows(frac=1.0 / 3.0, metric="lsh", seed=2)
        assert stats["effective_rows"] == 6  # keep = round(2/3 * 9) = 6
        assert torch.all(t.row_alias < 9)

    def test_durability_roundtrip_and_backfill(self):
        # A merged alias survives state_dict round-trip; a PRE-merge checkpoint
        # (no row_alias/_merged keys) backfills to identity via the patcher.
        from nano_scalemb.checkpoint_manager import _patch_missing_merge_buffers
        t = self._table([6, 6], D=4, count_gate=True)
        with torch.no_grad():
            t.embedding.weight.data.normal_()
            t.hit_rate.normal_().abs_()
        t.merge_rows(frac=0.5, metric="cosine", seed=0)
        merged_alias = t.row_alias.clone()

        fresh = self._table([6, 6], D=4, count_gate=True)
        sd = t.state_dict()
        assert "row_alias" in sd and "_merged" in sd
        fresh.load_state_dict(sd)
        assert torch.equal(fresh.row_alias, merged_alias)
        assert bool(fresh._merged)
        # load_state_dict does NOT touch the Python-bool _merged_flag, so the
        # resumed table still routes _shift through the no-op path until synced.
        # This is the regression the fix guards: branching on the tensor buffer
        # inside compiled forward perturbs the identity path numerically.
        assert fresh._merged_flag is False
        fresh.sync_merge_flags()
        assert fresh._merged_flag is True
        idx = torch.tensor([[[0, 0]]], dtype=torch.long)  # [B,T,H]
        assert torch.equal(fresh._shift(idx), t._shift(idx))

        # Pre-feature checkpoint: strip the new buffers, then backfill against a
        # freshly-built (identity) model, the way build_model does on load.
        old_sd = {k: v for k, v in t.state_dict().items()
                  if not (k.endswith("row_alias") or k.endswith("_merged"))}
        target = self._table([6, 6], D=4, count_gate=True)
        _patch_missing_merge_buffers(old_sd, target)
        assert "row_alias" in old_sd and "_merged" in old_sd
        assert torch.equal(old_sd["row_alias"], torch.arange(12))
        assert not bool(old_sd["_merged"])

    def test_merge_frac_validation(self):
        t = self._table([5], D=4)
        for bad in (0.0, 1.0, -0.1, 1.5):
            with pytest.raises(ValueError):
                t.merge_rows(frac=bad)
        with pytest.raises(ValueError):
            t.merge_rows(frac=0.5, metric="euclidean")

    def test_feature_override_clusters_on_supplied_matrix(self):
        # merge_source="semantic": clustering must use the EXTERNAL feature, not
        # the learned value table. Give the value table one clustering and the
        # override a DIFFERENT one; the alias must follow the override.
        t = self._table([8], D=4, count_gate=True)
        with torch.no_grad():
            # value table: pair rows {0,1},{2,3},{4,5},{6,7}
            vbase = torch.tensor([[3., 0, 0, 0], [0, 3., 0, 0],
                                  [0, 0, 3., 0], [0, 0, 0, 3.]])
            W = torch.empty(8, 4)
            for p in range(4):
                W[2 * p] = vbase[p]; W[2 * p + 1] = vbase[p] + 0.01
            t.embedding.weight.data.copy_(W)
            # override feature: a DIFFERENT pairing {0,2},{1,3},{4,6},{5,7}, each
            # cluster on its own orthogonal axis.
            feat = torch.zeros(8, 4)
            for a, b, d in [(0, 2, 0), (1, 3, 1), (4, 6, 2), (5, 7, 3)]:
                feat[a, d] = 3.0; feat[b, d] = 3.0 + 0.01
            # survivor = highest hit per cluster: make {0,1,4,5} the survivors so
            # every override cluster keeps exactly one, and the other member joins
            # it by cosine.
            hit = torch.tensor([9., 9., 1., 1., 9., 9., 1., 1.])
            t.hit_rate.copy_(hit)
        t.merge_rows(frac=0.5, metric="cosine", seed=0, feature_override=feat)
        alias = t.row_alias
        # rows that share an OVERRIDE cluster must share a survivor
        assert alias[0] == alias[2] and alias[1] == alias[3]
        assert alias[4] == alias[6] and alias[5] == alias[7]
        # and rows that share only a VALUE cluster must NOT (override won)
        assert alias[0] != alias[1] and alias[4] != alias[5]

    def test_keep_mask_identity_keeps_featureless_rows(self):
        # Rows with keep_mask=False must stay identity-aliased (never merged, never
        # a survivor) -- the semantic prior's featureless (hit==0) rows.
        t = self._table([10], D=4, count_gate=True)
        with torch.no_grad():
            t.embedding.weight.data.normal_()
            feat = torch.zeros(10, 4)
            # only rows 0..5 participate; give them 3 clear pairs on distinct axes
            for p, (a, b) in enumerate([(0, 1), (2, 3), (4, 5)]):
                feat[a, p] = 3.0; feat[b, p] = 3.0 + 0.01
            # one survivor per pair: {0,2,4} highest, so 6 -> 3 collapse cleanly
            hit = torch.tensor([9., 1., 9., 1., 9., 1., 0., 0., 0., 0.])
            t.hit_rate.copy_(hit)
            keep_mask = torch.zeros(10, dtype=torch.bool)
            keep_mask[:6] = True
        t.merge_rows(frac=0.5, metric="cosine", seed=0,
                     feature_override=feat, keep_mask=keep_mask)
        alias = t.row_alias
        # non-participating rows 6..9 map to themselves
        assert torch.equal(alias[6:], torch.arange(6, 10))
        # participating rows collapsed 6 -> 3 survivors, all inside 0..5
        part_surv = torch.unique(alias[:6])
        assert part_surv.numel() == 3
        assert torch.all(part_surv < 6)


    def test_semantic_config_roundtrips_through_checkpoint(self):
        # merge_source/semantic_prior_path must survive an EngramConfig round-trip
        # and backfill to defaults for pre-feature checkpoints. This is exactly
        # what checkpoint_manager does: EngramConfig(**saved_engram_kwargs), which
        # fills any absent field from its dataclass default.
        cfg = EngramConfig(layer_ids=(2, 6), merge_frac=0.5,
                           merge_source="semantic",
                           semantic_prior_path="/tmp/prior.pt")
        kwargs = dict(vars(cfg))
        rebuilt = EngramConfig(**kwargs)
        assert rebuilt.merge_source == "semantic"
        assert rebuilt.semantic_prior_path == "/tmp/prior.pt"
        # an OLD config dict without the new keys backfills to the defaults
        old = {k: v for k, v in kwargs.items()
               if k not in ("merge_source", "semantic_prior_path")}
        patched = EngramConfig(**old)
        assert patched.merge_source == "value"
        assert patched.semantic_prior_path == ""

    def test_compose_with_count_gate_and_mhc_forward_finite(self):
        # Full model with a fired merge + count gate + backbone mHC -> finite
        # logits of the right shape. Merge happens on the deduped shared table.
        from nano_scalemb.gpt import GPT, GPTConfig
        from nano_scalemb.mhc import MHCConfig
        config = GPTConfig(
            sequence_len=16, vocab_size=64, n_layer=2, n_head=2, n_kv_head=2, n_embd=32,
            engram=EngramConfig(
                layer_ids=(0,), use_tokenizer_compression=False, max_ngram_size=3,
                n_head_per_ngram=2, memory_dim=32, slot_multiplier=4, kernel_size=4,
                mhc_num_streams=4, count_gate=True, count_gate_decouple=True,
                merge_frac=0.5, merge_at_frac=0.5),
            mhc=MHCConfig(num_streams=4, sinkhorn_iters=3),
        )
        with torch.device("cpu"):
            model = GPT(config)
        model.init_weights()
        # fire the merge on every shared Engram table, as the training hook does
        n_tables = 0
        for table in model._engram_memory_tables():
            table.merge_rows(frac=0.5, metric="cosine", seed=0)
            assert bool(table._merged)
            n_tables += 1
        assert n_tables >= 1
        for block in model.transformer.h:
            block.attn.forward = lambda x, ve, cos_sin, window_size, kv_cache: x
        idx = torch.randint(0, 64, (2, 8), dtype=torch.long)
        fa_module._override_impl = "sdpa"
        try:
            logits = model(idx)
        finally:
            fa_module._override_impl = None
        assert logits.shape == (2, 8, 64)
        assert torch.isfinite(logits).all()
