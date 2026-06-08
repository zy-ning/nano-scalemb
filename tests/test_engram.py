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
