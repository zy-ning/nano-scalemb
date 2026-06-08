"""
Test Engine class. Example run:

python -m pytest tests/test_engine.py -v
"""

import json
import tempfile

import torch
import nano_scalemb.flash_attention as fa_module
from nano_scalemb.engine import KVCache, Engine
from dataclasses import dataclass, field
from nano_scalemb.gpt import GPT, GPTConfig
from nano_scalemb.engram import EngramConfig
from nano_scalemb.mhc import MHCConfig


# -----------------------------------------------------------------------------
# Mock classes for testing Engine without loading a real model


@dataclass
class MockConfig:
    """Minimal config for Engine tests."""

    n_kv_head: int = 4
    n_head: int = 4
    n_embd: int = 64
    n_layer: int = 2
    sequence_len: int = 128
    engram = None


@dataclass
class MockEngramConfig:
    max_ngram_size: int = 3
    pad_id: int = 0


@dataclass
class MockConfigWithEngram(MockConfig):
    engram: MockEngramConfig = field(default_factory=MockEngramConfig)


@dataclass
class MockConfigWithBackboneMHC(MockConfigWithEngram):
    mhc: MHCConfig = field(default_factory=lambda: MHCConfig(num_streams=3))


class MockModel:
    """
    Mock model that returns uniform logits over the vocab.
    This ensures that with temperature > 0, different samples should
    (with very high probability) produce different tokens.
    """

    def __init__(self, vocab_size=262):  # 256 bytes + 6 special tokens
        self.vocab_size = vocab_size
        self.config = MockConfig()
        self._device = torch.device("cpu")

    def get_device(self):
        return self._device

    def forward(self, ids, kv_cache=None):
        """Return uniform logits so sampling is spread across vocab."""
        B, T = ids.shape
        # With FA3, flash_attn_with_kvcache updates cache in-place and we advance position
        if kv_cache is not None:
            kv_cache.advance(T)
        # Uniform logits -> equal probability for all tokens
        logits = torch.zeros(B, T, self.vocab_size)
        return logits


class MockModelWithEngramDecodeContext:
    """Mock model that validates suffix+current context is passed during decode."""

    def __init__(self, vocab_size=64):
        self.vocab_size = vocab_size
        self.config = MockConfigWithEngram()
        self._device = torch.device("cpu")
        self.decode_contexts = []

    def get_device(self):
        return self._device

    def forward(self, ids, kv_cache=None, engram_input_ids=None):
        B, T = ids.shape
        if kv_cache is not None:
            kv_cache.advance(T)
            if T == 1 and engram_input_ids is not None:
                self.decode_contexts.append(engram_input_ids.detach().cpu().tolist())
        logits = torch.full((B, T, self.vocab_size), -1e9)
        logits[:, :, 7] = 0.0  # deterministic argmax for temperature=0
        return logits


class ContextSensitiveMockModel:
    """Mock model where next token depends on last 3-token context."""

    def __init__(self, vocab_size=32):
        self.vocab_size = vocab_size
        self.config = MockConfigWithEngram()
        self._device = torch.device("cpu")

    def get_device(self):
        return self._device

    def forward(self, ids, kv_cache=None, engram_input_ids=None):
        B, T = ids.shape
        if kv_cache is not None:
            kv_cache.advance(T)
        context = engram_input_ids if engram_input_ids is not None else ids
        last3 = context[:, -3:]
        next_token = (last3.sum(dim=1) % self.vocab_size).long()
        logits = torch.full((B, T, self.vocab_size), -1e9)
        logits[torch.arange(B), -1, next_token] = 0.0
        return logits


class MockModelWithBackboneMHCDecodeContext:
    """Mock model that validates visible-batch decode under backbone mHC."""

    def __init__(self, vocab_size=64):
        self.vocab_size = vocab_size
        self.config = MockConfigWithBackboneMHC()
        self._device = torch.device("cpu")
        self.forward_batches = []
        self.engram_batches = []
        self.cache_batches = []

    def get_device(self):
        return self._device

    def forward(self, ids, kv_cache=None, engram_input_ids=None):
        self.forward_batches.append(ids.size(0))
        if kv_cache is not None:
            self.cache_batches.append(kv_cache.batch_size)
            assert kv_cache.batch_size == ids.size(0)
            kv_cache.advance(ids.size(1))
        if engram_input_ids is not None:
            self.engram_batches.append(engram_input_ids.size(0))
            assert engram_input_ids.size(0) == ids.size(0)
        logits = torch.full((ids.size(0), ids.size(1), self.vocab_size), -1e9)
        logits[:, :, 7] = 0.0
        return logits


class ByteTokenizer:
    """
    Simple byte-level tokenizer for testing.
    Tokens 0-255 are raw bytes, 256+ are special tokens.
    """

    def __init__(self):
        # Special tokens start at 256
        self._special_tokens = {
            "<|python_start|>": 256,
            "<|python_end|>": 257,
            "<|output_start|>": 258,
            "<|output_end|>": 259,
            "<|assistant_end|>": 260,
            "<|bos|>": 261,
        }
        self._bos = 261

    def encode_special(self, s):
        return self._special_tokens[s]

    def get_bos_token_id(self):
        return self._bos

    def encode(self, s, prepend=None):
        tokens = list(s.encode("utf-8"))  # bytes 0-255
        if prepend is not None:
            tokens = [prepend] + tokens
        return tokens

    def decode(self, tokens):
        # Filter out special tokens before decoding
        byte_tokens = [t for t in tokens if t < 256]
        return bytes(byte_tokens).decode("utf-8", errors="replace")


def test_kv_cache_basic():
    """Test basic KVCache functionality for FA3."""
    batch_size = 2
    num_heads = 3
    seq_len = 64
    head_dim = 5
    num_layers = 6

    kv_cache = KVCache(
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len=seq_len,
        head_dim=head_dim,
        num_layers=num_layers,
        device="cpu",
        dtype=torch.float32,
    )

    # Check initial state
    assert kv_cache.get_pos() == 0
    assert kv_cache.k_cache.shape == (
        num_layers,
        batch_size,
        seq_len,
        num_heads,
        head_dim,
    )
    assert kv_cache.v_cache.shape == (
        num_layers,
        batch_size,
        seq_len,
        num_heads,
        head_dim,
    )

    # Test advance
    kv_cache.advance(10)
    assert kv_cache.get_pos() == 10

    kv_cache.advance(5)
    assert kv_cache.get_pos() == 15

    # Test reset
    kv_cache.reset()
    assert kv_cache.get_pos() == 0

    # Test get_layer_cache returns correct views
    k_layer0, v_layer0 = kv_cache.get_layer_cache(0)
    assert k_layer0.shape == (batch_size, seq_len, num_heads, head_dim)
    assert v_layer0.shape == (batch_size, seq_len, num_heads, head_dim)


def test_kv_cache_prefill():
    """Test KVCache.prefill() copies data correctly."""
    batch_size = 1
    num_heads = 4
    head_dim = 8
    num_layers = 2

    # Create source cache and advance it
    src_cache = KVCache(
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len=32,
        head_dim=head_dim,
        num_layers=num_layers,
        device="cpu",
        dtype=torch.float32,
    )
    # Write some data to source cache
    src_cache.k_cache[0, 0, :16, :, :] = 1.0
    src_cache.v_cache[0, 0, :16, :, :] = 2.0
    src_cache.advance(16)

    # Create destination cache with larger seq_len
    dst_cache = KVCache(
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len=64,
        head_dim=head_dim,
        num_layers=num_layers,
        device="cpu",
        dtype=torch.float32,
    )

    # Prefill
    dst_cache.prefill(src_cache)

    # Check position was copied
    assert dst_cache.get_pos() == 16

    # Check data was copied
    assert (dst_cache.k_cache[0, 0, :16, :, :] == 1.0).all()
    assert (dst_cache.v_cache[0, 0, :16, :, :] == 2.0).all()


def test_kv_cache_suffix_tokens_roll_and_reset():
    """Suffix buffer should initialize, roll, append, and reset correctly."""
    kv_cache = KVCache(
        batch_size=2,
        num_heads=2,
        seq_len=16,
        head_dim=4,
        num_layers=2,
        device="cpu",
        dtype=torch.float32,
        suffix_size=2,
        pad_id=0,
    )

    prompt = torch.tensor([[9, 8, 7], [5, 4, 3]], dtype=torch.long)
    kv_cache.set_suffix_from_tokens(prompt)
    assert kv_cache.suffix_tokens.tolist() == [[8, 7], [4, 3]]

    kv_cache.append_tokens(torch.tensor([[6], [2]], dtype=torch.long))
    assert kv_cache.suffix_tokens.tolist() == [[7, 6], [3, 2]]

    kv_cache.reset()
    assert kv_cache.suffix_tokens.tolist() == [[0, 0], [0, 0]]


def test_cached_decode_passes_suffix_context_for_engram():
    """Decode step should receive [suffix_tokens + current_token] context."""
    model = MockModelWithEngramDecodeContext()
    tokenizer = ByteTokenizer()
    engine = Engine(model, tokenizer)

    prompt_tokens = [261, 11, 12, 13]
    # max_tokens=1 still executes one decode forward at loop tail
    list(
        engine.generate(
            prompt_tokens, num_samples=2, max_tokens=1, temperature=0.0, seed=0
        )
    )

    # max_ngram_size=3 => suffix size 2, and sampled current token is 7
    assert model.decode_contexts, "Expected at least one decode context call"
    first_context = model.decode_contexts[0]
    assert first_context == [[12, 13, 7], [12, 13, 7]]


def test_cached_decode_matches_naive_full_sequence_for_engram_context():
    """Greedy cached decode should match naive full-sequence decode."""
    model = ContextSensitiveMockModel(vocab_size=32)
    tokenizer = ByteTokenizer()
    engine = Engine(model, tokenizer)

    prompt = [261, 3, 5, 7]
    max_tokens = 6

    # Naive autoregressive decoding with full sequence each step
    naive = prompt.copy()
    for _ in range(max_tokens):
        ids = torch.tensor([naive], dtype=torch.long)
        logits = model.forward(ids)
        tok = int(torch.argmax(logits[:, -1, :], dim=-1).item())
        naive.append(tok)

    # Engine cached decoding
    generated, _ = engine.generate_batch(
        prompt,
        num_samples=2,
        max_tokens=max_tokens,
        temperature=0.0,
        seed=0,
    )

    assert generated[0] == naive
    assert generated[1] == naive


def test_cached_decode_passes_suffix_context_for_engram():
    """Engram decode should use the suffix-context contract (path-independent)."""
    model = MockModelWithEngramDecodeContext()
    tokenizer = ByteTokenizer()
    engine = Engine(model, tokenizer)

    prompt_tokens = [261, 11, 12, 13]
    list(
        engine.generate(
            prompt_tokens, num_samples=2, max_tokens=1, temperature=0.0, seed=0
        )
    )

    assert model.decode_contexts, "Expected at least one decode context call"
    first_context = model.decode_contexts[0]
    assert first_context == [[12, 13, 7], [12, 13, 7]]


def test_multi_sample_first_token_diversity():
    """
    Test that when generating multiple samples, each sample gets an independently
    sampled first token (not a broadcast of the same token to all rows).

    Previously, the first token after prefill was sampled once and broadcast to all
    rows, causing all samples to start identically. The fix expands the prefill logits
    to num_samples and samples independently for each row.

    With uniform logits over 262 tokens and 16 samples, the probability that all
    samples independently pick the same token is (1/262)^15 ≈ 10^-36. So if they're
    all identical, it indicates tokens are being broadcast instead of independently sampled.
    """
    model = MockModel(vocab_size=262)
    tokenizer = ByteTokenizer()
    engine = Engine(model, tokenizer)

    # Generate 16 samples with temperature=1.0 (stochastic sampling)
    prompt_tokens = [261, 72, 101, 108, 108, 111]  # <bos> + "Hello"
    num_samples = 16

    # Collect the first generated token from each sample
    first_tokens = []
    gen = engine.generate(
        prompt_tokens,
        num_samples=num_samples,
        max_tokens=1,  # We only need the first token
        temperature=1.0,
        seed=42,
    )
    for token_column, token_masks in gen:
        first_tokens = token_column  # This is the first (and only) yield

    # With uniform distribution and 16 samples, they should NOT all be identical
    # If they are all identical, the bug exists (broadcasting instead of sampling)
    unique_tokens = set(first_tokens)
    assert len(unique_tokens) > 1, (
        f"All {num_samples} samples got the same first token ({first_tokens[0]}). "
        f"With uniform logits, this is statistically impossible (~10^-36 probability) "
        f"unless tokens are being broadcast instead of independently sampled."
    )


def test_seed_reproducibility():
    """Same seed must produce identical output."""
    model = MockModel()
    engine = Engine(model, ByteTokenizer())
    prompt = [261, 72, 101, 108, 108, 111]  # <bos> + "Hello"

    for seed in [1, 42, 123, 999]:
        r1, _ = engine.generate_batch(prompt, max_tokens=5, seed=seed)
        r2, _ = engine.generate_batch(prompt, max_tokens=5, seed=seed)
        r3, _ = engine.generate_batch(prompt, max_tokens=5, seed=seed)
        assert r1 == r2 == r3, (
            "Same seed must produce identical output for the same prompt."
        )


def test_temperature_zero_determinism():
    """Temperature=0 is deterministic regardless of seed."""
    model = MockModel()
    engine = Engine(model, ByteTokenizer())
    prompt = [261, 72, 101, 108, 108, 111]

    r1, _ = engine.generate_batch(prompt, temperature=0.0, max_tokens=5, seed=1)
    r2, _ = engine.generate_batch(prompt, temperature=0.0, max_tokens=5, seed=42)
    r3, _ = engine.generate_batch(prompt, temperature=0.0, max_tokens=5, seed=123)
    assert r1 == r2 == r3, (
        "Temperature=0 must result in the same output for the same prompt regardless of seed."
    )


def test_max_tokens_respected():
    """Generation stops at max_tokens limit."""
    model = MockModel()
    engine = Engine(model, ByteTokenizer())
    prompt = [261, 72, 101, 108, 108, 111]

    for max_tokens in [1, 4, 16, 64]:
        results, _ = engine.generate_batch(prompt, max_tokens=max_tokens)
        num_generated_tokens = len(results[0]) - len(prompt)
        assert num_generated_tokens <= max_tokens, (
            f"Generated {num_generated_tokens} tokens, expected max_tokens={max_tokens} or less."
        )


def test_num_samples_count():
    """num_samples=N produces exactly N sequences."""
    model = MockModel()
    engine = Engine(model, ByteTokenizer())
    prompt = [261, 72, 101, 108, 108, 111]

    for num_samples in [1, 4, 16, 64]:
        results, _ = engine.generate_batch(
            prompt, num_samples=num_samples, max_tokens=3
        )
        assert len(results) == num_samples, (
            f"Expected {num_samples} sequences from {num_samples} samples, got {len(results)}"
        )


def test_different_seeds_introduce_variation_when_temperature_nonzero():
    """With temperature > 0, different seeds should introduce sampling variation."""
    model = MockModel()
    engine = Engine(model, ByteTokenizer())
    prompt = [261, 72, 101, 108, 108, 111]  # <bos> + "Hello"

    outputs = set()

    for seed in [1, 42, 123, 999, 1000, 1001, 1002, 1003, 1004, 1005]:
        results, _ = engine.generate_batch(
            prompt,
            temperature=1.0,
            max_tokens=5,
            seed=seed,
        )
        outputs.add(tuple(results[0]))

    # Sanity check: sampling actually introduces variation
    assert len(outputs) > 1, (
        "All seeds produced the same output which is statistically highly improbable."
    )


def test_append_audit_record_writes_jsonl():
    from scripts.chat_eval import _append_audit_record

    record = {"task_name": "SpellingBee", "passed": True, "problem_index": 3}

    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/audit.jsonl"
        _append_audit_record(path, record)

        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

    assert len(lines) == 1
    assert json.loads(lines[0]) == record


def test_conversation_messages_for_audit_stringifies_parts():
    from scripts.chat_eval import _conversation_messages_for_audit

    conversation = {
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": [{"type": "text", "text": "#### 2"}]},
        ]
    }

    messages = _conversation_messages_for_audit(conversation)

    assert messages[0] == {"role": "user", "content": "hello"}
    assert messages[1]["role"] == "assistant"
    assert "#### 2" in messages[1]["content"]


def test_gpt_config_accepts_backbone_mhc():
    config = GPTConfig(mhc=MHCConfig(num_streams=4, sinkhorn_iters=5))
    assert config.mhc is not None
    assert config.mhc.num_streams == 4
    assert config.mhc.sinkhorn_iters == 5


def test_gpt_forward_with_backbone_mhc_preserves_shape():
    config = GPTConfig(
        sequence_len=16,
        vocab_size=64,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
        mhc=MHCConfig(num_streams=4, sinkhorn_iters=3),
    )
    with torch.device("cpu"):
        model = GPT(config)
    model.init_weights()

    for block in model.transformer.h:
        block.attn.forward = lambda x, ve, cos_sin, window_size, kv_cache: x

    idx = torch.randint(0, 64, (2, 8), dtype=torch.long)
    logits = model(idx)

    assert logits.shape == (2, 8, 64)


def test_gpt_forward_with_backbone_mhc_real_attention_paths():
    config = GPTConfig(
        sequence_len=16,
        vocab_size=64,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=32,
        mhc=MHCConfig(num_streams=4, sinkhorn_iters=3),
    )
    with torch.device("cpu"):
        model = GPT(config)
    model.init_weights()

    seen = {}
    block = model.transformer.h[1]
    original_block_forward = block.forward
    original_attn_forward = block.attn.forward

    def wrapped_block_forward(
        x,
        ve,
        cos_sin,
        window_size,
        kv_cache,
        input_ids=None,
        compressed_input_ids=None,
    ):
        seen["block_x_batch"] = x.size(0)
        seen["block_ve_batch"] = None if ve is None else ve.size(0)
        return original_block_forward(
            x,
            ve,
            cos_sin,
            window_size,
            kv_cache,
            input_ids=input_ids,
            compressed_input_ids=compressed_input_ids,
        )

    def wrapped_attn(x, ve, cos_sin, window_size, kv_cache):
        seen["attn_x_batch"] = x.size(0)
        seen["attn_ve_batch"] = None if ve is None else ve.size(0)
        return original_attn_forward(x, ve, cos_sin, window_size, kv_cache)

    block.forward = wrapped_block_forward
    block.attn.forward = wrapped_attn

    idx = torch.randint(0, 64, (2, 8), dtype=torch.long)
    fa_module._override_impl = "sdpa"
    try:
        logits = model(idx)
    finally:
        fa_module._override_impl = None

    assert logits.shape == (2, 8, 64)
    assert torch.isfinite(logits).all()
    assert seen == {
        "block_x_batch": 8,
        "block_ve_batch": 2,
        "attn_x_batch": 2,
        "attn_ve_batch": 2,
    }


def test_real_gpt_cached_decode_keeps_full_engram_context_for_backbone_mhc():
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
    engram_cfg = config.engram
    assert engram_cfg is not None

    seen = {}
    engram_block = model.transformer.h[0]
    original_forward_mhc_branches = engram_block.engram.forward_mhc_branches
    for block in model.transformer.h:
        block.attn.forward = lambda x, ve, cos_sin, window_size, kv_cache: x

    def wrapped_forward_mhc_branches(x, input_ids, compressed_input_ids=None):
        seen["hidden_len"] = x.size(1)
        seen["input_ids_len"] = input_ids.size(1)
        return original_forward_mhc_branches(
            x, input_ids, compressed_input_ids=compressed_input_ids
        )

    engram_block.engram.forward_mhc_branches = wrapped_forward_mhc_branches

    idx = torch.tensor([[7], [7]], dtype=torch.long)
    engram_input_ids = torch.tensor([[11, 12, 7], [21, 22, 7]], dtype=torch.long)
    kv_cache = KVCache(
        batch_size=2,
        num_heads=config.n_kv_head,
        seq_len=16,
        head_dim=config.n_embd // config.n_head,
        num_layers=config.n_layer,
        device="cpu",
        dtype=torch.float32,
        suffix_size=engram_cfg.max_ngram_size - 1,
        pad_id=engram_cfg.pad_id,
    )

    logits = model(idx, kv_cache=kv_cache, engram_input_ids=engram_input_ids)

    assert logits.shape == (2, 1, 64)
    assert seen == {"hidden_len": 1, "input_ids_len": 3}


def test_real_gpt_engram_mhc_cached_decode_matches_full_prefix_logits():
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

    prompt = torch.tensor([[11, 12, 7]], dtype=torch.long)
    full_logits = model(prompt)[:, -1, :]

    kv_cache = KVCache(
        batch_size=1,
        num_heads=config.n_kv_head,
        seq_len=16,
        head_dim=config.n_embd // config.n_head,
        num_layers=config.n_layer,
        device="cpu",
        dtype=torch.float32,
        suffix_size=2,
        pad_id=0,
    )
    kv_cache.advance(prompt.size(1) - 1)
    cached_logits = model(
        prompt[:, -1:], kv_cache=kv_cache, engram_input_ids=prompt
    )[:, -1, :]

    assert torch.allclose(cached_logits, full_logits, atol=1e-6, rtol=1e-5)


def test_engine_backbone_mhc_decode_uses_visible_batch_for_model_and_cache():
    model = MockModelWithBackboneMHCDecodeContext()
    tokenizer = ByteTokenizer()
    engine = Engine(model, tokenizer)

    prompt_tokens = [261, 11, 12, 13]
    list(
        engine.generate(
            prompt_tokens,
            num_samples=2,
            max_tokens=1,
            temperature=0.0,
            seed=0,
        )
    )

    assert model.forward_batches == [1, 2]
    assert model.cache_batches == [1, 2]
    assert model.engram_batches == [2]
