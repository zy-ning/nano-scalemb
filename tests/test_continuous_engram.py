"""Continuous Engram: addressing the memory by hidden state instead of tokens."""

import pytest
import torch

import nano_scalemb.flash_attention as fa_module
from nano_scalemb.discretize import (
    MAX_CODE_BITS,
    LSHDiscretizer,
    PQDiscretizer,
    code_alphabet_size,
)
from nano_scalemb.engram import (
    EngramConfig,
    Engram,
    NgramHasher,
    collect_engram_aux_loss,
    ngram_orders,
)
from nano_scalemb.gpt import GPT, GPTConfig
from nano_scalemb.mhc import MHCConfig
from nano_scalemb.product_key import ProductKeyMemory

SOURCES = {
    "hybrid": {"lsh_bits": 12},
    "tokens": {},
    "lsh": {"lsh_bits": 12},
    "pq": {"pq_subspaces": 2, "pq_codebook_size": 16},
    "pkm": {"pkm_n_keys": 16, "pkm_topk": 4, "pkm_query_dim": 32},
}


def tiny_engram_config(source="tokens", **overrides):
    kwargs = dict(
        layer_ids=(0, 2),
        max_ngram_size=3,
        n_head_per_ngram=2,
        memory_dim=32,
        slot_multiplier=2,
        use_tokenizer_compression=False,
        address_source=source,
    )
    kwargs.update(SOURCES[source])
    kwargs.update(overrides)
    return EngramConfig(**kwargs)


def build_gpt(engram, mhc=None, n_layer=4, n_embd=32):
    config = GPTConfig(
        sequence_len=16,
        vocab_size=64,
        n_layer=n_layer,
        n_head=2,
        n_kv_head=2,
        n_embd=n_embd,
        window_pattern="L",
        engram=engram,
        mhc=mhc,
    )
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device="cpu")
    model.init_weights()
    return model


# ---------------------------------------------------------------------------
# LSH
# ---------------------------------------------------------------------------


def test_lsh_codes_are_deterministic_and_in_range():
    torch.manual_seed(0)
    d = LSHDiscretizer(64, n_bits=10, seed=0)
    d.reset_parameters()
    x = torch.randn(2, 16, 64)

    codes = d(x)

    assert codes.shape == (2, 16)
    assert codes.dtype == torch.int64
    assert (codes >= 0).all() and (codes < d.num_codes).all()
    assert torch.equal(d(x), codes)


def test_lsh_is_scale_invariant_when_normalizing():
    """The hidden state's magnitude drifts a lot over training; only its
    direction should decide the address."""
    torch.manual_seed(0)
    x = torch.randn(2, 8, 64)

    normed = LSHDiscretizer(64, n_bits=10, seed=0, normalize=True)
    normed.reset_parameters()
    assert torch.equal(normed(x), normed(x * 9.7))

    # Without normalization a pure rescale still cannot flip a sign, so use a
    # shift to show the un-normalized code is not invariant to affine drift.
    raw = LSHDiscretizer(64, n_bits=10, seed=0, normalize=False)
    raw.reset_parameters()
    assert not torch.equal(raw(x), raw(x + 3.0))


def test_lsh_uses_many_distinct_codes():
    torch.manual_seed(0)
    d = LSHDiscretizer(64, n_bits=16, seed=0)
    d.reset_parameters()

    codes = d(torch.randn(4, 64, 64))

    # Random hyperplanes over random inputs should almost never collide.
    assert len(codes.unique()) > 0.9 * codes.numel()


def test_lsh_has_no_learned_parameters():
    """The point of the LSH arm: it changes the address SOURCE without adding
    learned addressing, so it isolates one variable against the token baseline."""
    d = LSHDiscretizer(64, n_bits=10, seed=0)

    assert list(d.parameters()) == []


def test_lsh_rejects_code_widths_that_would_overflow_the_hasher():
    with pytest.raises(ValueError, match="n_bits"):
        LSHDiscretizer(64, n_bits=MAX_CODE_BITS + 1)


# ---------------------------------------------------------------------------
# PQ / VQ
# ---------------------------------------------------------------------------


def test_pq_codes_in_range_and_codebook_learns():
    torch.manual_seed(0)
    d = PQDiscretizer(64, n_subspaces=2, codebook_size=16, seed=0)
    d.reset_parameters()
    d.train()
    x = torch.randn(4, 32, 64)

    codes = d(x)

    assert codes.shape == (4, 32)
    assert (codes >= 0).all() and (codes < d.num_codes).all()

    before = d.codebook.clone()
    d(x)
    assert not torch.equal(before, d.codebook), "EMA should move the codebook"


def test_pq_data_dependent_init_avoids_collapse_and_huge_commitment():
    """Gaussian-init centroids sit far from real (RMS-normed) activations: the
    commitment loss explodes and most centroids are never selected. Seeding from
    real sub-vectors on the first training batch fixes both."""
    torch.manual_seed(0)
    d = PQDiscretizer(64, n_subspaces=2, codebook_size=16, seed=0)
    d.reset_parameters()
    d.train()
    assert not bool(d._initialized)

    d(torch.randn(4, 32, 64))

    assert bool(d._initialized)
    assert d.aux_loss().item() < 1.0
    assert d.codebook_usage() > 0.5


def test_pq_commitment_loss_flows_gradient_to_the_encoder():
    torch.manual_seed(0)
    d = PQDiscretizer(64, n_subspaces=2, codebook_size=16, seed=0)
    d.reset_parameters()
    d.train()
    x = torch.randn(2, 16, 64, requires_grad=True)

    d(x)
    d.aux_loss().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0


def test_pq_codebook_is_not_an_optimizer_parameter():
    """Codebooks are EMA-updated buffers. Registering them as Parameters would
    have the optimizer fight the EMA."""
    d = PQDiscretizer(64, n_subspaces=2, codebook_size=16, seed=0)

    names = [n for n, _ in d.named_parameters()]
    buffers = [n for n, _ in d.named_buffers()]
    assert names == []
    assert "codebook" in buffers


def test_pq_emits_no_commitment_loss_in_eval():
    torch.manual_seed(0)
    d = PQDiscretizer(64, n_subspaces=2, codebook_size=16, seed=0)
    d.reset_parameters()
    x = torch.randn(2, 8, 64)

    d.train()
    d(x)
    assert d.aux_loss() is not None

    d.eval()
    d(x)
    assert d.aux_loss() is None, "a stale loss must not be collectable twice"


def test_pq_rejects_alphabets_that_would_overflow_the_hasher():
    # 256**4 = 2^32, one bit too wide for the int64 multiply in NgramHasher.
    with pytest.raises(ValueError, match="bits"):
        PQDiscretizer(64, n_subspaces=4, codebook_size=256)


def test_code_alphabet_size_matches_the_discretizers():
    assert code_alphabet_size(tiny_engram_config("lsh", lsh_bits=12)) == 1 << 12
    cfg = tiny_engram_config("pq", pq_subspaces=2, pq_codebook_size=16)
    assert code_alphabet_size(cfg) == 16**2


# ---------------------------------------------------------------------------
# Product-key memory
# ---------------------------------------------------------------------------


def test_pkm_is_zero_at_init_and_fully_differentiable():
    torch.manual_seed(0)
    m = ProductKeyMemory(64, num_heads=4, value_dim=8, n_keys=16, topk=4, query_dim=32)
    m.reset_parameters()
    x = torch.randn(2, 6, 64, requires_grad=True)

    assert torch.equal(m(x), torch.zeros(2, 6, 32)), "zero-init values => no-op branch"

    m.values.data.normal_(std=0.1)
    m(x).sum().backward()

    assert torch.isfinite(x.grad).all()
    # The distinguishing property vs LSH/PQ: gradient reaches the addressing.
    assert m.keys.grad is not None and m.keys.grad.abs().sum() > 0
    assert m.query_proj.weight.grad.abs().sum() > 0


def test_pkm_reads_only_topk_rows_per_head():
    torch.manual_seed(0)
    m = ProductKeyMemory(32, num_heads=2, value_dim=4, n_keys=8, topk=3, query_dim=16)
    m.reset_parameters()
    m.values.data.normal_(std=0.1)

    m(torch.randn(1, 1, 32)).sum().backward()

    # Exactly topk rows per head should receive gradient.
    touched = (m.values.grad.abs().sum(-1) > 0).sum(dim=-1)
    assert touched.tolist() == [3, 3]


def test_pkm_memory_grows_quadratically_in_n_keys():
    small = ProductKeyMemory(32, num_heads=1, value_dim=4, n_keys=8, topk=2, query_dim=16)
    big = ProductKeyMemory(32, num_heads=1, value_dim=4, n_keys=16, topk=2, query_dim=16)

    assert small.n_values == 64 and big.n_values == 256


# ---------------------------------------------------------------------------
# Unigram addressing
# ---------------------------------------------------------------------------


def test_ngram_orders_starts_at_bigrams_unless_unigram_requested():
    assert ngram_orders(3) == [2, 3]
    assert ngram_orders(2) == [2]
    # n=1 is only meaningful with contextual addressing, where the code already
    # summarizes context.
    assert ngram_orders(1) == [1]


def test_unigram_hasher_allocates_heads_and_hashes():
    cfg = tiny_engram_config("lsh", max_ngram_size=1)
    hasher = NgramHasher(cfg, tokenizer_vocab_size=256, compression=None)

    assert hasher.num_hash_heads == cfg.n_head_per_ngram

    codes = torch.randint(0, 100, (2, 8))
    hashes = hasher.hash(codes, 0, compressed_input_ids=codes)
    assert hashes.shape == (2, 8, cfg.n_head_per_ngram)


def test_unigram_address_depends_only_on_the_current_position():
    """n=1 must not mix in history: two sequences agreeing at position t must
    hash identically there, whatever precedes it."""
    cfg = tiny_engram_config("lsh", max_ngram_size=1)
    hasher = NgramHasher(cfg, tokenizer_vocab_size=256, compression=None)
    a = torch.tensor([[1, 2, 3, 4]])
    b = torch.tensor([[9, 9, 9, 4]])

    ha = hasher.hash(a, 0, compressed_input_ids=a)
    hb = hasher.hash(b, 0, compressed_input_ids=b)

    assert torch.equal(ha[:, -1], hb[:, -1])


def test_trigram_address_does_depend_on_history():
    cfg = tiny_engram_config("lsh", max_ngram_size=3)
    hasher = NgramHasher(cfg, tokenizer_vocab_size=256, compression=None)
    a = torch.tensor([[1, 2, 3, 4]])
    b = torch.tensor([[9, 9, 9, 4]])

    ha = hasher.hash(a, 0, compressed_input_ids=a)
    hb = hasher.hash(b, 0, compressed_input_ids=b)

    assert not torch.equal(ha[:, -1], hb[:, -1])


# ---------------------------------------------------------------------------
# Engram integration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("source", list(SOURCES))
def test_engram_forward_backward_for_every_address_source(source):
    torch.manual_seed(0)
    cfg = tiny_engram_config(source)
    engram = Engram(cfg=cfg, layer_id=0, d_model=64, tokenizer_vocab_size=256)
    engram.init_weights()
    engram.train()
    x = torch.randn(2, 8, 64, requires_grad=True)
    input_ids = torch.randint(0, 256, (2, 8))

    y = engram(x, input_ids)
    aux = engram.aux_loss()
    loss = y.sum() + (aux if aux is not None else 0.0)
    loss.backward()

    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert torch.isfinite(x.grad).all()
    assert (aux is not None) == (source == "pq")


@pytest.mark.parametrize("source", ["lsh", "pq", "pkm"])
def test_contextual_sources_ignore_token_ids(source):
    """The whole point: the address comes from the hidden state, so changing the
    token ids while holding the hidden state fixed must not change the output."""
    torch.manual_seed(0)
    engram = Engram(
        cfg=tiny_engram_config(source), layer_id=0, d_model=64, tokenizer_vocab_size=256
    )
    engram.init_weights()
    engram.eval()
    x = torch.randn(2, 8, 64)

    with torch.no_grad():
        a = engram(x, torch.randint(0, 256, (2, 8)))
        b = engram(x, torch.randint(0, 256, (2, 8)))

    assert torch.equal(a, b)


def test_token_source_does_depend_on_token_ids():
    torch.manual_seed(0)
    engram = Engram(
        cfg=tiny_engram_config("tokens"),
        layer_id=0,
        d_model=64,
        tokenizer_vocab_size=256,
    )
    engram.init_weights()
    # value_proj is zero-init, so give it something to read through.
    with torch.no_grad():
        engram.value_proj.weight.normal_(std=0.1)
    engram.eval()
    x = torch.randn(2, 8, 64)

    with torch.no_grad():
        a = engram(x, torch.zeros(2, 8, dtype=torch.long))
        b = engram(x, torch.full((2, 8), 7, dtype=torch.long))

    assert not torch.equal(a, b)


def test_pkm_engram_has_no_hash_table():
    engram = Engram(
        cfg=tiny_engram_config("pkm"), layer_id=0, d_model=64, tokenizer_vocab_size=256
    )

    assert engram.memory_table is None, "a hash table here would never get gradient"
    assert engram.product_key is not None


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("source", list(SOURCES))
def test_gpt_trains_with_every_address_source(source):
    torch.manual_seed(0)
    model = build_gpt(tiny_engram_config(source))
    model.train()
    idx = torch.randint(0, 64, (2, 8))

    prev = fa_module._override_impl
    fa_module._override_impl = "sdpa"
    try:
        loss = model(idx, idx)
        aux = collect_engram_aux_loss(model)
        if aux is not None:
            loss = loss + aux
        loss.backward()
    finally:
        fa_module._override_impl = prev

    assert torch.isfinite(loss)
    # Every trainable parameter must actually be trained: a dead table would
    # silently inflate the param count and trip the optimizer's assert.
    dead = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert dead == []
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
    model.setup_optimizer()


@pytest.mark.parametrize("source", list(SOURCES))
def test_buffers_survive_meta_device_init(source):
    """meta -> to_empty leaves buffers as garbage (~1e38). Discretizer state must
    be re-initialized in init_weights or the PQ commitment loss explodes."""
    model = build_gpt(tiny_engram_config(source))

    for engram in model.engram_modules.values():
        for _, buf in engram.named_buffers():
            if buf.is_floating_point():
                assert torch.isfinite(buf).all()
                assert buf.abs().max() < 1e4


@pytest.mark.parametrize("source", ["lsh", "pq"])
def test_composes_with_backbone_mhc(source):
    torch.manual_seed(0)
    model = build_gpt(
        tiny_engram_config(source), mhc=MHCConfig(num_streams=4, sinkhorn_iters=3)
    )
    idx = torch.randint(0, 64, (2, 8))

    prev = fa_module._override_impl
    fa_module._override_impl = "sdpa"
    try:
        logits = model(idx)
    finally:
        fa_module._override_impl = prev

    assert logits.shape == (2, 8, 64)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("source", ["lsh", "pq"])
def test_shared_memory_composes_with_contextual_addressing(source):
    model = build_gpt(tiny_engram_config(source, share_memory=True))

    t0 = model.engram_modules["0"].memory_table
    t2 = model.engram_modules["2"].memory_table
    assert t0 is t2
    assert t0.embedding.weight is t2.embedding.weight


def test_shared_memory_rejects_pkm():
    """PKM owns its keys and values; there is no MultiHeadEmbedding to share."""
    with pytest.raises(AssertionError, match="pkm"):
        build_gpt(tiny_engram_config("pkm", share_memory=True))


def test_collect_engram_aux_loss_only_fires_for_pq():
    idx = torch.randint(0, 64, (2, 8))
    prev = fa_module._override_impl
    fa_module._override_impl = "sdpa"
    try:
        for source in SOURCES:
            model = build_gpt(tiny_engram_config(source))
            model.train()
            model(idx, idx)
            aux = collect_engram_aux_loss(model)
            assert (aux is not None) == (source == "pq"), source
    finally:
        fa_module._override_impl = prev


def test_old_checkpoint_config_defaults_to_token_addressing():
    from nano_scalemb.checkpoint_manager import _patch_missing_config_keys

    config_dict = {
        "window_pattern": "L",
        "mhc": None,
        "moe": None,
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

    assert config_dict["engram"].address_source == "tokens"


@pytest.mark.parametrize("source", ["lsh", "pq", "pkm"])
def test_unigram_addressing_works_end_to_end(source):
    """n=1 is only sensible with contextual addressing, and is a config knob so
    the temporal window can be ablated separately from the address source."""
    torch.manual_seed(0)
    model = build_gpt(tiny_engram_config(source, max_ngram_size=1))
    idx = torch.randint(0, 64, (2, 8))

    prev = fa_module._override_impl
    fa_module._override_impl = "sdpa"
    try:
        loss = model(idx, idx)
        loss.backward()
    finally:
        fa_module._override_impl = prev

    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Second-generation variants: balanced LSH, per-head latents, detached PQ, hybrid
# ---------------------------------------------------------------------------


def test_balanced_lsh_raises_realized_code_entropy():
    """sign(<w,h>) is only an unbiased bit if the projection has zero median. On
    shifted/correlated hidden states many bits are near-constant, wasting most of
    the nominal width. Re-centring on an EMA of each bit's mean recovers it."""
    from nano_scalemb.discretize import LSHDiscretizer

    torch.manual_seed(0)
    basis = torch.randn(16, 128)
    h = (torch.randn(8192, 16) @ basis + 4.0).reshape(8, 1024, 128)  # shifted

    def entropy(codes):
        _, n = torch.unique(codes, return_counts=True)
        p = n.float() / n.sum()
        return float(-(p * p.log2()).sum())

    ents = {}
    for balance in (False, True):
        d = LSHDiscretizer(128, n_bits=15, seed=0, balance=balance)
        d.reset_parameters()
        d.train()
        for _ in range(30):
            d(h)
        d.eval()
        ents[balance] = entropy(d(h))

    assert ents[True] > ents[False] + 0.5, ents


def test_balanced_lsh_threshold_round_trips():
    """The thresholds ARE the addressing scheme; losing them on resume would
    re-address every slot in the table."""
    from nano_scalemb.discretize import LSHDiscretizer

    d = LSHDiscretizer(32, n_bits=8, seed=0, balance=True)
    assert "bit_threshold" in dict(d.named_buffers())
    assert "bit_threshold" in d.state_dict()


def test_per_head_latents_emit_one_code_per_head():
    torch.manual_seed(0)
    cfg = tiny_engram_config("lsh", n_head_per_ngram=4, address_latents_per_head=True)
    engram = Engram(cfg=cfg, layer_id=0, d_model=64, tokenizer_vocab_size=256)
    engram.init_weights()

    codes = engram.discretizer(torch.randn(2, 8, 64))

    assert codes.shape == (2, 8, engram.hasher.num_hash_heads)


def test_single_latent_heads_are_not_redundant():
    """A CRT property worth pinning: distinct primes per head mean one shared
    code still yields non-redundant addresses. (I originally assumed otherwise
    and designed per-head latents to fix a problem that did not exist.)"""
    torch.manual_seed(0)
    cfg = tiny_engram_config("lsh", n_head_per_ngram=4, lsh_bits=12)
    engram = Engram(cfg=cfg, layer_id=0, d_model=64, tokenizer_vocab_size=256)
    engram.init_weights()
    basis = torch.randn(10, 64)
    h = (torch.randn(4096, 10) @ basis).reshape(4, 1024, 64)

    codes = engram.discretizer(h)
    hashes = engram.hasher.hash(codes, 0, compressed_input_ids=codes).reshape(-1, 8)

    # For a fixed head-0 row, head-1 must still take many values.
    mask = hashes[:, 0] == hashes[0, 0]
    assert len(hashes[mask, 1].unique()) > 1


def test_detached_pq_emits_no_commitment_loss():
    """The commitment term pulls the residual stream toward its centroid; on a
    live layer it collapsed the representation to ~1 effective dimension."""
    from nano_scalemb.discretize import PQDiscretizer

    d = PQDiscretizer(
        64, n_subspaces=4, codebook_size=16, per_head=True, detach_encoder=True
    )
    d.reset_parameters()
    d.train()
    x = torch.randn(2, 16, 64, requires_grad=True)

    codes = d(x)

    assert d.aux_loss() is None
    assert codes.shape == (2, 16, 4)
    # ...and the codebook still adapts to the data.
    before = d.codebook.clone()
    d(torch.randn(2, 16, 64))
    assert not torch.equal(before, d.codebook)


def test_detached_pq_does_not_backprop_into_the_hidden_state():
    torch.manual_seed(0)
    cfg = tiny_engram_config(
        "pq", pq_codebook_size=16, address_latents_per_head=True, pq_detach_encoder=True
    )
    engram = Engram(cfg=cfg, layer_id=0, d_model=64, tokenizer_vocab_size=256)
    engram.init_weights()
    engram.train()
    # Zero value_proj is the only other path from x to the output, so with it
    # zeroed any gradient reaching x could only come through addressing.
    x = torch.randn(2, 8, 64, requires_grad=True)
    y = engram(x, torch.randint(0, 256, (2, 8)))
    y.sum().backward()

    assert engram.aux_loss() is None
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_hybrid_uses_both_token_and_hidden_heads():
    """The information hypothesis: token addressing injects exact token identity
    that h has partly discarded, so semantics should ADD to it rather than
    replace it. Hybrid gives some heads to each."""
    torch.manual_seed(0)
    cfg = tiny_engram_config("hybrid", n_head_per_ngram=4, lsh_bits=12)
    engram = Engram(cfg=cfg, layer_id=0, d_model=64, tokenizer_vocab_size=256)
    engram.init_weights()
    engram.eval()
    with torch.no_grad():
        engram.value_proj.weight.normal_(std=0.1)
    x = torch.randn(2, 8, 64)
    ids_a = torch.zeros(2, 8, dtype=torch.long)
    ids_b = torch.full((2, 8), 7, dtype=torch.long)

    with torch.no_grad():
        # token heads => changing ids must change the output
        assert not torch.equal(engram(x, ids_a), engram(x, ids_b))
        # hidden heads => changing h must change it too
        assert not torch.equal(engram(x, ids_a), engram(torch.randn(2, 8, 64), ids_a))


def test_hybrid_token_head_fraction_is_respected_and_clamped():
    for frac, n_heads in [(0.5, 4), (0.25, 4), (0.0, 4), (1.0, 4)]:
        cfg = tiny_engram_config(
            "hybrid", n_head_per_ngram=n_heads, hybrid_token_head_frac=frac
        )
        engram = Engram(cfg=cfg, layer_id=0, d_model=64, tokenizer_vocab_size=256)
        engram.init_weights()
        engram.eval()
        # Never all-token or all-hidden: the arm must stay a genuine mix.
        n_tok = max(1, min(n_heads - 1, round(frac * n_heads)))
        assert 1 <= n_tok <= n_heads - 1
        with torch.no_grad():
            assert torch.isfinite(engram(torch.randn(2, 6, 64), torch.randint(0, 256, (2, 6)))).all()


@pytest.mark.parametrize(
    "kw",
    [
        dict(address_source="lsh", lsh_balance=True),
        dict(address_source="lsh", address_latents_per_head=True),
        dict(address_source="pq", pq_codebook_size=16, address_latents_per_head=True,
             pq_detach_encoder=True),
        dict(address_source="hybrid"),
    ],
    ids=["lsh_balanced", "lsh_perhead", "pq_detached", "hybrid"],
)
def test_new_variants_train_end_to_end(kw):
    torch.manual_seed(0)
    base = dict(lsh_bits=12) if kw.get("address_source") in ("lsh", "hybrid") else {}
    cfg = tiny_engram_config(kw["address_source"], **{**base, **kw})
    model = build_gpt(cfg)
    model.train()
    idx = torch.randint(0, 64, (2, 8))

    prev = fa_module._override_impl
    fa_module._override_impl = "sdpa"
    try:
        loss = model(idx, idx)
        aux = collect_engram_aux_loss(model)
        if aux is not None:
            loss = loss + aux
        loss.backward()
    finally:
        fa_module._override_impl = prev

    assert torch.isfinite(loss)
    dead = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert dead == []
    model.setup_optimizer()


# ---------------------------------------------------------------------------
# share_memory x contextual addressing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("source", ["lsh", "pq", "hybrid"])
def test_share_memory_also_shares_the_discretizer(source):
    """A shared TABLE with per-layer projections is just a bigger table: the same
    hidden state would land on unrelated rows at different depths. Sharing the
    memory must therefore share the addressing function too."""
    extra = dict(SOURCES[source])
    if source == "pq":
        extra.update(address_latents_per_head=True, pq_detach_encoder=True)
    model = build_gpt(tiny_engram_config(source, share_memory=True, **extra))

    e0, e2 = model.engram_modules["0"], model.engram_modules["2"]
    assert e0.discretizer is e2.discretizer
    assert e0.memory_table is e2.memory_table
    assert model.engram_shared_discretizer is e0.discretizer
    # ...and no layer thinks it owns it, so init happens exactly once.
    assert not e0.owns_discretizer and not e2.owns_discretizer


def test_shared_addressing_maps_a_hidden_state_to_the_same_row_at_every_depth():
    shared = build_gpt(tiny_engram_config("lsh", share_memory=True))
    private = build_gpt(tiny_engram_config("lsh", share_memory=False))
    h = torch.randn(2, 8, 32)

    for model, expect_same in ((shared, True), (private, False)):
        e0, e2 = model.engram_modules["0"], model.engram_modules["2"]
        c0, c2 = e0.discretizer(h), e2.discretizer(h)
        r0 = e0.hasher.hash(c0, 0, compressed_input_ids=c0)
        r2 = e2.hasher.hash(c2, 2, compressed_input_ids=c2)
        assert torch.equal(c0, c2) is expect_same
        assert torch.equal(r0, r2) is expect_same


def test_shared_discretizer_state_is_stored_once():
    """PQ codebooks are persistent buffers; duplicating them per layer would both
    bloat the checkpoint and unshare on load_state_dict(assign=True)."""
    model = build_gpt(
        tiny_engram_config(
            "pq",
            share_memory=True,
            pq_subspaces=2,
            pq_codebook_size=16,
            address_latents_per_head=True,
            pq_detach_encoder=True,
        )
    )

    keys = [k for k in model.state_dict() if k.endswith("codebook")]
    assert len(keys) == 1, keys
    assert keys[0].startswith("engram_shared_discretizer")


@pytest.mark.parametrize("source", ["lsh", "hybrid"])
def test_shared_contextual_addressing_trains(source):
    torch.manual_seed(0)
    model = build_gpt(tiny_engram_config(source, share_memory=True))
    model.train()
    idx = torch.randint(0, 64, (2, 8))

    prev = fa_module._override_impl
    fa_module._override_impl = "sdpa"
    try:
        loss = model(idx, idx)
        aux = collect_engram_aux_loss(model)
        if aux is not None:
            loss = loss + aux
        loss.backward()
    finally:
        fa_module._override_impl = prev

    assert torch.isfinite(loss)
    assert [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None] == []
    model.setup_optimizer()
