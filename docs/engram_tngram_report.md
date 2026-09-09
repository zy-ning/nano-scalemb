# TN-gram (Tensorized Engram) — CP-factorized value table: a diagnosed negative

**Branch:** `engram-row-merge-experiments`  **Date:** 2026-09-08 → 2026-09-09
**Verdict:** CP factorization **underperforms** the native hash-indexed Engram value table at
**every budget tested** — from 0.6% to 19.4% of the model. A native slot15 table using **0.6%**
of the model (0.76486 bpb) beats a CP table using **19.4%** (R=400, 0.767101 bpb). Scaling CP
by 32× (into the paper's own 20%-of-model regime) narrows the gap from +6e-3 to +2.2e-3 but
never reaches parity. The native table's explicit per-order capacity is simply a stronger value
representation than the shared low-rank CP structure for this Engram read.

**Important framing (from a careful re-read of the paper).** The paper's headline is **not** a
BPB win — Table 1 (N=5) shows TN-gram **ties** Engram on val BPB (18L: 1.071 vs 1.070). Its
claimed advantages are **CORE (downstream) and parameter efficiency**. And its memory is a huge
fraction of the model (**22–26% of total**), where CP's param-saving matters. Our native Engram
table is already tiny (**0.6%** of the model), so CP's "saves params vs a huge hash table"
selling point has nothing to bite on here. The two studies operate in different regimes; this
report is the apples-to-apples BPB comparison **at our scale**, and at our scale native wins.

## Motivation

arXiv 2606.08347v1 *"Tensorizing Engram"* critiques exactly our design: a **separate hash
table per n-gram order** (orders {2,3}, `n_head_per_ngram=8` heads each), so nested n-grams
share no latent structure and every order pays its own collisions. TN-gram replaces the row
tables with **one rank-R CP tensor** whose token-position factors A₁..A_N are shared across
orders, indexed by the window tokens directly. Value params drop from O(Vⁿ·d) to
O(n·V·R + d·R). Theorem 4.1 claims lower-order n-gram vectors are linear combinations of
higher-order ones, so sharing should *help*. **Question:** does a shared-factor CP table match
the native bigger-is-better Engram at the same value-param budget?

## Design

`TensorizedNgramMemory(AddressedMemory)` in `engram.py`, selected by
`EngramConfig.value_table="tngram"` + `cp_rank`. K=8 independent CP models (own A/F/w/log_scale
per head); `n_orders·K = 16` heads of `head_dim=80` reconstruct `memory_dim=1280`, so the
`[B,T,16,80]` layout is byte-identical and value_proj / gate / short-conv / mHC are untouched.

- `A` = `ModuleList[nn.Embedding(V, K·R)]` (shared) or `A_orders` = per-order factor stacks
  (independent). `F` = `(K,d,R)` explicit value factor. `w`, `log_scale` stored **flat 1D** so
  the ndim-based optimizer split routes them to the no-decay AdamW group (a 2D `log_scale`
  would hit Muon and get orthogonalized; weight-decaying `w`→0 would kill the lower-order read).
- **forward:** Hadamard product over the n most-recent A slots, absorb leading slots with `w`
  (shared mode only), fp32 RMSNorm over R, `exp(log_scale)` scale, `einsum btkr,kdr->btkd`.
- Raw window tokens fed via `NgramHasher.compressed_windows()`; `_compute_embeddings`
  short-circuits the hash path. Merge hooks raise (no rows to cluster).

**Iso-param anchors (value params):** R=10 shared ≈ native slot15 (5,685,536); R=20 shared ≈
native slot30 (11,370,359). Independent-A R=6 has **exactly** the same A-param count as shared
R=10 (`5·V·K·6 == 3·V·K·10`), a clean sharing-vs-independent ablation at fixed budget.

## Results (d20, orders {2,3}, mHC-4, layers 2,6, shared memory, ~2.1h/arm)

### Iso-param-to-native sweep (the original, too-small budget)

| Variant | 500 | 1000 | 1500 | min bpb | vs native slot15 |
|---|---|---|---|---|---|
| native slot15 (anchor) | — | — | — | **0.76486** | — |
| native slot30 (anchor) | — | — | — | 0.76393 | −0.001 |
| shared A, R=10 (N(0,1) init) | 0.9578 | 0.8981 | 0.8713 | 0.770938 | +0.0061 |
| shared A, R=20 | — | — | — | ~tracks R10, ~1e-3 worse | +~0.007 |
| shared A, R=10 (init fix A=1+0.5·N) | 0.9606 | 0.9028 | 0.8769 | worse, aborted | — |
| independent A, R=6 (orders untied) | 0.9596 | 0.8987 | 0.8722 | 0.771356 | +0.0065 |

### Rank / budget scale sweep (the decisive test — memory as a fraction of the model)

| Variant | value params | % of model | 500 | 1000 | 1500 | min bpb | vs native slot15 |
|---|---|---|---|---|---|---|---|
| native slot15 | 5.7M | **0.6%** | — | — | — | **0.76486** | — |
| CP R=10 (F rank-limited) | 5.7M | 0.6% | 0.9578 | 0.8981 | 0.8713 | 0.770938 | +0.0061 |
| CP R=80 (F 80×80 full-rank) | 45.5M | 4.7% | 0.9552 | 0.8961 | 0.8708 | 0.770665 | +0.0058 |
| **CP R=400 (paper's ~20% regime)** | 228M | **19.4%** | 0.9513 | 0.8923 | 0.8665 | **0.767101** | **+0.0022** |

Throughput fine throughout (~258–264k tok/s, bf16_mfu ~8.1–8.3) — the extra
Hadamard/RMSNorm/einsum is not a regression, even at R=400.

## Four axes, all negative

1. **Rank / output bottleneck.** Our read contracts through `F` of shape `(K, head_dim=80, R)`,
   so R<80 caps each per-head read at rank-R in an 80-dim space. R=80 makes `F` full-rank
   (80×80), removing the bottleneck — yet it endpoints at 0.770665, only **0.00027** better than
   the rank-limited R=10 (0.770938). Rank is not the binding constraint.
2. **Budget / scale.** Scaling from 0.6% → 4.7% → 19.4% of the model gives 0.770938 → 0.770665
   → 0.767101. A **32× memory increase buys ~0.0038 bpb**, and the R=400 table (19.4% of model)
   **still loses to native slot15 at 0.6%**. Even at the paper's own ~20% regime, CP does not
   reach native parity.
3. **Init.** Centering factors at the multiplicative identity (A=1+0.5·N, to fight the Hadamard
   collapse where a product of N N(0,1) rows concentrates in ~2 of R coords) made results
   *mildly worse*. With `embedding_lr_mult=5` the A factors train far from init anyway → init is
   not the binding constraint.
4. **Cross-order sharing.** Independent-A R=6 (each order its own factors, iso-param with shared
   R=10) lies **on top of** the shared curve the whole way (end +0.0004). Untying the orders
   buys nothing → the paper's core design choice is not the liability.

## Conclusion

At our scale, **CP factorization is a strictly weaker value representation than the native
per-order hash table**, at every budget from 0.6% to 19.4% of the model. It is not a starved-rank
artifact (full-rank R=80 ≈ rank-limited R=10), not init, and not cross-order sharing (independent
≈ shared). The native table's explicit per-order capacity simply beats the shared low-rank CP
structure — even when CP is given 32× the parameters, into the paper's own 20%-of-model regime.

This does **not** contradict the paper: (1) the paper's headline is a **CORE + parameter-count**
win, and only a **BPB tie** (Table 1: 1.071 vs 1.070); (2) the paper's memory is **22–26% of the
model** where CP's param-saving matters, whereas our native table is already **0.6%** of the
model, so CP's core advantage has nothing to bite on here. The two studies operate in different
regimes. **Line closed.**

Code and tests remain on the branch (`test_tngram.py` 11/11, `test_tngram_e2e.py` 4/4, 191
existing engram+merge tests green) for reproducibility; the `tngram` value table is opt-in and
defaults untouched. An N=5 (paper's n-gram-depth sweet spot; ours is N=3) confirm arm was
considered but not run, since R=400 stalled clearly short of parity.
