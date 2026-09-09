# TN-gram (Tensorized Engram) — CP-factorized value table: a diagnosed negative

**Branch:** `engram-row-merge-experiments`  **Date:** 2026-09-08
**Verdict:** CP factorization does **not** match the native hash-indexed Engram at
equal value-param budget. Min bpb stalls at **~0.771** for every variant tried — a flat
**+6e-3 above native slot15 (0.76486)** and +6.5e-3 above the token floor (0.764396). The
ceiling is **intrinsic to the CP read**: it is invariant to rank, to init, and to whether the
token-position factors are shared across n-gram orders.

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

| Variant | 500 | 1000 | 1500 | min bpb | vs native slot15 |
|---|---|---|---|---|---|
| native slot15 (anchor) | — | — | — | 0.76486 | — |
| shared A, R=10 (N(0,1) init) | 0.9578 | 0.8981 | 0.8713 | **0.770938** | +0.0061 |
| shared A, R=20 | — | — | — | ~tracks R10, ~1e-3 worse | +~0.007 |
| shared A, R=10 (init fix A=1+0.5·N) | 0.9606 | 0.9028 | 0.8769 | worse, aborted | — |
| **independent A, R=6** (orders untied) | 0.9596 | 0.8987 | 0.8722 | **0.771356** | +0.0065 |

Throughput fine throughout (~262k tok/s, bf16_mfu ~8.2) — the extra Hadamard/RMSNorm/einsum
is not a regression.

## Three axes, all pinned negative

1. **Rank.** R=20 tracked R=10 checkpoint-for-checkpoint (~1e-3 *worse*, not better).
   Doubling rank does nothing → the ceiling is **not capacity**.
2. **Init.** Centering the factors at the multiplicative identity (to fight the Hadamard
   collapse where a product of N N(0,1) rows concentrates in ~2 of R coords) made results
   *mildly worse* with the gap widening. With `embedding_lr_mult=5` the A factors train far
   from init anyway → init-time collapse is **not the binding constraint**.
3. **Cross-order sharing.** Independent-A R=6 (each order its own factors, iso-param with
   shared R=10) lies **on top of** the shared curve the entire way (+0.0018 → +0.0006 →
   +0.0009 → +0.0004 at the end). Untying the orders buys nothing → **the paper's core design
   choice is not the liability**.

## Conclusion

The **+6e-3 ceiling is intrinsic to CP factorization** of this Engram's value read, invariant
to rank, init, and sharing. The CP read behaves like another lossy addressing scheme, landing
in the same +0.006–0.008 band as the LSH-addressing liability documented elsewhere. This is a
clean, well-diagnosed negative — not a bug (e2e gradients flow, shapes are correct, no
throughput regression). Theorem 4.1's "sharing helps" does not transfer to this setup; the
native per-order hash tables' explicit capacity beats the shared low-rank structure at equal
params. **Line closed.**

Code and tests remain on the branch (`test_tngram.py` 11/11, `test_tngram_e2e.py` 4/4, 191
existing engram+merge tests green) for reproducibility; the `tngram` value table is opt-in and
defaults untouched.
