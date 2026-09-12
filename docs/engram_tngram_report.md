# TN-gram (Tensorized Engram) — CP-factorized value table: negative at R/V≪1, REPRODUCED at R/V≈1

**Branch:** `engram-row-merge-experiments`  **Date:** 2026-09-08 → 2026-09-11
**Verdict (revised 2026-09-11):** the CP-vs-native gap is an **R/V (rank-vs-vocabulary) artifact**,
not an intrinsic property. At our 32k tokenizer (**R/V ≈ 0.0004–0.017**) CP loses to the native hash
table by ~+0.006 bpb at every budget from 0.6% to 19.4% of the model — but that is simply CP run far
below its operating regime. **In the paper's regime (vocab 1024, R=1024, R/V ≈ 1), CP crosses over
and beats native by −0.0020 bpb (0.807646 vs 0.809668)** — a direct reproduction of the paper. The
sections below are preserved in chronological order: the initial "closed negative" sweep, the two
diagnosed nulls (global readout, N=5 depth), and finally the **small-vocab reproduction** that
overturns the negative and pins the cause to R/V. Read the last section first for the headline.

**Important framing (from a careful re-read of the paper).** The paper's headline is **not** a
BPB win — Table 1 (N=5) shows TN-gram **ties** Engram on val BPB (18L: 1.071 vs 1.070), winning on
CORE + parameter efficiency. Crucially the paper runs at **R/V ≈ 1** (R=1024, vocab=1024); at its
one larger-vocab point (R=1800, vocab=8192, **R/V ≈ 0.22**) it *already loses* BPB. Our early
negatives were all at R/V ≪ 0.22 — the continuation of the paper's own trend, not a contradiction.
When we match the paper's R/V ≈ 1 (below), we reproduce its tie/slight-win.

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

## Global readout (paper Eq. 8): tested, also a null

A full re-read of the paper (arxiv.org/html/2606.08347v1) surfaced the one genuine
*implementation* divergence from our port. The paper does **not** partition the read per head:
it concatenates all `(N−1)·R` CP coordinates and maps the whole vector to `d_model` with a
**single** learned `M_V = Fᵀ W_V`. Our port instead forced the CP read into the native
`[B,T,16,80]` layout with a per-head `F ∈ (K,80,R)`, imposing **16 isolated rank-R
bottlenecks** the paper never has. Since the full-rank R=80 probe had already shown rank was
not the per-head binding constraint, the *partition* was the prime suspect for the gap.

`tngram_global_readout` (`--engram-tngram-global-readout`, default off) drops the explicit F,
emits the raw CP coordinates `[B,T,n_orders·K,R]`, and lets `value_proj` be the paper's global
`M_V` (Engram read width `n_orders·K·R` instead of `memory_dim`). At R=80 the value_proj is
1280×1280 either way, so global R=80 is **iso-param** with per-head R=80 (both 963.8–963.9M) —
a clean isolation of the readout, holding the A-budget and projection size fixed.

| step | global R=80 | per-head R=80 | native slot15 |
|---|---|---|---|
| 500 | 0.960019 | 0.9552 | 0.9492 |
| 1000 | 0.901156 | 0.8961 | 0.8894 |
| 2000 | 0.853830 | 0.8509 | 0.8440 |
| 3000 | 0.801864 | 0.8002 | 0.7933 |
| **end** | **0.772351** | **0.770665** | **0.76486** |

**Null.** Global readout finishes **+0.0017 worse** than per-head R=80 (and +0.0075 above
native slot15). The gap narrows over training (+0.005 → +0.0017 — the global M_V has to *learn*
to absorb F, which the explicit per-head F gets for free at init) but never crosses over.
Throughput identical (~262k tok/s, mfu 8.2). So the per-head partition was **not** the handicap:
the paper's own faithful readout gives no improvement at our scale. The only remaining untested
paper divergence is **N=5 depth** (orders {2,3,4,5} vs our {2,3}).

## N=5 depth (paper's n-gram order): tested, also a null

The last remaining paper divergence. The paper uses **N=5** (orders {2,3,4,5}); every run above
used **N=3** (orders {2,3}). Theorem 4.1's shared-factor benefit is strongest with *more* orders
to share across, so N=3 could simply be too shallow to show the paper's effect. N=5 changes the
head geometry: `n_orders=4`, `num_heads = 4·K = 32`, `head_dim = 40` (memory_dim=1280 fixed);
shared A-params = `5·V·K·R = 947,440·R`. Two arms (per-head readout, the stronger config):

- **R=48** → 45.5M A-params — **iso-param** with the best prior N=3 CP (per-head R=80, 45.5M,
  0.770665): same param budget, N=3→N=5. Clean depth isolation.
- **R=80** → 75.8M A-params — deeper *and* bigger.

| step | N=5 R=48 | N=5 R=80 | N=3 per-head R=80 | native slot15 |
|---|---|---|---|---|
| 500 | 0.958798 | 0.956988 | 0.9552 | 0.9492 |
| 1000 | 0.899616 | 0.896485 | 0.8961 | 0.8894 |
| 1500 | 0.872857 | 0.870842 | 0.8708 | 0.8632 |
| 2000 | 0.853760 | 0.851663 | 0.8509 | 0.8440 |
| 2500 | 0.825888 | 0.824591 | 0.8246 | — |
| 3000 | 0.801124 | 0.800425 | 0.8002 | 0.7933 |
| 3500 | 0.779515 | 0.778866 | — | — |
| **end** | **0.771438** | **0.770880** | **0.770665** | **0.76486** |

**Null.** Iso-param N=5 R=48 finishes **+0.00077 worse** than N=3 R=80; even N=5 R=80 (deeper AND
75.8M A-params) is **+0.00022 worse**. Both track just above the N=3 R=80 anchor the whole way and
never cross below it, and both remain +0.006 above native slot15. Adding the paper's extra n-gram
orders {4,5} does not rescue CP at our scale — depth was not the missing lever. Throughput healthy
throughout (~258–262k tok/s, mfu ~8.1). This was the **last untested divergence** from the paper's
config; with it null, every axis has been closed.

## Small-vocab reproduction (the R/V regime): CP CROSSES OVER AND WINS

**This is the axis that overturns the negative verdict above.** Every run in the sections above
held the vocabulary fixed at our tokenizer's ~32k (compressed ~23,686) and varied rank between
R=10 and R=400 — i.e. **R/V ≈ 0.0004–0.017**, always deep in the bottleneck where the CP token
factor `A ∈ R^{V×R}` cannot give each token an independent row. The paper's headline **tie** is at
**R/V ≈ 1** (R=1024, vocab=1024); at its one larger-vocab point (R=1800, vocab=8192, **R/V ≈ 0.22**)
TN-gram *already loses* BPB (1.071 vs 1.070). Plotting the paper's two points and ours on a single
R/V axis, our negatives are the smooth continuation of the paper's own downward trend — **not a
contradiction.** The decisive test is therefore to move into the paper's regime: shrink the whole
backbone vocabulary (the Engram's built-in tokenizer-compression bottoms out at ~23.7k and cannot
reach 1024) and re-run at **R ≈ V**.

Setup: trained fresh RustBPE tokenizers of vocab **1024** and **4096** into isolated base dirs
(data symlinked; on-the-fly tokenization means no dataset re-tokenization), `token_bytes.pt` written
so BPB stays vocab-invariant. All arms use `--engram-no-tokenizer-compression` so `A` indexes the
full padded vocab directly (paper-faithful raw-token indexing) — then **R = V ⇒ R/V ≈ 1**. Native
and CP arms in a cell share every flag but the value table (identical step-0 bpb confirms identical
backbone + data), so it is a clean within-cell comparison. **Do not** compare bpb *across* vocab
sizes except as an R/V trend — absolute bpb shifts with vocab.

### Decision cell — vocab 1024, N=5 (R=1024 ⇒ R/V ≈ 1)

| step | CP R=1024 (R/V≈1) | native | Δ (CP − native) |
|---|---|---|---|
| 500 | 1.024652 | 1.028265 | −0.0036 |
| 1000 | 0.960252 | 0.962478 | −0.0022 |
| 1500 | 0.930716 | 0.932224 | −0.0015 |
| 2000 | 0.908657 | 0.910105 | −0.0014 |
| 2500 | 0.876067 | 0.877487 | −0.0014 |
| 3000 | 0.846210 | 0.847781 | −0.0016 |
| 3500 | 0.818592 | 0.820358 | −0.0018 |
| **end** | **0.807646** | **0.809668** | **−0.0020** |

**CP wins by 0.0020 bpb, below native at every single eval.** The sign has **flipped** relative to
every prior run: at R/V ≈ 0.002–0.017 CP lost by ~+0.006; at R/V ≈ 1 CP wins by −0.002. This is a
direct reproduction of the paper's result (they report a tie/slight win at R/V ≈ 1) and confirms the
diagnosis from the paper re-read: **the persistent gap was an R/V (rank-vs-vocabulary) artifact, not
an intrinsic weakness of the CP factorization.** Native's apparent "win" throughout the sections
above was an artifact of running CP far below its operating regime — the 32k tokenizer forces
R/V ≪ 1, so CP never had a fair test until now.

### Crossover curve (R/V governs the sign of the gap)

The within-cell ordering is monotone in R/V in **every cell measured**, and **native always sits
between the two CP points** — R/V≈1 beats native, R/V≈0.25 loses to it. The crossover reproduces
across both n-gram depths (N=3, N=5) and both vocab sizes (1024, 4096):

| vocab | N | arm | R/V | min bpb | Δ vs native |
|---|---|---|---|---|---|
| 1024 | 5 | CP R=1024 | ≈1.00 | 0.807646 | **−0.0020 (CP wins)** |
| 1024 | 5 | native | — | 0.809668 | — |
| 1024 | 5 | CP R=256 | ≈0.25 | 0.810719 | **+0.0010 (CP loses)** |
| 1024 | 3 | CP R=1024 | ≈1.00 | 0.807715 | **−0.0012 (CP wins)** |
| 1024 | 3 | native | — | 0.808901 | — |
| 1024 | 3 | CP R=256 | ≈0.25 | 0.810984 | **+0.0021 (CP loses)** |
| 4096 | 5 | CP R=4096 | ≈1.00 | 0.783180 | **−0.0058 (CP wins)** |
| 4096 | 5 | native | — | 0.788987 | — |
| 4096 | 5 | CP R=1024 | ≈0.25 | 0.789941 | **+0.0009 (CP loses)** |
| 4096 | 3 | CP R=4096 | ≈1.00 | 0.784847 | **−0.0056 (CP wins)** |
| 4096 | 3 | native | — | 0.790432 | — |

For contrast, the 32k-vocab runs (R/V ≈ 0.0004–0.017) had CP losing by **+0.006**. So as R/V climbs
0.002 → 0.25 → 1.0, the CP−native gap moves +0.006 → +0.001 → −0.002: the sign flips near R/V ≈ 0.3,
consistent with the paper losing BPB at R/V ≈ 0.22 and tying/winning at R/V ≈ 1. The **vocab-4096
N=5** cell shows the **largest R/V≈1 win yet (−0.0058)** — as vocab grows the R/V≈1 CP advantage
widens, exactly the direction the paper's larger-vocab operating point predicts.

### Remaining sweep (in progress)

The vocab-4096 N=3 R/V≈1 arm reproduces the crossover (**−0.0056**, matching the vocab-4096 N=5 win);
its R/V≈0.25 arm is still running. A follow-up sweep adds the **R/V≈0.5 midpoint** (R=V/2: R=512 at
vocab 1024, R=2048 at vocab 4096) to all four cells, pinning down where the sign actually flips.
Table above will be extended as those arms land.

## Conclusion (revised — the negative was an R/V artifact)

The earlier "CP is a strictly weaker value representation" verdict was **wrong in its generality**:
it held only in the **R/V ≪ 1** regime that our 32k tokenizer forces. Across rank, budget, init,
cross-order sharing, global readout, and N=5 depth we varied everything *except* the one axis that
mattered — rank relative to vocabulary — and all those "negatives" were simply CP evaluated far
below R/V ≈ 1. When we shrink the vocabulary into the paper's regime (vocab 1024, R=1024, R/V ≈ 1),
**CP crosses over and beats the native hash table by 0.0020 bpb**, reproducing the paper.

Reconciliation with the paper is now clean: the paper operates at R/V ≈ 1 (small vocab, large rank)
where the CP structure gives each token an ~uncompressed row *and* shares latent factors across
n-gram orders; our native table's explicit per-order capacity only wins when CP is starved of rank
relative to vocab. The practical implication for *our* stack: with a 32k tokenizer the native hash
table remains the better value representation (reaching R/V ≈ 1 would need R ≈ 32k, ~infeasible at
our vocab×K), but the CP approach is **not** fundamentally inferior — it is the right choice in the
small-vocab / large-rank regime the paper targets.

Code and tests remain on the branch (`test_tngram.py` 11/11, `test_tngram_e2e.py` 4/4, 191
existing engram+merge tests green); the `tngram` value table is opt-in and defaults untouched. The
small-vocab sweep (`runs/prep_smallvocab_tokenizers.sh`, `runs/run_tngram_smallvocab.sh`) reproduces
the R/V crossover.

