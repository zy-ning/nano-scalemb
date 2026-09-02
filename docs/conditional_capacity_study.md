# Conditional capacity at d20: MoE, Mobius, and continuous Engram

**Status:** 88 runs, 61 distinct configs, rounds 1-18 complete (round 15 stopped at
3 of 5 arms by an external SIGTERM; its null was already decisive and the remaining
arms were deliberately not re-run).
**Setup, identical for every arm:** d20 (`model_dim` 1280, 20 layers, 10 heads,
`head_dim` 128), `seq_len` 2048, `window_pattern` SSSL, mHC 4 streams,
batch 524,288 tokens, **3814 steps ≈ 2.0B tokens**, 4× GB200, ClimbMix,
`--target-param-data-ratio=-1` (fixed budget, not compute-optimal).
Metric is minimum validation bpb, evaluated every 500 steps on 10.5M tokens.

Four questions, in sequence:

1. Does extra **conditional capacity** help, does the *mechanism* matter, and does
   each layer need its own copy of it? (§1)
2. Should the memory be addressed by **token identity** or by **what the model
   currently represents**? (§2–§4)
3. Does a memory row need to be a **function of the hidden state** rather than a
   constant? (§8 — asked via a refuted hypothesis, answered no)
4. Do the two mechanisms **compose**, or are they two ways of buying the same
   thing? (§9.2f — they compose, and it is the largest iso-parameter effect measured;
   §9.2g then found no further gain from either obvious follow-up, and §9.2h walked
   the table/expert split — the optimum is expert-heavier than the one combo_iso
   first tried, but the table stays necessary)

**Start at §9** for what the whole thing established; §8.1 is kept but refuted.

---

## 0. Read this first: what the numbers can and cannot support

- **There is no single error bar. It is per arm, and it spans 45×.** All measured
  by re-running with a different `--engram-seed` (which permutes hash multipliers,
  and for contextual arms redraws the LSH projections too):

  | arm | n | sd | range | class |
  |---|---|---|---|---|
  | `shared_rank1` | 2 | — | **3.8e-5** | token, shared, rank-1 |
  | *same config re-run* (CUDA nondeterminism only) | 2 | — | 4.3e-5 | floor |
  | `lsh` | 2 | 2.7e-4 | 3.8e-4 | contextual, rank-0 |
  | **`engram_shared`** | **3** | **3.5e-4** | **6.7e-4** | token, shared |
  | `hybrid_frac75` | 2 | — | 8.8e-4 | token/hidden mix |
  | `tokens_rank1` | 2 | — | 1.0e-3 | token, rank-1 |
  | `lsh_rank1` | 2 | — | 1.7e-3 | contextual, rank-1 |

  Two patterns, both measured: **rank-1 values inflate variance ~5×** on either
  address type, and **cross-layer sharing collapses it** — `shared_rank1` sits at
  the CUDA floor, 27× tighter than the per-layer arm with the same feature.
- **n=2 is not enough to set an error bar. This happened twice, and the second
  time it destroyed a headline result.**
  - `engram_shared`: sd 1.3e-4 at n=2 -> **3.5e-4** at n=3 (2.6x).
  - `count_gate`: sd 8.6e-5 at n=2 -> **7.5e-4** at n=3 (**8.8x**). At n=2 the arm
    read -0.00118 vs `engram_shared` at 3.7x with *complete separation*; the third
    seed took it to -0.00075 at **1.57x with the groups overlapping** (§9.2).

  This is expected statistically — the sampling distribution of a 2-point range is
  very wide — but it means every "2x the bar" judgement resting on an n=2 bar is
  softer than it looks, and a suspiciously *tight* n=2 pair is the most dangerous
  case of all. Prefer n>=3 before quoting a bar; quote sd, not range.
- **Anything added to the read path inflates seed variance.** Measured: rank-1
  values ~5x on either address type; the count gate 2.2x on the shared table and
  **15x** on the per-layer one (1.21e-3 vs 8.12e-5). So a new mechanism must be
  judged against its own inflated bar, not the baseline's.
- **...but a better-conditioned parameterisation can buy the variance back.**
  Splitting the count gate's single conflated input into two linear terms
  (`--engram-count-gate-decouple`) cut its sd from 7.50e-4 to **4.89e-4 at n=5**
  (1.53x), which is what turned a 1.57x non-result into a resolved 3.03x one
  (§9.2c-d). Its n=3 -> n=5 sd growth was **1.20x**, the only error bar in this
  study stable enough to quote with confidence. Variance is not purely a property
  of "how much machinery"; it is also a property of how the machinery is
  parameterised -- though the effect is base-specific (1.39x per-layer, where the
  arm resolves by a larger mean instead, §9.2d).
- **`--device-batch-size` is NOT a free knob.** Measured directly at fixed rows,
  parameters and token budget: dbs=4 beats dbs=8 by **0.00184** (§9.2c). The
  bos-bestfit loader packs documents per batch, so dbs changes the token stream.
  Never compare across it (§6).
- **~~Cross-layer sharing collapses seed variance 27x.~~ RETRACTED, and reversed.**
  That rested on two n=2 estimates (`shared_rank1` 3.8e-5 vs `tokens_rank1`
  1.04e-3). With n=3 on both *plain* arms the direction flips: shared sd 3.45e-4
  vs per-layer 8.12e-5, i.e. **sharing is 4.3x NOISIER**. Sharing's mean effect is
  untouched (-0.0037 to -0.0039, three framings, 9-12x); only the "and it is more
  reproducible" half dies.
- **n=2 sd estimates are unreliable by up to 13x, usually low.** Four measured
  cases: `engram_shared` 1.31e-4 -> 3.45e-4 (2.6x up), `count_gate` 8.56e-5 ->
  7.50e-4 (**8.8x up**), per-layer base 6.36e-6 -> 8.12e-5 (**12.8x up**), and
  `cg_perlayer` 1.71e-3 -> 1.21e-3 (1.4x *down*). So it is not a one-sided bias --
  it is that a 2-point spread carries almost no information about sd, and a
  suspiciously *tight* pair is the most dangerous case (it is what turned the
  count gate into a false headline).
- **Use the arm's own class bar, not a global one.** Applying a bar measured on a
  different arm class is the single error that produced every retraction here (§5.1
  and §8.4) — five of them, including one where the wrong bar turned a 1.1× result
  into a claimed 8.7×.
- **Most arms are n=1**, so treat any gap under ~2× the relevant bar as
  unresolved. With the corrected bars, a *new* n=1 arm against the n=3
  `engram_shared` baseline has a combined SE of 4.0e-4 and can only resolve
  ~0.0008; at n=2, ~0.0006.
- **MoE arms cannot have an error bar at all.** There is no `--moe-seed` and no
  global `manual_seed` in `base_train`, so re-running `moe`/`mobius` samples only
  CUDA nondeterminism (4.3e-5), not a structurally different draw. Every
  MoE-vs-Engram comparison here is therefore one-sided: the Engram side has a
  measured bar, the MoE side has none. Giving the router init a seed knob is
  unbuilt work.
- **The combo arms inherit that, so their bars are LOWER BOUNDS.** `--engram-seed`
  permutes the hash multipliers but does not touch the router, so `combo_iso`'s three
  seeds (sd 7.92e-4) share one router draw and its 4.66× is optimistic (§9.2f). The
  robust half of that result is **complete separation**: the worst combo seed beats
  the best of `cg_dec`'s five. This is now the top item in §9.3.
- **This repriced several earlier conclusions.** The first 21 arms were run
  against the 4.3e-5 figure, and mechanistic stories were built on differences of
  1–8e-4 that one extra seed dissolved. See §5.1 for what survived and what was
  retracted. The methodological lesson: measure the error bar before, not after,
  21 arms — and measure it per arm class.
- **2.0B tokens is not convergence.** Compute-optimal for d20 is ~9.5B
  (~13h/arm, ~65h for five arms). Every arm was still improving at cutoff. Read
  these as a mid-training slice; orderings could change with budget.
- **Matched on *active* params, not total,** except where noted. Active is the
  read view: conditional weight counts once per *reading* layer. All d20 arms sit
  at 918.3M active (Engram family) / 902.5–909.9M (MoE family) / 901.7M (dense).

---

## 1. Iso-parameter conditional-capacity sweep

`runs/run_d20_iso_sweep.sh` — matched on active params across all arms, and on
total across the four non-dense arms (1820.0M, to within +0.46%).

| arm | min bpb | vs dense | total | active | time | tok/s | peak |
|---|---|---|---|---|---|---|---|
| **engram_shared** | **0.763784** | **−0.0088** | 1828.0M | 918.3M | 128m | 258k | 77GB |
| mobius | 0.764727 | −0.0079 | 1820.0M | 909.9M | 202m | 166k | 96GB |
| moe | 0.766529 | −0.0061 | 1820.0M | 902.5M | 163m | 200k | 79GB |
| engram | 0.767555 | −0.0051 | 1828.4M | 918.3M | 127m | 262k | 79GB |
| dense | 0.772614 | — | 901.7M | 901.7M | 117m | 281k | 61GB |

**Findings.**

- All four conditional arms beat dense by 0.005–0.009, and land within 0.0038 of
  each other. **The spread between mechanisms is smaller than the gap to dense** —
  having the capacity matters more than which mechanism provides it.
- **Cross-layer sharing wins in both mechanisms**, at matched total *and* active:
  - `engram_shared` − `engram` = **−0.0038** (one memory table + one hash
    addressing scheme for all Engram layers, vs two independent tables)
  - `mobius` − `moe` = **−0.0018** (one bank of 640 experts shared by all 10 MoE
    layers, vs ten private banks of 64)
  Both held flat across the back half of training (engram at −0.0040 for four
  consecutive checkpoints; mobius at −0.0018 for six), so neither is a
  stopping-point artifact.
- **Practical pick: `engram_shared`** — best quality *and* near-dense speed
  (128m vs 117m, +9%). Mobius reaches similar quality for 202m (+72% over dense):
  a single 640-expert bank makes the grouped GEMM much less efficient than ten
  small ones, so sharing buys parameter reduction at a throughput cost.

**Caveat.** Iso-total forces Mobius off its natural operating point (640 experts
at sparsity-80, vs upstream Intern-S2-Mobius's 256 at sparsity-32), so this is
not a verdict on Mobius as designed.

---

## 2. Continuous Engram, round 1: replace the address source

`runs/run_d20_continuous_engram.sh`. The token Engram hashes raw token IDs, so it
can only key on *surface form*. These arms derive the address from the layer's
hidden state instead. `NgramHasher.hash()` already accepted precomputed codes in
place of token IDs, so every scheme reuses the n-gram mixing, prime tables,
`MultiHeadEmbedding`, gate and short-conv unchanged.

| arm | address source | learned? | differentiable? | min bpb | vs tokens |
|---|---|---|---|---|---|
| tokens | hash token IDs (paper) | no | — | **0.767512** | — |
| lsh | frozen random projection → sign bits | **no** | — | 0.771347 | +0.0038 |
| pkm | product keys, soft top-k read | yes | **fully** | 0.771378 | +0.0039 |
| pq | product-quantization codebooks | yes | via STE | 0.772703 | +0.0052 |
| lsh_n1 | LSH, **unigram** (no n-gram window) | no | — | 0.773514 | +0.0060 |
| *dense* | | | | *0.772614* | |

**Findings.**

- **Every contextual variant loses**, in a tight band (+0.0038 to +0.0052),
  consistent across all 8 checkpoints with no crossovers.
- **`pkm ≈ lsh` to 3e-5** — at the noise floor. Fully learned differentiable
  addressing with a soft 32-row read performs identically to frozen random
  addressing with a hard 1-row read. **Learning the addressing bought nothing.**
- **`pq ≈ dense`** — a ~926M-param memory contributed nothing measurable.
- **`lsh_n1` is worse than dense.** Removing the temporal window costs +0.0022 vs
  `lsh` and makes the memory a *net liability*.

`lsh` is the only clean single-variable comparison against `tokens`. `pq`
additionally changes learnability (and collapsed — see §3); `pkm` additionally
changes the read *and* was +8.9% over iso-total, so it had a parameter advantage
and still lost.

---

## 3. Diagnostics on the round-1 checkpoints

Measured on real validation batches, per Engram layer (L2, L6).

| arm | realized code entropy | distinct codes | row-uniqueness | hidden eff_dim (90% var) |
|---|---|---|---|---|
| tokens | 7.84 / 14.53 bits (54%) | 413 / 1024 | 0.862 | 157–205 / 1280 |
| lsh | 9.21–9.81 / 15 bits (61–65%) | 716–932 / 1024 | **0.994** | 232–239 / 1280 |
| pq | 2.44 / 12 bits (20%) | **38** / 1024 | 0.184 | **1** / 1280 |

**PQ collapsed the representation, and it was self-inflicted.** A single direction
held 90% of variance at the Engram layers, giving 38 effective addresses for a
926M-param table. Cause: the VQ **commitment loss pulls the hidden state toward
its centroid**, and it was applied to a live residual stream. `codebook_usage`
read 0.97 and was misleading — centroids retain EMA mass from early training
while the live states collapse onto a handful. The commitment loss falling to
1e-4 was the *pathological* reading, not the benign one.

**LSH is healthy on every axis and still loses.** It spreads *better* than tokens
(row-uniqueness 0.994 vs 0.862) and uses more of its nominal width. So address
capacity/entropy is **not** the binding constraint.

**Non-uniform VQ allocation would not help** (tested, since more keys in
high-variance subspaces is the textbook fix):

| n_subspaces | variance ratio max/min | optimal extra bits (spread) | max gain |
|---|---|---|---|
| 2 | 1.01× | 0.00 | 0.00 dB |
| 16 | 1.07× | 0.05 | 0.00 dB |
| 64 | 1.19× | 0.13 | 0.01 dB |

Subspace variances are already balanced to within 1–20%, because per-dim variance
is only 2.4× max/median after RMSNorm and pooling 320–640 dims averages away even
that. The anisotropy (eff_dim 232/1280) is **cross-dimensional correlation**, not
axis-aligned variance imbalance — invisible to per-subspace code budgets, and
addressable only by a rotation (OPQ), not by more keys.

---

## 4. Round 2–3: targeted fixes, and one that worked

`runs/run_d20_continuous_v2.sh`, `runs/run_d20_continuous_v3.sh`.

| arm | change | min bpb | vs tokens |
|---|---|---|---|
| **hybrid_frac75** | **75% token heads, 25% hidden-state** | **0.765593** | **−0.00192** |
| hybrid (frac50) | half the heads token, half hidden-state | 0.766783 | −0.00073 |
| hybrid_shared | hybrid + cross-layer sharing | 0.767041 | −0.00047 |
| hybrid_frac25 | 25% token heads, 75% hidden-state | 0.768086 | +0.00057 |
| lsh_perhead | one latent per head (~16× address bits) | 0.772223 | +0.0047 |
| lsh_balanced | EMA per-bit threshold (entropy 8.55→9.85 bits) | 0.772720 | +0.0052 |
| pq_detached | stop-grad encoder + per-head codes + K=256 | 0.775543 | +0.0080 |

### 4.1 Hybrid beats token addressing

**0.766783 vs 0.767512, −0.00073**, ahead at all 8 checkpoints, ~15× the noise
floor. The only configuration in the study where contextual addressing helps.

Contextual addressing as a **replacement** lost every way it was tried (frozen
random, learned quantized, learned differentiable — all ≈ +0.004). As an
**addition** alongside token n-grams, it wins. Consistent with the token n-gram
*injecting* information the residual stream has discarded (exact recent token
identity), which contextual addressing cannot substitute for — only supplement.

### 4.1b The mix has a genuine interior optimum

`hybrid_token_head_frac=0.5` was an untuned guess. Sweeping it gives a clean
U-shaped curve with a minimum at **75% token / 25% hidden-state**:

| token heads | arm | min bpb | delta |
|---|---|---|---|
| 0% | lsh | 0.771347 | — |
| 25% | hybrid_frac25 | 0.768086 | −0.00326 |
| 50% | hybrid (frac50) | 0.766783 | −0.00130 |
| **75%** | **hybrid_frac75** | **0.765593** | **−0.00119** |
| 100% | tokens | 0.767512 | **+0.00192** |

`frac` only changes which head reads which code source, so all four hybrid points
share byte-identical parameter geometry (1841.8M / 918.3M) — an unusually clean
single-variable sweep.

**RETRACTED IN PART.** This was originally reported as a clean U-shape with a
genuine interior optimum at 75/25, and called the study's strongest result. The
`hybrid_frac75` seed replicate (§5.1) then measured the error bar on these arms at
**8.8e-4**, which is larger than most of the curve's internal structure:

| comparison | delta | vs 8.8e-4 | status |
|---|---|---|---|
| frac75 vs lsh (0%) | −0.0053 | 6.0× | **holds** |
| frac75 vs tokens (100%) | −0.0015 | 1.7× | marginal |
| frac75 vs frac50 | −0.0008 | 0.8× | **not resolved** |
| frac75 vs frac875 | +0.0003 | 0.3× | **not resolved** |

What survives: **a mix beats pure contextual addressing by a wide margin**, and
pure contextual is bad. Whether any mix beats pure *token* addressing is
suggestive at 1.7× but not established, and the location of the optimum within
25–87.5% is not resolved at all. The pre-registered ordering
(frac75 > frac50 > frac25) was directionally correct but two of its three gaps sit
inside the error bar, so it should not be read as confirming a mechanism.

### 4.2 The two wins do not compose

| | vs tokens |
|---|---|
| sharing alone (`engram_shared`) | −0.0037 |
| hybrid alone (`hybrid`) | −0.00073 |
| both (`hybrid_shared`) | **−0.00047** |

Independent effects would give ≈ −0.0044. Observed −0.00047 — *less than either
alone*, and +0.0033 above `engram_shared` at all 8 checkpoints.

**Mechanism.** The shared table only pays off when different layers address it
*identically*. That holds for token n-grams (same codes at every depth) and fails
for hidden states (different vectors by construction). Sharing the projection —
which round 3 does — makes the addressing *function* shared but not the
*address*, because the inputs still differ by depth. So the contextual heads write
layer-specific content into the shared table and dilute exactly the property that
made sharing work.

### 4.3 Address granularity has an interior optimum

Every intervention that made the address *more* informative made the model
*worse*, and the one that made it *less* informative was worse too:

| change to address information | arm | min bpb |
|---|---|---|
| much **less** (unigram) | lsh_n1 | 0.773514 |
| **intermediate, Zipfian** | **tokens** | **0.767512** |
| more (15 bits) | lsh | 0.771347 |
| more (+59% distinct codes) | lsh_balanced | 0.772720 |
| ~16× more (per-head latents) | lsh_perhead | 0.772223 |
| ~16× more + detached | pq_detached | 0.775543 |

Proposed mechanism was **write reuse**: Zipfian token n-grams hit the same rows
thousands of times so those slots accumulate a strong signal, while high-entropy
codes spread writes thin.

**This account is FALSIFIED.** It predicted that *coarser* addressing would help.
Round 4 tested code widths below the token alphabet (23686) for the first time —
every prior arm had only made addressing finer, so the account had never been at
risk:

| bits | codes/pos | min bpb | vs b15 |
|---|---|---|---|
| 8 | 256 | 0.772758 | +0.0014 |
| 12 | 4096 | 0.773722 | +0.0024 |
| 15 | 32768 | **0.771347** | — |

Both coarser widths are worse, and b12 is worse than b8 — **not even monotone**.
The three points scatter within 0.0024 of each other, all near dense. Code width
does not order the LSH arms, so granularity cannot be the causal variable.

What the account was actually tracking is *distance from the token n-gram*. All
hidden-state addressing schemes are roughly interchangeable-and-mediocre
regardless of quantizer or width (b8, b12, b15, pkm, pq, balanced, perhead all
land within 0.0024, all near dense), which points at **information content**: the
token n-gram supplies exact recent identity that a hidden-state code cannot
substitute for at any width.

Note two of this account's predictions were nonetheless correct in advance
(`lsh_perhead` worse than `lsh`; the frac ordering). Correct predictions from a
wrong mechanism — the pattern was real, the explanation was not.

Two of these were **pre-registered predictions** (stated before the run):
`lsh_perhead` would land worse than `lsh` (it did, +0.00088), and `frac75` should
beat `frac50` should beat `frac25` (pending).

**`pq_detached` did not rescue PQ** — 0.775543, worst arm in the study, 0.0029
*below dense*. So the collapse was real but not what held PQ back. This arm
confounds three changes (detach + per-head codes + K=256 vs 2×64), so the damage
cannot be attributed to any one; that was a design error — the arm's stated job
was to isolate a cause and it was built to maximize the chance of a working arm
instead.

---

## 5. Summary of the whole study

88 runs, 61 distinct configs. Means where replicated; **bold** = n>=2;
*italic* = NOT iso-parameter (the §9.2b storage sweep, where more total params were
the independent variable -- those rows measured a slope, they did not win; plus
`combo_full`, which runs at +50% total).

| # | arm | min bpb | family |
|---|---|---|---|
| 1 | *combo_full* | *0.757196* (n=2, sd 5.1e-4) | *Engram+MoE, slot30 + 640 experts, 2746.3M — NOT iso* |
| 2 | **split_exp** | **0.760038** (n=3, sd 2.4e-4) | **combo_iso at 26/74** table/experts (slot9 + 468), iso total |
| 3 | **combo_iso** | **0.760625** (n=3, sd 7.9e-4) | **cg_dec table + routed experts**, 44/56, iso total (+0.50% active) |
| 4 | combo_mobius | **0.761117** (n=3, sd 1.04e-3) | combo_iso + private always-on expert (iso; top-7 of 347) |
| 5 | split_tbl | **0.761884** (n=3, sd 1.4e-4) | combo_iso at 73/27 table/experts (slot25 + 172), iso total |
| 6 | combo_nogate | **0.761898** (n=3, sd 7.05e-4) | combo_iso **minus** the decoupled count gate (iso) |
| 7 | cg_dec_four | **0.762787** (n=2) | shared + decoupled + Fourier gate |
| 8 | **cg_dec** | **0.762987** (n=5, sd 4.9e-4) | shared + **decoupled** count gate |
| 9 | **count_gate** | **0.763136** (n=3, sd 7.5e-4) | shared + learned count backoff |
| 10 | *shared_slot60* | *0.763329* | *shared, 2x rows, 2738M — NOT iso* |
| 11 | share_tbl | **0.763604** (n=2, sd 7.9e-5) | shared table, **per-layer hash** |
| 12 | **engram_shared** | **0.763884** (n=3: .763784/.763599/.764268, sd 3.5e-4) | cross-layer shared hash memory |
| 13 | *shared_slot15* | *0.764396* | *shared, 0.5x rows, 1373M — NOT iso* |
| 14 | cg_n5 | 0.764698 | count_gate + n-gram orders 2-5 (32 heads) |
| 15 | mobius | 0.764727 (n=1; no seed knob exists) | cross-layer shared MoE |
| 16 | hybrid_shared_f75 | 0.765106 | shared + 75/25 addressing |
| 17 | kv16 | 0.765461 | shared + 16-dim per-row key (gate input 256) |
| 18 | hybrid_frac875 | 0.765728 | 87.5/12.5 token + hidden |
| 19 | *perlayer_slot30_dbs4* | *0.765760* | *per-layer 2x rows at dbs=4 — NOT comparable* |
| 20 | cg_dec_perlayer | **0.765924** (n=3, sd 8.7e-4) | per-layer + decoupled count gate |
| 21 | cg_fourier | **0.766023** (n=2) | per-layer + Fourier count gate |
| 22 | hybrid_frac75 | **0.766035** (n=2: .765593/.766477) | 75/25 token + hidden |
| 23 | whiten_p03 | 0.766068 | shared + FIXED 1/freq^0.3 whitening |
| 24 | kv80 | 0.766208 | shared + 80-dim per-row key (iso, rows halved) |
| 25 | tokens_rank1 | **0.766381** (n=2, sd 7.3e-4) | token address, **rank-1 values** |
| 26 | shared_rank1 | **0.766457** (n=2, sd 2.7e-5) | shared + **rank-1 values** |
| 27 | moe | 0.766529 (n=1; no seed knob) | per-layer MoE |
| 28 | hybrid_frac50 | 0.766783 | 50/50 token + hidden |
| 29 | cg_perlayer | **0.766809** (n=3, sd 1.21e-3) | per-layer + count backoff |
| 30 | share_hash | **0.766863** (n=2, sd 3.7e-5) | per-layer tables, **shared hash** |
| 31 | hybrid_shared (f50) | 0.767041 | shared + 50/50 |
| 32 | tokens / engram | 0.767512 / 0.767555 | per-layer hash memory |
| 33 | *perlayer_slot30* | *0.767602* | *per-layer, 2x rows, 2738M — NOT iso* |
| 34 | hybrid_frac25 | 0.768086 | 25/75 token + hidden |
| 35 | pkm | 0.771378 | hidden, learned soft top-k |
| 36 | lsh (b15) | **0.771535** (n=2: .771347/.771722) | hidden, frozen random, 32768 codes |
| 37 | lsh_r1_isorow | 0.771666 | hidden, rank-1, **1.9x params** |
| 38 | lsh_rank4 | 0.771883 | hidden, rank-4, 33k rows |
| 39 | lsh_perhead | 0.772223 | hidden, 16 latents |
| 40 | dense (baseline) | 0.772614 | no conditional capacity |
| 41 | pq | 0.772703 | hidden, learned VQ |
| 42 | lsh_balanced | 0.772720 | hidden, balanced bits |
| 43 | lsh_b8 | 0.772758 | hidden, 256 codes |
| 44 | lsh_n1 | 0.773514 | hidden, unigram |
| 45 | lsh_b12 | 0.773722 | hidden, 4096 codes |
| 46 | lsh_r0_slot4 | 0.774389 | hidden, **131k rows** (row-count control) |
| 47 | lsh_rank1 | **0.774396** (n=2, sd 1.2e-3) | hidden, rank-1, 131k rows |
| 48 | pq_detached | 0.775543 | hidden, VQ detached |

Excluded as uninterpretable: `shared_slot120` (0.761128) and `perlayer_slot60`
(0.763210), the two round-11 arms that ran at `device-batch-size=4` while the rest
ran at 8 (§9.2b, §6). They cannot be attributed to anything.

**How to read the table.**

- **Ranks 1-6 are the arms that combine both mechanisms** (hash memory + routed
  experts, §9.2f-h). `split_exp` at rank 2 is now the best *iso* arm — combo_iso
  walked to a 26/74 table/expert split — but only at 1.23× over combo_iso, i.e. it
  is **not distinguishable** from rank 3; the resolved result of the split sweep is
  that the *table-heavy* side (`split_tbl`, rank 5) is worse, at 2.71× (§9.2h). Rank
  1 runs at +50% total params and is not iso to anything. Ranks 4 and 6 are round-17
  follow-ups that both came back *negative* — `combo_mobius` is 0.65× from combo_iso
  (unresolved) and `combo_nogate` (the count gate removed) costs 0.00127 at 2.08×.
  **Ranks 2-3 are not distinguishable; ranks 5-6 are not distinguishable.**
- **Ranks 7-11 span 8.2e-4**, less than the hybrid error bar (8.8e-4), so that part
  of the table is an unordered set rather than a ranking.
- **Ranks 35-48 span 0.0042** against a contextual bar of 3.8e-4 to 1.7e-3 and
  contain `dense` at 40. Read those fourteen as one cluster — "hidden-state
  addressing, near dense" — not as an ordering (see §5.1).

(The rank references in this paragraph were stale from an earlier revision and are
now recomputed against the table above.)

### 5.1 Error bars and retractions

Two arms were replicated with `--engram-seed=1`, the first real error bars in the
study (everything before rested on one same-config re-run):

| arm | seed 0 | seed 1 | spread |
|---|---|---|---|
| engram_shared | 0.763784 | 0.763599 | 1.85e-4 |
| hybrid_frac75 | 0.765593 | 0.766477 | **8.84e-4** |

The 5× difference is structural: for a token-addressed arm the seed only permutes
hash multipliers, whereas for a contextual arm it also redraws the LSH
projections, i.e. resamples *what the memory is addressed by*.

**Survives repricing:**

| result | delta | vs bar | |
|---|---|---|---|
| any conditional arm vs dense | 0.005–0.009 | 25–50× | solid |
| pure contextual vs token addressing (lsh vs tokens) | 0.0038 | 20× | solid |
| engram_shared vs engram (sharing) | 0.0038 | 4.3× | solid |
| a token/hidden mix vs pure contextual | 0.0053 | 6.0× | solid |
| hybrid_shared_f75 vs engram_shared (non-composition) | 0.0013 | 7× | solid |
| mobius vs moe (sharing, MoE side) | 0.0018 | 2.0× | marginal |

**Retracted or downgraded:**

| claim as originally reported | delta | vs bar | now |
|---|---|---|---|
| "75/25 is the optimum of the frac curve" | 0.0008 | 0.8× | not resolved |
| "frac875 confirms 75% as the minimum" | 0.0003 | 0.3× | not resolved |
| "hybrid beats token addressing" | 0.0015 | 1.7× | marginal |
| "engram_shared beats mobius" | 0.0010 | 1.2× | not resolved |
| ordering among near-dense contextual arms (lsh_b8/pq/lsh_balanced/dense, all within 2e-4) | ≤0.0002 | <1× | not meaningful |
| "function-valued rows should help contextual addressing" (§8.1, pre-registered < −0.003) | +0.0001 | 0.09× | **refuted**, §8.4 |
| "`tokens_rank1` gains at 8.7× the bar" | −0.0011 | 1.1× | wrong bar used |
| "rank helps token addressing, hurts contextual (2.1× asymmetry)" | — | — | **retracted**, §8.4 |
| "count_gate wins by 0.00118 at 3.7×, complete separation, new best" | −0.00076 pooled | 1.94× | **downgraded**, §9.2 — third seeds on both bases; unresolved |
| "cross-layer sharing collapses seed variance 27×" | — | — | **reversed**, §0 — at n=3 sharing is 4.3× *noisier* |
| "row capacity is worth 0.00195 per doubling (three confirmations)" | — | — | **overgeneralized**, §9.1 — one contextual measurement reused three times; token arms give −0.00053 (shared) and ~0 (per-layer) |
| "row capacity explains about half of the sharing win" | — | — | **retracted**, §9.2b — at matched rows/read sharing still wins 0.00372 at 9.3×, so none of it is capacity |
| "sharing works because one row is trained by gradients from every reading layer" | −0.00028 | 1.35× | **retracted**, §9.2c — `share_tbl` breaks the row correspondence and loses nothing |
| "the model ignores the decoupled freq/collision channels" | — | — | **retracted**, §9.2c — read off the Fourier arms, where the basis absorbs the freq channel; the plain arms use both with 16/16 sign agreement |
| the round-11 capacity slope above 1.42M rows (−0.00220/doubling) | −0.00036 corrected | — | **corrected**, §9.2c — the arm ran at dbs=4, worth −0.00184 by itself |
| "decoupling makes the count gate 1.8× tighter" | 1.53× / 1.39× | — | **corrected**, §9.2d |
| "per-layer `cg_dec` sd should be ~4e-4 if decoupling is a general conditioning fix" | 8.72e-4 | — | **prediction failed**, §9.2d — it transfers, but by doubling the mean, not by tightening variance |
| "`combo_iso` router health confirmed: max_vio 0.054 → 0.028" | — | — | **corrected**, §9.2f — those are the *regret* values; `max_vio` runs the other way, 1.94 → 8.20. Entropy at 93.6% of ceiling still supports "no collapse", but load imbalance grows rather than shrinks |
| "if the private always-on expert soaks up load, `combo_mobius`'s max_vio should IMPROVE" | 9.26 vs 7.97 | — | **prediction failed**, §9.2g — it got worse; always-on capacity did not relieve the router |
| "`combo_nogate` above 0.7619 means the gate contributes MORE with the MoE present" | −0.00127 vs −0.00090 | 0.6× | **band mislabelled**, §9.2g — at this power "survives unchanged" and "contributes more" are not separable; only the former is supported |

**Three conclusions.**

1. **Cross-layer sharing is the largest reproducible effect** (−0.0038 for Engram,
   −0.0018 for MoE), and it is nearly free in wall-clock for the Engram.
2. **Addressing by hidden state does not substitute for token identity.** Of 10
   *pure*-contextual arms, none beat token addressing and 7 are at or below dense —
   the memory becomes a net liability. Quantizer and code width barely matter (all
   within 0.0024, see 4.3). A token/hidden *mix* clearly beats pure contextual
   (6x the error bar) and may modestly beat pure token addressing (1.7x, not
   established); the best mixing ratio is not resolved.
3. **The two winning ideas conflict** rather than compose, because both trade
   against the same resource: how concentrated the writes into a memory row are.
   Read narrowly — this is about *sharing* vs *hybrid addressing*, both of which act
   on the same memory. It does **not** generalize to "nothing composes": the hash
   memory composes with routed experts at −0.00236, 4.66× (§9.2f), precisely because
   those two do *not* contend for the same resource.

**Done in round 4** — all four previously-listed items: coarse LSH falsified the
granularity account (4.3); `frac875` did not resolve the optimum;
`hybrid_shared_f75` improved on frac50 but stayed 0.0013 behind `engram_shared`;
the two seed replicates repriced the study (5.1).

**Open, in priority order** — *this list is the round-4 snapshot, kept for the
record. The current priorities are in §9.3.*

1. **More seeds, not more arms.** With n=2 on two arms and n=1 on twenty-one, the
   binding limit is statistical, not architectural. Ranks 1-5 are separated by less
   than the hybrid error bar. 3-4 seeds on `engram_shared`, `mobius` and one
   hybrid would settle the top of the table.
2. **Why is contextual addressing high-variance?** The 8.8e-4 spread from redrawing
   an LSH projection is a result in its own right and is unexplained. If a
   *learned* projection reduced it, that would be a genuine improvement even
   without a mean gain.
3. **Longer horizon.** Every arm was still improving at 2.0B tokens;
   compute-optimal is ~9.5B. Sharing effects could grow or vanish.
4. **d26** — configs solved (iso-total 3707.6M; moe E=64/dff=832, mobius E=832
   share=1, engram slot=33, engram_shared slot=66); Mobius OOMs at
   `device_batch_size=8` and needs 4. Untested.
5. **Round 5, built and ready** (§8.4): `--engram-value-rank` makes a memory row a
   function of the hidden state rather than a constant — the axis rounds 1-4 never
   varied, and on the §8.1 reading the one that mattered. Differential prediction
   pre-registered, 2 arms × 2 seeds, `runs/run_d20_value_rank.sh`. Also built and
   untested at scale: `--engram-readout-whiten` (§8.2). Still unbuilt: margin as a
   pre-training predictor, and an absolute capacity yardstick (§8.5).

---

## 6. Operational notes (things that cost time)

- **`/tmp` is periodically wiped.** The base dir lived at `/tmp/nsmb_out` holding
  symlinks to data + tokenizer and all checkpoints. It vanished mid-run, crashing
  two arms (SIGABRT / exit 1) and destroying **every checkpoint** in the study.
  Results survived only because logs are written into `runs/logs/` in the repo.
  Base dir is now `/mnt/3fs/.../nsmb_out`. *Diagnostics in §3 are not reproducible
  without re-running the arms.*
- **Never test liveness with `pgrep -f <plain name>`.** The checking shell's own
  command line contains the pattern, so it always matches. This produced three
  separate false positives and idled 4 GPUs for hours: (1) a chaining waiter that
  matched itself, (2) a monitor that matched the waiter, (3) a
  `pgrep -f scripts.base_train` liveness probe that matched its own shell.
  **The bracketed-pattern workaround also fails.** A fourth instance: the checking
  shell ran the probe through `zsh -c 'eval ...'`, so the literal string
  `base_tra[i]n` appeared in *its own* command line and `ps | grep` matched it
  again. There is no pattern that is safe, because the pattern is always in the
  checker's argv. **Use GPU memory as the liveness signal** (`nvidia-smi
  --query-gpu=memory.used`); 0 MiB on every device is the only trustworthy "not
  running". For chaining, poll for a **marker string in the predecessor's log**
  rather than any process state — `kill -0` succeeds on zombies and pids get
  reused, but a log line does neither. `runs/run_d20_value_rank_v2.sh` and
  `run_d20_kv_backoff.sh` do this (`WAIT_FOR_LOG`/`WAIT_FOR_MARKER`).
- **`kill -0 <pid>` succeeds on zombies.** A finished-but-unreaped driver kept a
  waiter looping forever. Check process state, not just PID existence.
- **Never vary `--device-batch-size` inside a comparison.** Three script headers
  in this repo called dbs=4 vs dbs=8 "numerically identical, just slower" because
  the total batch is held at 524288. That is wrong: the loader is
  `tokenizing_distributed_data_loader_bos_bestfit`, which best-fit-packs documents
  into sequences, so the number of sequences in flight changes the *packing* and
  therefore the token stream -- not just the accumulation schedule. Round 11 mixed
  dbs=8 and dbs=4 in one capacity sweep and both dbs=4 points came in 0.0022-0.0044
  better than the fit predicted, which cannot be separated from capacity after the
  fact (§9.2). Two of five arms in that round are unusable as a result.
- Engram runs need `TORCHDYNAMO_CACHE_SIZE_LIMIT=64` (many distinct param shapes
  recompile the fused AdamW step).
- The system `torchrun` resolves to a python whose torch import fails on
  `libucc`; use the venv's `torchrun`.

## 7. Reproduce

```bash
export NANOCHAT_BASE_DIR=/mnt/3fs/dots-pretrain/ningzhiyuan/local_exps/nsmb_out
bash runs/run_d20_iso_sweep.sh            # §1, 5 arms, ~14h
bash runs/run_d20_continuous_engram.sh    # §2, 5 arms, ~11h
bash runs/run_d20_continuous_v2.sh        # §4, 4 arms, ~9h
bash runs/run_d20_continuous_v3.sh        # §4.2, 3 arms, ~6.5h
bash runs/run_d20_continuous_v4.sh        # §4.3 + 5.1, 6 arms, ~13h
bash runs/run_d20_value_rank.sh           # §8.4 round 5, 7 arms, ~17h
bash runs/run_d20_value_rank_v2.sh        # §8.4 round 6, 4 arms, ~9h
bash runs/run_d20_kv_backoff.sh           # §9.2 round 7, 4 arms, ~9h
bash runs/run_d20_count_gate_followups.sh # §9.2 round 8, 3 arms, ~7h
bash runs/run_d20_count_gate_generalize.sh # §9.2 round 9, 1 arm, ~2.2h
bash runs/run_d20_count_gate_fourier.sh   # §9.2b round 10, 6 arms, ~14h
bash runs/run_d20_row_capacity.sh         # §9.2b round 11, 5 arms, ~13h (2 unusable)
bash runs/run_d20_count_gate_decouple.sh  # §9.2b round 12, 5 arms, ~12h
python -m pytest                          # 283 passed, 10 skipped
```

Rounds 6 and 7 chain onto their predecessor by polling for a marker string in its
log (see §6):

```bash
WAIT_FOR_LOG=runs/logs/<prev>/driver.log WAIT_FOR_MARKER="Round 6 complete" \
  bash runs/run_d20_kv_backoff.sh
```

Round 5 (§8.4) is built but not yet run:

```bash
bash runs/run_d20_value_rank.sh           # §8.4, 2 arms x 2 seeds, ~9h
RUN_SHARED=1 bash runs/run_d20_value_rank.sh   # + the arm that could take rank 1
```

The readout whitening from §8.2 is a flag on any Engram arm, untested at scale:

```bash
# add to the engram_shared arm of run_d20_iso_sweep.sh
--engram-readout-whiten                    # 1/frequency, downweight-only
--engram-readout-whiten-power=0.5          # softer, for a Zipfian alphabet
```

---

## 8. Reinterpretation: the memory is a Hebbian kernel, and margin is the axis

Everything above treats "what the memory is addressed by" as the design variable.
[*MLPs are Hebbians* (Garcia et al. 2026, arXiv:2607.10034)](https://arxiv.org/abs/2607.10034)
suggests that was the wrong axis, and — more usefully — that the study already
measured the right one without noticing.

**The frame.** Their Theorem 3.1 is an identity: any MLP `Bφ(x)` *is* a Hebbian
kernel memory `H(z) = (1/F)Σᵢ yᵢ K(xᵢ,z)` under the whitened kernel
`K(x,z) = φ(x)ᵀΣ̂⁻¹φ(z)`. So a hash memory and an MLP are the same object with
different kernels, and they can be compared on one scale. That scale is the
**decoding margin**, which decomposes as

```
γᵢⱼ = ⟨v_f(i)−vⱼ, v_f(i)⟩·K(kᵢ,kᵢ)        signal
    + Σ_{t≠i} ⟨v_f(i)−vⱼ, v_f(t)⟩·K(k_t,kᵢ)  cross-talk
```

Their central empirical point (their Fig 2a) is that **storage is not usability**:
an inserted memory stores its fact set as soon as `γ_min > 0`, but end-to-end
accuracy lags until the margin is bounded *away* from zero — 11,708 vs 53,396
parameters for the same facts. The reason is that the query arriving at the memory
is perturbed, quantified as an attention noise ceiling
`ε_attn = maxᵢ max_q ‖q − kᵢ‖₂`, with usability requiring
`ε_attn ≲ c₀/(L·√(F log F/d))` (their Thm 5.2).

### 8.1 What this says about our arms — **REFUTED, see §8.4**

> This section's central claim — that the constant-vs-function-valued distinction
> is what separates the families — was tested directly in rounds 5-6 and is
> **wrong**. Making rows functions of `h` is neutral at fixed row count (two
> independent tests, 0.02× and 0.09× of the bar) and *harmful* on the best arm
> (+0.00257 at 12.8×). The section is kept because the observation it starts from
> is still correct and still unexplained: the two contextual-capacity families
> differ by 0.0066. The *explanation* offered here is not the reason. §9 gives the
> account that survives.

Token addressing has `ε_attn = 0`: the address is a token id, delivered exactly,
with nothing in the path to perturb it. Hidden-state addressing has `ε_attn > 0` by
construction. Under this frame the arms sort not by address source but by whether
the retrieved value is a **constant** or a **function of the input**:

| family | what a read returns | query noise | best bpb |
|---|---|---|---|
| MoE / Mobius | `W_out·act(W_in·h)`, a function of `h` | > 0 | **0.764727** |
| token Engram | a constant row, but the address is exact | 0 | **0.763599** |
| all 10 pure contextual Engram arms | a constant row | > 0 | 0.771347 |
| dense | — | — | 0.772614 |

**The gap between the two contextual families is 0.0066 — 7.5× the contextual
error bar, and larger than every effect §4 and §5.1 argue over.** It was in the
data from §1 onward, filed as "MoE vs Engram" (two mechanisms) rather than as
"constant-valued vs function-valued memory over the same continuous key space"
(one mechanism, two settings).

This also accounts for two things the earlier explanations did not:

- **`lsh` had better address spread than tokens (0.994 vs 0.862, §3) and still
  lost by 0.0038.** Spread is a write-distribution statistic; it is not in the
  margin decomposition at all. A well-spread but noise-brittle address is the
  worst case, not the best one.
- **`pkm` (0.771378) tied `lsh` (0.771347) to 3e-5.** PKM is the only contextual
  arm with a *soft* read — learned keys, softmax top-k, fully differentiable — so
  softening the address changed nothing measurable. That rules out "hard
  quantization is the problem", which would otherwise be the obvious reading, and
  leaves the constant-vs-function distinction, which no arm has tested. Both are
  constant-valued.

A constant-per-row memory over a continuous key space discards all within-cell
variation, and the cell is the only resolution available. For token n-grams a
constant is the right thing to store — the key genuinely is discrete. All 10
contextual arms built fact memories over a space that needed a function memory.

### 8.2 What was implemented: diagonal readout whitening

Their Lemma B.3 shows that among PSD preconditioners with fixed average
self-kernel, whitening by the empirical feature covariance minimizes an upper
bound on the *key crowding* statistic `E_K = maxᵢ‖Kᵢ‖²` that sets the cross-talk
scale — and it is what turns their construction from near-optimal into the first
closed-form one to actually reach `W = Θ(F log F)`. For a one-hot feature map like
the Engram's, `Σ̂` is diagonal with the per-row hit frequencies on it, so the whole
correction is their Algorithm 1 `diag` mode: divide row *r*'s contribution by its
own frequency.

`--engram-readout-whiten` does this with an EMA of per-row hit rates, normalized
per head so 1.0 means "addressed as often as a uniform row of this head" (that
per-head mean is exactly invariant, which the tests pin). Two deliberate
divergences from the paper, both because their fact sets hit each key exactly once
while token n-grams are Zipfian:

- `--engram-readout-whiten-max` defaults to 1.0, i.e. **downweight-only**. A row
  that is never read cannot crowd anything, so upweighting cold rows buys no
  margin while amplifying their random init straight into the read. The faithful
  uncapped form is reachable by raising the cap.
- `--engram-readout-whiten-power` softens the correction; full `1/frequency` on a
  power law effectively deletes the most frequent n-grams from the read.

This is also the *correct* version of an intuition that already failed here once:
§4.3's write-reuse account, falsified by `lsh_b8`/`lsh_b12`. Hot rows dominating
the summed read is `E_K` inflation, and frequency normalization is the closed-form
fix for it — a different claim from "coarser addresses are better", which the data
refuted.

**Expected weakly, and here is why.** The memory rows train under AdamW, which
normalizes gradient scale, so a row can partly re-absorb its weight by growing.
In the paper the readout is solved in closed form and cannot adapt. The
intervention bites hardest early, and the hit rates keep moving, so it will not
cancel exactly — but this is a softer lever here than there. **Untested at scale:**
30-step smoke runs confirm correctness and zero throughput cost (76.1k vs 75.0k
tok/s, and zero `torch.compile` graph breaks), but whitening ramps in over
~1/(1−decay) = 100 steps, so a 30-step bpb comparison measures nothing.

### 8.3 Limits of the transfer

Single layer, synthetic fixed fact sets, *constructed* rather than trained
weights, and the objective is memorizing a known map, not bpb. Their own
GD-trained MLPs beat the best construction by 6–10× parameters, i.e. gradient
descent already finds a better kernel than the theory builds. Every arm here is
GD-trained, so we are in the regime where the construction is not the binding
constraint. What transfers is the *diagnostic* framing (margin under query noise,
and the four geometry penalties `P_key`/`P_val`/`P_align`/`S_sig`), not the
construction. Porting the closed-form construction itself should not be expected
to lower bpb.

### 8.4 Rounds 5-6: the prediction failed, and the axis is dead

11 arms, ~24h. `runs/run_d20_value_rank.sh` and `runs/run_d20_value_rank_v2.sh`.
`--engram-value-rank=k` makes row *r* a rank-k map
`W_out[r] @ act(W_in[r] @ q)`, `q = query_proj(rms_norm(h))`, instead of a
constant. rank 0 is the Engram, high rank with few rows is an MoE expert bank.

**Pre-registered:** `lsh_rank1 − lsh < −0.003`, and `tokens_rank1 − tokens ≈ 0`
or positive, with "if both gain equally, §8.1 is wrong" as the falsifier.

| arm | rows/head | bpb (seeds) | mean | vs its rank-0 baseline |
|---|---|---|---|---|
| `lsh` | 360k | 0.771347 / 0.771722 | 0.771535 | — |
| `lsh_r0_slot4` | 131k | 0.774389 | — | +0.00285 (row loss alone) |
| `lsh_rank1` | 131k | 0.775254 / 0.773538 | 0.774396 | +0.00286 |
| `lsh_r1_isorow` | 360k | 0.771666 | — | **+0.00013** (1.9× the params) |
| `lsh_rank4` | 33k | 0.771883 | — | +0.00035 |
| `tokens` | 355k | 0.767512 | — | — |
| `tokens_rank1` | 142k | 0.765862 / 0.766900 | 0.766381 | −0.00113 (1.1× its bar) |
| `engram_shared` | 710k | 0.763784 / 0.763599 / 0.764268 | 0.763884 | — |
| `shared_rank1` | 284k | 0.766476 / 0.766438 | 0.766457 | **+0.00257 (12.8×)** |

**The 2×2 that settles it.** Holding rows fixed and varying only rank:

|  | rows 131k | rows 360k |
|---|---|---|
| rank 0 | 0.774389 | 0.771535 |
| rank 1 | 0.774396 | 0.771666 |

```
rank effect @ 131k rows:  +0.000007   (0.02× bar)
rank effect @ 360k rows:  +0.000131   (0.09× bar)
row  effect @ rank 0:     +0.002854   (12×   bar)   <- the real effect
row  effect @ rank 1:     +0.002730
```

Additive, with a **zero rank effect measured twice**. `lsh_r1_isorow` carried
**1.9× the parameters** of `lsh` with identical addressing — the most favourable
test constructible — and returned +0.00013. The falsifier stated in advance was
"if it fails to beat 0.771347 with 1.9× the params and the same addresses, the
account is dead." It failed. **§8.1 is dead.**

**Verdict on the rank axis:** neutral at fixed rows (two independent tests), and
**harmful** where the measurement is tightest (`shared_rank1`, +0.00257 at 12.8×,
on the best arm in the study). It also **inflates seed variance ~5×** on both
address types (§0). Function-valued rows never helped anything.

**What the round did establish, and it is the more useful result:**

- **Row count is the binding constraint.** +0.00285 for 2.6× fewer rows at 12×
  its bar, i.e. **~0.00195 per doubling**. This is the first directly measured
  capacity law in the study, and it retro-explains cross-layer sharing: sharing
  doubles rows per read at fixed params. Doubling predicts −0.00195; sharing
  measured −0.0039, so **capacity explains about half of the sharing win** and
  the rest is something else (one table serving all depths cannot specialize).
- **Sharing collapses seed variance 27×**, to the CUDA floor (§0). Sharing is
  robust on two independent axes, not just the mean.
- The one loose end: `lsh_rank4` (33k rows, rank 4) is level with `lsh` at 360k
  rows despite **11× fewer rows**, which the row-count law does not predict. n=1
  against a ~1.5e-3 bar, so 1.6× and unresolved. It is the only hint that high
  rank behaves unlike rank 1, approaching the MoE regime of few expressive units.
  Do not build on it without a replicate.

**Two retractions of my own, both mid-round, both the same error:**

| claimed | actual | cause |
|---|---|---|
| "`tokens_rank1` gains −0.00165 at 8.7× the bar, solid" | −0.00113 at **1.1×** | used the 1.9e-4 constant-row token bar; the arm's own bar is 1.0e-3 |
| "a 2.1× asymmetry: rank helps token, hurts contextual" | no asymmetry | built on the above; `shared_rank1` at 12.8× shows rank is neutral-to-harmful everywhere |

Both are the §5.1 error repeated one round later: reading signal out of a gap
smaller than that arm's own error bar. The bar must be selected per arm class
*before* interpreting, not after.

**Throughput.** Rank costs nothing at scale: 261–263k tok/s across rank 0/1,
matching the rank-0 Engram band (258–262k), versus the −1.5% a d8 smoke suggested.
Only `lsh_rank4` paid (256k, 4× the read matmuls) and `lsh_r1_isorow` (198k, but
that is `device-batch-size=4` for a 3458M-param model, not the rank arithmetic).
Zero `torch.compile` graph breaks at any rank.

### 8.5 Still open from the reframing

- **Margin as a pre-training predictor.** Their per-key margin diagnostic tracks
  end-to-end accuracy (their Fig 7). Given that this round cost three retractions,
  a cheap predictor of whether an addressing scheme is usable is worth more than
  another arm. It would also give open item 2 (why contextual addressing is
  high-variance) a mechanism to test: redrawing an LSH projection moves where
  quantization boundaries fall, which moves per-key margin directly, while a token
  hash cannot.
- **A capacity yardstick.** `W = Θ(F log F)` against the Ω(|K| log|V|) counting
  bound gives an absolute scale for whether `slot_multiplier=30` is near-optimal
  for the number of distinct n-grams a table actually sees. Never asked.

---

## 9. What 88 runs actually established, and where to go

§8's Hebbian reading is refuted (§8.1, §8.4). This section is what survives it.

### 9.1 The three axes, ranked by evidence

**Axis 1 — does the address carry information the residual stream has discarded?
This is where all the value is.** Token n-grams beat hidden-state addressing by
**0.0040** (20× the bar). A token id is *data*: exact surface form that `h`, two
layers of attention and MLP later, has partly thrown away. A quantized hidden
state is a re-encoding of information the layer already has.

The strongest evidence is the null result, not the positive one: **13 distinct
contextual mechanisms land in a 0.0042 band that contains dense** (ranks 13-26).
LSH at 3 code widths, product quantization, product keys, per-head latents, EMA
bit balancing, rank 0/1/4 values, rows from 33k to 360k. The
mechanism-to-mechanism spread is *smaller* than the gap to dense. That is not 13
implementation failures; it is a ceiling on the information available.

And the reason is unflattering to the design: **a memory addressed by quantized
`h` returning learned parameters is a worse parameterization of "a learned
function of `h`" than the MLP beside it** — piecewise-constant over a random hard
partition, versus smooth. MoE, which is that function done properly with learned
routing, beats every contextual arm by 0.0068.

**Axis 2 — address capacity. Real, but arm-specific and much shallower than
first claimed.** The 0.00195-per-doubling figure was measured on ONE contextual
pair (`lsh` 360k rows vs `lsh_r0_slot4` 131k) and then reused to subtract "row
cost" from `lsh_rank1` and `kv80` — three confirmations that were one measurement
wearing three hats. Round 11 swept rows directly on the token arms at constant
compute (§9.2):

| arm | rows/head swept | slope per doubling |
|---|---|---|
| contextual `lsh` | 131k -> 360k | −0.00285 (the original) |
| **shared token** | 355k -> 710k -> 1.42M | **−0.000533**, log-linear over 4x |
| **per-layer token** | 355k -> 710k | **−0.0000040** (flat) |

So capacity binds hard on the contextual family, mildly on the shared token arm,
and **not at all** on the per-layer token arm. Two independent shared slopes agree
to 4.3e-5, so the shallow shared slope is real even though each step alone is only
1.3-1.4x its bar. Reading: token addressing at slot 15 already has enough slots
for the n-grams that matter in 2B tokens; sharing packs both layers' reads into one
table and so sits closer to its capacity edge, which is where the residual slope
comes from.

**Axis 3 — row content expressiveness. Dead.** Rank-k values: neutral at fixed
rows (0.02× and 0.09× of the bar, two independent tests), harmful on the best arm
(+0.00257 at 12.8×), and they inflate seed variance ~5×. Do not spend parameters
making a row richer.

### 9.2 The organizing principle, and what it predicts

Sort every arm — and the external literature — by *where the information enters*:

| | address | value | outcome |
|---|---|---|---|
| token Engram | **external** (token ids) | learned | −0.0040 vs contextual |
| kNN-LM / RETRO | internal (`h`) | **external** (corpus continuations) | works (published) |
| contextual Engram | internal (`h`) | learned | ≈ dense, 13 mechanisms deep |
| dense MLP | — | learned | baseline |

**At least one of the address or the value must carry information from outside the
model's own forward pass.** Our contextual arms had neither, which is exactly why
no quantizer, code width, row count or row rank rescued them.

Note what this does *not* say: addressing by `h` is not inherently wrong — kNN-LM
does it and works, because its *values* are empirical next-token distributions
from real data. We addressed by `h` and returned learned vectors. Different object.

**Rounds 7-9 tested the two things on the live axes that were never tried, plus
three follow-ups. 9 arms, ~21h, all iso-sized.**

| arm | bpb | vs its baseline | verdict |
|---|---|---|---|
| **`count_gate`** | **0.763136** (n=3, sd 7.5e-4) | −0.00075 vs `engram_shared` | **1.57×, not resolved** |
| `count_gate_perlayer` | 0.765645 | −0.00189 vs `engram`/`tokens` | 2.3× (conservative), suggestive |
| `cg_n5` (orders 2-5) | 0.764698 | +0.00156 vs `count_gate` | 1.8× worse — no help |
| `whiten_p03` (fixed exponent) | 0.766068 | +0.00218 vs `engram_shared` | **5.5× worse** |
| `kv16` (key_dim 16) | 0.765461 | +0.00106 *excess* over row law | 2.7× worse |
| `kv80` (key_dim 80, iso) | 0.766208 | +0.00037 *excess* over row law | **0.9× — neutral** |

**The k/v split is neutral (`kv80`).** Exactly iso on total and active, gate
projection byte-identical, only rows halved. Its whole loss is predicted by the
0.00195-per-doubling row law from §8.4: observed +0.00232, predicted +0.00195,
**excess 0.9× the bar**. So a row's key being a fixed linear function of its value
is real but was *not a binding constraint* — the gate was fine deriving its key
from the payload. `kv16` looked worse (excess 2.7×) because it also narrowed the
gate's input 1280 → 256; the two arms together isolate that **narrowing the gate's
input costs ~0.0011**, which is a side finding neither gives alone.

This unifies with the rank result into one statement:

| intervention | at fixed rows | at iso-total |
|---|---|---|
| rank-1 values | neutral (0.02×, 0.09×) | loses = row cost |
| separate k/v | neutral (0.9×) | loses = row cost |

**Spending row bytes on anything other than more rows is neutral-to-negative.**
Two independent features, four measurements, and a third confirmation of the row
capacity law.

**The count gate: a real learned mechanism whose benefit does not resolve.** At
n=2 it read −0.00118 at 3.7× with complete separation and I called it the new
best. The third seed came in at 0.764000, taking its sd from 8.6e-5 to **7.5e-4**
(8.8×) and the result to **−0.00075 at 1.57×, with the groups now overlapping**.
Downgraded to unresolved. It remains the *best mean* in the study and the only
intervention that spends no row bytes at all (32 scalars), but "best mean" at
1.57× is not a result.

What *does* replicate, perfectly, is what the gate learns. Read from four
independent gates across two runs (64 heads total), **every single head learned
negative scale**:

| gate | mean scale `a` | all negative? |
|---|---|---|
| `count_gate_s0`, shared | −0.29 (order-2 −0.32, order-3 −0.25) | yes, 16/16 |
| `count_gate_s2`, shared | −0.28 | yes, 16/16 |
| `count_gate_perlayer`, layer 2 | −0.32 | yes, 16/16 |
| `count_gate_perlayer`, layer 6 | −0.47 | yes, 16/16 |

Negative `a` in `w = 2·sigmoid(a·log1p(rate) + b)` means *the more often a row is
addressed, the less it is trusted* — Kneser-Ney backoff, learned from scratch, with
the sparser order-3 heads discounted ~1.7× harder than order-2 and the deeper layer
(6) backing off harder than the shallower (2). In the sigmoid tail
`w ≈ 2e^b·(1+rate)^a`, so the learned inverse-frequency **exponent is ≈ −0.3**. Only
`a` is meaningful: `value_proj` can apply its own per-head scale and could
reproduce any fixed `b`, but not a frequency-*dependent* weight.

So there is a clean dissociation: **the model reliably learns to back off, and
backing off does not reliably move bpb.** Both halves are findings.

**A fixed exponent is actively harmful (`whiten_p03`, +0.00218 at 5.5×).** Setting
whitening to the learned −0.3 — the same monotone rule, without per-head or
per-order freedom — is *much worse* than either the learned gate or no gate at all.
So whatever value the gate has is not "the right exponent"; imposing that exponent
uniformly destroys 0.002. This also means the `readout_whiten` default of power 1.0
that shipped is worse still, and the feature should be considered a negative result
rather than untested.

**Longer n-gram orders did not help (`cg_n5`, +0.00156 at 1.8×).** §9.3 previously
ranked this as the top thing to build, on the grounds that orders 4-5 add exact
surface information `h` has discarded — the only axis that pays. Paired with the
gate (which demonstrably learns to discount sparser orders) it still came in worse
than orders 2-3 alone. n=1 against an inflated bar, so not decisively refuted, but
the priority was wrong. Sizing note for anyone repeating it: 32 heads × 40 dims at
slot 30 is iso with 16 × 80, and `max_ngram_size=4` is *impossible* at
`memory_dim=1280` because 24 does not divide 1280.

**The one positive signal left is `count_gate_perlayer`** (−0.00189, 2.3×
conservative / 4.7× if both sides are baseline-class). It matches the
pre-registered prediction — per-layer tables hold half the rows, so they collide
more, and backoff is worth more where collisions are worse — and it is a *larger*
effect than on the shared table. It is the one thing here worth a replicate.

### 9.2b Rounds 10-11: the count gate at n=3, and sharing is not capacity

**Round 10 — count gating does not resolve.** Both bases taken to n=3:

| base | gate arm (n=3) | baseline (n=3) | delta | own bar | ratio |
|---|---|---|---|---|---|
| shared | 0.763136 (sd 7.5e-4) | 0.763884 (sd 3.5e-4) | −0.00075 | | 1.57× |
| per-layer | 0.766809 (sd 1.21e-3) | 0.767606 (sd 8.1e-5) | −0.00080 | | 1.14× |
| **pooled** (6 vs 6 runs) | | | **−0.00076** | 3.9e-4 | **1.94×** |

The two independent estimates agree to 5e-5, which is the strongest thing in the
gate's favour, but pooled it still misses the 2x threshold and each base alone is
1.1-1.6x. **Verdict: suggestive, unresolved** — and it costs a **15x variance
inflation** on the per-layer arm (1.21e-3 vs 8.12e-5) for 32 scalars. Whatever
small benefit exists is bought with an order of magnitude more run-to-run noise,
which is a bad trade for anyone deciding from one run.

**The gate's learned behaviour replicates perfectly even though its benefit does
not.** Six independent gates across three runs, 96 heads, **every head negative**
scale — Kneser-Ney backoff learned from scratch, with sparser orders discounted
harder and deeper layers backing off harder. Effective inverse-frequency exponent
~ -0.3. That dissociation is itself the finding: *the model reliably learns to
back off, and backing off does not reliably move bpb.*

**A fixed exponent is actively harmful** (`whiten_p03`, +0.00218 at 5.5x): setting
whitening to the learned -0.3, same monotone rule without per-head freedom, is much
worse than either the learned gate or no gate. `readout_whiten` is a negative
result, not untested, and its shipped default of power 1.0 is worse still.

**The Fourier gate came back MONOTONE.** Given a basis that provably fits a
unimodal trust curve (loss 5e-3, where the linear gate cannot beat 5e-2), the
learned curve peaks at count 0 and decreases throughout, in both layers; the only
non-monotonicity is a ~1%-of-range wiggle. bpb unresolved (0.86x), as
pre-committed. Diagnosis: a row's hit rate CONFLATES "this n-gram is frequent"
with "this row is crowded", and both argue for downweighting, so no
non-monotonicity can appear. Katz discounting is about the reliability of a count
for a *specific* n-gram, which we never measured. Round 12 (`--engram-count-gate-
decouple`) splits the channel using the fact that every head of an n-gram order
hashes the same n-gram under a different prime, so `f_o = min over the order's
heads` is that n-gram's own rate and `c_h = rate_h - f_o` is head *h*'s crowding,
with `rate_h == f_o + c_h` exactly and no new buffers.

**Round 11 — sharing is not capacity, and the capacity law is arm-specific.**
A pure storage sweep at constant compute: active params 918.34M and FLOPs 3.135e9
identical for every arm, only rows and total params moving. **These arms are NOT
iso-total; a number here bought with 2.5x the parameters has measured a slope, not
beaten a baseline.**

| arm | rows/head | total | dbs | bpb |
|---|---|---|---|---|
| perlayer_slot15 | 355,290 | 1828M | 8 | 0.767606 (n=3) |
| perlayer_slot30 | 710,580 | 2738M | 8 | 0.767602 |
| shared_slot15 | 355,290 | 1373M | 8 | 0.764396 |
| shared_slot30 | 710,580 | 1828M | 8 | 0.763884 (n=3) |
| shared_slot60 | 1,421,160 | 2738M | 8 | 0.763329 |
| *shared_slot120* | *2,842,320* | *4557M* | *4* | *0.761128* (unusable) |
| *perlayer_slot60* | *1,421,160* | *4557M* | *4* | *0.763210* (unusable) |

- **Sharing is not capacity.** At **matched rows per read** (710,580 both sides),
  `shared_slot30` beats `perlayer_slot30` by **0.00372 at 9.3x while using 67% of
  the storage**. Not half of the sharing win is capacity — *none* of it is. What
  sharing does is put the same n-gram on the same row at every depth, so one row is
  trained by gradients from all reading layers: a jointly-trained memory, not more
  memory.
- **The token capacity slope is −0.000533 per doubling and log-linear over 4x**
  (two independent steps agreeing to 4.3e-5) on the shared arm, and **flat** on
  per-layer (−4.0e-6). The 0.00195 "law" was contextual-only.
- **Two arms are unusable, and that is a design error of mine.** `shared_slot120`
  and `perlayer_slot60` ran at `device-batch-size=4` while the rest ran at 8, and
  both came in 0.0022-0.0044 better than the fit predicted. A single additive dbs
  effect does not fit either (the two shifts differ 2x), and a *prior* dbs=4 arm
  (`lsh_r1_isorow`, round 6) showed no benefit at all. So the 4x steepening at the
  top of the shared curve cannot be attributed to capacity or to batch geometry
  after the fact. See §6: never vary dbs inside a comparison. The clean fix is one
  arm — `perlayer_slot30` re-run at dbs=4 — which isolates dbs at fixed rows.

### 9.2c Rounds 12-13: the first resolved gate result, and sharing is *pooling*

**Round 12 — decoupling the count gate's input resolves it, by halving its
variance.** `--engram-count-gate-decouple` splits a row's hit rate into the
n-gram's own frequency and that head's collision excess, using the fact that every
head of an n-gram order hashes the SAME n-gram under a different prime:
`f_o = min over the order's heads`, `c_h = rate_h - f_o`, with `rate_h == f_o + c_h`
exactly. No cardinality sketch, no new buffers, 64 extra scalars.

| arm (n=3, shared, slot 30, iso) | bpb | sd | vs `engram_shared` |
|---|---|---|---|
| `engram_shared` | 0.763884 | 3.45e-4 | — |
| `count_gate` | 0.763136 | 7.50e-4 | −0.00075, 1.57× |
| **`cg_dec`** | **0.762826** | **4.07e-4** | **−0.00106, 3.43×** |
| `cg_dec_four` (n=2) | 0.762787 | — | −0.00110, ~1.9× conservative |

The *mean* barely moved — `cg_dec` vs `count_gate` is 0.63×, indistinguishable.
What changed is conditioning: **sd 7.50e-4 -> 4.07e-4, 1.8x tighter**, and its
n=2 -> n=3 sd growth was only **1.24x** against 2.6-12.8x for every earlier arm.
Decoupling did not make the gate better; it made it measurable. This is the first
count-gate-family result to clear 2x on its own n=3 bar, after 17 runs.

**The decoupling prediction holds, and an earlier reading of it was wrong.** From
the *Fourier* arms the decoupled coefficients looked like noise, and this document
briefly said the model ignored them. That was an artifact: a Fourier basis on
`log1p(f)` absorbs the frequency channel and leaves its linear coefficient
degenerate. The plain arms are unambiguous:

| run | rate `a` | freq `e` | collide `d` |
|---|---|---|---|
| `cg_dec_s0` | −0.623 (16/16 neg) | **+0.421 (16/16 pos)** | **−0.062 (16/16 neg)** |
| `cg_dec_s1` | −0.612 (16/16 neg) | **+0.416 (16/16 pos)** | **−0.069 (16/16 neg)** |
| `cg_dec_four_s0` | −0.702 (16/16) | +0.060 (8/16) | +0.012 (11/16) |
| `cg_dec_four_s1` | −0.693 (16/16) | +0.136 (11/16) | −0.008 (12/16) |

`collide_scale` is negative with **16/16 head agreement in both plain seeds** --
crowded rows *are* distrusted, which is exactly what the decomposition predicted.
Netting the terms at `f = c = 1`: `d(logit)/dc = -0.343` versus
`d(logit)/df = -0.101`, so **collision is penalised 3.4x more steeply than genuine
frequency**. The memory does want the two separated; it just needed them in
separate linear terms rather than behind a basis that could absorb one of them.

The Katz question itself stays closed: both Fourier arms are monotone in `f` with
the peak at 0, and the two reconstructed curves correlate at **0.9996** across
seeds. Trust is monotone in a specific n-gram's own frequency too, not just in the
conflated rate.

**Round 13 arm 1 — the `dbs` confound is real, large, and now measured.**
`perlayer_slot30` re-run at `device-batch-size=4`: **0.765760** against 0.767602 at
dbs=8. **-0.00184 at fixed rows, fixed parameters, fixed token budget.** So dbs is
not a free knob and cross-dbs comparisons are invalid (§6). Applying the correction
rescues the round-11 capacity curve:

| shared slope | uncorrected | corrected |
|---|---|---|
| 710k -> 1.42M rows (dbs 8->8) | −0.000555 | — |
| 1.42M -> 2.84M rows (dbs 8->4) | −0.002201 | **−0.000359** |

The 4x "steepening" at the top of the curve was almost entirely batch geometry.
The log-linear **−0.000533 per doubling** slope survives, now over 8x of rows.
`perlayer_slot60` still leaves −0.00255 after correction, so the per-layer curve is
not fully explained, but it is no longer a 4x anomaly.

**Round 13 arms 2-5 — the sharing effect is POOLING, and the 2x2 closes.**
`--engram-share-hash` separates the two things `share_memory` bundles. All four
cells iso: 710,580 total rows, 909.5M of table bytes, 918.34M active, dbs=8.

| | per-layer hash | shared hash |
|---|---|---|
| **per-layer table** (2 x 355,290) | `engram` **0.767606** (n=3, sd 8.1e-5) | `share_hash` **0.766863** (n=2, sd 3.7e-5) |
| **shared table** (1 x 710,580) | `share_tbl` **0.763604** (n=2, sd 7.9e-5) | `engram_shared` **0.763884** (n=3, sd 3.5e-4) |

```
MAIN EFFECT table pooling : -0.003491
MAIN EFFECT hash sharing  : -0.000232   (15x smaller)
INTERACTION               : +0.001023   (larger than the hash main effect)
```

**Pooling the table is essentially the whole effect.** Sharing the addressing
scheme is worth 15x less, and the interaction is a genuine crossover with a
readable mechanism:

- **private tables: sharing the hash HELPS**, -0.00074 at **10x** (conservative;
  13.9x on its own n=2 sd, which is 3.7e-5, at the CUDA floor).
- **shared table: sharing the hash HURTS**, +0.00028 (1.35x, unresolved). Making
  an n-gram hit the *same* row at both depths means each n-gram occupies one row
  instead of two, so consistent addressing *under-uses* a shared pool.

That last point kills the "one row trained by gradients from every reading layer"
story outright: the configuration that maximises per-n-gram row sharing is the
*worse* of the two shared-table cells.

**What pooling is not.** Three eliminations, each with its own arm:
- not capacity (round 11: at matched rows-per-read, sharing still wins 0.00372 at
  9.3x on 67% of the storage);
- not addressing consistency (this round: 15x smaller, and the wrong sign on a
  shared table);
- not joint training of matched rows (`share_tbl` breaks the correspondence and
  loses nothing).

**The sharp statement of what is left.** `share_tbl` gives each layer 710,580
addressable rows out of ONE 909.5M table and reaches 0.7636. Round 11's
`perlayer_slot30` gives each layer the same 710,580 rows out of TWO tables
totalling 1819M and reaches 0.7676. **Same rows per read, half the parameters,
0.0040 better, and no per-n-gram row correspondence.** A row in the pooled case is
read by layer 2 for one n-gram and by layer 6 for a different one, so its content
is in superposition -- and that is apparently *better* than giving each layer its
own private row.

One testable hypothesis, not yet run: a pooled row receives gradient from every
reading layer, so it is simply better trained per parameter at a fixed token
budget. That predicts **the pooling benefit shrinks with longer training**, which
2.0B tokens (well short of the ~9.5B compute-optimal point, §0) cannot distinguish
from a representational effect.

### 9.2d Round 14: `cg_dec` hardened to n=5 and shown to transfer

The only resolved mechanism in the study, taken to n=5 on the base it was found on
and to n=3 on a second base. All arms iso to their own baselines (shared
1828.0M/918.34M; per-layer 1828.4M/918.34M), dbs=8 throughout.

| track | arm | n | mean | sd | vs its baseline |
|---|---|---|---|---|---|
| shared | `cg_dec` | **5** | 0.762987 | **4.89e-4** | −0.00090, **3.03×** |
| per-layer | `cg_dec_perlayer` | 3 | 0.765924 | 8.72e-4 | −0.00168, **3.33×** |
| | **pooled** | | | | **−0.00110, 4.29×** |

The two estimates are statistically consistent (they differ by 1.34×), so this is
one effect of about −0.001 measured twice on independent bases. **It is the first
mechanism in the study to resolve on two bases.**

**The pre-registered variance test passed.** `cg_dec`'s shared sd went 4.07e-4
(n=3) -> 4.89e-4 (n=5), a growth of **1.20x**. Every other sd in this study grew
2.6x-12.8x when a seed landed (§0), so this is the only arm whose error bar can be
quoted with confidence. It did *not* inflate to gate-class 7.5e-4, which was the
stated failure mode.

**Two of my own numbers need correcting.**

| claimed | actual |
|---|---|
| decoupling makes the gate 1.8x tighter | **1.53x** on shared (7.50e-4 -> 4.89e-4), **1.39x** per-layer (1.21e-3 -> 8.72e-4) |
| per-layer sd should come in ~4e-4 if decoupling is a general conditioning fix | **8.72e-4** — the prediction failed |

So the mechanism transfers, but **not by the route I claimed**. On the shared base
it resolves by tightening variance at a flat mean; on the per-layer base it
resolves by *doubling the mean* (−0.00168 vs the plain gate's −0.00080) while the
variance stays high. I have no account for that asymmetry, and "decoupling is a
conditioning fix" is too simple a description of it.

**The learned coefficients replicate everywhere, and one of them is quantitatively
informative.** Seven gates across two bases and two seeds, **16/16 head sign
agreement in every one**:

| run | gate | rate `a` | freq `e` | collide `d` |
|---|---|---|---|---|
| `cg_dec_perlayer_s0` | layer 2 | −0.479 | +0.319 | **−0.372** |
| `cg_dec_perlayer_s0` | layer 6 | −0.482 | +0.109 | **−0.278** |
| `cg_dec_perlayer_s1` | layer 2 | −0.497 | +0.309 | **−0.345** |
| `cg_dec_perlayer_s1` | layer 6 | −0.464 | +0.107 | **−0.260** |
| `cg_dec_s3` | shared | −0.599 | +0.402 | −0.087 |

`collide_scale` is **3-4x larger on private tables** (−0.26 to −0.37) than on the
shared one (−0.087). Private tables hold half the rows per layer, so they collide
more, and the gate penalises crowding correspondingly harder. That is a
*quantitative* correspondence between a learned parameter and a structural fact,
and it is the strongest evidence that the decomposition measures what it claims.
It also explains the larger per-layer mean effect: there is more crowding to fix.
Layer 6 consistently learns a lower `freq_scale` than layer 2 (+0.11 vs +0.31,
both seeds).

**New diagnostic: the tables are saturated.** Read from the `hit_rate` buffers of a
round-14 checkpoint, per n-gram order:

| | rows empty | p99 hit rate | max |
|---|---|---|---|
| order-2 heads | 0.1% | 8.6 | **1670x** |
| order-3 heads | 0.0% | 3.8 | 575x |

Essentially every row is occupied, at ~7 (order 2) to ~21 (order 3) hits per row
per EMA window: the number of distinct n-grams vastly exceeds the row count. This
closes off two obvious responses to the collision finding. A better hash cannot
help -- uniform hashing already minimises max load to within the power-of-choices
log-log factor -- and more rows barely help either (round 11: −0.000533 per
doubling shared, flat per-layer). Order-2's much heavier tail (1670x vs 575x) is
where frequent bigrams collide with each other, which is the damaging case and
matches `collide_scale` being largest exactly where rows are scarcest.

What *is* available is decorrelating the interference across more independent
hashes, since each head hashes the same n-gram under a different prime. Round 15
tests it (`--engram-heads-per-ngram`), and it is exactly free: table bytes =
`heads x rows x head_dim` with `head_dim = memory_dim/heads`, so the table size is
independent of head count (1827.94M / 1828.02M / 1828.16M at 8 / 16 / 32 heads,
with active params and FLOPs identical).

### 9.2e Round 15: more independent hashes do nothing — the collision line is closed

Pre-registered as §9.3 item 2. Head count is **exactly free**: table bytes =
`heads x rows x head_dim` with `head_dim = memory_dim / heads`, so at fixed
`memory_dim` and fixed `slot_multiplier` the table size does not depend on it.
Verified on meta — 1827.94M / 1828.02M / 1828.16M total at 8 / 16 / 32 heads
(0.025% spread), with **active params and FLOPs/token identical** (918.340M,
3.135e9). One of the few genuinely single-variable sweeps in this study.

| arm | heads/order | total heads | head_dim | n | mean | vs baseline |
|---|---|---|---|---|---|---|
| `heads4` | 4 | 8 | 160 | 1 | 0.763833 | −5.1e-5, **0.15×** |
| `engram_shared` (baseline) | 8 | 16 | 80 | 3 | 0.763884 | — |
| `heads16` | 16 | 32 | 40 | 2 | 0.763854 | −3.0e-5, **0.13×** |

The three group means span **5.1e-5** against the baseline's 3.45e-4 seed sd. That
is the flattest result in the study: 0.15× of one error bar across a **4× range** of
the independent variable, and not even monotone. Throughput is flat too (257.6k /
257.9k / 258.0k tok/sec).

The pre-registered prediction was `heads16 < baseline < heads4` if
interference-averaging is the mechanism. **Neither direction appeared**, so H-head
averaging was already saturated at 16 heads and collisions are *tolerated* rather
than cancelled. Two arms (`heads4_s1`, `heads16_s2`) were lost to an external
SIGTERM and deliberately **not** re-run — they could only tighten a null.

**What this closes.** Together with §9.2d's saturation diagnostic, every available
response to the collision finding is now eliminated:

| response | verdict |
|---|---|
| a better hash function | impossible — tables saturated (0.1% / 0.0% of rows empty), uniform hashing already near-optimal for max load |
| more rows | −0.000533 per doubling shared, flat per-layer (§9.2b) |
| more independent hashes | this round, 0.13× |

So `collide_scale`'s consistent negative sign with 16/16 head agreement (§9.2d) says
collisions cost real bpb, and *nothing available at this scale reduces them*.
**The gate's response is the fix, not de-crowding.** Decoupling lets the read
*discount* a crowded row instead of un-crowding it, which is exactly the one
mechanism that resolved (−0.00110 pooled, 4.29×). Read §9.2d and this section
together: the decoupled count gate is not a refinement of an addressing scheme, it
is the only handle on a saturated table.

### 9.2f Round 16: the two mechanisms are complementary — the first composition result

Fifteen rounds compared hash memory against routed experts and **never put them in
the same model**. §1's conclusion ("the spread between mechanisms is smaller than
the gap to dense") makes composition the largest question the study's own framing
implies, and by round 15 ten mechanism ideas *inside* the hash-memory design had
produced exactly one resolved win — a poor base rate for an eleventh. This round
composes two things already known to work instead.

Both arms are `--engram-share-memory` + `--engram-count-gate-decouple` (the
`cg_dec` recipe) *plus* a Mobius-style shared expert pool (`--moe-share-blocks=1`,
`--moe-every=2`, top-8 of N, `d_ff_expert=640`). dbs=8 throughout.

| arm | engram slot | experts | total | active | FLOPs/tok | n | mean | sd | tok/sec |
|---|---|---|---|---|---|---|---|---|---|
| `cg_dec` (reference) | 30 | — | 1828.0M | 918.34M | 3.135e9 | 5 | 0.762987 | 4.89e-4 | ~257k |
| **`combo_iso`** | 15 | 357 | **1827.5M** | 922.91M | 3.162e9 | 3 | **0.760625** | 7.92e-4 | 170.8k |
| *`combo_full`* | 30 | 640 | *2746.3M* | 926.53M | 3.184e9 | 2 | *0.757196* | 5.06e-4 | 155.8k |

Seeds: `combo_iso` 0.760619 / 0.759836 / 0.761419; `combo_full` 0.756838 / 0.757553.

**`combo_iso` is the disciplined arm** — matched to `cg_dec` on total params within
**0.03%**, splitting the same budget 44/56 between table and experts instead of
putting all of it in one.

| comparison | delta | Welch SE | ratio |
|---|---|---|---|
| `cg_dec` − `combo_iso` | **−0.002363** | 5.07e-4 | **4.66×** |
| `combo_iso` − `combo_full` | −0.003429 | 5.80e-4 | 5.91× |
| `cg_dec` − `combo_full` | −0.005792 | 4.19e-4 | 13.82× |

**There is also complete separation**: the *worst* `combo_iso` seed (0.761419) beats
the *best* of `cg_dec`'s five (0.762400) by 9.81e-4. All three seeds land in
0.7598–0.7614, well below the pre-registered 0.7630 threshold.

**The pre-registered outcome that holds is "complementary."** At a fixed parameter
budget, a mix of hash memory and routed experts beats spending it all on the memory.
At −0.00236 this is the **second-largest iso-parameter mechanism effect** in the
study, behind table pooling's −0.0035 and ahead of the decoupled count gate's
−0.00110. Cumulatively `dense - combo_iso` = **0.01199**, which is 1.37× the §1
winner's 0.00873.

This supports the §9.1–§9.2 account rather than the "capacity is capacity" reading.
A token n-gram address carries *external* information — the exact surface form the
residual stream has discarded — while an MoE expert is a learned input-dependent
*function of the hidden state*. Different information, different machinery, so they
add. Had they been two ways of buying the same conditional capacity, the mix would
have landed at ~0.7630, and it did not.

**Three caveats, stated before the run and all still live.**

1. **The combo carries +0.50% active params and +0.87% FLOPs/token**, all of it the
   MoE router (1280 × n_experts, read by 10 layers). It cannot be removed without
   moving `moe_every` off the Mobius reference, so a combo *win* is partly
   confounded — the pre-registration said only a combo *loss* would be decisive.
   The bound against it explaining the result: the router is ~4.6M params, whereas
   doubling the entire shared table (+910M) buys only −0.00056 (§9.2b). To produce
   −0.00236 those 4.6M would need ~500× the per-parameter efficiency of table rows.
   Suggestive, not proof.
2. **The seed sd is a lower bound.** `--engram-seed` permutes the Engram's hash
   multipliers but does **not** reseed the MoE router — there is no `--moe-seed` and
   no global `manual_seed` in `base_train` (§0). All three `combo_iso` seeds share
   one router draw, so 4.66× is optimistic. The complete-separation observation is
   the more robust half of the claim.
3. **`combo_full` is not iso** (+50% total) and per the pre-registration must be
   read as partly "more parameters help." What it does establish is that the
   combination has **not saturated**: `dense − combo_full` = 0.01578, **1.81×** the
   §1 winner. Its +918.8M bought −0.00343, ~6× better per parameter than the
   table-doubling slope — but it scales the table (455→909M) *and* the experts
   (585→1049M) at once, so that is not a single-axis capacity measurement and should
   not be pushed further.

**Cost.** This is the study's first result that is not free. `combo_iso` runs at
170.8k tok/sec against `cg_dec`'s ~257k — **1.51× slower wall-clock** (194 min vs
~128) — and `combo_full` at 155.8k, 1.65× slower. Every earlier mechanism moved bpb
at roughly constant throughput.

**Router health: no collapse.** The `loss_free` balancer carries no aux term
(`aux_loss: 0.00000` throughout, as designed), so all balancing is bias-based.

| arm | entropy start → end | ceiling `ln(n_experts)` | regret start → peak → end | `max_vio` start → end |
|---|---|---|---|---|
| `combo_iso` | 5.794 → 5.502 | 5.878 | 0.054 → 0.090 → 0.032 | 1.94 → 8.20 |
| `combo_full` | 6.296 → 5.981 | 6.462 | 0.076 → 0.146 → 0.044 | 2.88 → 9.60 |

Load entropy ends at **93.6%** and **92.5%** of the uniform ceiling and regret ends
*below* where it started, so the routers are not degenerate and the Engram's
presence does not starve them. But `max_vio` **rises** from ~2 to 8–10 over training
(the most-loaded expert ends at ~9–11× uniform load), which is real imbalance.

**A limitation that cannot be repaired from the surviving artifacts:** §1's
Engram-free `mobius` logs did not survive the `/tmp` wipe (§6), so there is **no
baseline router trajectory to compare against**. Whether `max_vio ≈ 9` is normal for
this config or an Engram-induced degradation is *unknown*. The intended
mechanism-interaction reading of this round is therefore unavailable, and only
absolute health can be reported. Preserving router stats for a no-Engram control is
now a prerequisite for asking that question.

### 9.2g Round 17: both follow-ups on the composition come back negative

Two arms, both **exactly iso** to `combo_iso` — total 0.000%, active ≤0.008%,
FLOPs/token ≤0.020%, verified on meta *and* in each run's own printout
(1827.5M / 922.8M vs 922.91M). dbs=8 throughout.

| arm | config | n | mean | sd | range | vs `combo_iso` |
|---|---|---|---|---|---|---|
| `combo_iso` (ref) | 357 experts, top-8, cg_dec | 3 | 0.760625 | 7.92e-4 | .759836–.761419 | — |
| `combo_mobius` | 347 experts, top-**7**, + private always-on 640 expert/layer | 3 | 0.761117 | **1.04e-3** | .759994–.762051 | +0.000493, **0.65×** |
| `combo_nogate` | 357 experts, top-8, **no count gate** | 3 | 0.761898 | 7.05e-4 | .761113–.762476 | +0.001273, **2.08×** |

#### `combo_mobius`: the real Mobius component does nothing — and it had never been run

**The study called an arm "mobius" for 16 rounds while omitting its defining
part.** §1's `mobius` and round 16's combo both used only
`--moe-share-blocks=1`. Neither passed `--moe-shared-d-ff`, i.e. neither had the
*private per-layer gated dense expert* that upstream Intern-S2-Mobius adds on top
of a shared pool — the thing that makes a shared pool more than plain weight
tying. This arm adds it, and pays for it exactly: one always-on 640-wide expert
costs precisely one routed expert, so `top_k` 8→7 and `n_experts` 357→347 balance
it to the parameter.

**Result: +0.000493 at 0.65× — unresolved, and if anything slightly worse.** The
pre-registered **"interchangeable"** outcome holds: at a fixed budget, *where* the
conditional capacity lives does not matter, only how much of it there is. This
**retires `--moe-shared-d-ff` as a knob** and means §1's omission was harmless —
the arm was under-specified, but not in a way that changed its number.

**The model does use the thing; it just gains nothing from it.** The shared-expert
gate is zero-init (`sigmoid(x·g)` = 0.5 for every input at step 0). After training
it has moved a long way: ‖g‖ per layer, three seeds —

| | layer 0 | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 18 |
|---|---|---|---|---|---|---|---|---|---|---|
| s0 | 31.8 | 31.5 | 8.2 | 30.6 | 7.6 | 31.7 | 16.2 | 6.8 | 7.3 | 14.1 |
| s1 | 32.4 | 31.7 | 30.3 | 30.8 | 28.3 | 16.1 | 30.1 | 31.7 | 23.6 | 11.3 |
| s2 | 33.2 | — | — | — | — | 20.8 | 29.8 | — | — | — |

Every layer learns a strongly input-dependent gate (‖g‖ 7–33 against 0 at init),
with mean ≈ 0 (±0.04), so it swings to both sides of 0.5 and at that norm is close
to a hard on/off switch. Which layers get large gates is **not consistent across
seeds** (layer 4: 8.2 / 30.3; layer 10: 31.7 / 16.1 / 20.8), so there is no depth
pattern to report. The coherent reading: *a gated always-on expert is functionally
just one more router, and the routed pool already does that job.*

**It also inflated the seed variance to 1.04e-3**, the widest bar of any combo arm
— consistent with the study-wide rule that anything added to the read path widens
the bar (§0), and a reminder that the "free" arms are the ones that change nothing.

#### `combo_nogate`: the count gate survives composition

Removing `--engram-count-gate{,-decouple}` costs **+0.001273 at 2.08×** — the arm
gets worse, so the gate is still doing work with the MoE present.

**But read the magnitude carefully, and do not claim the strong version.** The
gate is worth −0.00127 ± 6.1e-4 inside the combo against −0.00090 ± (its own bar)
standalone. Those two differ by 0.00037, far below either error bar, so the honest
statement is: **the gate's contribution survives composition at a magnitude
statistically indistinguishable from its standalone value.** The pre-registered
"contributes MORE with the MoE present" branch is **not** established — my
pre-registration labelled the >0.7619 band that way and the mean landed at
0.761898, just under it, with the ratio clearing 2× only marginally. The band was
mislabelled: at this power, "survives unchanged" and "contributes more" are not
separable, and only the former is supported.

This does answer the question the arm was built for. Round 16's −0.00236 was
measured as "cg_dec + MoE" vs "cg_dec", and the gate's own half had never been
re-checked inside the combo. It is **not** absorbed by the MoE. §9.3's "cg_dec on
a third base" item is thereby partly satisfied: the combo is a third base, and the
gate transfers to it.

**The learned coefficients replicate a fourth time, and the crowding
correspondence sharpens.** All three seeds, read from the checkpoints:

| seed | `count_gate_scale` | `freq_scale` | `collide_scale` |
|---|---|---|---|
| s0 | −0.563 | **+0.305** | **−0.140** |
| s1 | −0.550 | **+0.287** | **−0.159** |
| s2 | −0.571 | **+0.314** | **−0.163** |

Signs are as in §9.2d — frequency good, collision bad — now on a fourth base. And
`collide_scale` is **−0.14 to −0.16 here vs −0.087 on the plain shared table**.
These combo arms run `slot_multiplier=15`, i.e. *half* the rows of `cg_dec`'s slot
30, so there is more crowding, and the gate penalises it correspondingly harder.
That is the §9.2d crowding correspondence confirmed a third time, quantitatively
and in the predicted direction.

#### Router: one clean confirmation, one falsified prediction

| arm | entropy start → end | ceiling | regret start → end | `max_vio` start → end |
|---|---|---|---|---|
| `combo_iso` | 5.7939 → 5.4920 | 5.878 | 0.05422 → 0.02792 | 1.9427 → **7.9702** |
| `combo_nogate` | 5.7939 → 5.4527 | 5.878 | 0.05422 → 0.03507 | 1.9427 → **9.1237** |
| `combo_mobius` | 5.7516 → 5.4614 | 5.849 | 0.04792 → 0.02947 | 2.1981 → **9.2619** |

**The pre-registered `max_vio` prediction failed.** I predicted that if the private
always-on expert soaked up load, `combo_mobius`'s `max_vio` would *improve* on
`combo_iso`'s. It got **worse** (9.26 vs 7.97). Adding always-on capacity did not
relieve the router; entropy is unchanged at 93.4% of ceiling either way.

**And a clean confirmation of the frozen-router caveat.** `combo_nogate_s0`'s first
logged MoE line is *byte-identical* to `combo_iso_s0`'s (regret 0.05422, max_vio
1.9427, entropy 5.7939) across a different Engram config and a different run. That
is direct evidence that `--engram-seed` does not touch the router and that every
error bar in the combo family is a **lower bound** (§0). It also means the
7.97 → 9.12 divergence between those two arms is attributable to the Engram count
gate alone: removing it left the router *more* imbalanced, which is a mechanism
interaction worth one line but is n=1 and not worth an arm.

#### What this round costs the composition story

Nothing about round 16's headline: `combo_iso` remains −0.00236 at 4.66× over
`cg_dec` with complete seed separation. What round 17 removes is *two of the three
obvious ways to push it further*. The remaining one — sweeping the table/expert
split, whose endpoints are both already measured — is now the only cheap item with
a known-positive gradient, and it moves to the top of §9.3. **(Round 18 then ran it
— §9.2h: asymmetric, optimum expert-heavier than 44/56, and the split retired as a
knob.)**

### 9.2h Round 18: walking the table/expert split — the optimum is expert-heavier than 44/56

Round 16 put the hash memory and the routed experts in one model at one budget
division (44/56 table/experts) and it won. That split was the first one tried, not
a chosen one, so the last cheap question in the combo was whether it is near the
peak or merely a sample. Both far endpoints were already measured (`cg_dec` = 100/0;
`mobius` ≈ 0/100), so this is interpolation on a curve known to have an interior
optimum. Two flanking arms, each **iso to combo_iso on total** (2330.7–2330.9M,
±0.005%) with the memory rows and expert count traded against each other, turn one
point into a shape. dbs=8 throughout; three Engram seeds each.

**The five-point curve** (table% / expert% of the conditional budget):

| split | arm | slot | experts | n | mean | sd | vs `combo_iso` |
|---|---|---|---|---|---|---|---|
| 100/0 | `cg_dec` | — | 0 | 5 | 0.762987 | 4.89e-4 | +0.002362 |
| **73/27** | **`split_tbl`** | 25 | 172 | 3 | **0.761884** | 1.41e-4 | **+0.001259, 2.71×** |
| 44/56 | `combo_iso` (ref) | 15 | 357 | 3 | 0.760625 | 7.92e-4 | — |
| **26/74** | **`split_exp`** | 9 | 468 | 3 | **0.760038** | 2.42e-4 | **−0.000587, 1.23×** |
| 0/100 | `mobius` | — | ≈∞ | 1 | 0.764727 | — | +0.004102 |

**The pre-registered ASYMMETRIC outcome holds.** Table-heavy (73/27) is
**resolvably worse** at +2.71× (all three seeds above 0.7617, sd 1.4e-4 — the
tightest bar in the family). Expert-heavy (26/74) is **flat-to-better** at −1.23×
— *not* resolved (a one-step flank plausibly costs 0.0005–0.0010, below the ~0.0010
this n=3-vs-n=3 pairing can resolve), but it is the best sampled point and no seed
of it exceeds combo_iso's worst. Read the shape, not the individual gaps: **the
interior is monotone** — 0.762987 → 0.761884 → 0.760625 → 0.760038 as the budget
shifts from table to experts — turning back up only at the pure-expert wall
(`mobius`, no table, the worst point in the whole family).

**What this says, and what it does not.** The split *is* a live knob and it points
toward experts: 44/56 was not the peak, and the study should have sampled the
expert-heavy side. But the table is not optional — deleting it entirely is the
worst arm measured. The optimum sits between 26/74 and the pure-expert extreme,
expert-heavier than combo_iso happened to try. What the round **cannot** say is how
much better than 44/56 the expert-heavy side is: −0.00059 at 1.23× is below
resolution, and honesty requires reporting it as "at least as good", not "better".

**The confound cannot rescue table-heavy.** `router_active` scales with `n_experts`
(1280 × n_experts, read by 10 layers), so split_exp carries +0.100% active and
split_tbl −0.166% — a ~3.8M spread that systematically favours the expert-heavy arm.
But for scale, +910M of *table* bought only 0.00056 (§9.2b), so 3.8M is worth ~1e-6:
it cannot explain even the unresolved 0.00059, let alone the resolved 0.00126 by
which table-heavy *loses*. The gradient is real, not an accounting artefact.

**`collide_scale` confirms the crowding mechanism — and hits its study maximum.**
The Engram's learned `collide_scale` coefficient tracks table crowding: fewer rows
per read means more hash collisions, and the model leans harder on the collision
channel to compensate. Read straight from the six checkpoints (no forward pass),
the mean |collide_scale| is monotone in table size across every slot ever measured:

| slot (rows/read) | 30 | 25 (split_tbl) | 15 (combo_iso) | 9 (split_exp) |
|---|---|---|---|---|
| mean collide_scale | −0.087 | −0.093 | −0.15 | **−0.219** |

Slot 9 is the smallest table the study has run and shows its **largest
|collide_scale|** (−0.20 to −0.23 across three seeds), exactly as pre-registered.
`count_gate_scale` and `freq_scale` also shrink as the table shrinks (−0.60→−0.50
and +0.36→+0.22), consistent with a smaller table having less clean per-n-gram
signal to gate on.

**MoE routing is healthy and comparable on both arms once normalized.** Final
balance stats (loss_free, `--moe-log-every=200`):

| arm | experts | regret | max_vio | entropy | entropy / ln(n) |
|---|---|---|---|---|---|
| `split_tbl` | 172 | 0.0262 | 5.15 | 4.79 | 93.0% |
| `split_exp` | 468 | 0.0387 | 8.83 | 5.72 | 93.1% |

Both sit at **93% of their own entropy ceiling** — routing does not collapse and is
not the bottleneck on either side. `max_vio` rises with `n_experts` (more experts,
more room for one to be overloaded), which is expected and does not indicate worse
utilization. This closes the split as a knob for the study's purposes: the answer is
"lean expert-heavy, keep a minority table", and the exact peak is below the
resolution this compute budget can buy per arm.

### 9.3 What to build next, in order

Revised after round 18. Six items are now closed (`cg_dec` hardened and shown to
transfer to a third base; the "collide less" question answered twice;
composition tested and positive; `--moe-shared-d-ff` retired; the gate confirmed
not absorbed by the MoE; and the table/expert split walked — asymmetric, optimum
expert-heavier than 44/56, §9.2h). Every remaining item now needs new machinery
(`--moe-seed`, a no-Engram control) or more compute (a longer horizon, CORE);
there is no cheap iso arm left inside the combo.

1. **`--moe-seed`, then re-run `combo_iso`.** Every bar in the combo family is a
   lower bound, and round 17 proved it directly: `combo_nogate_s0`'s first logged
   router stats are byte-identical to `combo_iso_s0`'s across a different Engram
   config (§9.2g). Until the router is reseeded, no MoE-side comparison in this
   study can be resolved, including the top of the ranking table. Implementation
   work, no GPU cost to build.
2. **A no-Engram router control, with stats preserved.** Still unbuilt, and §9.2g
   makes it more interesting rather than less: `max_vio` ends at 8-9.3 on every
   combo arm, always-on capacity does *not* relieve it, and removing the Engram
   count gate makes it worse (7.97 -> 9.12 at a byte-identical router init). Whether
   any of that is Engram-induced is unanswerable without one 3814-step `mobius` run
   with `--moe-log-every=200`, because §1's logs did not survive the `/tmp` wipe (§6).
3. **Explain pooling — still the only large unexplained mechanism.** `share_tbl`
   reaches 0.7636 with each layer addressing 710,580 rows out of ONE 909.5M table;
   `perlayer_slot30` reaches 0.7676 with each layer addressing the same 710,580 rows
   out of TWO tables totalling 1819M. Same rows per read, half the parameters,
   0.0040 better, no per-n-gram row correspondence. Capacity, addressing consistency
   and matched-row joint training are all eliminated (§9.2c). Best probe: **a longer
   horizon on that one pair.** The leading hypothesis is that a pooled row is better
   *trained* per parameter because it collects gradient from every reader, which
   predicts the benefit shrinks toward convergence; 2.0B tokens cannot separate that
   from a representational effect, ~6B would.
4. **CORE downstream eval on the surviving checkpoints.** Inference-only, and there
   is now a point large enough to be worth testing: `combo_full` is 0.01578 below
   dense. Everything else in the study sits at ~0.001 (0.13% relative), which I
   expect CORE cannot resolve — that expectation is itself worth recording.
5. **External values: kNN-LM / RETRO-lite.** Still the only untried direction with
   published support, and §9.2f strengthens the pointer: the composition gain comes
   from the *address* carrying information the residual stream does not have, which
   is exactly what an external retrieval index supplies.

**Do not:** more quantizers or code widths for hidden-state addressing (13 arms);
richer row contents (4 measurements, neutral at fixed rows); fixed-exponent
whitening (5.5x harmful); longer n-gram orders in this form (1.8x worse); Fourier
bases on a count channel (monotone twice, curves correlating 0.9996 across seeds,
and the basis *hides* the linear coefficients it competes with); **a better hash
function** (tables saturated -- uniform hashing already near-optimal for max load,
§9.2d); **more hash heads** (0.13× across a 4x range, §9.2e); **a private always-on
expert alongside a routed pool** (0.65×, and it widens the seed bar to 1.04e-3 --
§9.2g retires `--moe-shared-d-ff`).

### 9.4 The methodological result

Six retractions, one cause: interpreting a gap smaller than that arm's own error
bar. The bars are **arm-dependent by 45×** (3.8e-5 to 1.7e-3), and they move with
the feature under test — rank-1 inflates them ~5×, the count gate 2.2× on the shared
table and 15× on the per-layer one, and cross-layer sharing inflates them 4.3× (the
earlier "sharing *collapses* variance 27×" claim was itself a casualty of this
error and is retracted in §0). Two n=2 estimates ran low by 2.6× and **8.8×**, the
second one destroying a headline result that had looked like 3.7× with complete
separation.

The sharpest version: **a suspiciously tight n=2 pair is the most dangerous
observation in the whole study.** `count_gate`'s two seeds landed 1.21e-4 apart,
which read as an unusually stable arm and licensed a confident claim; the truth was
7.5e-4 and the claim did not survive. There is no global error bar in this kind of
sweep, and there is no cheap one either. Measure per arm class, on the first two
arms, and prefer n>=3 before quoting anything.

What survived all of it, after rounds 10-13 pruned and then partly rebuilt it:
**token addressing over hidden-state addressing** (0.0040, 20x), and **cross-layer
sharing's mean effect** (-0.0037 to -0.0039, 9-12x, three framings, and *not*
reducible to capacity). Row capacity survives only in a much weaker form:
-0.000533 per doubling on the shared token arm (now over 8x of rows, after the
dbs correction), flat on per-layer, and the 0.00195 "law" was one contextual
measurement generalized too far. Sharing's mechanism has been narrowed by
elimination to *pooling*: not capacity (round 11), not addressing consistency
(15x smaller, and the wrong sign on a shared table), not joint training of matched
rows. The 2x2 that establishes this is fully iso and has n>=2 in every cell.

And one mechanism finally resolved: the **decoupled count gate**, -0.00110 pooled at
4.29x on two independent bases, which worked not by changing the mean but by halving
the variance of a mechanism that had been unmeasurable at 1.57x for 12 runs. It is
the only arm whose error bar has been shown *stable* (n=3 -> n=5 sd growth 1.20x).

**Then round 16 changed the shape of the answer.** Fifteen rounds searched inside
the hash-memory design and produced one resolved win out of eleven mechanism ideas.
Composing the memory with routed experts, at the same total parameter budget, gave
**-0.00236 at 4.66x with complete seed separation** (§9.2f) -- twice the effect of
the best mechanism found by searching inside, on the first attempt. The
methodological reading is uncomfortable and worth stating plainly: **after a
mechanism family has returned ten neutral results, the expected value of an
eleventh is lower than the expected value of composing it with something else that
already works.** The base rate was visible by round 10 and it took six more rounds
to act on.

Round 17 then tried the two obvious ways to push the composition further and both
came back negative: the *actual* Mobius component -- a private always-on expert per
layer, which neither §1's "mobius" arm nor round 16 had ever used -- is 0.65× from
the combo it was added to, and removing the count gate costs 0.00127, confirming
the gate survives composition rather than adding to it (§9.2g). So the composition
result is real but not a new frontier of its own: it is one point, and the only
cheap unexplored direction left in it is *how* to split the budget.

Four results, one qualified slope, and one sharply-posed open mechanism, from 82
runs. The largest of the four is the one that required no new mechanism at all --
and the twelve mechanism ideas tried inside the memory design are now thirteen,
still with one win.
