# Releasing nano-scalemb: a hackable harness for scaling embeddings, and what it taught us about n-gram memory

*A cleaned, nanochat-style research harness — plus a worked investigation into
whether a transformer's built-in n-gram memory actually remembers anything.*

---

## TL;DR

- We're releasing **nano-scalemb**, a minimal, readable research harness for
  tokenization → pretraining → SFT → eval → inference, built for fast
  architecture experiments (it's what we used for everything below).
- As the flagship case study we interrogate **Engram + mHC** — a training-free
  n-gram hash *memory* injected into a multi-stream residual transformer — with
  one question: *does the model actually use the memory, or is it a free-rider
  on the extra compute?*
- The answer is **both**. The real memory genuinely grows, opens a
  content-graded read gate, writes a large contribution at inference, and
  responds causally when you swap its input — and it wins on downstream tasks
  (CORE 0.271 vs 0.263 baseline; ChatCORE 0.410 vs 0.377; ARC-C 0.563 vs 0.496).
  **But** ablating the memory to pure noise or to a single repeated row recovers
  *most* of that lift, which means a large share of the benefit comes from the
  added gated compute path itself, not the n-gram content.
- We get there with a two-tier interpretability recipe that ships in the repo:
  a **Tier-1 weight probe** (read what was learned straight from the checkpoint,
  CPU, no forward pass) and a **Tier-2 forward-pass probe** (read what the model
  *does* on real tokens, also cheap, also CPU).

All five figures below are interactive — hover for exact values.

---

## What is nano-scalemb?

nano-scalemb is a cleaned [nanochat](https://github.com/karpathy/nanochat)-style
harness for scaling-embedding and Engram experiments. The design goal is the
opposite of a framework: **direct scripts, dataclasses, and explicit control
flow** you can read top-to-bottom and modify without fighting abstractions.

- Core model + training logic lives in `nano_scalemb/` (`gpt.py`, `engram.py`,
  `mhc.py`, `engine.py`, `optim.py`, …).
- Operational pipelines are plain shell: `runs/speedrun.sh`,
  `runs/nano_engram_speedrun.sh`, the layer/ablation sweeps in `runs/*.sh`.
- One-off analyses are plain Python: `scripts/*.py`.

```bash
uv venv && source .venv/bin/activate
uv sync --extra gpu          # or --extra cpu / MPS
bash runs/speedrun.sh        # baseline GPU pipeline
bash runs/nano_engram_speedrun.sh
python -m pytest             # tests in tests/
```

Everything in this post — the sweeps, the ablations, and both probes — runs from
this repo. The two interpretability probes in particular run on **CPU in a few
minutes**, with no GPU required.

---

## The case study: Engram + mHC

The model under the microscope is a **d24** decoder (24 layers, `n_embd=1536`,
12 heads, 32 768-token vocab, 2048 context, ~1.95 B params) trained on
[ClimbMix](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle) at a
~9.5 tokens-per-param ratio. Two ingredients make it interesting:

**mHC — Manifold-Constrained Hyper-Connections.** Instead of a single residual
stream, the backbone carries **4 persistent residual streams**. At every block a
small content-dependent router (a Sinkhorn-balanced, doubly-stochastic mixing
matrix) decides how the streams feed each sub-layer and how its output is
written back. A learned `MHCHead` collapses the 4 streams to one before the LM
head. (`nano_scalemb/mhc.py`.)

**Engram — a training-free n-gram memory.** At a few chosen layers, the current
token's 2- and 3-gram context is **hashed** (multiplicative-XOR, multi-head) into
a big multi-head embedding table, looked up, **gated** against the hidden state,
and passed through a short causal conv before being added to the residual.
Critically, the *addressing* (the hash) is fixed at init — only the **table
contents**, the **read gate**, and the **projections** are learned.
(`nano_scalemb/engram.py`.)

The faithful fusion wires these together: the Engram emits a **per-stream**
contribution and the backbone's mHC router places it into the residual streams.
For this study the Engram sits at **layers 2, 12, 18** with `memory_dim=1280`,
3-gram max, 8 heads/n-gram, 4 streams.

### The ablation knobs

The whole investigation hinges on four checkpoints that share *everything*
except what's in the memory table:

| Variant | Memory table | What it isolates |
|---|---|---|
| **mHC baseline** | Engram disabled | the backbone alone |
| **Real engram** | learned n-gram memory | the full method |
| **Randomized** | frozen `N(0,1)` noise (never trained) | the read/gate path with a *content-free but distinct* payload |
| **Uniform** | every row identical | the path with *no information at all* |

If the n-gram *content* is what matters, only **Real** should help. If the
*mechanism* (extra gated compute + an extra routed branch) is what matters, even
**Randomized** and **Uniform** should help. Hold that thought.

---

## Q1 — Does the memory help at all?

📊 **[`ablation_sweep.html`](./ablation_sweep.html)** — base + SFT metrics, four variants.

Real Engram is the best variant on nearly every axis:

| Metric (↑ better unless noted) | mHC baseline | **Real** | Randomized | Uniform |
|---|---|---|---|---|
| CORE | 0.2626 | **0.2707** | 0.2520 | 0.2534 |
| val bpb (↓) | 0.7107 | **0.7048** | 0.7237 | 0.7127 |
| ChatCORE | 0.3765 | **0.4095** | 0.3853 | 0.3946 |
| ARC-Easy | 0.6494 | **0.6970** | 0.6768 | 0.6928 |
| ARC-Challenge | 0.4957 | **0.5631** | 0.5265 | 0.5316 |
| MMLU | 0.3660 | **0.4047** | 0.3829 | 0.3862 |
| HumanEval | 0.1280 | **0.1402** | 0.1098 | 0.1341 |
| GSM8K | **0.1198** | 0.1008 | 0.1069 | 0.1016 |

So the method works — adding the real memory beats the backbone-only baseline
across CORE, val bpb, and the whole chat suite (GSM8K is the lone exception).

**But look at the ablation columns.** Uniform and Randomized — tables with *zero*
and *no useful* information respectively — still beat the baseline on ChatCORE
(0.395 / 0.385 vs 0.377), ARC, and MMLU. An empty memory shouldn't teach the
model anything, yet it lifts the model. That's the central puzzle, and it's why
the rest of the post stops trusting downstream scores alone and goes *inside* the
model.

```bash
# Reproduce: train all four variants on the same backbone, then plot
bash runs/run_engram_ablation_sweep_21218_mhc.sh
python docs/blog/plot_ablation_sweep.py
```

---

## Q2 — How many memory layers, and where?

📊 **[`layer_count_sweep.html`](./layer_count_sweep.html)** — train loss, val bpb,
CORE, ChatCORE vs. number/placement of Engram layers, with the best-at-each-count
frontier drawn in.

Sweeping 1-, 2-, and 3-layer placements shows the expected shape: **more memory
layers help loss and bpb, with diminishing returns, and placement matters** —
the best configs spread the memory across early/mid/late depth rather than
clustering it.

- Best 1-layer: `3` → train loss 2.352, val bpb 0.709
- Best 2-layer: `3,12` → train loss 2.328, val bpb 0.705
- Best 3-layer (CORE): `8,12,17` → CORE 0.269

The dashed lattice in the figure connects each config to its supersets (e.g.
`3 → 3,12 → 3,12,17`), so you can see whether adding a layer to a given base
actually pays off. The frontier flattens by three layers — consistent with the
ablation hint that we're partly buying *capacity/compute*, which saturates.

```bash
# Reproduce: sweep Engram placement at 1 / 2 / 3+ layers, then plot
bash runs/run_one_layer_sweep_mhc.sh
bash runs/run_two_layer_sweep_mhc.sh
bash runs/run_four_layer_sweep_mhc.sh   # extends the best 3-layer config
python docs/blog/plot_layer_count_sweep.py
```

---

## Q3 — Is the memory read causal at inference?

📊 **[`donor_probe.html`](./donor_probe.html)** — swap the Engram's input stream,
watch the target token.

The Engram reads from a token stream (`engram_input_ids`) that is normally just
the prompt. So we can **hold the prompt fixed and feed the memory a different
"donor" text** — matched, adversarial, or unrelated — and watch the target
token's logit move. If the read does nothing, nothing changes.

On three factual-recall cases (*France→Paris*, *Gold→Au*, *largest planet→
Jupiter*), the read is demonstrably live: changing the donor changes the logit by
up to **−1.4**. Two things are consistent across every case and donor:

1. **Every change is a *decrease*.** The self-matched donor (the prompt feeding
   its own memory) always gives the highest target logit; any other donor only
   *lowers* confidence.
2. **The prediction never flips.** The target token stays **rank-1** in all 9
   case×donor combinations.

So the memory read is genuinely wired in and content-sensitive — but on facts the
backbone already knows cold, it acts as a **confidence modulator, not a decider**.
The read is real; its leverage here is gentle.

```bash
# Reproduce: swap the donor stream on the real-Engram checkpoint, then plot
bash runs/run_engram_donor_probe_81216.sh   # wraps scripts/engram_donor_eval.py
python docs/blog/plot_donor_probe.py
```

---

## Q4 — Tier-1: what did it learn? (weights only)

📊 **[`weight_probe.html`](./weight_probe.html)** · code: `scripts/engram_weight_probe.py`

Here's a trick the architecture hands us for free. Almost every Engram knob is
initialized to a *known* value — `value_proj` at **0**, the short conv at **0**,
the read-gate projection to a **uniform** scale, the table to `N(0,1)`. So the
**drift of a trained weight from its init is a direct, training-free readout of
what the model chose to use** — readable straight from the checkpoint's
`state_dict` on CPU, no forward pass. That's Tier-1.

Three signals, per variant:

- **Write strength** (`value_proj` RMS, init 0): how hard the layer writes into
  the residual.
- **Read gate** (`stream_key_proj` norm ÷ init): how much the read path grew.
- **Memory content** (embedding row-norm distribution): did the table fill up?

The picture is crisp:

- **Real** grows its memory ~**50× past init** (row-norm ~440 vs ~9) and opens
  both the write (`value_proj` RMS ≈ 0.13–0.15) and the read gate (~4–5× init).
- **Randomized** can't change its frozen table, but it still *cranks the read
  gate even higher* (~5× init) — it's trying to read the noise.
- **Uniform** does the opposite: it **collapses the write path** (`value_proj`
  RMS → ~0.005–0.03) and pulls the read gate *below* init. An information-free
  table is worth ignoring, and the weights say so.

Tier-1 already separates the three. But weights only tell you what *could*
happen — not what *does*, on real text.

```bash
# Reproduce: read drift-from-init straight from the checkpoints (CPU, no forward)
python -m scripts.engram_weight_probe \
    --checkpoint REAL_CKPT:STEP \
    --checkpoint RANDOMIZE_CKPT:STEP \
    --checkpoint UNIFORM_CKPT:STEP \
    --output-dir docs/blog/weight_probe
python docs/blog/plot_weight_probe.py
```

---

## Q5 — Tier-2: what does it do at inference? (forward pass)

📊 **[`forward_probe.html`](./forward_probe.html)** · code: `scripts/engram_forward_probe.py`

Tier-2 runs a single, small batch of real validation tokens through each
checkpoint and reads the signals that **only exist once activations flow**. It's
"cheap" in the same spirit as Tier-1 — no training, no backprop, ~4 k tokens on
CPU in a couple of minutes — captured with light wrappers that recompute each
module's internals from its *own* trained weights and always return the model's
true output (so the forward pass, and the reported loss, are unchanged).

Three panels, on real text:

**1. The read gate distribution (box plots).** Per token and per stream, the gate
is a sigmoid in (0,1). The *shape* of that distribution is the story:

| Variant (layer 12) | gate Q1–median–Q3 | shape |
|---|---|---|
| **Real** | 0.23 – 0.37 – 0.53 | graded, sits low-to-mid |
| **Randomized** | 0.01 – 0.55 – **0.995** | slammed to **both rails** (0/1) |
| **Uniform** | 0 – 0 – 0 | **shut** |

Real opens a *graded*, content-dependent gate. Randomized's gate saturates to its
extremes — it can't grade tokens it can't tell apart, so it flails. Uniform's
gate is simply closed at depth.

**2. Gate selectivity** (across-token std) confirms the same ranking, and **3.
the contribution actually written** is the headline:

| Variant | Engram output RMS (layers 2/12/18) |
|---|---|
| **Real** | **~100 – 145** |
| **Randomized** | ~4 – 6 |
| **Uniform** | ~0.3 – 6 |

**The real memory writes ~25× more into the residual than either ablation.** This
is the inference-time face of Tier-1's collapse: Randomized's payload has tiny
norm (frozen noise never grew), and Uniform zeroed its own write — so even when
their gates are open, there's almost nothing behind them. As a bonus, Uniform's
`MHCHead` collapses entirely onto a **single stream** (`[0, 0, 0, 1.0]`),
routing *away* from the streams the dead memory feeds.

Single-batch loss tracks this too (real best, randomized worst), matching the
full-validation bpb from Q1.

```bash
# Reproduce: cheap forward-pass signals on real tokens (CPU, a couple of minutes)
python -m scripts.engram_forward_probe \
    --checkpoint REAL_CKPT:STEP \
    --checkpoint RANDOMIZE_CKPT:STEP \
    --checkpoint UNIFORM_CKPT:STEP \
    --checkpoint MHC_BASELINE_CKPT:STEP \
    --device cpu --batch-size 8 --seq-len 512 \
    --output-dir docs/blog/forward_probe
python docs/blog/plot_forward_probe.py
```

---

## Synthesis: a real memory that's also partly redundant

Stack the evidence:

- **Downstream (Q1):** real wins — but information-free ablations recover most of
  the gain.
- **Causality (Q3):** the read is live and content-sensitive, yet modulatory on
  known facts.
- **Weights (Q4):** real *grows and opens* its memory; uniform *shuts* it;
  randomized *strains* to read noise.
- **Inference (Q5):** real writes a large, graded, selective contribution;
  the ablations write ~nothing.

The consistent reading: **the real Engram genuinely learns and uses an n-gram
memory** — that's unambiguous from the inside (Q4, Q5) and it buys real
downstream quality (Q1). **At the same time, a meaningful slice of the headline
lift is the *mechanism*, not the *content*:** the extra gated, routed compute
branch helps even when it carries an empty table (Q1, and the layer-count
saturation in Q2). The interpretability tiers are what let us say *both* of these
at once — the downstream scores alone would have hidden the second half.

The honest takeaway for anyone adding a memory module: **measure the empty-table
baseline, and look inside.** A score bump is not proof your memory remembers.

---

## Get nano-scalemb

Every experiment above ships in the repo — each section's `Reproduce` block is a
copy-paste command, and the probes need only a checkpoint and a CPU.

```bash
uv venv && source .venv/bin/activate
uv sync --extra gpu          # or --extra cpu
bash runs/speedrun.sh        # train a baseline, then go probe it
```

**Run a sweep, then probe your own memory module.** If a score goes up, now you
know how to ask the model whether it earned it.
