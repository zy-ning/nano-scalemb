# Does Engram memory actually remember? Releasing nano-scalemb

*We're open-sourcing nano-scalemb, a small nanochat-style research harness, and to
show what it's for, we use it to chase down one nagging question: when you bolt an
n-gram memory onto a transformer and the scores go up, is the model really *using*
the memory, or just enjoying the extra compute?*

> This is the reproducible follow-up to our
> [reproduction-and-reassessment of Engram](https://zhuanlan.zhihu.com/p/2027403480428558123).
> That write-up was itself a response to 栀染's widely-shared
> [《DeepSeek Engram 里没有记忆，就像 MoE 里没有专家》](https://zhuanlan.zhihu.com/p/2026419832371836848),
> which argued the celebrated memory table is mostly an elaborate regularizer.

## TL;DR

nano-scalemb is a minimal, readable harness for the whole training loop, and
everything below was done with it.

The idea we put under the microscope is **Engram + mHC**: an n-gram hash *memory*
(fixed hash addressing, learned contents) dropped into a multi-stream residual
transformer. On paper it helps. The real memory beats a memory-free baseline on
CORE (0.271 vs 0.263), ChatCORE (0.410 vs 0.377), ARC-Challenge (0.563 vs 0.496),
and more.

The catch shows up when you fill that memory with garbage. Swap the learned table
for frozen noise, or for a single row copied a million times, and on the chat-suite
scores a surprising amount of the lift still survives, even though those tables hold
nothing useful. (On the base metrics it doesn't survive, which is its own clue.) So
something subtler is going on, and downstream scores alone can't tell you what.

To find out we go inside the model two ways, both cheap and both CPU-only: a
**weight probe** that reads what each checkpoint learned straight from its
`state_dict`, and a **forward probe** that watches what the model actually does on
real tokens. Between them they tell a two-part story. The real memory genuinely
learns and uses an n-gram store, *and* a good chunk of the headline number is
plumbing, not contents.

All five figures below are interactive; hover for exact values.

## What nano-scalemb is

It's a cleaned-up [nanochat](https://github.com/karpathy/nanochat)-style harness for
embedding-scaling and Engram experiments, built to be the opposite of a framework:
plain scripts, dataclasses, and control flow you can read top to bottom and edit in
place. The model lives in `nano_scalemb/` (`gpt.py`, `engram.py`, `mhc.py`, and
friends), the pipelines are shell in `runs/`, and the analyses are Python in
`scripts/`. Every experiment below runs from here, and the two probes doing the
interpretability heavy lifting don't even need a GPU: a checkpoint and a few CPU
minutes is enough. (Install and quickstart are at the end.)

## The architecture under test: Engram + mHC

The model on the table is a **d24** decoder (24 layers, `n_embd=1536`, 12 heads,
32,768-token vocab, 2048 context, ~1.95 B params) trained on
[ClimbMix](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle) at ~9.5
tokens per param. It stacks two recent DeepSeek ideas, both reproduced faithfully:
[Engram](https://arxiv.org/abs/2601.07372) and
[mHC](https://arxiv.org/abs/2512.24880).

**mHC (Manifold-Constrained Hyper-Connections)** runs four persistent residual
streams (expansion rate *n* = 4) instead of one. Per block, a content-dependent
router emits three transforms: **H_pre** mixes the streams into the sub-layer's
input, **H_res** remixes the streams among themselves, and **H_post** writes the
output back. The "manifold-constrained" trick is that H_res is forced *doubly
stochastic* via Sinkhorn-Knopp (a Birkhoff-polytope projection), which conserves
signal energy across depth and dodges the instability plain Hyper-Connections can
hit. A learned `MHCHead` collapses the four streams before the LM head.
(`nano_scalemb/mhc.py`.)

**Engram** is the n-gram memory (DeepSeek's "conditional memory," a lookup-based
complement to MoE's compute sparsity). At a few layers it hashes the current 2- and
3-gram context into a big embedding table, looks it up, gates it against the hidden
state, and adds the result through a short depthwise causal conv. The key point for
what follows: the addressing (the hash) is fixed at init; only the table contents,
read gate, and projections are learned. "Fixed where to look, learned what's there."
(`nano_scalemb/engram.py`.)

In this study the memory sits at layers 2, 12, 18 (`memory_dim=1280`, 3-gram max,
8 heads per n-gram, 4 streams), emitting a per-stream contribution that the mHC
router places into the residual streams.

### The four knobs

The whole investigation rests on four checkpoints that are identical in every way
except for what's sitting in the memory table:

| Variant | Memory table | What it isolates |
|---|---|---|
| **mHC baseline** | Engram disabled | the backbone alone |
| **Real engram** | learned n-gram memory | the full method |
| **Randomized** | frozen `N(0,1)` noise, never trained | the read/gate path with a content-free but *distinct* payload |
| **Uniform** | every row identical | the path with no information at all |

The logic is simple. If what matters is the n-gram *content*, only Real should
help. If what matters is the *mechanism* (the extra gated compute and the extra
routed branch), then even Randomized and Uniform should help. Keep that fork in
mind; the rest of the post is really about which side wins.

## Does the memory help at all?

📊 **[`ablation_sweep.html`](./ablation_sweep.html)**: base + SFT metrics, four variants.

Start with the scoreboard. The real Engram is the best variant on almost
everything:

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

So far, so good: the memory earns its keep across CORE, val bpb, and the whole chat
suite, with GSM8K the only place the baseline wins.

Now look one column over. Uniform and Randomized hold *zero* and *no useful*
information, and they still beat the baseline on ChatCORE (0.395 and 0.385 vs
0.377), on ARC, on MMLU. An empty table has nothing to teach the model, yet the
model comes out ahead anyway. (Not everywhere, though: on the base metrics both
ablations fall *below* baseline, CORE 0.252/0.253 vs 0.263 and val bpb worse too.
The first hint that content and pathway pull on different scores.) That's the puzzle
that sets up the rest of the post: the scores say "memory good," but they clearly
can't be measuring only the memory. Time to stop trusting the scoreboard and go look
inside.

```bash
# Reproduce: train all four variants on the same backbone, then plot
bash runs/run_engram_ablation_sweep_21218_mhc.sh
python docs/blog/plot_ablation_sweep.py
```

## How many memory layers, and where?

📊 **[`layer_count_sweep.html`](./layer_count_sweep.html)**: train loss, val bpb,
CORE, and ChatCORE against the number and placement of Engram layers, with the
best-at-each-count frontier drawn in.

Sweeping 1-, 2-, and 3-layer placements gives roughly the shape you'd expect: more
memory layers help loss and bpb with diminishing returns, and where you put them
matters. The strongest configs spread the memory across early, mid, and late depth
instead of bunching it up.

- Best 1-layer: `3` → train loss 2.352, val bpb 0.709
- Best 2-layer: `3,12` → train loss 2.328, val bpb 0.705
- Best 3-layer (CORE): `8,12,17` → CORE 0.269

The dashed lattice in the figure connects each config to its supersets
(`3 → 3,12 → 3,12,17`), so you can see at a glance whether adding a layer to a
given base actually pays for itself.

The frontier does flatten by three layers, but there's a caveat worth stating
plainly, because it's easy to misread the flattening as the architecture topping
out. Every config here trains for the same number of steps. A 3-layer Engram has
about three times as many memory tables to fill on the same token budget, so each
table sees fewer effective updates. Some of the reason the 3-layer configs don't
pull further ahead is almost certainly that the extra memory is simply
*undertrained*, not that it has nothing left to give. Read the flat frontier as a
lower bound: with a longer schedule, or one that matches updates per table, the gap
could widen. With that caveat in hand, the flattening is still consistent with the
hint from the ablation: part of what we're buying is capacity and compute, and that
part saturates first.

```bash
# Reproduce: sweep Engram placement at 1 / 2 / 3+ layers, then plot
bash runs/run_one_layer_sweep_mhc.sh
bash runs/run_two_layer_sweep_mhc.sh
bash runs/run_four_layer_sweep_mhc.sh   # extends the best 3-layer config
python docs/blog/plot_layer_count_sweep.py
```

## Is the read actually causal?

📊 **[`donor_probe.html`](./donor_probe.html)**: swap the memory's input, watch
the target token.

Here's a clean test the architecture makes easy. The Engram reads from a token
stream (`engram_input_ids`) that's normally just the prompt. Nothing stops us from
holding the prompt fixed and feeding the memory a *different* "donor" text (one that
matches, one that's adversarial, one that's unrelated) and watching the target
token's logit move. If the read is doing nothing, the logit won't budge.

It budges. Across three factual-recall cases (*France → Paris*, *Gold → Au*,
*largest planet → Jupiter*) the read is clearly live: swapping the donor moves the
target's logit by as much as −1.4, a clear and repeatable dent, though (as the two
invariants below show) never quite enough to close the margin to rank-2. And two
things hold in every single case-and-donor combination:

1. The change is always a *decrease*. Feeding the prompt its own memory gives the
   highest target logit; any other donor only lowers confidence.
2. The prediction never flips. The target token stays rank-1 in all nine
   combinations.

So the read is genuinely wired in and content-sensitive, but on facts the backbone
already knows cold it behaves as a confidence dial, not a decision-maker. Real, but
gentle.

```bash
# Reproduce: swap the donor stream on the real-Engram checkpoint, then plot
bash runs/run_engram_donor_probe_21218.sh   # wraps scripts/engram_donor_eval.py
python docs/blog/plot_donor_probe.py
```

## What did it learn? Reading the weights

📊 **[`weight_probe.html`](./weight_probe.html)** · code: `scripts/engram_weight_probe.py`

The architecture hands us a freebie here. Almost every Engram knob starts at a
value we know exactly: `value_proj` at 0, the short conv at 0, the read-gate
projection at a uniform scale, the table at `N(0,1)`. So how far a trained weight
has drifted from its init is a direct, training-free readout of what the model
chose to lean on. No forward pass needed; you can read it straight off the
checkpoint on a CPU. That's the weight probe.

It tracks three things per variant: how hard the layer writes into the residual
(`value_proj` RMS, init 0), how much the read path grew (`stream_key_proj` norm
over init), and whether the table actually filled up (the embedding row-norm
distribution).

The three variants separate cleanly:

- **Real** grows its memory about 50× past init (row-norm ~440 vs ~9) and opens
  both the write (`value_proj` RMS ≈ 0.13–0.15) and the read gate (~4–5× init).
- **Randomized** can't change its frozen table, so instead it cranks the read gate
  *even harder* (~5× init), straining to read noise it can't improve.
- **Uniform** goes the other way entirely: it collapses the write path
  (`value_proj` RMS down to ~0.005–0.03) and pulls the read gate below init. An
  information-free table is worth ignoring, and the weights say so out loud.

That's a clean three-way split. But weights only tell you what the model *could*
do, not what actually happens when text flows through.

```bash
# Reproduce: read drift-from-init straight from the checkpoints (CPU, no forward)
python -m scripts.engram_weight_probe \
    --checkpoint REAL_CKPT:STEP \
    --checkpoint RANDOMIZE_CKPT:STEP \
    --checkpoint UNIFORM_CKPT:STEP \
    --output-dir docs/blog/weight_probe
python docs/blog/plot_weight_probe.py
```

## What does it do? The forward pass

📊 **[`forward_probe.html`](./forward_probe.html)** · code: `scripts/engram_forward_probe.py`

The forward probe pushes a single small batch of real validation tokens through
each checkpoint and reads the signals that only exist once activations are flowing.
It stays cheap in the same spirit as the weight probe: no training, no backprop,
about 4k tokens on CPU in a couple of minutes. The capture works through light
wrappers that recompute each module's internals from its own trained weights and
always hand back the model's true output, so the forward pass and the reported loss
are untouched.

Three panels, all on real text.

**The read gate.** Per token and per stream the gate is a sigmoid in (0,1), and
its *shape* is the tell:

| Variant (layer 12) | gate Q1–median–Q3 | shape |
|---|---|---|
| **Real** | 0.23 – 0.37 – 0.53 | graded, sits low-to-mid |
| **Randomized** | 0.01 – 0.55 – 0.995 | slammed to both rails (0/1) |
| **Uniform** | 0 – 0 – 0 | shut |

Real opens a graded, content-dependent gate. Randomized's gate flies to its
extremes; it can't grade tokens it can't tell apart, so it flails between fully open
and fully closed. Uniform just keeps the gate shut at depth.

**Gate selectivity** (the across-token std) confirms the same ranking. And the
third panel, how much the memory actually *writes*, is the punchline:

| Variant | Engram output RMS (layers 2/12/18) |
|---|---|
| **Real** | ~100 – 145 |
| **Randomized** | ~4 – 6 |
| **Uniform** | ~0.3 – 6 |

The real memory writes on the order of 25× more into the residual than either
ablation. This is the inference-time mirror of what the weights showed:
Randomized's payload has a tiny norm because frozen noise never grew, and Uniform
zeroed its own write, so even when their gates crack open there's almost nothing on
the other side. And as a kicker, Uniform's `MHCHead` collapses onto a single stream
(`[0, 0, 0, 1.0]`), routing *away* from the very streams its dead memory feeds into.

The single-batch loss lines up with all of this too (real best, randomized worst),
matching the full-validation bpb from the ablation.

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

## Putting it together

Line the evidence up and a coherent picture appears:

- **Downstream:** real wins, but the information-free ablations recover much of the
  *chat-suite* lift while falling *below* baseline on the base metrics (CORE, val
  bpb). Content and pathway pull on different scores.
- **Inside the model:** the read is live and content-sensitive (if only modulatory
  on facts already known); real grows and opens its memory while uniform shuts it
  down and randomized strains against frozen noise; and on real tokens real writes a
  big, graded, selective contribution where the ablations write next to nothing.

Both halves of the answer are true at once. The real Engram genuinely learns and
uses an n-gram memory; that much is unambiguous from the inside (the weight and
forward probes), and it does buy real downstream quality. At the same time, a
meaningful slice of the headline lift is the *mechanism* rather than the *content*:
the extra gated, routed compute branch helps even when the table is empty. The
layer sweep keeps us honest here: don't lean too hard on the saturation as proof of
that, since the deeper stacks are also undertrained on a step-matched budget. Either
way, it's the inside-the-model probes that let us hold both claims together; the
scoreboard alone would have quietly hidden the second one.

This is also where we land on the claim that started the thread, that the memory
table is "just regularization." We think that's directionally right but too strong.
The content is *not* inert: the real table grows ~50× and writes ~25× more than any
ablation, and the base-side metrics move with it. But most of the *headline* lift
really is the pathway, not the lookup. And the balance isn't fixed; it shifts with
where you put the memory. In companion runs at other placements (the
[知乎 report](https://zhuanlan.zhihu.com/p/2027403480428558123)) a sparse,
mid-to-late layout makes the content's contribution clean and clearly real, while
spreading the memory densely across depth narrows the real-vs-empty gap until the
pathway dominates.

So, to answer the title head-on: yes, it remembers. The probes leave no doubt that a
real, content-dependent store gets learned, read, and written. But on facts the
backbone already knows cold that remembering only nudges confidence, much of the
headline score is the routed pathway rather than the recall, and how much the content
itself shows up in your metrics has no topology-free answer. Only the inside view
pulls those threads apart, which is the whole point.

If there's one takeaway for anyone bolting a memory module onto a model, it's this:
measure the empty-table baseline, and look inside. A score that went up is not, by
itself, proof that your memory remembers anything.

---

## Get nano-scalemb

Everything above ships in the repo. Each section's `Reproduce` block is a
copy-paste command, and the two probes need nothing more than a checkpoint and a
CPU.

```bash
uv venv && source .venv/bin/activate
uv sync --extra gpu          # or --extra cpu
bash runs/speedrun.sh        # train a baseline, then go probe it
```

Run a sweep, then turn the probes on your own memory module. If a score goes up,
you'll now know how to ask the model whether it actually earned it.
