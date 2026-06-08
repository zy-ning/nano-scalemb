# nano-scalemb

**A minimal, hackable research harness for scaling embeddings and Engram experiments** —
a cleaned [nanochat](https://github.com/karpathy/nanochat)-style pipeline you can
read top-to-bottom and modify without fighting a framework.

> 📝 **Release write-up & case study:** [`docs/blog/blog.md`](docs/blog/blog.md)
> — *Does a transformer's n-gram memory actually remember?* A full Engram + mHC
> investigation (ablations → layer sweep → causal donor probe → two-tier
> interpretability probes), with interactive figures.

## What it is

nano-scalemb covers the whole loop — **tokenization → pretraining → SFT → RL →
evaluation → inference/chat** — in direct scripts, dataclasses, and explicit
control flow. The design goal is the opposite of a framework: code you can read
in one sitting and edit in place. It's the harness behind the Engram + mHC study
in the blog post above.

**Highlights**

- **d-series transformers** with FP8 training, sliding-window attention, Muon +
  AdamW optimizers (`nano_scalemb/gpt.py`, `optim.py`, `engine.py`).
- **mHC** — Manifold-Constrained Hyper-Connections: persistent multi-stream
  residuals with a Sinkhorn-balanced, content-dependent router
  (`nano_scalemb/mhc.py`).
- **Engram** — a training-free n-gram hash *memory* injected at chosen layers,
  fused as a per-stream branch through mHC (`nano_scalemb/engram.py`).
- **Two-tier interpretability probes** for the Engram (both cheap, both CPU):
  - **Tier-1 weight probe** — `scripts/engram_weight_probe.py`: reads what was
    learned straight from the checkpoint (no forward pass).
  - **Tier-2 forward probe** — `scripts/engram_forward_probe.py`: reads what the
    model *does* on real tokens (read gate, contribution, head mix, bpb).
- **Ready-made sweeps** for Engram layer count/placement and payload ablations
  (`runs/run_*_sweep_*.sh`).

## Setup

Dependencies are managed with [`uv`](https://docs.astral.sh/uv/); Torch is pinned
to 2.9.1 with CPU or CUDA 12.8 selected via extras.

```bash
uv venv
source .venv/bin/activate
uv sync --extra gpu        # GPU (CUDA 12.8)
# uv sync --extra cpu      # CPU / MPS
```

## Quickstart

```bash
bash runs/speedrun.sh              # baseline GPU pipeline (train → eval)
bash runs/nano_engram_speedrun.sh  # the Engram + mHC pipeline
bash runs/runcpu.sh                # CPU / MPS path
python -m scripts.chat_web         # chat with a trained checkpoint
```

## Repository layout

| Path | What's there |
|---|---|
| `nano_scalemb/` | Core library — `gpt.py`, `engram.py`, `mhc.py`, `engine.py`, `optim.py`, `dataloader.py`, `tokenizer.py`, `checkpoint_manager.py`, … |
| `scripts/` | Entry points — `base_train.py`, `base_eval.py`, `chat_sft.py`, `chat_rl.py`, `chat_eval.py`, `chat_web.py`, and the Engram probes |
| `runs/` | Shell pipelines and experiment sweeps; `runs/reports/` holds generated run reports (gitignored) |
| `tests/` | `pytest` suite (`test_engine.py`, `test_engram.py`, `test_mhc.py`, …) |
| `docs/` | The release blog and interactive figures (`docs/blog/`) |
| `AGENTS.md` | Conventions for contributors and coding agents |

## The Engram + mHC case study

The flagship investigation lives in [`docs/blog/blog.md`](docs/blog/blog.md)
and asks whether the model actually *uses* its n-gram memory or just free-rides on
the extra compute. Each step has an interactive figure and a copy-paste reproduce
command:

| Question | Figure | Reproduce |
|---|---|---|
| Does the memory help? | `docs/blog/ablation_sweep.html` | `runs/run_engram_ablation_sweep_21218_mhc.sh` |
| How many layers, and where? | `docs/blog/layer_count_sweep.html` | `runs/run_{one,two,four}_layer_sweep_mhc.sh` |
| Is the read causal? | `docs/blog/donor_probe.html` | `runs/run_engram_donor_probe_81216.sh` |
| What did it learn? (weights) | `docs/blog/weight_probe.html` | `scripts/engram_weight_probe.py` |
| What does it do? (inference) | `docs/blog/forward_probe.html` | `scripts/engram_forward_probe.py` |

The probes are intentionally cheap — a checkpoint and a CPU are enough:

```bash
# Tier-1: drift-from-init, straight from the state_dict (no forward pass)
python -m scripts.engram_weight_probe --checkpoint CKPT:STEP \
    --output-dir docs/blog/weight_probe

# Tier-2: forward-pass signals on a small batch of real tokens
python -m scripts.engram_forward_probe --checkpoint CKPT:STEP \
    --device cpu --batch-size 8 --seq-len 512 \
    --output-dir docs/blog/forward_probe
```

Figures are Plotly; re-render any of them with the matching
`docs/blog/plot_*.py`.

## Tests

```bash
python -m pytest                       # full suite
python -m pytest tests/test_engine.py -v
python -m pytest tests/test_engram.py tests/test_mhc.py -v
```

## Contributing

See [`AGENTS.md`](AGENTS.md) for repository conventions: prefer the smallest
coherent change, match nearby patterns, and keep readability over cleverness. No
new config/DI frameworks.

## License

[MIT](LICENSE).
