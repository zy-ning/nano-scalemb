# AGENTS.md

This file guides coding agents working in `nano_scalemb/`.
It is intentionally repository-specific and grounded in the current codebase.
When in doubt, prefer matching existing local patterns over introducing new abstractions.

## Repository Summary

- `nano_scalemb` is a minimal, hackable Python research harness for tokenization, pretraining, SFT, evaluation, inference, and chat UI.
- The repo favors direct scripts, dataclasses, and explicit control flow over framework-heavy architecture.
- Most operational workflows are driven by `runs/*.sh` and `scripts/*.py`.
- Core reusable logic lives in `nano_scalemb/`.
- Tests live in `tests/` and use `pytest`.

## Source of Truth

Use these files as the primary authority before making changes:

- `README.md` for the main workflow and reference commands.
- `pyproject.toml` for Python version, dependencies, and pytest discovery.
- `runs/nano_engram_speedrun.sh` for the main nano-engram experiment pipeline.
- `runs/speedrun.sh` for the baseline GPU pipeline.
- `runs/runcpu.sh` for the CPU/MPS path.
- `runs/miniseries.sh` and `runs/scaling_laws.sh` for experiment patterns.
- `scripts/base_train.py`, `scripts/base_eval.py`, `scripts/chat_sft.py`, `scripts/chat_eval.py`, `scripts/chat_web.py` for script conventions.
- `nano_scalemb/common.py`, `nano_scalemb/gpt.py`, `nano_scalemb/engine.py`, `nano_scalemb/engram.py` for core code style.
- `nano_scalemb/moe/` for the MoE / Mobius arms (`MoEConfig.share_blocks` picks per-layer vs cross-layer-shared experts); ported from the sibling `nanochat-moe` harness, Apache-2.0, attributed per file.
- `runs/run_conditional_capacity_sweep_mhc.sh` for the five-arm dense/engram/engram-shared/moe/mobius comparison.
- `tests/test_engine.py` and `tests/test_attention_fallback.py` for mature test style.

## Environment and Tooling

- Python requirement: `>=3.10` from `pyproject.toml`.
- Dependency management uses `uv`.
- Optional dependency groups are `cpu` and `gpu` for Torch index selection.
- Tests use `pytest`.
- There is no repo-evidenced `ruff`, `black`, `isort`, `mypy`, `tox`, `nox`, or `make` workflow.
- There is no packaging build flow documented in the repo.

## Setup Commands

```bash
uv venv
source .venv/bin/activate
uv sync --extra gpu
```

For CPU-only or MPS setups:

```bash
uv venv
source .venv/bin/activate
uv sync --extra cpu
```

## Primary Run Commands

```bash
bash runs/nano_engram_speedrun.sh
bash runs/speedrun.sh
bash runs/runcpu.sh
python -m scripts.chat_web
```

## Test Commands

- `python -m pytest`
- `python -m pytest tests/test_engine.py -v`
- `python -m pytest tests/test_attention_fallback.py -v -s`

## Change Strategy

- Make the smallest coherent change that solves the requested problem.
- Preserve readability over cleverness.
- Match nearby patterns before introducing a new one.
- Do not introduce a new dependency injection framework or config framework.
- Weights shared across layers (Mobius expert pools, the shared Engram memory
  table) must be registered on the `GPT` exactly once and handed to layers by
  reference in a plain list — never as a submodule on each consumer. The model is
  built on `meta` and then `to_empty()`-ed, and checkpoints load with
  `assign=True`, so a second registration silently gives each copy its own
  storage and unshares the weights.
- Third-party autograd kernels (e.g. scattermoe) are often not autocast-aware:
  they mix the autocast dtype with fp32 saved tensors and fail in *backward*, not
  forward. Wrap such calls in `torch.amp.autocast(..., enabled=False)` and cast
  operands explicitly. Watch out for autocast's fp32-promotion ops (`Tensor.sum`)
  silently changing a fallback path's output dtype relative to the fast path.
- When adding a variant that changes how much weight is *active* per token,
  update `GPT.estimate_flops` and `GPT.num_scaling_params` with it. Otherwise MFU,
  the reported training FLOPs, and the scaling-law token horizon are all wrong
  for that arm, and it stops being comparable to the others.
- **Every new buffer needs an explicit reset on the `init_weights` path.**
  `to_empty()` fills buffers with garbage and `nn.Module.__init__` defaults do not
  survive it, so a buffer nothing re-initializes starts at whatever was in that
  memory. This shipped once: the PQ codebook began at ~1e38 and the commitment
  loss read 3.2e26. Buffers that *define* behaviour (LSH bit thresholds, Engram
  hit rates) should also be persistent, or a resumed run silently changes model
  behaviour at the resume step.
