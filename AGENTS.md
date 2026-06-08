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
