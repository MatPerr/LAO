# LAO

Latent Architecture Optimization (LAO) is a research codebase for neural architecture search in a latent space.

## What This Repo Does

The system is organized into four parts:

1. Architecture graphs (`lao/graph`)
- Graph representation (`ArcGraph`), validity checks, constraints
- Graph-to-blueprint compilation
- Graph-to-PyTorch / graph-to-Keras builders
- Graph visualization and memory animation utilities

2. Architecture embedding (`lao/embedding`)
- `ArcAE` autoencoder for architecture vectors
- Latent-space predictor heads for params/FLOPs/BBGP
- Dataset generation/loading utilities for graph vectors

3. Candidate training/evaluation (`lao/candidate_eval`)
- Candidate model training loops and baselines
- FLOPs/latency/profile helpers
- Tensor release / memory-analysis helpers

4. NAS outer loop (`lao/nas`)
- Latent-space Bayesian optimization loop
- EI / logEI / PiEI acquisition variants
- Candidate proposal, evaluation, GP updates, logging

## Repository Layout

```text
lao/
  graph/
  embedding/
  candidate_eval/
  nas/
```

## Environment Setup (uv + direnv)

Prerequisites:
- `uv`
- `direnv`
- Homebrew `graphviz` (needed for `pygraphviz` headers)

Already configured in this repo:
- `pyproject.toml`
- `uv.lock`
- `.python-version` (`3.11`)
- `.envrc` (auto-activates `.venv`)

First-time setup:

```bash
uv venv --python 3.11 .venv
CPPFLAGS='-I/opt/homebrew/opt/graphviz/include' \
CFLAGS='-I/opt/homebrew/opt/graphviz/include' \
LDFLAGS='-L/opt/homebrew/opt/graphviz/lib' \
UV_PROJECT_ENVIRONMENT=.venv uv sync --python 3.11 --group dev

direnv allow .
```

After that, `cd` into the repo auto-loads the environment.

## Development Workflow

Install git hooks:

```bash
UV_PROJECT_ENVIRONMENT=.venv uv run pre-commit install
```

Run all hooks manually:

```bash
UV_PROJECT_ENVIRONMENT=.venv uv run pre-commit run --all-files
```

## Typical Workflows

### 1) Train the architecture autoencoder

```bash
UV_PROJECT_ENVIRONMENT=.venv uv run python -m lao.embedding.autoencoder
```

### 2) Run latent-space NAS

```bash
UV_PROJECT_ENVIRONMENT=.venv uv run python -m lao.nas.nas
```

### 3) Run latency/candidate training experiments

```bash
UV_PROJECT_ENVIRONMENT=.venv uv run python -m lao.candidate_eval.latency
```

### 4) Latent-space analysis/plots

```bash
UV_PROJECT_ENVIRONMENT=.venv uv run python -m lao.embedding.viz_test
```

## Notes

- Several scripts still contain experiment-specific checkpoint/data paths; update these before running on a new machine or dataset.
- `deepspeed` is optional and placed under the `profiling` extra.
