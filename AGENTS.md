# AGENTS.md

## Purpose
This repo tests one claim: at matched rater accuracy, optimizing against a diverse, randomly sampled ensemble of raters produces less reward hacking than optimizing against a single rater. The construct is toxicity, framed as detoxification, and a hack is a rewrite the training signal scores as clean when it is not. The work is built in four reviewed stages: skeleton, data, raters, search and evaluation.

## Commands
- Install: `uv sync`
- Lint: `uv run ruff check .`
- Format: `uv run ruff format .`
- Test: `uv run pytest`

## Invariants
- Held-out sets are carved out before any partitioning and never leak into training.
- Every condition gets an equal search budget.
- Every candidate is scored by every rater.
- `PREDICTIONS.md` is frozen after the first ensemble run.
- Trusted-score sources (held-out annotators, LLM judge, human sample) are reported separately, never merged.

## Rules
- No data, keys, or workspace URLs in the repo.
- Every run logs its config, git commit, and dataset versions to MLflow.
- Library code goes in `src/dre/`; notebooks stay thin.
- Configs live in `conf/`.
- Seeds are fixed.
- One stage or plan step per PR.
- No paid API calls in tests.
