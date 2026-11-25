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

## Clean code
This project follows clean-code principles. Apply them to every change, every time. They are not optional and not something to tidy up later. If a principle conflicts with a task, say so and ask rather than quietly dropping it.

- **KISS:** write the simplest thing that works. Prefer plain functions and data over clever abstractions, frameworks, or metaprogramming.
- **DRY:** one source of truth for every piece of knowledge (names, constants, logic). Duplicated logic gets extracted. Don't extract after one use, but do after the second.
- **YAGNI:** build only what the current stage needs. No speculative parameters, hooks, stubs, or config for later stages.
- **Single responsibility:** each module, class, and function does one thing and has one reason to change. Keep functions short enough to read in one screen.
- **Meaningful names:** names say what a thing is or does, with no abbreviations or cryptic one-letter names outside tiny scopes. If a name needs a comment to explain it, rename it.
- **Comments explain why, not what:** the code should say what it does. Delete comments that restate code and any commented-out code.
- **Explicit over implicit:** type-annotate public functions, pass dependencies as arguments, and avoid hidden global state and side effects at import time. Seeds, paths, and names come from `Settings` or `conf/`, never hardcoded inline.
- **Fail fast:** validate inputs at the boundary and raise clear errors. Never swallow exceptions or return silent fallbacks.
- **Pure core, thin edges:** keep computation free of I/O so it can be tested without Databricks, the network, or paid APIs. I/O lives at the edges.
- **Tested behavior:** new logic ships with tests that check behavior, not implementation details. Fix the cause of a failing test rather than weakening the test.
- **Small, focused changes:** leave code cleaner than you found it, but don't refactor unrelated code inside a PR. Keep each diff reviewable.
