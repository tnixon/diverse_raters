# AGENTS.md

## Purpose
This repo tests one claim: at matched rater accuracy, optimizing against a diverse, randomly sampled ensemble of raters produces less reward hacking than optimizing against a single rater. The construct is toxicity, framed as detoxification, and a hack is a rewrite the training signal scores as clean when it is not. The work is built in four reviewed stages: skeleton, data, raters, search and evaluation.

## Roles and workflow
The human is the **director**; the agent is the **actor**. The agent codes, the director reviews, and the two design together.

1. **Direction:** the director says what we want to build.
2. **Plan:** the agent proposes how to build it. We may iterate on the plan several times.
3. **Approval:** the director approves the plan. Do not write code before this.
4. **Execution:** the agent implements the approved plan and produces candidate code, running lint and tests as it goes.
5. **Review:** the director reviews all changes. Leave them uncommitted in the working tree until this is done.
6. **Go-ahead:** only when the director says so may the agent commit, push, and open a PR. The PR then gets further review and automated checks.

Rules that follow from this:
- Never commit unreviewed code. Report what changed and wait.
- Never push a branch or open a PR until directed to. A go-ahead to commit is not a go-ahead to push, and approval for one step does not carry over to the next.
- If the plan needs to change during execution, stop and discuss it instead of widening scope on your own.
- Design is shared: raise concerns, alternatives, and uncertainties rather than silently picking one. The director decides.

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
