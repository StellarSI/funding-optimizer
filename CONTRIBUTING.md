# Contributing to funding-optimizer

Thanks for your interest in contributing! This document describes the local
workflow and the expectations for issues and pull requests.

## Getting started

1. Fork and clone the repository:

   ```bash
   git clone https://github.com/<your-username>/funding-optimizer.git
   cd funding-optimizer
   ```

2. Create a virtual environment and install the project with development
   dependencies:

   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -e ".[dev]"
   ```

3. Copy the environment template and adjust it for your setup:

   ```bash
   cp .env.example .env
   ```

   The required variables are documented in `.env.example` and
   `README.md` §Configuration (`STELLAR_NETWORK`, `HORIZON_URL`,
   `ANALYTICS_API_URL`, `DATABASE_URL`).

## Development workflow

- **Open an issue first** for larger changes so the approach can be agreed
  before implementation.
- Create a feature branch from `main`:

  ```bash
  git fetch upstream
  git checkout -b feature/<descriptive-name> upstream/main
  ```

- Make your change, then run the checks before committing:

  ```bash
  make check
  ```

  `make check` runs lint (`ruff`), type checking (`mypy`), and tests
  (`pytest`). Individual targets are `make lint`, `make format`,
  `make typecheck`, and `make test`.
- Commit with a clear message. Reference the issue in the body, e.g.
  `Closes #12`.

### Optional pre-commit hooks

```bash
pre-commit install
```

This runs lint and formatting checks on each commit.

## Checks

| Check | Command |
| --- | --- |
| Lint | `ruff check funding_optimizer tests` |
| Format | `black funding_optimizer tests` |
| Type check | `mypy funding_optimizer tests` |
| Tests | `pytest` |
| All | `make check` |

The same checks run in CI on every pull request.

## Pull requests

- Keep one PR focused on one issue or one coherent change.
- Describe the change, the motivation, and any tradeoffs.
- Reference the issue with a closing keyword in the PR body
  (`Closes #12`).
- Make sure `make check` passes before requesting review.

## Responsible use

Allocation models affect real people. Treat outputs as decision support:
keep a human in the loop, audit for bias, and collect only the data you need.
See `README.md` §Responsible use.
