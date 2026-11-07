# Release checklist

Use this checklist when preparing a release of `funding-optimizer`.

## Before release

- [ ] All merged PRs reference their issues (`Closes #N`).
- [ ] `make check` passes on `main` (ruff, black, mypy, pytest).
- [ ] CI is green on `main` for all matrix Python versions.
- [ ] The CHANGELOG / release notes draft summarizes user-facing changes.
- [ ] Version bump in `pyproject.toml` follows semantic versioning.
- [ ] README and docs reflect any user-visible changes.
- [ ] SECURITY.md supported-versions table is up to date.
- [ ] No secrets, credentials, or real recipient data are committed.

## Release

- [ ] Create a release branch from `main`.
- [ ] Open a release PR and collect review/approvals.
- [ ] Tag the release (e.g. `v0.1.0`) after the PR merges.
- [ ] Publish the release notes on GitHub.

## After release

- [ ] Announce the release to stakeholders.
- [ ] Monitor for issues and regressions.
- [ ] Update the roadmap in README if priorities changed.
