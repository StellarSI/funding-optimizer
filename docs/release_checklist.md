# Release Checklist

This checklist outlines the verification steps required before tagging and publishing a new release of `funding-optimizer`.

---

## 1. Pre-Release Verification

- [ ] **Clean Git Working Tree**: Ensure no uncommitted changes or untracked temporary files exist (`git status`).
- [ ] **Branch Sync**: Verify that `main` is completely up-to-date with upstream (`git fetch upstream && git checkout main && git merge upstream/main`).
- [ ] **Full Test Suite Passing**:
  ```bash
  pytest -v
  ```
  All unit, property-based (Hypothesis), and end-to-end integration tests must pass cleanly.
- [ ] **Linting & Code Quality**:
  ```bash
  ruff check .
  ruff format --check .
  ```
  Ensure zero warnings or style violations.
- [ ] **Type Checking (if applicable)**: Ensure signatures and dataclasses maintain backward compatibility.

---

## 2. Documentation & Metadata

- [ ] **Version Bump**: Update version string in `pyproject.toml` according to [Semantic Versioning (SemVer)](https://semver.org/).
- [ ] **Changelog**:
  - Update `CHANGELOG.md` with features, bug fixes, performance improvements, and breaking changes.
  - Credit community pull requests and external contributors.
- [ ] **Environment & Docs Sync**: Verify `.env.example` and `README.md` reflect any new configuration options or solver flags.

---

## 3. Security & Dependency Audit

- [ ] **Dependency Check**: Audit installed packages for security advisories:
  ```bash
  pip-audit # or safety check
  ```
- [ ] **Sanitization & Secrets Check**: Ensure no API keys, private Stellar seeds, or development tokens are checked into repo or fixtures.

---

## 4. Release Tagging & Build

- [ ] **Commit Version Bump**:
  ```bash
  git add pyproject.toml CHANGELOG.md
  git commit -m "chore(release): vX.Y.Z"
  ```
- [ ] **Create Annotated Git Tag**:
  ```bash
  git tag -a vX.Y.Z -m "Release vX.Y.Z"
  git push upstream main --tags
  ```
- [ ] **Package Build & Distribution**:
  ```bash
  python -m build
  twine check dist/*
  ```

---

## 5. Post-Release & Announcement

- [ ] **GitHub Release Notes**: Draft GitHub Release from the newly pushed tag, linking closed issues and merged PRs.
- [ ] **Docker / Container Deployment**: If container images are built, verify CI image build and push to registry.
- [ ] **Monitor Post-Deploy**: Verify initial downstream consumer telemetry or health check status.
