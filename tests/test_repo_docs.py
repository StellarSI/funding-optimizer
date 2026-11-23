"""Path checks for repository documentation and contribution templates.

These verify that the documentation and templates added by the DX issues are
present at the paths the issues target. They are existence checks only: the
targets are documentation and templates, so there is no runtime behavior to
exercise.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_security_md_exists() -> None:
    """SECURITY.md (issue #98) is present at the repository root."""
    assert (ROOT / "SECURITY.md").is_file()


def test_architecture_doc_exists() -> None:
    """docs/architecture.md (issue #99) is present at the documented path."""
    assert (ROOT / "docs" / "architecture.md").is_file()


def test_issue_templates_exist() -> None:
    """Issue templates (issue #100) are present under .github/ISSUE_TEMPLATE."""
    templates = ROOT / ".github" / "ISSUE_TEMPLATE"
    assert (templates / "bug_report.md").is_file()
    assert (templates / "feature_request.md").is_file()
    assert (templates / "config.yml").is_file()


def test_pull_request_template_exists() -> None:
    """PULL_REQUEST_TEMPLATE.md (issue #100) is present at .github/."""
    assert (ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md").is_file()


def test_release_checklist_exists() -> None:
    """Release checklist (issue #100) is present at .github/."""
    assert (ROOT / ".github" / "release-checklist.md").is_file()


def test_makefile_exists_and_declares_check_target() -> None:
    """Makefile (issue #97) is present at repo root and declares valid tab-indented check targets."""
    makefile = ROOT / "Makefile"
    assert makefile.is_file(), "Makefile must exist at the repository root"
    content = makefile.read_text(encoding="utf-8")
    raw_lines = content.splitlines()

    # Every indented recipe line in a Makefile MUST begin with a literal tab character
    recipe_lines = [line for line in raw_lines if line.startswith((" ", "\t"))]
    assert recipe_lines, "Makefile must contain recipe commands"
    for line in recipe_lines:
        assert line.startswith(
            "\t"
        ), f"Makefile recipe line must start with a tab, not spaces: {line!r}"

    # Verify required targets and commands
    assert ".PHONY:" in content
    for target in (
        "install:",
        "check:",
        "lint:",
        "format:",
        "format-check:",
        "typecheck:",
        "test:",
    ):
        assert target in content, f"Missing {target} in Makefile"

    assert 'python -m pip install -e ".[dev]"' in content
    assert "ruff check funding_optimizer tests" in content
    assert "black --check funding_optimizer tests" in content
    assert "mypy funding_optimizer tests" in content
    assert "\tpytest" in content

    check_lines = [line for line in raw_lines if line.startswith("check:")]
    assert len(check_lines) == 1
    check_deps = check_lines[0].split(":", 1)[1].split()
    assert {"lint", "format-check", "typecheck", "test"}.issubset(set(check_deps))
