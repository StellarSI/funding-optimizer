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
