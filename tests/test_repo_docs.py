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
