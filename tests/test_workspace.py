"""Tests for materialize_workspace.

Synthesizes small unified diffs, materializes them into tmp_path, and verifies
the resulting files. Covers: empty diff, single-file with context+added lines,
removal exclusion, multi-file, and nested-directory creation.
"""

from pathlib import Path

from sentry.workspace import materialize_workspace


def test_empty_diff_creates_nothing(tmp_path: Path) -> None:
    """Empty or whitespace-only diffs leave the workspace empty."""
    materialize_workspace("", tmp_path)
    assert list(tmp_path.iterdir()) == []

    materialize_workspace("   \n\n", tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_single_file_with_context_and_added_lines(tmp_path: Path) -> None:
    """Both context and added lines appear in the materialized file."""
    diff = (
        "diff --git a/src/users.py b/src/users.py\n"
        "--- a/src/users.py\n"
        "+++ b/src/users.py\n"
        "@@ -1,2 +1,3 @@\n"
        " class UserRepo:\n"
        "     def __init__(self, db):\n"
        "+        self.db = db\n"
    )
    materialize_workspace(diff, tmp_path)

    out = tmp_path / "src" / "users.py"
    assert out.exists()
    content = out.read_text()
    assert "class UserRepo:" in content
    assert "def __init__(self, db):" in content
    assert "self.db = db" in content


def test_removed_lines_are_dropped(tmp_path: Path) -> None:
    """Lines with '-' prefix are excluded from the materialized file."""
    diff = (
        "diff --git a/src/x.py b/src/x.py\n"
        "--- a/src/x.py\n"
        "+++ b/src/x.py\n"
        "@@ -1,2 +1,2 @@\n"
        " keep_me\n"
        "-remove_me\n"
        "+add_me\n"
    )
    materialize_workspace(diff, tmp_path)

    content = (tmp_path / "src" / "x.py").read_text()
    assert "keep_me" in content
    assert "add_me" in content
    assert "remove_me" not in content


def test_multiple_files_all_materialized(tmp_path: Path) -> None:
    """A diff touching multiple files produces all of them."""
    diff = (
        "diff --git a/a.py b/a.py\n"
        "--- a/a.py\n"
        "+++ b/a.py\n"
        "@@ -1 +1 @@\n"
        "-old_a\n"
        "+new_a\n"
        "diff --git a/b.py b/b.py\n"
        "--- a/b.py\n"
        "+++ b/b.py\n"
        "@@ -1 +1 @@\n"
        "-old_b\n"
        "+new_b\n"
    )
    materialize_workspace(diff, tmp_path)

    assert (tmp_path / "a.py").read_text().strip() == "new_a"
    assert (tmp_path / "b.py").read_text().strip() == "new_b"


def test_nested_directories_are_created(tmp_path: Path) -> None:
    """Files in nested paths have their parent dirs auto-created."""
    diff = (
        "diff --git a/src/services/auth/handler.py b/src/services/auth/handler.py\n"
        "--- a/src/services/auth/handler.py\n"
        "+++ b/src/services/auth/handler.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )
    materialize_workspace(diff, tmp_path)

    out = tmp_path / "src" / "services" / "auth" / "handler.py"
    assert out.exists()
    assert out.read_text().strip() == "new"