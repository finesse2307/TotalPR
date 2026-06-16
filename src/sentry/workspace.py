"""Workspace abstraction: a directory containing the post-diff version of changed files.
"""

from pathlib import Path

from unidiff import PatchSet  # type: ignore[import-untyped]


def materialize_workspace(raw_diff: str, root: Path) -> None:
    """Write each changed file's post-diff content under ``root``.

    For every file in the diff, the new content is reconstructed from the
    hunks: context (' ' prefix) and added ('+' prefix) lines are kept;
    removed ('-' prefix) lines are dropped. Files are written under ``root``
    at the same relative paths the diff used; parent directories are created
    as needed.

    Empty or whitespace-only ``raw_diff`` is a no-op.
    """
    if not raw_diff.strip():
        return

    patch_set = PatchSet(raw_diff)

    for patched_file in patch_set:
        out_path = root / patched_file.path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        content_lines: list[str] = []
        for hunk in patched_file:
            # str(hunk).splitlines() gives [@@ header, body line 1, ...].
            # Skip the header; process body lines by prefix.
            for line in str(hunk).splitlines()[1:]:
                if not line:
                    continue
                prefix = line[0]
                if prefix in (" ", "+"):
                    content_lines.append(line[1:])

        out_path.write_text("\n".join(content_lines) + "\n")