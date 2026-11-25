"""Confine host directories named in langgraph.json to a developer-chosen root.

`langgraph.json` travels with the project, so the directories it names are not
trusted to pick what gets copied into an image or a deploy archive. By default
they must stay inside the directory that holds `langgraph.json`. Only the person
running the CLI can widen that boundary, with `--workspace-root`.
"""

from __future__ import annotations

import os
import pathlib

import click

WORKSPACE_ROOT_OPTION = "--workspace-root"


def _home_dir() -> pathlib.Path | None:
    try:
        return pathlib.Path.home().resolve()
    except (RuntimeError, KeyError, OSError):
        return None


def resolve_workspace_root(
    project_dir: pathlib.Path,
    workspace_root: str | os.PathLike[str] | None = None,
) -> pathlib.Path:
    """Return the directory that local paths from langgraph.json must stay inside.

    `project_dir` is the directory that contains `langgraph.json`. Without an
    explicit `workspace_root` the boundary is `project_dir` itself.
    """
    project_dir = project_dir.resolve()
    if workspace_root is None:
        return project_dir

    root = pathlib.Path(workspace_root).resolve()
    if root == pathlib.Path(root.anchor) or root == _home_dir():
        raise click.UsageError(
            f"{WORKSPACE_ROOT_OPTION} cannot be {root}. Pass the root directory "
            "of the monorepo or uv workspace that contains your project."
        )
    if root != project_dir and root not in project_dir.parents:
        raise click.UsageError(
            f"{WORKSPACE_ROOT_OPTION} {root} must contain the directory of the "
            f"LangGraph config file ({project_dir})."
        )
    return root


def ensure_within_workspace(
    resolved: pathlib.Path,
    *,
    project_dir: pathlib.Path,
    workspace_root: pathlib.Path,
    label: str,
) -> None:
    """Refuse `resolved` unless it is `workspace_root` or lies beneath it.

    `resolved` must already be the fully resolved path (no symlinks, no `..`)
    that the caller goes on to use.
    """
    if resolved == workspace_root or workspace_root in resolved.parents:
        return
    if workspace_root == project_dir.resolve():
        boundary = f"the project directory {workspace_root}"
        hint = (
            "Paths in langgraph.json must stay inside the directory that "
            "contains it. To use a directory elsewhere in a monorepo, pass "
            f"{WORKSPACE_ROOT_OPTION} <monorepo root> on the command line."
        )
    else:
        boundary = f"the workspace root {workspace_root}"
        hint = f"Pass a {WORKSPACE_ROOT_OPTION} that contains it."
    raise click.UsageError(
        f"{label} resolves to {resolved}, which is outside {boundary}. {hint}"
    )
