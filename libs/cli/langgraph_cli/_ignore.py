"""Shared ignore-file handling for local source filtering."""

import fnmatch
import os
import pathlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pathspec

_ALWAYS_EXCLUDE = [
    "__pycache__/",
    ".git/",
    ".venv/",
    "venv/",
    "node_modules/",
    ".tox/",
    ".mypy_cache/",
]
_ALWAYS_EXCLUDE_NAMES = frozenset(
    pattern.rstrip("/").split("/")[-1] for pattern in _ALWAYS_EXCLUDE
)
_GLOB_CHARS = frozenset("*?[")
# Matched against the lowercased basename after ignore files are applied, so
# `!` negations in `.gitignore` / `.dockerignore` cannot re-include them.
_CREDENTIAL_NAME_PATTERNS = (".env", ".env.*", ".npmrc", ".netrc")


def _is_credential_name(name: str) -> bool:
    """Whether a file or directory basename is an env or credential file."""
    lowered = name.lower()
    return any(
        fnmatch.fnmatchcase(lowered, pattern) for pattern in _CREDENTIAL_NAME_PATTERNS
    )


@dataclass(frozen=True, slots=True)
class _CredentialFileFilter:
    """Refuses env/credential files by name and the configured env file by inode.

    The configured env file is identified once by `(st_dev, st_ino)`, so it is
    still recognised under any name or through a hard link.
    """

    env_file_ids: frozenset[tuple[int, int]] = frozenset()

    @classmethod
    def from_config(
        cls, config_path: pathlib.Path, config: Mapping[str, Any]
    ) -> "_CredentialFileFilter":
        ids: set[tuple[int, int]] = set()
        env_field = config.get("env")
        if isinstance(env_field, str) and env_field:
            try:
                st = os.stat(config_path.parent / env_field)
            except OSError:
                pass
            else:
                ids.add((st.st_dev, st.st_ino))
        return cls(env_file_ids=frozenset(ids))

    def excludes_name(self, name: str) -> bool:
        return _is_credential_name(name)

    def excludes_stat(self, st: os.stat_result) -> bool:
        return (st.st_dev, st.st_ino) in self.env_file_ids


@dataclass(frozen=True, slots=True)
class _NegatedDockerignoreHints:
    exact_dirs: frozenset[pathlib.PurePosixPath] = frozenset()
    wildcard_prefixes: frozenset[pathlib.PurePosixPath] = frozenset()
    recurse_all: bool = False

    def requires_dir_walk(self, path: pathlib.PurePosixPath) -> bool:
        if self.recurse_all or path in self.exact_dirs:
            return True
        return any(
            path == prefix or path in prefix.parents or prefix in path.parents
            for prefix in self.wildcard_prefixes
        )


def _build_ignore_spec(
    directory: pathlib.Path, *, include_gitignore: bool = True
) -> pathspec.PathSpec:
    """Build a PathSpec combining built-in exclusions with ignore files.

    Always excludes common non-source directories (`_ALWAYS_EXCLUDE`). On top
    of that, patterns from `.dockerignore` are merged in. `.gitignore` patterns
    are optional because some callers need Docker build-context semantics,
    while archive creation wants both files.
    """
    lines: list[str] = list(_ALWAYS_EXCLUDE)
    ignore_files = [".dockerignore"]
    if include_gitignore:
        ignore_files.append(".gitignore")
    for name in ignore_files:
        ignore_file = directory / name
        if ignore_file.is_file():
            lines.extend(ignore_file.read_text(encoding="utf-8").splitlines())
    return pathspec.PathSpec.from_lines("gitwildmatch", lines)


def _is_always_excluded(path: pathlib.PurePosixPath, *, is_dir: bool) -> bool:
    """Whether `path` lives inside a built-in excluded directory."""
    parent_parts = path.parts if is_dir else path.parts[:-1]
    return any(part in _ALWAYS_EXCLUDE_NAMES for part in parent_parts)


def _build_dockerignore_negation_hints(
    directory: pathlib.Path,
) -> _NegatedDockerignoreHints:
    """Summarize which ignored directories must still be traversed.

    Most negations only require walking a small, concrete chain of parent
    directories (for example `!assets/keep.txt` requires entering `assets/`).
    Broader glob negations may force a wider walk.
    """
    ignore_file = directory / ".dockerignore"
    if not ignore_file.is_file():
        return _NegatedDockerignoreHints()

    exact_dirs: set[pathlib.PurePosixPath] = set()
    wildcard_prefixes: set[pathlib.PurePosixPath] = set()
    recurse_all = False

    for raw_line in ignore_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("\\!"):
            continue
        if line.startswith("\\#"):
            line = line[1:]
        if not line.startswith("!"):
            continue

        pattern = line[1:].lstrip("/")
        while pattern.startswith("./"):
            pattern = pattern[2:]
        pattern = pattern.rstrip("/")
        parts = [part for part in pattern.split("/") if part and part != "."]
        if not parts:
            recurse_all = True
            continue

        wildcard_index = next(
            (
                idx
                for idx, part in enumerate(parts)
                if any(char in part for char in _GLOB_CHARS)
            ),
            None,
        )
        if wildcard_index is not None:
            literal_parts = parts[:wildcard_index]
            if not literal_parts:
                recurse_all = True
                continue
            wildcard_prefixes.add(pathlib.PurePosixPath(*literal_parts))
            continue

        parent_parts = parts[:-1]
        for idx in range(1, len(parent_parts) + 1):
            exact_dirs.add(pathlib.PurePosixPath(*parent_parts[:idx]))

    return _NegatedDockerignoreHints(
        exact_dirs=frozenset(exact_dirs),
        wildcard_prefixes=frozenset(wildcard_prefixes),
        recurse_all=recurse_all,
    )
