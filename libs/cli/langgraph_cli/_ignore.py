"""Shared ignore-file handling for local source filtering."""

import fnmatch
import os
import pathlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import click
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

# Environment files hold credentials and are never copied into an image or a
# deploy archive, regardless of `.dockerignore` / `.gitignore` contents.
_ENV_FILE_PATTERNS = (".env", ".env.*", "*.env")
# Characters that pass unchanged through Dockerfile flag parsing and are not
# pattern syntax in `--exclude`.
_PLAIN_EXCLUDE_PATH = re.compile(r"[A-Za-z0-9._@+,=/-]+")
_GITWILDMATCH_SPECIAL = re.compile(r"([\\*?\[\] ])")


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


def _configured_env_paths(
    config_path: pathlib.Path, config: Mapping
) -> tuple[pathlib.Path, ...]:
    """Absolute paths of the env file named by `config["env"]`, if any.

    Both the lexical path and its symlink-resolved target are returned so that
    either one can be matched against a directory being copied.
    """
    env = config.get("env")
    if not isinstance(env, str) or not env:
        return ()
    path = pathlib.Path(os.path.abspath(config_path.parent / env))
    return tuple(dict.fromkeys((path, path.resolve())))


def _env_paths_within(
    directory: pathlib.Path, env_paths: Sequence[pathlib.Path]
) -> list[str]:
    """POSIX paths, relative to `directory`, of env files located inside it."""
    roots = dict.fromkeys(
        (pathlib.Path(os.path.abspath(directory)), directory.resolve())
    )
    relative: dict[str, None] = {}
    for env_path in env_paths:
        for root in roots:
            try:
                rel = env_path.relative_to(root)
            except ValueError:
                continue
            if rel.parts:
                relative[rel.as_posix()] = None
    return list(relative)


def _matches_env_file_pattern(name: str) -> bool:
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in _ENV_FILE_PATTERNS)


def _docker_exclude_flags(
    directory: pathlib.Path, env_paths: Sequence[pathlib.Path] = ()
) -> str:
    """`--exclude` flags that keep env files out of an ADD/COPY of `directory`.

    Patterns are relative to the copied directory, as BuildKit applies them.
    """
    patterns = [f"**/{pattern}" for pattern in _ENV_FILE_PATTERNS]
    for rel in _env_paths_within(directory, env_paths):
        if _matches_env_file_pattern(pathlib.PurePosixPath(rel).name):
            continue
        if _PLAIN_EXCLUDE_PATH.fullmatch(rel):
            patterns.append(rel)
        else:
            raise click.UsageError(
                f"The env file '{rel}' is inside '{directory}', which is copied "
                "into the image, and its name cannot be excluded safely. Rename "
                "it to '.env' or '<name>.env', or move it out of that directory."
            )
    return " ".join(f"--exclude={pattern}" for pattern in patterns)


def _build_ignore_spec(
    directory: pathlib.Path,
    *,
    include_gitignore: bool = True,
    env_paths: Sequence[pathlib.Path] = (),
) -> pathspec.PathSpec:
    """Build a PathSpec combining built-in exclusions with ignore files.

    Always excludes common non-source directories (`_ALWAYS_EXCLUDE`). On top
    of that, patterns from `.dockerignore` are merged in. `.gitignore` patterns
    are optional because some callers need Docker build-context semantics,
    while archive creation wants both files. Env files (`_ENV_FILE_PATTERNS`
    and any of `env_paths` inside `directory`) are excluded last so that no
    negation in an ignore file can re-include them.
    """
    lines: list[str] = list(_ALWAYS_EXCLUDE)
    ignore_files = [".dockerignore"]
    if include_gitignore:
        ignore_files.append(".gitignore")
    for name in ignore_files:
        ignore_file = directory / name
        if ignore_file.is_file():
            lines.extend(ignore_file.read_text(encoding="utf-8").splitlines())
    lines.extend(_ENV_FILE_PATTERNS)
    lines.extend(
        "/" + _GITWILDMATCH_SPECIAL.sub(r"\\\1", rel)
        for rel in _env_paths_within(directory, env_paths)
    )
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
