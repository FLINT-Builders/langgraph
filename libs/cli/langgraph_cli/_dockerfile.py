import json
import unicodedata

import click

# Docker splits instructions on line breaks and arguments on whitespace, treats
# `\` as its escape character, expands `$` variables, and processes quotes in
# ADD, COPY, and WORKDIR arguments. Paths are written into those instructions
# unquoted, so any of these characters would change what Docker executes.
_DOCKERFILE_SPECIAL_CHARS = frozenset("\\$'\"")


def _find_unsafe_char(value: str) -> str | None:
    for char in value:
        if (
            char.isspace()
            or unicodedata.category(char) == "Cc"
            or char in _DOCKERFILE_SPECIAL_CHARS
        ):
            return char
    return None


def validate_dockerfile_path(path: str, *, description: str) -> str:
    """Return `path` if it can be written unquoted into a Dockerfile instruction.

    Raises `click.UsageError` instead of emitting a path that could split the
    instruction, start a new one, or be read as an instruction flag.
    """
    unsafe_char = _find_unsafe_char(path)
    if unsafe_char is not None:
        raise click.UsageError(
            f"{description} {path!r} contains {unsafe_char!r}, which cannot be "
            "written safely into a Dockerfile instruction. Rename the file or "
            "directory so its path has no whitespace, control characters, "
            "quotes, '$', or '\\'."
        )
    if path.startswith("-"):
        raise click.UsageError(
            f"{description} {path!r} starts with '-', which Docker would read "
            "as an instruction flag. Rename the file or directory."
        )
    return path


def validate_build_context(name: str, path: str) -> None:
    """Reject additional build contexts that cannot be referenced safely."""
    validate_dockerfile_path(name, description="Build context name")
    if any(unicodedata.category(char) == "Cc" for char in path):
        raise click.UsageError(
            f"Build context path {path!r} contains a control character. "
            "Move the project to a path without line breaks or control characters."
        )


def dockerfile_env_json(value: object) -> str:
    """Serialize `value` as JSON that is safe inside a single-quoted ENV value."""
    return json.dumps(value).replace("'", "\\u0027")
