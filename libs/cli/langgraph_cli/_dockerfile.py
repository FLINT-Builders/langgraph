"""Guards for values interpolated into generated Dockerfiles.

Dockerfiles are line oriented: any line break inside an interpolated value
starts a new instruction. Shell-form `ADD` and `COPY` arguments are also split
on whitespace, and Docker processes quotes, `\\` escapes, and `$` variables in
`ADD`, `COPY`, and `WORKDIR` values, so a single value can otherwise turn into
several arguments, flags, or a different path.
"""

import json

import click

# `\` escapes characters and continues lines, quotes are stripped by Docker's
# word processing, and `$` triggers variable expansion (by Docker, and by
# Compose when the Dockerfile is inlined in a compose file).
_PATH_METACHARACTERS = frozenset("\\\"'$")


def dockerfile_text(value: str, what: str) -> str:
    """Return `value` if it cannot break out of the line it is written to.

    Rejects every kind of line break (including those Docker, Python, and YAML
    treat as line separators) and other control characters.
    """
    if not value.isprintable():
        raise click.UsageError(
            f"{what} {value!r} contains a line break or control character, "
            "which cannot be written into a Dockerfile. Rename it or remove "
            "the character."
        )
    return value


def dockerfile_path(value: str, what: str) -> str:
    """Return `value` if Docker reads it as one literal path when it is a whole field.

    Spaces are allowed: `WORKDIR` keeps them, and `add_or_copy` switches to the
    JSON form for them.
    """
    dockerfile_text(value, what)
    if (
        not value
        or value != value.strip()
        or any(ch in _PATH_METACHARACTERS for ch in value)
    ):
        raise click.UsageError(
            f"{what} {value!r} cannot be used as a path in a Dockerfile: it "
            "must be non-empty, must not start or end with whitespace, and "
            "must not contain quotes, '\\', or '$'. Rename it to use it in an "
            "image build."
        )
    return value


def dockerfile_token(value: str, what: str) -> str:
    """Return `value` if it is safe as a single shell-form argument or flag value."""
    dockerfile_path(value, what)
    if value.startswith("-") or any(ch.isspace() for ch in value):
        raise click.UsageError(
            f"{what} {value!r} cannot be used in a Dockerfile: it must not "
            "start with '-' or contain whitespace."
        )
    return value


def add_or_copy(instruction: str, *paths: str, flags: tuple[str, ...] = ()) -> str:
    """Render an `ADD` or `COPY` instruction whose sources and destination stay literal."""
    for path in paths:
        dockerfile_path(path, f"{instruction} path")
    if any(" " in path for path in paths) or paths[0].startswith(("-", "[")):
        args = json.dumps(list(paths))
    else:
        args = " ".join(paths)
    return " ".join([instruction, *flags, args])
