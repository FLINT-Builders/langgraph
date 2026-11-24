"""Faux-package directory names must not become shell commands during image builds.

`langgraph build` and `langgraph dockerfile` generate a Dockerfile through
`config_to_docker`. `langgraph up` embeds that Dockerfile through
`config_to_compose`. For a local dependency with no `pyproject.toml` or
`setup.py`, the directory name is copied into `RUN` instructions.

These tests fail while that name can break out of the generated shell. They
pass when the name is shell-quoted, or when generation refuses it, without
writing or deleting a file first.
"""

import json
import pathlib
import re
import subprocess

from langgraph_cli.config import (
    config_to_compose,
    config_to_docker,
    validate_config_file,
)

_MARKER = "injectedmarker"
_BASE_IMAGE = "example.invalid/langgraph-api"
# BuildKit joins a continued Dockerfile line when a backslash is the last
# non-space token. See moby/buildkit v0.35.1 parser.go setEscapeToken.
_LINE_CONTINUATION = re.compile(rb"([^\\])\\[ \t]*$|^\\[ \t]*$")
_COMMAND_SPLIT = re.compile(r"[\t\v\f\r ]+")
_DESTRUCTIVE_SNIPPETS = (
    "rm -rf",
    "pip uninstall",
    "uv pip uninstall",
    "find /usr",
)


def _quote_payload() -> str:
    """Close the single-quoted `for` word and run a command in that loop."""
    return f"pkg'; do touch {_MARKER}; done; for line in '"


def _semicolon_payload() -> str:
    """Break an unquoted path in a `RUN` instruction and run a command."""
    return f"pkg; touch {_MARKER}; #"


def _newline_payload() -> str:
    """Insert a new Dockerfile `RUN` instruction after the directory name."""
    return f"pkg\nRUN touch {_MARKER}\n#"


def _write_project(
    tmp_path: pathlib.Path,
    payload: str,
    *,
    layout: str,
    requirements: bool = False,
    outside: bool = False,
) -> pathlib.Path:
    if outside:
        config_dir = tmp_path / "project" / "app"
        package_dir = tmp_path / "project" / payload
        dependency = "../" + payload
    else:
        config_dir = tmp_path / "app"
        package_dir = config_dir / payload
        dependency = "./" + payload
    config_dir.mkdir(parents=True)
    package_dir.mkdir(parents=True)
    (package_dir / "graph.py").write_text("graph = None\n", encoding="utf-8")
    if layout == "flat":
        (package_dir / "__init__.py").write_text("", encoding="utf-8")
    if requirements:
        (package_dir / "requirements.txt").write_text("# none\n", encoding="utf-8")
    config_path = config_dir / "langgraph.json"
    config_path.write_text(
        json.dumps(
            {
                "dependencies": [dependency],
                "graphs": {"agent": f"{dependency}/graph.py:graph"},
            }
        ),
        encoding="utf-8",
    )
    return config_path


def _snapshot(root: pathlib.Path) -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    for path in root.rglob("*"):
        kind = "dir" if path.is_dir() else "file"
        found.add((str(path.relative_to(root)), kind))
    return found


def _trim_leading_ws(token: bytes) -> bytes:
    text = token.decode("utf-8")
    index = 0
    while index < len(text) and text[index].isspace():
        index += 1
    return text[index:].encode("utf-8")


def _is_comment(token: bytes) -> bool:
    return _trim_leading_ws(token.rstrip(b"\r\n")).startswith(b"#")


def _process_line(token: bytes, *, strip_left: bool) -> bytes:
    token = token.rstrip(b"\r\n")
    if strip_left:
        token = _trim_leading_ws(token)
    if _is_comment(token):
        return b""
    return token


def _trim_continuation(line: bytes) -> tuple[bytes, bool]:
    match = _LINE_CONTINUATION.search(line)
    if match is None:
        return line, True
    prefix = match.group(1) or b""
    return line[: match.start()] + prefix, False


def _logical_lines(dockerfile: str) -> list[str]:
    """Join escaped Dockerfile lines the way BuildKit's parser does.

    moby/buildkit v0.35.1 `frontend/dockerfile/parser/parser.go` `Parse`,
    `processLine`, and `trimContinuationCharacter`.
    """
    raw_lines = dockerfile.splitlines(keepends=True)
    index = 0

    def next_raw() -> bytes | None:
        nonlocal index
        if index >= len(raw_lines):
            return None
        item = raw_lines[index].encode("utf-8")
        index += 1
        return item

    lines: list[str] = []
    while True:
        raw = next_raw()
        if raw is None:
            break
        processed = _process_line(raw, strip_left=True)
        processed, is_end = _trim_continuation(processed)
        if is_end and not processed:
            continue
        buf = bytearray(processed)
        while not is_end:
            nxt = next_raw()
            if nxt is None:
                break
            if _is_comment(nxt):
                continue
            processed = _process_line(nxt, strip_left=False)
            if not _trim_leading_ws(processed):
                continue
            processed, is_end = _trim_continuation(processed)
            buf.extend(processed)
        lines.append(buf.decode("utf-8"))
    return lines


def _run_invocations(dockerfile: str) -> list[tuple[str, str | list[str]]]:
    """Return shell-form and exec-form `RUN` instructions.

    Shell form is the `RUN` body BuildKit passes to `/bin/sh -c` when the base
    image does not set `Config.Shell` (`defaultshell.go` and `withShell` in
    moby/buildkit v0.35.1). Exec form is the JSON argument vector, which
    BuildKit runs without a shell (`PrependShell` is false).
    """
    invocations: list[tuple[str, str | list[str]]] = []
    for line in _logical_lines(dockerfile):
        parts = _COMMAND_SPLIT.split(line.strip(), maxsplit=1)
        if not parts or not parts[0]:
            continue
        command = parts[0]
        rest = parts[1].strip() if len(parts) == 2 else ""
        if command.upper() != "RUN" or not rest:
            continue
        if rest.startswith("["):
            try:
                argv = json.loads(rest)
            except json.JSONDecodeError:
                invocations.append(("shell", rest))
                continue
            if isinstance(argv, list) and all(isinstance(item, str) for item in argv):
                invocations.append(("exec", argv))
                continue
        invocations.append(("shell", rest))
    return invocations


def _inline_dockerfiles(compose: str) -> list[str]:
    lines = compose.splitlines()
    dockerfiles: list[str] = []
    index = 0
    while index < len(lines):
        if lines[index].strip() != "dockerfile_inline: |":
            index += 1
            continue
        index += 1
        block: list[str] = []
        while index < len(lines):
            line = lines[index]
            if line.strip() == "":
                block.append("")
                index += 1
                continue
            if line.startswith("                "):
                block.append(line[16:])
                index += 1
                continue
            break
        dockerfiles.append("\n".join(block))
    return dockerfiles


def _invocation_text(kind: str, spec: str | list[str]) -> str:
    if kind == "shell":
        assert isinstance(spec, str)
        return spec
    assert isinstance(spec, list)
    return " ".join(spec)


def _execute(kind: str, spec: str | list[str], cwd: pathlib.Path) -> None:
    if kind == "shell":
        assert isinstance(spec, str)
        subprocess.run(
            ["/bin/sh", "-c", spec],
            cwd=cwd,
            check=False,
            capture_output=True,
        )
        return
    assert isinstance(spec, list)
    subprocess.run(spec, cwd=cwd, check=False, capture_output=True)


def _assert_dirname_is_not_executed(
    tmp_path: pathlib.Path, payload: str, dockerfiles_fn
) -> None:
    marker = tmp_path / _MARKER
    before = _snapshot(tmp_path)
    try:
        dockerfiles = dockerfiles_fn()
    except Exception:
        assert not marker.exists()
        assert _snapshot(tmp_path) == before
        return

    assert not marker.exists()
    assert _snapshot(tmp_path) == before
    assert dockerfiles
    executed = False
    for dockerfile in dockerfiles:
        assert dockerfile.strip()
        for kind, spec in _run_invocations(dockerfile):
            text = _invocation_text(kind, spec)
            if _MARKER not in text and payload not in text:
                continue
            assert not any(snippet in text for snippet in _DESTRUCTIVE_SNIPPETS)
            executed = True
            _execute(kind, spec, tmp_path)
            assert not marker.exists()
    if not executed:
        for dockerfile in dockerfiles:
            assert _MARKER not in dockerfile
            assert payload not in dockerfile


def _dockerfiles_for_build(config_path: pathlib.Path) -> list[str]:
    config = validate_config_file(config_path)
    dockerfile, _additional_contexts = config_to_docker(
        config_path,
        config,
        base_image=_BASE_IMAGE,
    )
    return [dockerfile]


def _dockerfiles_for_up(
    config_path: pathlib.Path, *, engine_runtime_mode: str
) -> list[str]:
    config = validate_config_file(config_path)
    compose = config_to_compose(
        config_path,
        config,
        base_image=_BASE_IMAGE,
        engine_runtime_mode=engine_runtime_mode,
    )
    dockerfiles = _inline_dockerfiles(compose)
    assert dockerfiles
    assert all("FROM " in dockerfile for dockerfile in dockerfiles)
    return dockerfiles


def test_build_src_layout_quote_in_dirname_is_not_executed(
    tmp_path: pathlib.Path,
) -> None:
    payload = _quote_payload()
    config_path = _write_project(tmp_path, payload, layout="src")
    _assert_dirname_is_not_executed(
        tmp_path,
        payload,
        lambda: _dockerfiles_for_build(config_path),
    )


def test_build_flat_layout_quote_in_dirname_is_not_executed(
    tmp_path: pathlib.Path,
) -> None:
    payload = _quote_payload()
    config_path = _write_project(tmp_path, payload, layout="flat")
    _assert_dirname_is_not_executed(
        tmp_path,
        payload,
        lambda: _dockerfiles_for_build(config_path),
    )


def test_build_requirements_path_metacharacters_are_not_executed(
    tmp_path: pathlib.Path,
) -> None:
    payload = _semicolon_payload()
    config_path = _write_project(
        tmp_path,
        payload,
        layout="src",
        requirements=True,
    )
    _assert_dirname_is_not_executed(
        tmp_path,
        payload,
        lambda: _dockerfiles_for_build(config_path),
    )


def test_build_parent_directory_quote_in_dirname_is_not_executed(
    tmp_path: pathlib.Path,
) -> None:
    payload = _quote_payload()
    config_path = _write_project(tmp_path, payload, layout="src", outside=True)
    _assert_dirname_is_not_executed(
        tmp_path,
        payload,
        lambda: _dockerfiles_for_build(config_path),
    )


def test_build_newline_in_dirname_does_not_run(tmp_path: pathlib.Path) -> None:
    payload = _newline_payload()
    config_path = _write_project(tmp_path, payload, layout="src")
    _assert_dirname_is_not_executed(
        tmp_path,
        payload,
        lambda: _dockerfiles_for_build(config_path),
    )


def test_up_compose_quote_in_dirname_is_not_executed(tmp_path: pathlib.Path) -> None:
    payload = _quote_payload()
    config_path = _write_project(tmp_path, payload, layout="src")
    _assert_dirname_is_not_executed(
        tmp_path,
        payload,
        lambda: _dockerfiles_for_up(
            config_path,
            engine_runtime_mode="combined_queue_worker",
        ),
    )


def test_up_distributed_quote_in_dirname_is_not_executed(
    tmp_path: pathlib.Path,
) -> None:
    payload = _quote_payload()
    config_path = _write_project(tmp_path, payload, layout="src")
    _assert_dirname_is_not_executed(
        tmp_path,
        payload,
        lambda: _dockerfiles_for_up(
            config_path,
            engine_runtime_mode="distributed",
        ),
    )
