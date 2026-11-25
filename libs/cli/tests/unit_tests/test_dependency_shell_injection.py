"""Spec tests for shell injection through langgraph.json dependency strings.

`python_config_to_docker` places every dependency that does not start with a
dot into a shell-form `RUN` line. These tests drive the `langgraph build` and
`langgraph up` paths and then apply the Dockerfile shell-form rules from
BuildKit v0.21.1:

- `parser/parser.go` splits the file on newlines and joins only backslash
  continuations.
- `parser/split_command.go` separates the instruction from builder flags.
- `parser/line_parsers.go` `parseMaybeJSON` keeps a non-JSON `RUN` as one
  string.
- `instructions/support.go` `handleJSONArgs` joins that string for the shell.
- `dockerfile2llb/defaultshell.go` and `withShell` execute it as `/bin/sh -c`.
- A non-zero `RUN` stops the build, so later instructions are not executed.

The injection tests fail while a dependency string can still become shell or a
new `RUN`. They pass when generation rejects that string, or the installer
receives it as data (quoted shell or a JSON exec-form array) and the shell
never runs the payload.
"""

from __future__ import annotations

import json
import pathlib
import shlex
import subprocess
import textwrap
from dataclasses import dataclass

import click
import pytest

from langgraph_cli.cli import prepare
from langgraph_cli.config import validate_config_file
from langgraph_cli.docker import DockerCapabilities, Version, build_docker_image
from langgraph_cli.exec import Runner

_CAPABILITIES = DockerCapabilities(
    version_docker=Version(26, 1, 1),
    version_compose=Version(2, 27, 0),
    healthcheck_start_interval=False,
)


@dataclass(frozen=True)
class RunInstruction:
    """One Dockerfile RUN, after builder flags are removed."""

    shell_body: str | None
    exec_args: tuple[str, ...] | None


def _write_project(
    directory: pathlib.Path,
    dependencies: list[str],
    *,
    pip_installer: str | None = None,
) -> pathlib.Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "agent.py").write_text("graph = None\n", encoding="utf-8")
    config: dict = {
        "python_version": "3.11",
        "dependencies": [".", *dependencies],
        "graphs": {"agent": "./agent.py:graph"},
    }
    if pip_installer is not None:
        config["pip_installer"] = pip_installer
    config_path = directory / "langgraph.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def _logical_lines(dockerfile: str) -> list[str]:
    """Split a Dockerfile the way BuildKit's parser scans lines.

    Continuation matches `directives.lineContinuationRegex` for the default
    backslash escape token: a trailing `\\` joins the next physical line, and
    the backslash is removed. Quote characters do not join lines.
    """
    logical: list[str] = []
    pending = ""
    continued = False
    for physical in dockerfile.splitlines():
        stripped_newline = physical.rstrip("\r")
        if not continued:
            comment_check = stripped_newline.lstrip()
            if comment_check.startswith("#") or comment_check == "":
                continue
            piece = stripped_newline.lstrip()
        else:
            piece = stripped_newline
        if piece.endswith("\\") and not piece.endswith("\\\\"):
            pending += piece[:-1]
            continued = True
            continue
        pending += piece
        logical.append(pending)
        pending = ""
        continued = False
    if pending:
        logical.append(pending)
    return logical


def _run_instructions(dockerfile: str) -> list[RunInstruction]:
    instructions: list[RunInstruction] = []
    for line in _logical_lines(dockerfile):
        parts = line.split(None, 1)
        if not parts or parts[0].lower() != "run":
            continue
        rest = parts[1] if len(parts) > 1 else ""
        rest = _strip_builder_flags(rest)
        if rest.startswith("["):
            try:
                parsed = json.loads(rest)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list) and all(
                isinstance(item, str) for item in parsed
            ):
                instructions.append(
                    RunInstruction(shell_body=None, exec_args=tuple(parsed))
                )
                continue
        instructions.append(RunInstruction(shell_body=rest, exec_args=None))
    return instructions


def _strip_builder_flags(rest: str) -> str:
    """Drop leading `--flag` words, matching `extractBuilderFlags`."""
    remaining = rest.lstrip()
    while remaining.startswith("--"):
        flag, separator, tail = remaining.partition(" ")
        if separator == "":
            return ""
        if flag == "--":
            return tail.lstrip()
        remaining = tail.lstrip()
    return remaining


def _execute_shell(body: str, *, cwd: pathlib.Path, env: dict[str, str]) -> int:
    completed = subprocess.run(
        ["/bin/sh", "-c", body],
        cwd=cwd,
        env=env,
        check=False,
    )
    return completed.returncode


def _host_unsafe(body: str) -> bool:
    """Template cleanup and package-metadata RUNs must not run on the host."""
    return (
        body.lstrip().startswith("set -ex")
        or body.lstrip().startswith("for dep in /deps")
        or body.lstrip().startswith("mkdir ")
        or " pip uninstall" in body
        or body.lstrip().startswith("rm ")
        or " rm -rf " in f" {body} "
        or ">>" in body
    )


def _run_build_steps(
    dockerfile: str,
    *,
    cwd: pathlib.Path,
    env: dict[str, str],
) -> list[RunInstruction]:
    """Run shell-form RUNs in order, stopping after the first non-zero status.

    Instructions that delete files or write package metadata are the fixed
    template appended after the dependency install. They are not executed on
    the host; a failing install still prevents them from running, matching a
    build that stops at the first failed `RUN`.
    """
    ran: list[RunInstruction] = []
    for instruction in _run_instructions(dockerfile):
        ran.append(instruction)
        if instruction.exec_args is not None:
            continue
        body = instruction.shell_body or ""
        if _host_unsafe(body):
            break
        if _execute_shell(body, cwd=cwd, env=env) != 0:
            break
    return ran


def _installer_args(instructions: list[RunInstruction]) -> list[str] | None:
    for instruction in instructions:
        args = instruction.exec_args
        if args is None:
            continue
        if "install" in args:
            return list(args)
    return None


def _base_env(cwd: pathlib.Path) -> dict[str, str]:
    return {
        "PATH": "/usr/bin:/bin",
        "HOME": str(cwd),
        "LANG": "C",
    }


def _dockerfile_from_build(
    config_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> str | None:
    """Return the Dockerfile `langgraph build` passes to `docker build` on stdin."""
    captured: dict[str, object] = {}

    async def fake_subp_exec(*args, input=None, **kwargs):  # type: ignore[no-untyped-def]
        captured["args"] = args
        captured["input"] = input
        return None, None

    monkeypatch.setattr("langgraph_cli.docker.subp_exec", fake_subp_exec)
    config_json = validate_config_file(config_path)
    try:
        with Runner() as runner:
            build_docker_image(
                runner,
                lambda _message: None,
                config_path,
                config_json,
                None,
                None,
                False,
                "dependency-injection-spec",
                verbose=False,
            )
    except (click.UsageError, ValueError):
        return None
    assert captured["args"][0] == "docker"
    assert captured["args"][1] == "build"
    assert "-f" in captured["args"] and "-" in captured["args"]
    dockerfile = captured["input"]
    assert isinstance(dockerfile, str)
    return dockerfile


def _dockerfiles_from_up(
    config_path: pathlib.Path,
    *,
    engine_runtime_mode: str = "combined_queue_worker",
) -> list[str] | None:
    """Return Dockerfiles embedded for `langgraph up`."""
    try:
        _args, stdin = prepare(
            None,
            capabilities=_CAPABILITIES,
            config_path=config_path,
            docker_compose=None,
            port=8123,
            pull=False,
            watch=False,
            verbose=False,
            engine_runtime_mode=engine_runtime_mode,
        )
    except (click.UsageError, ValueError):
        return None
    return _extract_inline_dockerfiles(stdin)


def _extract_inline_dockerfiles(compose_text: str) -> list[str]:
    lines = compose_text.splitlines()
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
            if line and not line.startswith(" ") and not line.startswith("\t"):
                break
            # The next compose key is indented less than the Dockerfile body.
            if line.strip() and not line.startswith("                ") and ":" in line:
                break
            block.append(line)
            index += 1
        dockerfiles.append(textwrap.dedent("\n".join(block)))
    if not dockerfiles:
        raise AssertionError("langgraph up did not embed a Dockerfile")
    return dockerfiles


def _assert_payload_does_not_run(dockerfiles: list[str], marker: pathlib.Path) -> None:
    env = _base_env(marker.parent)
    for dockerfile in dockerfiles:
        _run_build_steps(dockerfile, cwd=marker.parent, env=env)
        assert not marker.exists(), (
            "A dependency string became a shell command during the image build.\n"
            f"Marker: {marker}\nDockerfile:\n{dockerfile}"
        )


def test_build_semicolon_dependency_does_not_run_shell(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "semicolon-marker"
    config_path = _write_project(
        tmp_path / "project",
        [f"requests; touch {shlex.quote(str(marker))}"],
    )
    dockerfile = _dockerfile_from_build(config_path, monkeypatch)
    if dockerfile is None:
        assert not marker.exists()
        return
    _assert_payload_does_not_run([dockerfile], marker)


def test_build_newline_dependency_does_not_add_a_run(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "newline-marker"
    # `; true` makes the install line succeed even when the installer is
    # missing, which is what lets a following RUN execute in `docker build`.
    # The extra `#` line keeps a middle RUN intact if the whole string is
    # wrapped in one pair of shell quotes: the Dockerfile parser splits on
    # newlines before the shell parses quotes.
    payload = f"; true\nRUN touch {shlex.quote(str(marker))}\n#"
    config_path = _write_project(tmp_path / "project", [payload])
    dockerfile = _dockerfile_from_build(config_path, monkeypatch)
    if dockerfile is None:
        assert not marker.exists()
        return
    _assert_payload_does_not_run([dockerfile], marker)


@pytest.mark.parametrize("pip_installer", ["pip", "uv"])
def test_build_installer_choice_does_not_change_injection(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    pip_installer: str,
) -> None:
    marker = tmp_path / "installer-marker"
    config_path = _write_project(
        tmp_path / "project",
        [f"requests; touch {shlex.quote(str(marker))}"],
        pip_installer=pip_installer,
    )
    dockerfile = _dockerfile_from_build(config_path, monkeypatch)
    if dockerfile is None:
        assert not marker.exists()
        return
    _assert_payload_does_not_run([dockerfile], marker)


@pytest.mark.parametrize(
    "engine_runtime_mode",
    ["combined_queue_worker", "distributed"],
)
def test_up_embeds_dependencies_without_running_shell(
    tmp_path: pathlib.Path,
    engine_runtime_mode: str,
) -> None:
    marker = tmp_path / "up-marker"
    config_path = _write_project(
        tmp_path / "project",
        [f"requests; touch {shlex.quote(str(marker))}"],
    )
    dockerfiles = _dockerfiles_from_up(
        config_path, engine_runtime_mode=engine_runtime_mode
    )
    if dockerfiles is None:
        assert not marker.exists()
        return
    _assert_payload_does_not_run(dockerfiles, marker)


def test_version_specifier_reaches_the_installer_as_one_argument(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`httpx>=0.28` is a real requirement. `>` must not be a shell redirect."""
    project = tmp_path / "project"
    config_path = _write_project(project, ["httpx>=0.28"])
    dockerfile = _dockerfile_from_build(config_path, monkeypatch)
    assert dockerfile is not None, "A version specifier is a valid dependency"

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    argv_file = tmp_path / "argv"
    stub = bin_dir / "uv"
    stub.write_text(
        '#!/bin/sh\nprintf \'%s\\n\' "$@" > "$STUB_ARGV"\n',
        encoding="utf-8",
    )
    stub.chmod(0o755)
    (bin_dir / "pip").write_text(stub.read_text(encoding="utf-8"), encoding="utf-8")
    (bin_dir / "pip").chmod(0o755)

    work = tmp_path / "work"
    work.mkdir()
    env = _base_env(work)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["STUB_ARGV"] = str(argv_file)
    ran = _run_build_steps(dockerfile, cwd=work, env=env)

    assert not (work / "=0.28").exists(), (
        "The unquoted requirement was parsed as a shell redirection.\n"
        f"Dockerfile:\n{dockerfile}"
    )
    recorded = (
        argv_file.read_text(encoding="utf-8").splitlines() if argv_file.exists() else []
    )
    exec_args = _installer_args(ran)
    received = exec_args if exec_args is not None else recorded
    assert "httpx>=0.28" in received, (
        "The installer did not receive the version specifier as one argument.\n"
        f"Received: {received}\nDockerfile:\n{dockerfile}"
    )


def test_environment_marker_is_not_a_shell_separator(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PEP 508 markers use `;` and must be one installer argument, not a command."""
    specifier = 'requests; python_version < "3.12"'
    # A shell that splits on `;` would try to run `python_version`. Point that
    # name at a script that records the split. Quoted or exec-form data must
    # not invoke it. Rejecting every semicolon is not a fix: markers are valid.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    invoked = tmp_path / "python-version-invoked"
    (bin_dir / "python_version").write_text(
        f"#!/bin/sh\ntouch {shlex.quote(str(invoked))}\n",
        encoding="utf-8",
    )
    (bin_dir / "python_version").chmod(0o755)
    for name in ("uv", "pip"):
        path = bin_dir / name
        path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        path.chmod(0o755)

    config_path = _write_project(tmp_path / "project", [specifier])
    dockerfile = _dockerfile_from_build(config_path, monkeypatch)
    assert dockerfile is not None, "A PEP 508 environment marker is a valid dependency"

    env = _base_env(tmp_path)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    ran = _run_build_steps(dockerfile, cwd=tmp_path, env=env)
    assert not invoked.exists(), (
        "A PEP 508 environment marker was executed as a shell command.\n"
        f"Dockerfile:\n{dockerfile}"
    )
    exec_args = _installer_args(ran)
    if exec_args is not None:
        assert specifier in exec_args
    else:
        # Shell form must keep the marker in the installer command line as data.
        # Re-run the install body with a stub that records its own arguments.
        argv_file = tmp_path / "argv"
        recorder = bin_dir / "uv"
        recorder.write_text(
            '#!/bin/sh\nprintf \'%s\\n\' "$@" > "$STUB_ARGV"\nexit 0\n',
            encoding="utf-8",
        )
        recorder.chmod(0o755)
        (bin_dir / "pip").write_text(
            recorder.read_text(encoding="utf-8"), encoding="utf-8"
        )
        env["STUB_ARGV"] = str(argv_file)
        install_bodies = [
            instruction.shell_body
            for instruction in _run_instructions(dockerfile)
            if instruction.shell_body and " install " in instruction.shell_body
        ]
        assert install_bodies, dockerfile
        _execute_shell(install_bodies[0], cwd=tmp_path, env=env)
        recorded = (
            argv_file.read_text(encoding="utf-8").splitlines()
            if argv_file.exists()
            else []
        )
        assert specifier in recorded, (
            "The installer did not receive the environment marker as one argument.\n"
            f"Received: {recorded}\nDockerfile:\n{dockerfile}"
        )
