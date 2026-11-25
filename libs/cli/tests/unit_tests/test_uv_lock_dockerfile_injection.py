"""Dockerfile lines built from uv workspace paths must reject unsafe names.

`langgraph build`, `langgraph up`, and `langgraph dockerfile` all render the
image from `config_to_docker` (`python_config_to_docker_uv_lock` when
`source.kind` is `uv`). Workspace directory and file names are written into
`ADD`, `COPY`, and `WORKDIR` lines with no quoting. A newline or carriage
return in a name splits the line into a further Dockerfile instruction, and a
space splits one instruction into extra arguments.

These tests fail while that embedding is still present. They pass once
generation refuses the path, with no Dockerfile or other file written first.

The non-uv Dockerfile builders in `config.py` (`python_config_to_docker` for
local dependencies, and `node_config_to_docker`) also interpolate paths into
`ADD` and `COPY` lines. Those builders are not on the uv attack path and are
not covered here.
"""

import inspect
import json
import pathlib
import tempfile
import textwrap
from collections.abc import Iterator
from contextlib import contextmanager

import click
import pytest
from click.testing import CliRunner

from langgraph_cli.cli import cli
from langgraph_cli.config import (
    config_to_compose,
    config_to_docker,
    validate_config,
)
from langgraph_cli.docker import build_docker_image

_BASE_IMAGE = "langchain/langgraph-api:0.2.47"

# The first physical line of the emitted ADD/COPY is a complete instruction
# (`ADD pyproject.toml .` or `COPY --from=uv-workspace-root pyproject.toml .`).
# The next line is a separate `RUN` that BuildKit's Dockerfile parser accepts.
_NEWLINE_DIR = "pyproject.toml .\nRUN echo PWNED\n#"
_CRLF_DIR = "pyproject.toml .\r\nRUN echo PWNED\r\n#"
_BARE_NEWLINE_DIR = "member\nRUN echo PWNED"
_CR_DIR = "member\rRUN"
_SPACE_DIR = "evil pkg"
_FILE_NAME = "readme .\nRUN echo PWNED\n#"


class _DockerRunner:
    """Records image-build invocations without starting a container runtime."""

    def __init__(self) -> None:
        self.calls: list[object] = []

    def run(self, command: object) -> None:
        self.calls.append(command)
        if inspect.iscoroutine(command):
            command.close()


def _package_toml(name: str, dependencies: list[str] | None = None) -> str:
    return (
        textwrap.dedent(
            f"""
            [project]
            name = "{name}"
            version = "0.1.0"
            dependencies = {(dependencies or [])!r}

            [build-system]
            requires = ["setuptools>=61"]
            build-backend = "setuptools.build_meta"
            """
        ).strip()
        + "\n"
    )


def _snapshot(root: pathlib.Path) -> tuple[dict[str, bytes], frozenset[str]]:
    files: dict[str, bytes] = {}
    directories: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            directories.add(relative)
        elif path.is_file():
            files[relative] = path.read_bytes()
    return files, frozenset(directories)


@contextmanager
def _workspace(
    *,
    members: list[str],
    config_dir: str = "",
    source_root: str | None = None,
    package: str | None = None,
    app_dependencies: list[str] | None = None,
    app_sources: str = "",
    extra_members: dict[str, str] | None = None,
    extra_files: dict[str, str] | None = None,
) -> Iterator[tuple[pathlib.Path, pathlib.Path]]:
    """Yield `(project_root, langgraph.json)` for a temporary uv workspace."""
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        (root / "uv.lock").write_text("# uv lock file\n", encoding="utf-8")
        (root / "pyproject.toml").write_text(
            textwrap.dedent(
                f"""
                [project]
                name = "app"
                version = "0.1.0"
                dependencies = {(app_dependencies or [])!r}

                [tool.uv.workspace]
                members = {members!r}

                {app_sources}

                [build-system]
                requires = ["setuptools>=61"]
                build-backend = "setuptools.build_meta"
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        (root / "src").mkdir()
        (root / "src" / "agent.py").write_text("graph = object()\n", encoding="utf-8")

        for relative, body in (extra_members or {}).items():
            package_root = root / relative
            package_root.mkdir(parents=True)
            (package_root / "pyproject.toml").write_text(body, encoding="utf-8")
        for relative, body in (extra_files or {}).items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body, encoding="utf-8")

        config_path = (
            root / config_dir / "langgraph.json"
            if config_dir
            else root / "langgraph.json"
        )
        config_path.parent.mkdir(parents=True, exist_ok=True)
        source: dict[str, str] = {"kind": "uv"}
        if source_root is not None:
            source["root"] = source_root
        if package is not None:
            source["package"] = package
        config_path.write_text(
            json.dumps(
                {
                    "python_version": "3.11",
                    "graphs": {"agent": "agent:graph"},
                    "source": source,
                }
            ),
            encoding="utf-8",
        )
        yield root, config_path


def _load(config_path: pathlib.Path) -> dict:
    return validate_config(json.loads(config_path.read_text(encoding="utf-8")))


def _assert_refused(root: pathlib.Path, config_path: pathlib.Path) -> None:
    before = _snapshot(root)
    with pytest.raises(click.UsageError):
        config_to_docker(config_path, _load(config_path), base_image=_BASE_IMAGE)
    assert _snapshot(root) == before


def test_safe_workspace_member_is_copied() -> None:
    """A normal member path is still embedded. Refusal must be limited to unsafe names."""
    with _workspace(
        members=["libs/*"],
        extra_members={"libs/shared": _package_toml("shared")},
    ) as (_root, config_path):
        dockerfile, contexts = config_to_docker(
            config_path,
            _load(config_path),
            base_image=_BASE_IMAGE,
        )

    assert contexts == {}
    assert (
        "ADD libs/shared/pyproject.toml /tmp/uv_export/project/libs/shared/pyproject.toml"
        in dockerfile
    )
    assert "\nRUN echo PWNED\n" not in dockerfile


def test_newline_in_workspace_member_directory_is_rejected() -> None:
    with _workspace(
        members=["*"],
        extra_members={_NEWLINE_DIR: _package_toml("evilpkg")},
    ) as (root, config_path):
        _assert_refused(root, config_path)


def test_bare_newline_in_workspace_member_directory_is_rejected() -> None:
    with _workspace(
        members=["*"],
        extra_members={_BARE_NEWLINE_DIR: _package_toml("evilpkg")},
    ) as (root, config_path):
        _assert_refused(root, config_path)


def test_crlf_in_workspace_member_directory_is_rejected() -> None:
    with _workspace(
        members=["*"],
        extra_members={_CRLF_DIR: _package_toml("evilpkg")},
    ) as (root, config_path):
        _assert_refused(root, config_path)


def test_carriage_return_in_workspace_member_directory_is_rejected() -> None:
    with _workspace(
        members=["*"],
        extra_members={_CR_DIR: _package_toml("evilpkg")},
    ) as (root, config_path):
        _assert_refused(root, config_path)


def test_space_in_workspace_member_directory_is_rejected() -> None:
    with _workspace(
        members=["*"],
        extra_members={_SPACE_DIR: _package_toml("evil-pkg")},
    ) as (root, config_path):
        _assert_refused(root, config_path)


def test_newline_member_used_as_dependency_is_rejected() -> None:
    """Dependency members also feed the package comment, COPY/ADD, and WORKDIR."""
    with _workspace(
        members=["*"],
        app_dependencies=["evilpkg"],
        app_sources="[tool.uv.sources]\nevilpkg = { workspace = true }\n",
        extra_members={_NEWLINE_DIR: _package_toml("evilpkg")},
    ) as (root, config_path):
        _assert_refused(root, config_path)


def test_newline_member_uses_copy_from_additional_context() -> None:
    """langgraph.json outside the project root selects `COPY --from`, not `ADD`."""
    with _workspace(
        members=["*"],
        config_dir="deploy",
        source_root="..",
        package="app",
        extra_members={_NEWLINE_DIR: _package_toml("evilpkg")},
    ) as (root, config_path):
        _assert_refused(root, config_path)


def test_newline_in_copied_workspace_file_is_rejected() -> None:
    with _workspace(
        members=[],
        extra_files={"readme": "ok\n", _FILE_NAME: "x\n"},
    ) as (root, config_path):
        _assert_refused(root, config_path)


def test_build_docker_image_refuses_before_invoking_docker() -> None:
    with _workspace(
        members=["*"],
        extra_members={_NEWLINE_DIR: _package_toml("evilpkg")},
    ) as (root, config_path):
        runner = _DockerRunner()
        before = _snapshot(root)
        with pytest.raises(click.UsageError):
            build_docker_image(
                runner,
                lambda _message: None,
                config_path,
                _load(config_path),
                _BASE_IMAGE,
                None,
                False,
                "langgraph-test",
            )
        assert runner.calls == []
        assert _snapshot(root) == before


def test_compose_for_up_refuses_newline_member() -> None:
    with _workspace(
        members=["*"],
        extra_members={_NEWLINE_DIR: _package_toml("evilpkg")},
    ) as (root, config_path):
        before = _snapshot(root)
        with pytest.raises(click.UsageError):
            config_to_compose(
                config_path,
                _load(config_path),
                base_image=_BASE_IMAGE,
            )
        assert _snapshot(root) == before


def test_dockerfile_command_does_not_write_a_file() -> None:
    with _workspace(
        members=["*"],
        extra_members={_NEWLINE_DIR: _package_toml("evilpkg")},
    ) as (root, config_path):
        save_path = root / "Dockerfile"
        before = _snapshot(root)
        result = CliRunner().invoke(
            cli,
            ["dockerfile", str(save_path), "--config", str(config_path)],
        )
        assert result.exit_code != 0
        assert not save_path.exists()
        assert _snapshot(root) == before
