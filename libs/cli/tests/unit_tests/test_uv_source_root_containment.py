"""Containment spec for `source.root` in uv image builds.

`langgraph.json` with `source.kind` `uv` resolves `source.root` on the machine
that runs `langgraph build`, `langgraph up`, `langgraph dockerfile`, or a local
`langgraph deploy`. When that directory is not the config directory, a child of
it, or a parent that contains it, the CLI still passes the directory to Docker
as the `uv-workspace-root` build context and copies the uv project's files into
the image. Files listed only in `.gitignore` are copied. Files listed in
`.dockerignore` are not.

These tests require that case to be rejected with `click.UsageError` before
Docker is invoked and before any file is created or deleted. Absolute
`source.root` values are rejected even when they resolve to the config
directory or to a parent of it. A relative root that resolves to a parent of
`langgraph.json` stays allowed: that is the in-repo uv workspace layout.

The path Docker would read is the path that must be judged. Retargeting a
symlink after `validate_config_file` returns, which is before `build`, `up`,
`dockerfile`, and local deploy build the image, still has to be rejected.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import textwrap
from collections.abc import Callable

import click
import pytest
from click.testing import CliRunner

import langgraph_cli.config as config_mod
import langgraph_cli.docker as docker_mod
from langgraph_cli.cli import cli
from langgraph_cli.config import config_to_docker, validate_config_file
from langgraph_cli.deploy import _run_local_build
from langgraph_cli.docker import build_docker_image
from langgraph_cli.exec import Runner

_BASE_IMAGE = "langchain/langgraph-api:0.2.47"
_OUTSIDE_MARKER = "outside-project-marker"
_INSIDE_MARKER = "inside-project-marker"

# Shape of `docker info -f {{json .}}` read by
# `langgraph_cli.docker.check_capabilities`.
_DOCKER_INFO = json.dumps(
    {
        "ServerVersion": "26.1.1",
        "ClientInfo": {"Plugins": [{"Name": "compose", "Version": "2.27.0"}]},
    }
)


class DockerInvoked(Exception):
    """A Docker command other than `docker info` was started."""

    def __init__(self, args: tuple, stdin: str | None) -> None:
        self.docker_args = args
        self.stdin = stdin
        super().__init__(" ".join(str(part) for part in args))


class CommandAccepted(Exception):
    """A CLI command returned without rejecting the source root."""

    def __init__(self, output: str) -> None:
        self.output = output
        super().__init__(output)


def _pyproject(name: str) -> str:
    return (
        textwrap.dedent(
            f"""
            [project]
            name = "{name}"
            version = "0.1.0"
            dependencies = ["httpx>=0.28"]

            [build-system]
            requires = ["setuptools>=61"]
            build-backend = "setuptools.build_meta"
            """
        ).strip()
        + "\n"
    )


def _write_single_project(
    root: pathlib.Path, name: str, marker: str | None = None
) -> pathlib.Path:
    """Write a one-package uv project.

    `not_ignored.txt` is listed in `.gitignore` only. `skipped.txt` is listed
    in `.dockerignore`. The uv image path applies `.dockerignore` and the
    built-in exclusions, and does not apply `.gitignore`.
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / "uv.lock").write_text("# uv lock file\n", encoding="utf-8")
    (root / "pyproject.toml").write_text(_pyproject(name), encoding="utf-8")
    (root / "src").mkdir()
    (root / "src" / "agent.py").write_text("graph = object()\n", encoding="utf-8")
    (root / ".dockerignore").write_text("skipped.txt\n", encoding="utf-8")
    (root / ".gitignore").write_text("not_ignored.txt\n", encoding="utf-8")
    (root / "skipped.txt").write_text("dropped-by-dockerignore\n", encoding="utf-8")
    if marker is not None:
        (root / "not_ignored.txt").write_text(marker + "\n", encoding="utf-8")
    return root


def _write_workspace(root: pathlib.Path) -> pathlib.Path:
    root.mkdir(parents=True)
    (root / "uv.lock").write_text("# uv lock file\n", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        textwrap.dedent(
            """
            [project]
            name = "workspace-root"
            version = "0.1.0"

            [tool.uv.workspace]
            members = ["apps/*", "libs/*"]

            [build-system]
            requires = ["setuptools>=61"]
            build-backend = "setuptools.build_meta"
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    agent = root / "apps" / "agent"
    shared = root / "libs" / "shared"
    agent.mkdir(parents=True)
    shared.mkdir(parents=True)
    (agent / "pyproject.toml").write_text(
        textwrap.dedent(
            """
            [project]
            name = "agent"
            version = "0.1.0"
            dependencies = ["shared", "httpx>=0.28"]

            [tool.uv.sources]
            shared = { workspace = true }

            [build-system]
            requires = ["setuptools>=61"]
            build-backend = "setuptools.build_meta"
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    (shared / "pyproject.toml").write_text(
        textwrap.dedent(
            """
            [project]
            name = "shared"
            version = "0.1.0"
            dependencies = ["anyio>=4"]

            [build-system]
            requires = ["setuptools>=61"]
            build-backend = "setuptools.build_meta"
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    graph = agent / "src" / "agent"
    graph.mkdir(parents=True)
    (graph / "graph.py").write_text("graph = object()\n", encoding="utf-8")
    (agent / "not_ignored.txt").write_text(_OUTSIDE_MARKER + "\n", encoding="utf-8")
    (root / ".dockerignore").write_text("skipped.txt\n", encoding="utf-8")
    (root / "skipped.txt").write_text("dropped-by-dockerignore\n", encoding="utf-8")
    return agent / "src" / "agent" / "graph.py"


def _write_config(
    config_dir: pathlib.Path,
    *,
    source_root: str,
    graph: str,
    package: str | None = None,
) -> pathlib.Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    source: dict[str, str] = {"kind": "uv", "root": source_root}
    if package is not None:
        source["package"] = package
    config_path = config_dir / "langgraph.json"
    config_path.write_text(
        json.dumps(
            {
                "python_version": "3.11",
                "graphs": {"agent": graph},
                "source": source,
            }
        ),
        encoding="utf-8",
    )
    return config_path


def _snapshot(root: pathlib.Path) -> dict[str, str]:
    if not root.exists():
        return {}
    recorded: dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        for name in sorted(filenames):
            path = pathlib.Path(dirpath) / name
            rel = path.relative_to(root).as_posix()
            if path.is_symlink():
                recorded[rel] = "symlink:" + os.readlink(path)
            else:
                recorded[rel] = path.read_bytes().hex()
    return recorded


def _install_docker_stub(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    """Record Docker invocations.

    `docker info` returns the JSON object `check_capabilities` parses. Every
    other command fails the test: a fix must reject the source root before
    `docker build`, `docker compose`, or `docker push`.
    """
    recorded: list[tuple] = []

    async def fake_subp_exec(*args, **kwargs):
        stdin = kwargs.get("input")
        recorded.append((args, stdin))
        if args[:2] == ("docker", "info"):
            return _DOCKER_INFO, None
        raise DockerInvoked(args, stdin)

    def which(cmd: str) -> str | None:
        if cmd == "docker":
            return "/usr/bin/docker"
        return None

    monkeypatch.setattr(docker_mod, "subp_exec", fake_subp_exec)
    monkeypatch.setattr("langgraph_cli.cli.subp_exec", fake_subp_exec)
    monkeypatch.setattr(docker_mod.shutil, "which", which)
    monkeypatch.setattr("langgraph_cli.cli.shutil.which", which)
    return recorded


def _retarget_after_validate(
    monkeypatch: pytest.MonkeyPatch, retarget: Callable[[], None]
) -> None:
    """Swap the source root after validation and before the image is built.

    `build`, `up`, `dockerfile`, and local deploy all call
    `validate_config_file` and then, later, resolve `source.root` again while
    constructing the Docker build.
    """
    real_validate = config_mod.validate_config_file

    def wrapped(config_path: pathlib.Path):
        validated = real_validate(config_path)
        retarget()
        return validated

    monkeypatch.setattr(config_mod, "validate_config_file", wrapped)


def _second_source_root_lookup(
    monkeypatch: pytest.MonkeyPatch,
    config_dir: pathlib.Path,
    source_root: str,
    decoy: pathlib.Path,
) -> None:
    """Return `decoy` from the second `Path.resolve` of `source.root`.

    `decoy` is the outside project the symlink is retargeted to. A fix that
    resolves the root, checks that result, then resolves the same path again
    and builds from the second result must still be rejected.
    """
    real_resolve = pathlib.Path.resolve
    lookups = {"count": 0}
    unresolved = config_dir / source_root

    def wrapped(self: pathlib.Path, *args, **kwargs):
        if pathlib.Path(self) == unresolved:
            lookups["count"] += 1
            if lookups["count"] >= 2:
                return real_resolve(decoy)
        return real_resolve(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "resolve", wrapped)


def _failure_detail(recorded: list[tuple], extra: str = "") -> str:
    lines = [extra] if extra else []
    for args, stdin in recorded:
        lines.append("command: " + " ".join(str(part) for part in args))
        if stdin and (_OUTSIDE_MARKER in stdin or "uv-workspace-root" in stdin):
            interesting = [
                line
                for line in stdin.splitlines()
                if "uv-workspace-root" in line
                or "not_ignored" in line
                or "skipped" in line
                or _OUTSIDE_MARKER in line
            ]
            lines.extend(interesting[:40])
    return "\n".join(lines)


def _assert_rejected(
    fn: Callable[[], object],
    *,
    recorded: list[tuple] | None = None,
    roots: list[pathlib.Path] | None = None,
    absent_files: list[pathlib.Path] | None = None,
) -> None:
    recorded = recorded if recorded is not None else []
    roots = roots or []
    absent_files = absent_files or []
    before = [_snapshot(root) for root in roots]
    try:
        fn()
    except click.UsageError:
        after = [_snapshot(root) for root in roots]
        assert before == after, "files changed before the source root was rejected"
        assert recorded == [] or all(
            args[:2] == ("docker", "info") for args, _stdin in recorded
        ), _failure_detail(
            recorded, "Docker was invoked before the source root was rejected"
        )
        for path in absent_files:
            assert not path.exists(), (
                f"{path} was created before the source root was rejected"
            )
        return
    except DockerInvoked as exc:
        pytest.fail(
            "Docker was invoked for a source root outside the config tree.\n"
            + _failure_detail([(exc.docker_args, exc.stdin)])
        )
    except CommandAccepted as exc:
        pytest.fail(
            "source root outside the config tree was accepted.\n"
            + exc.output
            + "\n"
            + _failure_detail(recorded)
        )
    else:
        pytest.fail(
            "source root outside the config tree was accepted.\n"
            + _failure_detail(recorded)
        )


def _invoke(args: list[str]) -> None:
    result = CliRunner().invoke(cli, args)
    if isinstance(result.exception, click.UsageError):
        raise result.exception
    if result.exception is not None:
        raise result.exception
    raise CommandAccepted(result.output)


def _assert_config_rejected(
    config_path: pathlib.Path, roots: list[pathlib.Path]
) -> None:
    def generate() -> None:
        config = validate_config_file(config_path)
        try:
            dockerfile, contexts = config_to_docker(
                config_path, config, base_image=_BASE_IMAGE
            )
        except click.UsageError:
            raise
        copied = "not_ignored.txt" in dockerfile
        ignored = "skipped.txt" in dockerfile
        pytest.fail(
            "source root was accepted.\n"
            f"contexts={contexts}\n"
            f"not_ignored.txt copied={copied}\n"
            f"skipped.txt copied={ignored}\n"
            + "\n".join(
                line
                for line in dockerfile.splitlines()
                if "COPY" in line or "ADD" in line or "uv-workspace" in line
            )
        )

    _assert_rejected(generate, roots=roots)


def _layout_outside_config(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    victim = _write_single_project(tmp_path / "victim", "victim-app", _OUTSIDE_MARKER)
    config_dir = tmp_path / "attacker"
    config_dir.mkdir()
    return victim, config_dir


def _link_projects(
    config_dir: pathlib.Path, initial: pathlib.Path, name: str = "root-link"
) -> pathlib.Path:
    link = config_dir / name
    link.symlink_to(initial, target_is_directory=True)
    return link


def test_absolute_source_root_is_rejected(tmp_path: pathlib.Path) -> None:
    victim, config_dir = _layout_outside_config(tmp_path)
    config_path = _write_config(
        config_dir,
        source_root=str(victim),
        graph=str(victim / "src" / "agent.py") + ":graph",
    )
    _assert_config_rejected(config_path, [victim, config_dir])


def test_absolute_source_root_of_the_config_directory_is_rejected(
    tmp_path: pathlib.Path,
) -> None:
    project = _write_single_project(tmp_path / "project", "local-app", _INSIDE_MARKER)
    config_path = _write_config(
        project,
        source_root=str(project.resolve()),
        graph="./src/agent.py:graph",
    )
    _assert_config_rejected(config_path, [project])


def test_absolute_parent_source_root_is_rejected(tmp_path: pathlib.Path) -> None:
    workspace = _write_single_project(
        tmp_path / "workspace", "workspace-app", _INSIDE_MARKER
    )
    config_dir = workspace / "app"
    config_dir.mkdir()
    config_path = _write_config(
        config_dir,
        source_root=str(workspace.resolve()),
        graph="../src/agent.py:graph",
    )
    _assert_config_rejected(config_path, [workspace])


def test_sibling_source_root_is_rejected(tmp_path: pathlib.Path) -> None:
    victim, config_dir = _layout_outside_config(tmp_path)
    config_path = _write_config(
        config_dir,
        source_root="../victim",
        graph="../victim/src/agent.py:graph",
    )
    _assert_config_rejected(config_path, [victim, config_dir])


def test_sibling_inside_the_same_git_repo_is_rejected(tmp_path: pathlib.Path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    victim, config_dir = _layout_outside_config(tmp_path)
    config_path = _write_config(
        config_dir,
        source_root="../victim",
        graph="../victim/src/agent.py:graph",
    )
    _assert_config_rejected(config_path, [victim, config_dir])


def test_climbed_source_root_is_rejected(tmp_path: pathlib.Path) -> None:
    victim, config_dir = _layout_outside_config(tmp_path)
    (config_dir / "nested").mkdir()
    config_path = _write_config(
        config_dir,
        source_root="nested/../../victim",
        graph="../victim/src/agent.py:graph",
    )
    _assert_config_rejected(config_path, [victim, config_dir])


def test_symlink_source_root_outside_the_config_tree_is_rejected(
    tmp_path: pathlib.Path,
) -> None:
    victim, config_dir = _layout_outside_config(tmp_path)
    _link_projects(config_dir, victim)
    config_path = _write_config(
        config_dir,
        source_root="root-link",
        graph="./root-link/src/agent.py:graph",
    )
    _assert_config_rejected(config_path, [victim, config_dir])


def test_workspace_source_root_outside_the_config_tree_is_rejected(
    tmp_path: pathlib.Path,
) -> None:
    workspace = tmp_path / "workspace"
    graph = _write_workspace(workspace)
    config_dir = tmp_path / "attacker"
    config_path = _write_config(
        config_dir,
        source_root="../workspace",
        graph=os.path.relpath(graph, config_dir) + ":graph",
        package="agent",
    )
    _assert_config_rejected(config_path, [workspace, config_dir])


def test_config_to_docker_rejects_root_moved_after_validation(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    victim, _inside, link, config_path, config_dir = _movable_root(tmp_path)
    _second_source_root_lookup(monkeypatch, config_dir, "root-link", victim)

    def generate() -> None:
        config = validate_config_file(config_path)
        link.unlink()
        link.symlink_to(victim, target_is_directory=True)
        try:
            dockerfile, contexts = config_to_docker(
                config_path, config, base_image=_BASE_IMAGE
            )
        except click.UsageError:
            raise
        pytest.fail(
            "source root moved after validation was accepted.\n"
            f"contexts={contexts}\n"
            f"outside marker copied={_OUTSIDE_MARKER in dockerfile}\n"
            + "\n".join(
                line
                for line in dockerfile.splitlines()
                if "COPY" in line or "ADD" in line or "uv-workspace" in line
            )
        )

    _assert_rejected(generate, roots=[victim, config_dir])


def test_build_docker_image_rejects_sibling_root(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    victim, config_dir = _layout_outside_config(tmp_path)
    config_path = _write_config(
        config_dir,
        source_root="../victim",
        graph="../victim/src/agent.py:graph",
    )
    recorded = _install_docker_stub(monkeypatch)
    config = validate_config_file(config_path)

    def build() -> None:
        with Runner() as runner:
            build_docker_image(
                runner,
                lambda _message: None,
                config_path,
                config,
                _BASE_IMAGE,
                None,
                False,
                "langgraph-test",
            )

    _assert_rejected(build, recorded=recorded, roots=[victim, config_dir])


def test_build_command_rejects_sibling_root(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    victim, config_dir = _layout_outside_config(tmp_path)
    config_path = _write_config(
        config_dir,
        source_root="../victim",
        graph="../victim/src/agent.py:graph",
    )
    recorded = _install_docker_stub(monkeypatch)
    _assert_rejected(
        lambda: _invoke(
            [
                "build",
                "--config",
                str(config_path),
                "--tag",
                "langgraph-test",
                "--no-pull",
            ]
        ),
        recorded=recorded,
        roots=[victim, config_dir],
    )


def test_build_command_rejects_root_moved_after_validation(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    victim, _inside, link, config_path, config_dir = _movable_root(tmp_path)
    _second_source_root_lookup(monkeypatch, config_dir, "root-link", victim)
    recorded = _install_docker_stub(monkeypatch)

    def retarget() -> None:
        link.unlink()
        link.symlink_to(victim, target_is_directory=True)

    _retarget_after_validate(monkeypatch, retarget)
    _assert_rejected(
        lambda: _invoke(
            [
                "build",
                "--config",
                str(config_path),
                "--tag",
                "langgraph-test",
                "--no-pull",
            ]
        ),
        recorded=recorded,
        roots=[victim, config_dir],
    )


def test_up_command_rejects_sibling_root(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    victim, config_dir = _layout_outside_config(tmp_path)
    config_path = _write_config(
        config_dir,
        source_root="../victim",
        graph="../victim/src/agent.py:graph",
    )
    recorded = _install_docker_stub(monkeypatch)
    _assert_rejected(
        lambda: _invoke(["up", "--config", str(config_path), "--no-pull"]),
        recorded=recorded,
        roots=[victim, config_dir],
    )


def test_up_command_rejects_root_moved_after_validation(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    victim, _inside, link, config_path, config_dir = _movable_root(tmp_path)
    _second_source_root_lookup(monkeypatch, config_dir, "root-link", victim)
    recorded = _install_docker_stub(monkeypatch)

    def retarget() -> None:
        link.unlink()
        link.symlink_to(victim, target_is_directory=True)

    _retarget_after_validate(monkeypatch, retarget)
    _assert_rejected(
        lambda: _invoke(["up", "--config", str(config_path), "--no-pull"]),
        recorded=recorded,
        roots=[victim, config_dir],
    )


def test_dockerfile_command_rejects_sibling_root(
    tmp_path: pathlib.Path,
) -> None:
    victim, config_dir = _layout_outside_config(tmp_path)
    config_path = _write_config(
        config_dir,
        source_root="../victim",
        graph="../victim/src/agent.py:graph",
    )
    save_path = tmp_path / "Dockerfile"
    _assert_rejected(
        lambda: _invoke(["dockerfile", "--config", str(config_path), str(save_path)]),
        roots=[victim, config_dir],
        absent_files=[save_path],
    )


def test_dockerfile_command_rejects_root_moved_after_validation(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    victim, _inside, link, config_path, config_dir = _movable_root(tmp_path)
    _second_source_root_lookup(monkeypatch, config_dir, "root-link", victim)
    save_path = tmp_path / "Dockerfile"

    def retarget() -> None:
        link.unlink()
        link.symlink_to(victim, target_is_directory=True)

    _retarget_after_validate(monkeypatch, retarget)
    _assert_rejected(
        lambda: _invoke(["dockerfile", "--config", str(config_path), str(save_path)]),
        roots=[victim, config_dir],
        absent_files=[save_path],
    )


def test_local_deploy_rejects_sibling_root_before_push(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    victim, config_dir = _layout_outside_config(tmp_path)
    config_path = _write_config(
        config_dir,
        source_root="../victim",
        graph="../victim/src/agent.py:graph",
    )
    recorded = _install_docker_stub(monkeypatch)
    config = validate_config_file(config_path)

    def deploy() -> None:
        _run_local_build(
            client=object(),
            deployment_id="deployment",
            step=1,
            config=config_path,
            config_json=config,
            verbose=False,
            pull=False,
            api_version=None,
            base_image=_BASE_IMAGE,
            image_name=None,
            prebuilt_image=None,
            name="name",
            tag="tag",
            install_command=None,
            build_command=None,
            docker_build_args=(),
            secrets=[],
            tracked_packages=None,
        )

    _assert_rejected(deploy, recorded=recorded, roots=[victim, config_dir])


def test_local_deploy_rejects_root_moved_after_validation(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    victim, _inside, link, config_path, config_dir = _movable_root(tmp_path)
    _second_source_root_lookup(monkeypatch, config_dir, "root-link", victim)
    recorded = _install_docker_stub(monkeypatch)

    def retarget() -> None:
        link.unlink()
        link.symlink_to(victim, target_is_directory=True)

    def deploy() -> None:
        config = validate_config_file(config_path)
        retarget()
        _run_local_build(
            client=object(),
            deployment_id="deployment",
            step=1,
            config=config_path,
            config_json=config,
            verbose=False,
            pull=False,
            api_version=None,
            base_image=_BASE_IMAGE,
            image_name=None,
            prebuilt_image=None,
            name="name",
            tag="tag",
            install_command=None,
            build_command=None,
            docker_build_args=(),
            secrets=[],
            tracked_packages=None,
        )

    _assert_rejected(deploy, recorded=recorded, roots=[victim, config_dir])


def test_relative_parent_workspace_root_is_allowed(tmp_path: pathlib.Path) -> None:
    """A relative root that contains `langgraph.json` is the workspace layout."""
    workspace = _write_single_project(
        tmp_path / "workspace", "workspace-app", _INSIDE_MARKER
    )
    config_dir = workspace / "app"
    config_path = _write_config(
        config_dir,
        source_root="..",
        graph="../src/agent.py:graph",
    )
    config = validate_config_file(config_path)
    dockerfile, contexts = config_to_docker(config_path, config, base_image=_BASE_IMAGE)
    assert contexts == {"uv-workspace-root": str(workspace.resolve())}
    assert "not_ignored.txt" in dockerfile
    assert "skipped.txt" not in dockerfile
    assert _INSIDE_MARKER not in dockerfile


def _movable_root(
    tmp_path: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path, pathlib.Path, pathlib.Path]:
    victim = _write_single_project(tmp_path / "victim", "victim-app", _OUTSIDE_MARKER)
    config_dir = tmp_path / "attacker"
    inside = _write_single_project(
        config_dir / "nested-app", "inside-app", _INSIDE_MARKER
    )
    link = _link_projects(config_dir, inside)
    config_path = _write_config(
        config_dir,
        source_root="root-link",
        graph="./root-link/src/agent.py:graph",
    )
    return victim, inside, link, config_path, config_dir
