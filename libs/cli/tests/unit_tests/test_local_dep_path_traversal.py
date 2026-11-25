"""Local dependency paths must stay inside the project being built.

`langgraph up`, `langgraph build`, `langgraph dockerfile`, and `langgraph deploy`
load `dependencies` from the project's langgraph.json. Entries that start with
`.` are host paths. These tests are the spec for confining those paths to the
project directory: they fail while a dependency can resolve outside the project
and succeed once that dependency is rejected before its files are staged.
"""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
import tarfile
from collections.abc import Callable

import pytest

import langgraph_cli.docker as docker_mod
from langgraph_cli.archive import create_archive
from langgraph_cli.cli import prepare_args_and_stdin
from langgraph_cli.config import (
    config_to_compose,
    config_to_docker,
    validate_config_file,
)
from langgraph_cli.docker import DockerCapabilities, Version, build_docker_image

MARKER_NAME = "marker.txt"
MARKER = b"OUTSIDE_PROJECT_MARKER"
OUTSIDE_NAME = "hosttree"

_CAPABILITIES = DockerCapabilities(
    version_docker=Version(26, 1, 1),
    version_compose=Version(2, 27, 0),
    healthcheck_start_interval=False,
)


def _escape_relative(target: pathlib.Path) -> str:
    """Relative dependency that reaches `target` by way of the filesystem root.

    Extra `..` segments stay at `/` after normalization, so the same string
    escapes no matter how deep the project directory is.
    """
    relative_from_root = target.resolve().relative_to(pathlib.Path("/"))
    return ("../" * 40) + relative_from_root.as_posix()


def _write_config(project: pathlib.Path, dependencies: list[str]) -> pathlib.Path:
    (project / "agent.py").write_text("graph = object()\n", encoding="utf-8")
    config_path = project / "langgraph.json"
    config_path.write_text(
        json.dumps(
            {
                "dependencies": dependencies,
                "graphs": {"agent": "./agent.py:graph"},
            }
        ),
        encoding="utf-8",
    )
    return config_path


def _populate_outside(outside: pathlib.Path, *, real_package: bool) -> None:
    outside.mkdir(parents=True, exist_ok=True)
    (outside / MARKER_NAME).write_bytes(MARKER)
    (outside / "requirements.txt").write_text("example-package==0\n", encoding="utf-8")
    if real_package:
        (outside / "pyproject.toml").write_text(
            '[project]\nname = "hosttree"\nversion = "0.0.0"\n',
            encoding="utf-8",
        )


def _project_with_dependency(
    tmp_path: pathlib.Path,
    dependency: str,
    *,
    outside: pathlib.Path,
    real_package: bool,
) -> tuple[pathlib.Path, pathlib.Path]:
    project = tmp_path / "repo" / "nested" / "app"
    project.mkdir(parents=True)
    config_path = _write_config(project, [".", dependency])
    _populate_outside(outside, real_package=real_package)
    resolved = (project / dependency).resolve()
    assert resolved == outside.resolve()
    assert project.resolve() not in resolved.parents
    assert resolved != project.resolve()
    return config_path, outside.resolve()


def _cases(tmp_path: pathlib.Path) -> dict[str, tuple[pathlib.Path, pathlib.Path]]:
    """Dependencies that resolve outside the project, plus the directory they name."""
    parent = tmp_path / "workspace"
    project = parent / "app"
    project.mkdir(parents=True)
    parent_config = _write_config(project, [".", ".."])
    (parent / MARKER_NAME).write_bytes(MARKER)
    assert (project / "..").resolve() == parent.resolve()

    outside = tmp_path / OUTSIDE_NAME
    faux_config, faux_outside = _project_with_dependency(
        tmp_path / "faux",
        _escape_relative(outside),
        outside=outside,
        real_package=False,
    )
    real_root = tmp_path / "realcase"
    real_outside = real_root / OUTSIDE_NAME
    real_config, real_resolved = _project_with_dependency(
        real_root,
        _escape_relative(real_outside),
        outside=real_outside,
        real_package=True,
    )
    link_root = tmp_path / "linkcase"
    link_outside = link_root / OUTSIDE_NAME
    link_project = link_root / "repo" / "nested" / "app"
    link_project.mkdir(parents=True)
    _populate_outside(link_outside, real_package=False)
    alias = link_project / "alias"
    alias.symlink_to(link_outside, target_is_directory=True)
    link_config = _write_config(link_project, [".", "./alias"])
    assert alias.resolve() == link_outside.resolve()

    return {
        "ancestor": (parent_config, parent.resolve()),
        "root_relative_faux": (faux_config, faux_outside),
        "root_relative_real_package": (real_config, real_resolved),
        "symlink": (link_config, link_outside.resolve()),
    }


@pytest.fixture
def staged_archives(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, bytes]]:
    """Record every file written into a deploy archive, including a later refusal."""
    staged: list[tuple[str, bytes]] = []
    original = tarfile.TarFile.addfile

    def wrapped(self, tarinfo, fileobj=None):
        data = b""
        if fileobj is not None:
            data = fileobj.read()
            if hasattr(fileobj, "seek"):
                fileobj.seek(0)
        staged.append((tarinfo.name, data))
        return original(self, tarinfo, fileobj)

    monkeypatch.setattr(tarfile.TarFile, "addfile", wrapped)
    return staged


def _assert_refused(
    operation: Callable[[], object],
    outside: pathlib.Path,
    staged: list[tuple[str, bytes]],
) -> None:
    try:
        result = operation()
    except AssertionError:
        raise
    except Exception:
        result = None
        refused = True
    else:
        refused = False

    leaked = [
        name
        for name, data in staged
        if data == MARKER or name.endswith("/" + MARKER_NAME) or name == MARKER_NAME
    ]
    rendered = repr(result)
    assert leaked == [], f"outside file was staged before refusal: {leaked}"
    assert str(outside) not in rendered
    assert MARKER.decode() not in rendered
    assert refused, f"outside local dependency was accepted: {rendered[:2000]}"


def _fresh(config_path: pathlib.Path) -> dict:
    return validate_config_file(config_path)


def _up_stdin(config_path: pathlib.Path, *, distributed: bool) -> str:
    _args, stdin = prepare_args_and_stdin(
        capabilities=_CAPABILITIES,
        config_path=config_path,
        config=_fresh(config_path),
        docker_compose=None,
        port=8123,
        watch=True,
        engine_runtime_mode="distributed" if distributed else "combined_queue_worker",
    )
    return stdin


def _archive_names(config_path: pathlib.Path) -> str:
    with create_archive(config_path, _fresh(config_path)) as (
        archive_path,
        _size,
        _rel,
    ):
        with tarfile.open(archive_path, "r:gz") as tar:
            names = tar.getnames()
    return repr(names)


@pytest.mark.parametrize(
    "kind",
    ["ancestor", "root_relative_faux", "root_relative_real_package", "symlink"],
)
class TestOutsideLocalDependencyRejected:
    def test_dockerfile_rejects_outside_dependency(self, tmp_path, kind):
        config_path, outside = _cases(tmp_path)[kind]
        _assert_refused(
            lambda: config_to_docker(config_path, _fresh(config_path)),
            outside,
            [],
        )

    def test_up_compose_rejects_outside_dependency(self, tmp_path, kind):
        config_path, outside = _cases(tmp_path)[kind]
        _assert_refused(lambda: _up_stdin(config_path, distributed=False), outside, [])

    def test_up_distributed_compose_rejects_outside_dependency(self, tmp_path, kind):
        config_path, outside = _cases(tmp_path)[kind]
        _assert_refused(lambda: _up_stdin(config_path, distributed=True), outside, [])

    def test_direct_compose_rejects_outside_dependency(self, tmp_path, kind):
        config_path, outside = _cases(tmp_path)[kind]
        _assert_refused(
            lambda: config_to_compose(
                config_path,
                _fresh(config_path),
                watch=True,
                engine_runtime_mode="distributed",
            ),
            outside,
            [],
        )

    def test_build_docker_image_rejects_outside_dependency(self, tmp_path, kind):
        config_path, outside = _cases(tmp_path)[kind]

        async def fake_exec(*args, input=None, **kwargs):
            return "", ""

        _assert_refused(
            lambda: _build_invocation_with(config_path, fake_exec),
            outside,
            [],
        )

    def test_deploy_archive_rejects_outside_dependency(
        self, tmp_path, staged_archives, kind
    ):
        config_path, outside = _cases(tmp_path)[kind]
        _assert_refused(lambda: _archive_names(config_path), outside, staged_archives)


def _build_invocation_with(config_path: pathlib.Path, fake_exec) -> str:
    """Run build_docker_image up to the docker invocation.

    `fake_exec` stands in for the process launcher only. The argv and the
    Dockerfile stdin are the ones `build_docker_image` assembled from
    `config_to_docker`. Invoking docker at all, before or after a refusal,
    fails the test.
    """
    captured: list[tuple[object, ...]] = []

    async def recording(*args, input=None, **kwargs):
        captured.append((*args, input))
        return await fake_exec(*args, input=input, **kwargs)

    class _Runner:
        def run(self, coro):
            return asyncio.run(coro)

    original = docker_mod.subp_exec
    docker_mod.subp_exec = recording
    try:
        build_docker_image(
            _Runner(),
            lambda _message: None,
            config_path,
            _fresh(config_path),
            None,
            None,
            False,
            "langgraph-test:local",
        )
    except AssertionError:
        raise
    except Exception:
        if captured:
            raise AssertionError(
                f"docker was invoked before the dependency was refused: {captured!r}"
            ) from None
        raise
    finally:
        docker_mod.subp_exec = original
    return repr(captured)


def test_in_project_local_dependency_is_accepted(tmp_path, staged_archives):
    """A dependency that stays inside the project is still packaged."""
    project = tmp_path / "app"
    inner = project / "inner"
    inner.mkdir(parents=True)
    (inner / "pyproject.toml").write_text(
        '[project]\nname = "inner"\nversion = "0.0.0"\n',
        encoding="utf-8",
    )
    (inner / "mod.py").write_text("value = 1\n", encoding="utf-8")
    config_path = _write_config(project, [".", "./inner"])
    config = validate_config_file(config_path)

    _dockerfile, contexts = config_to_docker(config_path, config)
    assert contexts == {}

    with create_archive(config_path, validate_config_file(config_path)) as (
        archive_path,
        _size,
        _rel,
    ):
        with tarfile.open(archive_path, "r:gz") as tar:
            names = tar.getnames()
    assert any(name.endswith("inner/mod.py") for name in names)
    assert all(MARKER not in data for _name, data in staged_archives)


def test_second_resolution_cannot_escape(tmp_path, monkeypatch, staged_archives):
    """A later lookup of the same dependency must not select a different directory.

    The first resolution stays inside the project. Any later resolution of that
    same path, within one image or archive build, points outside it. Using the
    later result fails this test. `langgraph up` in distributed mode builds two
    images; each image is a separate build and resolves the dependency again.
    """
    outside = (tmp_path / OUTSIDE_NAME).resolve()
    _populate_outside(outside, real_package=True)
    project = tmp_path / "app"
    package = project / "pkg"
    package.mkdir(parents=True)
    (package / "pyproject.toml").write_text(
        '[project]\nname = "pkg"\nversion = "0.0.0"\n',
        encoding="utf-8",
    )
    (package / "mod.py").write_text("value = 1\n", encoding="utf-8")
    config_path = _write_config(project, [".", "./pkg"])
    package_norm = os.path.normpath(package)
    real_resolve = pathlib.Path.resolve
    state = {"calls": 0}

    def resolve(self, *args, **kwargs):
        found = real_resolve(self, *args, **kwargs)
        if os.path.normpath(self) == package_norm:
            state["calls"] += 1
            if state["calls"] >= 2:
                return outside
        return found

    monkeypatch.setattr(pathlib.Path, "resolve", resolve)

    def run_once(operation: Callable[[], object]) -> None:
        state["calls"] = 0
        try:
            result = operation()
        except AssertionError:
            raise
        except Exception:
            result = None
        leaked = [
            name
            for name, data in staged_archives
            if data == MARKER or name.endswith(MARKER_NAME)
        ]
        rendered = repr(result)
        assert leaked == []
        assert str(outside) not in rendered
        assert MARKER.decode() not in rendered

    run_once(lambda: config_to_docker(config_path, _fresh(config_path)))
    staged_archives.clear()
    run_once(lambda: _archive_names(config_path))
    staged_archives.clear()
    run_once(lambda: _up_stdin(config_path, distributed=False))
