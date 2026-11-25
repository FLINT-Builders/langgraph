"""Local dependency paths must not become Docker Compose service config.

`langgraph up` writes extra build-context names and host paths into the Compose
document with string concatenation. A dependency directory chosen by the
repository author can contain newlines. Those newlines are pasted into the
document, and Docker Compose then applies the resulting service keys, including
a bind mount of the host root.

Generation must reject context names and paths that contain newlines or other
control characters before it returns a Compose document, and it must not write
a Compose file first. A benign extra context still has to be emitted.
"""

import json
import pathlib

import click
import pytest

from langgraph_cli.cli import prepare_args_and_stdin
from langgraph_cli.config import (
    config_to_compose,
    validate_config,
    validate_config_file,
)
from langgraph_cli.docker import DockerCapabilities, Version

_CAPABILITIES = DockerCapabilities(
    version_docker=Version(26, 1, 1),
    version_compose=Version(2, 27, 0),
    healthcheck_start_interval=True,
)

# compose-go v2.4.8 ParseVolume treats this short syntax as a bind of `/`.
_HOST_ROOT_MOUNT = "/:/hostroot=ok"


def _files(root: pathlib.Path) -> list[str]:
    return sorted(
        str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()
    )


def _assert_control_chars_rejected(root: pathlib.Path, generate) -> None:
    before = _files(root)
    with pytest.raises(
        (ValueError, click.UsageError),
        match=r"(?i)newline|control character",
    ):
        generate()
    assert _files(root) == before


def _write_json(path: pathlib.Path, payload: dict) -> None:
    path.write_text(json.dumps(payload))


def _host_root_project(root: pathlib.Path) -> pathlib.Path:
    """Build the directory layout whose pasted path mounts the host root.

    `langgraph.json` lives in `app/`. The dependency resolves outside that
    directory, so it becomes an additional build context. The context name is
    the package directory name. The host path is the absolute path of that
    directory. Neither value is quoted. The components below make Compose see
    a `langgraph-api` volume with source `/`.
    """
    volume_parent = (
        "IGNORED\nservices:\n    langgraph-api:\n        volumes:\n            - "
    )
    package_name = "hostroot=ok\n---\nx-swallow: |\n    z"
    app = root / "app"
    app.mkdir()
    (app / "agent.py").write_text("graph = None\n")
    package = root / volume_parent / ":" / package_name
    package.mkdir(parents=True)
    (package / "pyproject.toml").write_text(
        '[project]\nname = "evilpkg"\nversion = "0.0.1"\n'
    )
    config_path = app / "langgraph.json"
    _write_json(
        config_path,
        {
            "dependencies": [".", "../" + "/".join((volume_parent, ":", package_name))],
            "graphs": {"agent": "./agent.py:graph"},
            "python_version": "3.11",
        },
    )
    return config_path


def _prepare(config_path: pathlib.Path, **kwargs):
    config = validate_config_file(config_path)
    return prepare_args_and_stdin(
        capabilities=_CAPABILITIES,
        config_path=config_path,
        config=config,
        docker_compose=None,
        port=8000,
        watch=kwargs.get("watch", False),
        engine_runtime_mode=kwargs.get("engine_runtime_mode", "combined_queue_worker"),
    )


@pytest.mark.parametrize(
    ("watch", "engine_runtime_mode"),
    [
        (False, "combined_queue_worker"),
        (True, "combined_queue_worker"),
        (False, "distributed"),
    ],
)
def test_up_rejects_host_root_mount_injected_by_context_path(
    tmp_path: pathlib.Path, watch: bool, engine_runtime_mode: str
) -> None:
    config_path = _host_root_project(tmp_path)
    _assert_control_chars_rejected(
        tmp_path,
        lambda: _prepare(
            config_path, watch=watch, engine_runtime_mode=engine_runtime_mode
        ),
    )


def test_faux_package_context_name_with_newline_is_rejected(
    tmp_path: pathlib.Path,
) -> None:
    app = tmp_path / "app"
    app.mkdir()
    (app / "agent.py").write_text("graph = None\n")
    package_name = "bad\npkg"
    package = tmp_path / package_name
    package.mkdir()
    (package / "mod.py").write_text("value = 1\n")
    config_path = app / "langgraph.json"
    _write_json(
        config_path,
        {
            "dependencies": [".", "../" + package_name],
            "graphs": {"agent": "./agent.py:graph"},
            "python_version": "3.11",
        },
    )

    _assert_control_chars_rejected(tmp_path, lambda: _prepare(config_path))


def test_real_package_context_name_with_carriage_return_is_rejected(
    tmp_path: pathlib.Path,
) -> None:
    app = tmp_path / "app"
    app.mkdir()
    (app / "agent.py").write_text("graph = None\n")
    package_name = "bad\rpkg"
    package = tmp_path / package_name
    package.mkdir()
    (package / "pyproject.toml").write_text(
        '[project]\nname = "badpkg"\nversion = "0.0.1"\n'
    )
    config_path = app / "langgraph.json"
    _write_json(
        config_path,
        {
            "dependencies": [".", "../" + package_name],
            "graphs": {"agent": "./agent.py:graph"},
            "python_version": "3.11",
        },
    )

    _assert_control_chars_rejected(tmp_path, lambda: _prepare(config_path))


def test_uv_workspace_root_with_newline_is_rejected(tmp_path: pathlib.Path) -> None:
    project = tmp_path / "ws\nbad"
    agent = project / "apps" / "agent"
    agent.mkdir(parents=True)
    (project / "pyproject.toml").write_text(
        '[project]\nname = "ws"\nversion = "0.1.0"\n'
    )
    (project / "uv.lock").write_text("# uv lock file\n")
    (agent / "graph.py").write_text("graph = None\n")
    config_path = agent / "langgraph.json"
    config_path.touch()
    config = validate_config(
        {
            "python_version": "3.11",
            "graphs": {"agent": "./graph.py:graph"},
            "source": {"kind": "uv", "root": "../.."},
        }
    )

    def generate():
        config_to_compose(config_path, config, base_image="langchain/langgraph-api")

    _assert_control_chars_rejected(tmp_path, generate)


def test_benign_additional_context_is_still_emitted(tmp_path: pathlib.Path) -> None:
    app = tmp_path / "app"
    app.mkdir()
    (app / "agent.py").write_text("graph = None\n")
    sibling = tmp_path / "sibling"
    sibling.mkdir()
    (sibling / "pyproject.toml").write_text(
        '[project]\nname = "sibling"\nversion = "0.0.1"\n'
    )
    config_path = app / "langgraph.json"
    _write_json(
        config_path,
        {
            "dependencies": [".", "../sibling"],
            "graphs": {"agent": "./agent.py:graph"},
            "python_version": "3.11",
        },
    )

    _args, stdin = _prepare(config_path)
    assert f"- sibling: {sibling.resolve()}" in stdin
    assert _HOST_ROOT_MOUNT not in stdin
