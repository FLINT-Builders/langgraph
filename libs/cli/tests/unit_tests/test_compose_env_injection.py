"""Security spec for `langgraph up` compose generation.

`prepare` reads `env` from langgraph.json and `config_to_compose` pastes a
string value into the compose document. A string that is a path, then a
newline, then service-level keys, must not become volume mounts on the API
container. These tests fail while that paste is still in place.
"""

import json
import pathlib
from collections.abc import Mapping

import click
import pytest
import yaml

from langgraph_cli.cli import prepare
from langgraph_cli.docker import DockerCapabilities, Version

_CAPABILITIES = DockerCapabilities(
    version_docker=Version(26, 1, 1),
    version_compose=Version(2, 40, 3),
    healthcheck_start_interval=True,
    compose_type="plugin",
)

# A normal env-file path, then compose keys at the service indent (8 spaces).
_INJECTED_ENV = (
    ".env\n"
    "        volumes:\n"
    "            - /var/run/docker.sock:/var/run/docker.sock\n"
    "            - /:/hostroot"
)

_API_SERVICES = ("langgraph-api",)
_DISTRIBUTED_SERVICES = (
    "langgraph-api",
    "langgraph-orchestrator",
    "langgraph-executor",
)


class _Runner:
    def run(self, *args, **kwargs):
        raise AssertionError("prepare pulled or executed a command")


def _snapshot(root: pathlib.Path) -> set[str]:
    return {path.relative_to(root).as_posix() for path in root.rglob("*")}


def _write_project(root: pathlib.Path, env: object) -> pathlib.Path:
    (root / "agent.py").write_text("graph = None\n", encoding="utf-8")
    config = {
        "dependencies": ["."],
        "graphs": {"agent": "./agent.py:graph"},
        "env": env,
    }
    config_path = root / "langgraph.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def _prepare(
    config_path: pathlib.Path,
    *,
    engine_runtime_mode: str,
    image: str | None,
    watch: bool,
) -> tuple[list[str], str]:
    return prepare(
        _Runner(),
        capabilities=_CAPABILITIES,
        config_path=config_path,
        docker_compose=None,
        port=8123,
        pull=False,
        watch=watch,
        verbose=False,
        engine_runtime_mode=engine_runtime_mode,
        image=image,
    )


def _assert_services_have_no_volumes(
    document: str, service_names: tuple[str, ...]
) -> None:
    loaded = yaml.safe_load(document)
    assert isinstance(loaded, dict)
    services = loaded["services"]
    assert isinstance(services, dict)
    for name in service_names:
        service = services[name]
        assert isinstance(service, dict)
        assert not service.get("volumes"), (name, service.get("volumes"))


def _assert_env_file_is_plain_path(service: Mapping[str, object]) -> None:
    env_file = service.get("env_file")
    assert env_file in (".env", [".env"]), env_file


@pytest.mark.parametrize(
    ("engine_runtime_mode", "image", "watch", "service_names"),
    [
        ("combined_queue_worker", None, False, _API_SERVICES),
        (
            "combined_queue_worker",
            "example.invalid/langgraph-api:local",
            False,
            _API_SERVICES,
        ),
        ("combined_queue_worker", None, True, _API_SERVICES),
        ("distributed", None, False, _DISTRIBUTED_SERVICES),
    ],
)
def test_env_path_newline_cannot_mount_host(
    tmp_path: pathlib.Path,
    engine_runtime_mode: str,
    image: str | None,
    watch: bool,
    service_names: tuple[str, ...],
) -> None:
    config_path = _write_project(tmp_path, _INJECTED_ENV)
    before = _snapshot(tmp_path)

    try:
        args, stdin = _prepare(
            config_path,
            engine_runtime_mode=engine_runtime_mode,
            image=image,
            watch=watch,
        )
    except (click.ClickException, ValueError):
        assert _snapshot(tmp_path) == before
        return

    assert _snapshot(tmp_path) == before
    assert args[-2:] == ["-f", "-"]
    _assert_services_have_no_volumes(stdin, service_names)


@pytest.mark.parametrize(
    ("engine_runtime_mode", "image", "service_names"),
    [
        ("combined_queue_worker", None, _API_SERVICES),
        ("combined_queue_worker", "example.invalid/langgraph-api:local", _API_SERVICES),
        ("distributed", None, _DISTRIBUTED_SERVICES),
    ],
)
def test_plain_env_path_is_still_accepted(
    tmp_path: pathlib.Path,
    engine_runtime_mode: str,
    image: str | None,
    service_names: tuple[str, ...],
) -> None:
    config_path = _write_project(tmp_path, ".env")
    _args, stdin = _prepare(
        config_path,
        engine_runtime_mode=engine_runtime_mode,
        image=image,
        watch=False,
    )
    loaded = yaml.safe_load(stdin)
    services = loaded["services"]
    for name in service_names:
        service = services[name]
        assert not service.get("volumes")
        _assert_env_file_is_plain_path(service)
