"""Spec for how `langgraph up` publishes ports.

`langgraph up` and `langgraph dockerfile --add-docker-compose` must publish
the API, debugger, and bundled Postgres ports on loopback unless the operator
explicitly chooses another address. These tests fail while those mappings are
emitted without a host IP (which Docker Compose publishes on every interface).
"""

from __future__ import annotations

import json
import pathlib
import re
from collections.abc import Iterable

import pytest
from click.testing import CliRunner

from langgraph_cli.cli import cli, prepare_args_and_stdin
from langgraph_cli.config import validate_config
from langgraph_cli.docker import (
    DockerCapabilities,
    Version,
    compose,
    compose_as_dict,
    debugger_compose,
)

LOOPBACK = "127.0.0.1"
NON_LOOPBACK_WARNING = (
    "exposes it on every network interface. Only use a non-loopback address "
    "in trusted networks."
)

CAPABILITIES = DockerCapabilities(
    version_docker=Version(26, 1, 1),
    version_compose=Version(2, 27, 0),
    healthcheck_start_interval=True,
)

_PORT_ITEM = re.compile(r"""^-\s+["']?([^"']+)["']?\s*$""")


def _published_mappings(compose_text: str) -> list[str]:
    """Return every published port mapping in generated compose YAML."""
    mappings: list[str] = []
    in_ports = False
    for raw_line in compose_text.splitlines():
        stripped = raw_line.strip()
        if stripped == "ports:":
            in_ports = True
            continue
        if not in_ports:
            continue
        match = _PORT_ITEM.match(stripped)
        if match:
            mappings.append(match.group(1))
            continue
        if stripped:
            in_ports = False
    return mappings


def _assert_bound(
    mappings: Iterable[str],
    host_ip: str,
    expected: set[tuple[str, str]],
) -> None:
    found: set[tuple[str, str]] = set()
    for mapping in mappings:
        parts = mapping.split(":")
        assert len(parts) == 3, (
            f"published port {mapping!r} has no host address; "
            f"expected {host_ip}:host_port:container_port"
        )
        host, host_port, container_port = parts
        assert host == host_ip, mapping
        found.add((host_port, container_port))
    assert found == expected


def _minimal_config() -> dict:
    return validate_config(
        {
            "dependencies": ["."],
            "graphs": {"agent": "agent.py:graph"},
            "env": ".env",
        }
    )


def test_compose_as_dict_binds_loopback_by_default() -> None:
    document = compose_as_dict(
        CAPABILITIES,
        port=8123,
        debugger_port=8001,
        engine_runtime_mode="distributed",
    )
    services = document["services"]
    assert "ports" not in services["langgraph-redis"]
    _assert_bound(
        [item.strip('"') for item in services["langgraph-api"]["ports"]],
        LOOPBACK,
        {("8123", "8000")},
    )
    _assert_bound(
        [item.strip('"') for item in services["langgraph-postgres"]["ports"]],
        LOOPBACK,
        {("5433", "5432")},
    )
    _assert_bound(
        [item.strip('"') for item in services["langgraph-debugger"]["ports"]],
        LOOPBACK,
        {("8001", "3968")},
    )


def test_debugger_compose_binds_requested_host() -> None:
    debugger = debugger_compose(port=8001, base_url="http://127.0.0.1:8123")
    _assert_bound(
        [item.strip('"') for item in debugger["langgraph-debugger"]["ports"]],
        LOOPBACK,
        {("8001", "3968")},
    )
    exposed = debugger_compose(port=8001, host="0.0.0.0")
    _assert_bound(
        [item.strip('"') for item in exposed["langgraph-debugger"]["ports"]],
        "0.0.0.0",
        {("8001", "3968")},
    )


def test_compose_text_uses_requested_host() -> None:
    loopback = compose(CAPABILITIES, port=8123, debugger_port=8001)
    _assert_bound(
        _published_mappings(loopback),
        LOOPBACK,
        {("8123", "8000"), ("5433", "5432"), ("8001", "3968")},
    )
    exposed = compose(CAPABILITIES, port=8123, host="10.1.2.3")
    _assert_bound(
        _published_mappings(exposed),
        "10.1.2.3",
        {("8123", "8000"), ("5433", "5432")},
    )


def test_prepare_args_and_stdin_publishes_loopback_and_keeps_env_file() -> None:
    config_path = pathlib.Path(__file__).parent / "cli" / "langgraph.json"
    _args, stdin = prepare_args_and_stdin(
        capabilities=CAPABILITIES,
        config_path=config_path,
        config=_minimal_config(),
        docker_compose=None,
        port=8123,
        watch=False,
        debugger_port=8001,
    )
    _assert_bound(
        _published_mappings(stdin),
        LOOPBACK,
        {("8123", "8000"), ("5433", "5432"), ("8001", "3968")},
    )
    assert "env_file: .env" in stdin


def test_prepare_args_and_stdin_honors_explicit_non_loopback_host() -> None:
    config_path = pathlib.Path(__file__).parent / "cli" / "langgraph.json"
    _args, stdin = prepare_args_and_stdin(
        capabilities=CAPABILITIES,
        config_path=config_path,
        config=_minimal_config(),
        docker_compose=None,
        port=9000,
        watch=False,
        host="0.0.0.0",
        postgres_uri="postgres://db.internal:5432/postgres",
    )
    _assert_bound(
        _published_mappings(stdin),
        "0.0.0.0",
        {("9000", "8000")},
    )
    assert "5433:5432" not in stdin
    assert "env_file: .env" in stdin


def test_up_help_defaults_host_to_loopback() -> None:
    result = CliRunner().invoke(cli, ["up", "--help"])
    assert result.exit_code == 0, result.output
    host_help = result.output.split("--host", 1)[1].split("\n--", 1)[0]
    assert "127.0.0.1" in host_help


def _project(directory: pathlib.Path) -> None:
    directory.joinpath("langgraph.json").write_text(
        json.dumps(
            {
                "dependencies": ["."],
                "graphs": {"agent": "agent.py:graph"},
                "env": ".env",
            }
        ),
        encoding="utf-8",
    )
    directory.joinpath("agent.py").write_text("graph = None\n", encoding="utf-8")
    directory.joinpath(".env").write_text(
        "OPENAI_API_KEY=placeholder\n",
        encoding="utf-8",
    )


def _patch_docker(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    captured: list[str] = []

    async def fake_subp_exec(cmd, *args, **kwargs):
        if cmd == "docker" and args and args[0] == "info":
            payload = json.dumps(
                {
                    "ServerVersion": "26.1.1",
                    "ClientInfo": {
                        "Plugins": [{"Name": "compose", "Version": "2.27.0"}]
                    },
                }
            )
            return payload, ""
        compose_input = kwargs.get("input")
        if compose_input is not None and "compose" in args:
            captured.append(compose_input)
            on_stdout = kwargs.get("on_stdout")
            if on_stdout is not None:
                on_stdout("Application startup complete")
            return None, None
        if kwargs.get("collect"):
            return "", ""
        return None, None

    monkeypatch.setattr(
        "langgraph_cli.docker.shutil.which", lambda _name: "/usr/bin/docker"
    )
    monkeypatch.setattr("langgraph_cli.docker.subp_exec", fake_subp_exec)
    monkeypatch.setattr("langgraph_cli.cli.subp_exec", fake_subp_exec)
    return captured


def test_up_command_publishes_loopback_by_default(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _patch_docker(monkeypatch)
    _project(tmp_path)
    result = CliRunner().invoke(
        cli,
        [
            "up",
            "--config",
            str(tmp_path / "langgraph.json"),
            "--no-pull",
            "--port",
            "8123",
            "--debugger-port",
            "8001",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured, "langgraph up did not pass a compose file to docker compose"
    stdin = captured[-1]
    _assert_bound(
        _published_mappings(stdin),
        LOOPBACK,
        {("8123", "8000"), ("5433", "5432"), ("8001", "3968")},
    )
    assert "env_file: .env" in stdin
    assert NON_LOOPBACK_WARNING not in result.output
    assert "http://0.0.0.0:8123" not in result.output


def test_up_command_warns_when_host_is_not_loopback(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _patch_docker(monkeypatch)
    _project(tmp_path)
    result = CliRunner().invoke(
        cli,
        [
            "up",
            "--config",
            str(tmp_path / "langgraph.json"),
            "--no-pull",
            "--port",
            "8123",
            "--host",
            "0.0.0.0",
            "--debugger-port",
            "8001",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured, "langgraph up did not pass a compose file to docker compose"
    _assert_bound(
        _published_mappings(captured[-1]),
        "0.0.0.0",
        {("8123", "8000"), ("5433", "5432"), ("8001", "3968")},
    )
    assert NON_LOOPBACK_WARNING in result.output
    assert "http://0.0.0.0:8123" in result.output
    assert "http://localhost:8123" not in result.output


def test_dockerfile_add_docker_compose_binds_loopback(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_docker(monkeypatch)
    _project(tmp_path)
    result = CliRunner().invoke(
        cli,
        [
            "dockerfile",
            str(tmp_path / "Dockerfile"),
            "--config",
            str(tmp_path / "langgraph.json"),
            "--add-docker-compose",
        ],
    )
    assert result.exit_code == 0, result.output
    compose_text = (tmp_path / "docker-compose.yml").read_text(encoding="utf-8")
    _assert_bound(
        _published_mappings(compose_text),
        LOOPBACK,
        {("8123", "8000"), ("5433", "5432")},
    )
    assert "env_file" in compose_text
