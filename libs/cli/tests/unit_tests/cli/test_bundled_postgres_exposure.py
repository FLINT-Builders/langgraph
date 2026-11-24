"""Security spec for the bundled Postgres service in generated Compose stacks.

These tests exercise `langgraph up` and `langgraph dockerfile --add-docker-compose`,
the two commands that build the Compose project containing `langgraph-postgres`.
They fail while that service is published on every host interface or uses one
shared superuser password. A fix should keep the database on the Compose
network, stop publishing it except on loopback, and give each project its own
password that the API (and, in distributed mode, the orchestrator and executor)
actually use.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import shutil
import stat
import tempfile
import textwrap
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import unquote, urlparse

import pytest
import yaml
from click.testing import CliRunner

from langgraph_cli.cli import cli

_MIN_PASSWORD_LENGTH = 16
_INTERNAL_DB_HOST = "langgraph-postgres"
_INTERNAL_DB_PORT = 5432


def _write_project(root: Path) -> Path:
    (root / "agent.py").write_text("graph = None\n", encoding="utf-8")
    config_path = root / "langgraph.json"
    config_path.write_text(
        json.dumps(
            {
                "python_version": "3.11",
                "dependencies": ["."],
                "graphs": {"agent": "./agent.py:graph"},
            }
        ),
        encoding="utf-8",
    )
    return config_path


@contextmanager
def _docker_stub(stdin_path: Path) -> Iterator[None]:
    """Serve the docker CLI calls `up` and `dockerfile` make before Compose runs.

    `docker info` reports a Compose plugin so capability detection succeeds.
    `docker pull` is a no-op. `docker compose` records the Compose document the
    CLI passes on stdin, which is the stack `langgraph up` would launch.
    """
    bin_dir = Path(tempfile.mkdtemp())
    script = bin_dir / "docker"
    script.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import os
            import sys

            command = sys.argv[1] if len(sys.argv) > 1 else ""
            if command == "info":
                sys.stdout.write(
                    '{"ServerVersion":"27.3.1","ClientInfo":{"Plugins":[{"Name":"compose","Version":"2.32.0"}]}}\\n'
                )
            elif command == "pull":
                sys.exit(0)
            elif command == "compose":
                payload = sys.stdin.read()
                destination = os.environ["DOCKER_COMPOSE_STDIN_PATH"]
                with open(destination, "w", encoding="utf-8") as handle:
                    handle.write(payload)
            else:
                sys.stderr.write("unexpected docker command\\n")
                sys.exit(1)
            """
        ),
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    env = {
        "PATH": str(bin_dir) + os.pathsep + os.environ.get("PATH", ""),
        "DOCKER_COMPOSE_STDIN_PATH": str(stdin_path),
    }
    with pytest.MonkeyPatch.context() as patch:
        for key, value in env.items():
            patch.setenv(key, value)
        try:
            yield
        finally:
            shutil.rmtree(bin_dir)


def _run_up(config_path: Path, *extra: str) -> str:
    stdin_path = config_path.parent / "compose-stdin.yml"
    runner = CliRunner()
    with _docker_stub(stdin_path):
        result = runner.invoke(
            cli,
            ["up", "--config", str(config_path), *extra],
        )
    assert result.exit_code == 0, result.output
    assert stdin_path.is_file(), (
        "langgraph up did not pass a Compose file to docker compose"
    )
    return stdin_path.read_text(encoding="utf-8")


def _run_dockerfile(config_path: Path) -> tuple[str, str]:
    stdin_path = config_path.parent / "unused-stdin.yml"
    dockerfile_path = config_path.parent / "Dockerfile"
    runner = CliRunner()
    with _docker_stub(stdin_path):
        result = runner.invoke(
            cli,
            [
                "dockerfile",
                str(dockerfile_path),
                "--config",
                str(config_path),
                "--add-docker-compose",
            ],
        )
    assert result.exit_code == 0, result.output
    compose_path = config_path.parent / "docker-compose.yml"
    assert compose_path.is_file(), (
        "dockerfile --add-docker-compose did not write a Compose file"
    )
    env_path = config_path.parent / ".env"
    env_text = env_path.read_text(encoding="utf-8") if env_path.is_file() else ""
    return compose_path.read_text(encoding="utf-8"), env_text


def _load_env(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _interpolate(value: str, env: Mapping[str, str]) -> str:
    def replace(match: re.Match[str]) -> str:
        token = match.group(1)
        if ":-" in token:
            name, default = token.split(":-", 1)
            return env.get(name, default)
        if "-" in token:
            name, default = token.split("-", 1)
            return env.get(name, default)
        return env.get(token, match.group(0))

    return re.sub(r"\$\{([^}]+)\}", replace, value)


def _services(compose_text: str, env_text: str) -> dict:
    document = yaml.safe_load(compose_text)
    assert isinstance(document, dict), "Compose document is not a mapping"
    services = document.get("services")
    assert isinstance(services, dict), "Compose document has no services"
    env = _load_env(env_text)
    for service in services.values():
        environment = service.get("environment")
        if isinstance(environment, dict):
            service["environment"] = {
                key: _interpolate(str(val), env) for key, val in environment.items()
            }
    return services


def _split_port_spec(spec: str) -> tuple[str, str, str]:
    """Split a Compose short port spec the way docker/go-connections nat.splitParts does."""
    parts = spec.split(":")
    if len(parts) == 1:
        return "", "", parts[0]
    if len(parts) == 2:
        return "", parts[0], parts[1]
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    host_ip = ":".join(parts[:-2])
    return host_ip, parts[-2], parts[-1]


def _published_host_ip(spec: object) -> str | None:
    if isinstance(spec, dict):
        host_ip = str(spec.get("host_ip") or "")
    elif isinstance(spec, str):
        host_ip, _host_port, container = _split_port_spec(spec.split("/")[0])
        if container == "":
            raise AssertionError("published port spec has no container port")
    else:
        raise AssertionError("published port spec is not a string or mapping")
    if host_ip.startswith("[") and host_ip.endswith("]"):
        host_ip = host_ip[1:-1]
    return host_ip or None


def _assert_loopback_or_unpublished(service_name: str, service: Mapping) -> None:
    ports = service.get("ports") or []
    if not isinstance(ports, list):
        raise AssertionError(f"{service_name} ports value is not a list")
    for spec in ports:
        host_ip = _published_host_ip(spec)
        if host_ip is None:
            raise AssertionError(
                f"{service_name} publishes a port on every host interface"
            )
        try:
            address = ipaddress.ip_address(host_ip)
        except ValueError as exc:
            raise AssertionError(
                f"{service_name} publishes a port on a non-loopback address"
            ) from exc
        if not address.is_loopback:
            raise AssertionError(
                f"{service_name} publishes a port on a non-loopback address"
            )


def _password_from_uri(uri: str) -> tuple[str, str, int | None]:
    parsed = urlparse(uri)
    port = parsed.port
    return parsed.hostname or "", unquote(parsed.password or ""), port


def _assert_database_credential(service: Mapping, label: str) -> str:
    environment = service.get("environment")
    if not isinstance(environment, dict) or "POSTGRES_PASSWORD" not in environment:
        raise AssertionError(f"{label} does not set a bundled database password")
    password = str(environment["POSTGRES_PASSWORD"])
    username = str(environment.get("POSTGRES_USER", ""))
    if (
        not password
        or password == username
        or len(password) < _MIN_PASSWORD_LENGTH
        or password.startswith("${")
    ):
        raise AssertionError(
            f"{label} uses a fixed or guessable bundled database password"
        )
    return password


def _assert_client_uri(uri: str, password: str, label: str) -> None:
    host, uri_password, port = _password_from_uri(uri)
    if (
        host != _INTERNAL_DB_HOST
        or port != _INTERNAL_DB_PORT
        or uri_password != password
        or not uri_password
    ):
        raise AssertionError(
            f"{label} does not connect to the bundled database over the "
            "internal Compose network with that database's password"
        )


def _assert_bundled_stack(
    compose_text: str,
    env_text: str,
    *,
    other_password: str | None,
    expect_debugger: bool,
    expect_distributed: bool,
) -> str:
    services = _services(compose_text, env_text)
    postgres = services.get("langgraph-postgres")
    if not isinstance(postgres, Mapping):
        raise AssertionError("bundled database service is missing")
    _assert_loopback_or_unpublished("langgraph-postgres", postgres)
    password = _assert_database_credential(postgres, "langgraph-postgres")
    if other_password is not None and password == other_password:
        raise AssertionError(
            "two generated projects share the same bundled database password"
        )

    api = services.get("langgraph-api")
    if not isinstance(api, Mapping):
        raise AssertionError("langgraph-api service is missing")
    api_env = api.get("environment")
    if not isinstance(api_env, dict) or "POSTGRES_URI" not in api_env:
        raise AssertionError("langgraph-api does not set POSTGRES_URI")
    _assert_client_uri(str(api_env["POSTGRES_URI"]), password, "langgraph-api")

    if expect_debugger:
        debugger = services.get("langgraph-debugger")
        if not isinstance(debugger, Mapping):
            raise AssertionError("debugger service is missing")
        _assert_loopback_or_unpublished("langgraph-debugger", debugger)

    if expect_distributed:
        for name in ("langgraph-orchestrator", "langgraph-executor"):
            service = services.get(name)
            if not isinstance(service, Mapping):
                raise AssertionError(f"{name} service is missing")
            environment = service.get("environment")
            if not isinstance(environment, dict) or "DATABASE_URI" not in environment:
                raise AssertionError(f"{name} does not set DATABASE_URI")
            _assert_client_uri(str(environment["DATABASE_URI"]), password, name)
    return password


def test_langgraph_up_does_not_expose_bundled_database_or_debugger() -> None:
    passwords: list[str] = []
    for _ in range(2):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = _write_project(Path(tmp))
            compose_text = _run_up(
                config_path,
                "--debugger-port",
                "8124",
            )
            env_path = config_path.parent / ".env"
            env_text = (
                env_path.read_text(encoding="utf-8") if env_path.is_file() else ""
            )
            passwords.append(
                _assert_bundled_stack(
                    compose_text,
                    env_text,
                    other_password=passwords[0] if passwords else None,
                    expect_debugger=True,
                    expect_distributed=False,
                )
            )
    assert len(passwords) == 2


def test_langgraph_up_distributed_uses_the_same_unexposed_database() -> None:
    passwords: list[str] = []
    for _ in range(2):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = _write_project(Path(tmp))
            compose_text = _run_up(
                config_path,
                "--engine-runtime-mode",
                "distributed",
            )
            env_path = config_path.parent / ".env"
            env_text = (
                env_path.read_text(encoding="utf-8") if env_path.is_file() else ""
            )
            passwords.append(
                _assert_bundled_stack(
                    compose_text,
                    env_text,
                    other_password=passwords[0] if passwords else None,
                    expect_debugger=False,
                    expect_distributed=True,
                )
            )
    assert len(passwords) == 2


def test_dockerfile_add_docker_compose_does_not_expose_bundled_database() -> None:
    passwords: list[str] = []
    for _ in range(2):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = _write_project(Path(tmp))
            compose_text, env_text = _run_dockerfile(config_path)
            passwords.append(
                _assert_bundled_stack(
                    compose_text,
                    env_text,
                    other_password=passwords[0] if passwords else None,
                    expect_debugger=False,
                    expect_distributed=False,
                )
            )
    assert len(passwords) == 2
