"""Verbose subprocess logging must not echo stdin.

`subp_exec` is the single helper that runs child processes for deploy, image
build, and `langgraph up`. When `verbose=True` it prints the command and, today,
every line of the stdin it is about to pipe to the child. Callers put secrets
on that stdin:

- `langgraph deploy` pipes the host-backend push token to
  `docker login --password-stdin`.
- `build_docker_image` pipes the generated Dockerfile.
- `langgraph up` pipes the generated compose file, which includes config env
  values and the Postgres URI.

These tests run those product paths. They fail while the helper echoes stdin
and pass once the stdin body is no longer written to the process output.
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from click.testing import CliRunner

from langgraph_cli.cli import cli
from langgraph_cli.config import validate_config_file
from langgraph_cli.docker import build_docker_image
from langgraph_cli.exec import Runner, subp_exec

# Obvious placeholders. These are not credentials.
PUSH_TOKEN_PLACEHOLDER = "PUSH_TOKEN_PLACEHOLDER"
ENV_SECRET_PLACEHOLDER = "ENV_SECRET_PLACEHOLDER"
POSTGRES_PASSWORD_PLACEHOLDER = "POSTGRES_PASSWORD_PLACEHOLDER"
DOCKERFILE_SECRET_PLACEHOLDER = "DOCKERFILE_SECRET_PLACEHOLDER"
STDIN_SECRET_PLACEHOLDER = "STDIN_SECRET_PLACEHOLDER"
API_KEY_PLACEHOLDER = "API_KEY_PLACEHOLDER"

_REGISTRY_URL = "https://registry.example.test/tenant"
_REGISTRY_HOST = "registry.example.test"
_DEPLOYMENT_ID = "dep-placeholder"


def _write_fake_docker(bin_dir: Path, state_dir: Path) -> None:
    """Stand-in for the docker CLI the product execs.

    Login stdin is turned into the password the same way Docker CLI 28.3.3
    does in `cli/command/registry/login.go` (`verifyLoginOptions` reads stdin
    and trims one trailing newline, then a trailing CR) and
    `cli/command/registry.go` (`PromptUserForCredentials` trims surrounding
    whitespace and stores the result as `AuthConfig.Password`). The stand-in
    records that password and does not write it to its own stdout, so anything
    that shows up in the parent output was printed by the product.
    """
    script = textwrap.dedent(
        f"""\
        #!/usr/bin/env python3
        import json
        import sys
        from pathlib import Path

        state = Path({str(state_dir)!r})

        def command_name(argv):
            index = 0
            while index < len(argv):
                arg = argv[index]
                if arg == "--config":
                    index += 2
                    continue
                if arg.startswith("-"):
                    index += 1
                    continue
                return arg, argv[index + 1 :]
            return "", []

        args = sys.argv[1:]
        with (state / "argv.log").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(args) + "\\n")
        name, rest = command_name(args)

        def read_stdin():
            return sys.stdin.read()

        if name == "info":
            sys.stdout.write(
                json.dumps(
                    {{
                        "ServerVersion": "28.3.3",
                        "ClientInfo": {{
                            "Plugins": [{{"Name": "compose", "Version": "2.32.0"}}]
                        }},
                    }}
                )
            )
            raise SystemExit(0)
        if name == "image" and rest[:1] == ["inspect"]:
            fmt = ""
            if "--format" in rest:
                fmt = rest[rest.index("--format") + 1]
            if "RepoDigests" in fmt:
                image = rest[-1]
                repo = image.rsplit(":", 1)[0]
                sys.stdout.write(json.dumps([repo + "@sha256:" + ("ab" * 32)]) + "\\n")
            else:
                sys.stdout.write("linux/amd64\\n")
            raise SystemExit(0)
        if name == "login":
            password = read_stdin()
            if password.endswith("\\n"):
                password = password[:-1]
            if password.endswith("\\r"):
                password = password[:-1]
            password = password.strip()
            (state / "login-password").write_text(password, encoding="utf-8")
            raise SystemExit(0)
        if name == "build":
            (state / "build-stdin").write_text(read_stdin(), encoding="utf-8")
            raise SystemExit(0)
        if name == "buildx":
            if rest[:1] == ["version"]:
                sys.stdout.write("github.com/docker/buildx v0.0.0\\n")
                raise SystemExit(0)
            (state / "build-stdin").write_text(read_stdin(), encoding="utf-8")
            raise SystemExit(0)
        if name == "compose":
            (state / "compose-stdin").write_text(read_stdin(), encoding="utf-8")
            raise SystemExit(0)
        if name in {{"tag", "push", "pull", "volume"}}:
            raise SystemExit(0)
        sys.stderr.write("unexpected docker invocation\\n")
        raise SystemExit(1)
        """
    )
    path = bin_dir / "docker"
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)


def _prepend_path(monkeypatch: pytest.MonkeyPatch, bin_dir: Path) -> None:
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.setenv("LANGGRAPH_CLI_NO_ANALYTICS", "1")


class _HostBackend(threading.Thread):
    """Local stand-in for the host backend endpoints the deploy command calls.

    Responses follow the fields `HostBackendClient` and `_run_local_build` read:
    `get_deployment` only needs a successful JSON body, `request_push_token`
    reads `token` and `registry_url`, and `update_deployment` returns the JSON
    object used as the deployment record.
    """

    def __init__(self, token: str) -> None:
        super().__init__(daemon=True)
        self.token = token
        self.paths: list[str] = []
        parent = self

        class Handler(BaseHTTPRequestHandler):
            def _read_body(self) -> bytes:
                length = int(self.headers.get("Content-Length", "0"))
                if length:
                    return self.rfile.read(length)
                return b""

            def _send(self, status: int, payload: object) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _dispatch(self) -> None:
                parent.paths.append(f"{self.command} {self.path}")
                self._read_body()
                deployment = f"/v2/deployments/{_DEPLOYMENT_ID}"
                if self.command == "GET" and self.path == deployment:
                    self._send(200, {"id": _DEPLOYMENT_ID, "name": "demo"})
                    return
                if self.command == "POST" and self.path == deployment + "/push-token":
                    self._send(
                        200,
                        {
                            "token": parent.token,
                            "registry_url": _REGISTRY_URL,
                        },
                    )
                    return
                if self.command == "PATCH" and self.path == deployment:
                    self._send(200, {"id": _DEPLOYMENT_ID})
                    return
                self._send(404, {"error": f"unexpected {self.command} {self.path}"})

            def do_GET(self) -> None:
                self._dispatch()

            def do_POST(self) -> None:
                self._dispatch()

            def do_PATCH(self) -> None:
                self._dispatch()

            def log_message(self, format: str, *args: object) -> None:
                return

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self._httpd.server_address[1]}"

    def run(self) -> None:
        self._httpd.serve_forever()

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


@pytest.fixture
def host_backend():
    server = _HostBackend(PUSH_TOKEN_PLACEHOLDER)
    server.start()
    try:
        yield server
    finally:
        server.close()


def _project(tmp_path: Path, *, dockerfile_secret: str | None = None) -> Path:
    config: dict[str, object] = {
        "dependencies": ["."],
        "graphs": {"agent": "./agent.py:graph"},
    }
    if dockerfile_secret is not None:
        config["dockerfile_lines"] = [f"ENV APP_BUILD_SECRET={dockerfile_secret}"]
    (tmp_path / "agent.py").write_text("graph = None\n", encoding="utf-8")
    path = tmp_path / "langgraph.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _invoke(args: list[str]) -> tuple[str, object]:
    result = CliRunner(env={"LANGGRAPH_CLI_NO_ANALYTICS": "1"}).invoke(cli, args)
    combined = result.output
    if result.stderr_bytes:
        combined += result.stderr_bytes.decode()
    return combined, result


def _assert_secret_not_in_output(secret: str, output: str) -> None:
    assert secret not in output, "process output contained a secret passed as stdin"


def test_subp_exec_verbose_does_not_echo_stdin() -> None:
    """The shared helper prints the command in verbose mode and not the stdin body."""
    captured: list[str] = []
    original = sys.stdout

    class _Buffer:
        def write(self, data: bytes) -> int:
            captured.append(data.decode())
            return len(data)

        def flush(self) -> None:
            return None

    class _Stream:
        def write(self, data: str) -> int:
            captured.append(data)
            return len(data)

        def flush(self) -> None:
            return None

        @property
        def buffer(self) -> _Buffer:
            return _Buffer()

    sys.stdout = _Stream()  # type: ignore[assignment]
    try:
        with Runner() as runner:
            runner.run(
                subp_exec(
                    sys.executable,
                    "-c",
                    "import sys; sys.stdin.read()",
                    input=f"keep-me\n{STDIN_SECRET_PLACEHOLDER}\n",
                    verbose=True,
                )
            )
    finally:
        sys.stdout = original

    output = "".join(captured)
    assert "+ " in output and sys.executable in output
    _assert_secret_not_in_output(STDIN_SECRET_PLACEHOLDER, output)
    assert "keep-me" not in output


def test_deploy_verbose_does_not_echo_push_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, host_backend: _HostBackend
) -> None:
    bin_dir = tmp_path / "bin"
    state_dir = tmp_path / "docker-state"
    bin_dir.mkdir()
    state_dir.mkdir()
    _write_fake_docker(bin_dir, state_dir)
    _prepend_path(monkeypatch, bin_dir)
    config = _project(tmp_path)

    output, result = _invoke(
        [
            "deploy",
            "--verbose",
            "--json",
            "--no-input",
            "--no-wait",
            "--deployment-id",
            _DEPLOYMENT_ID,
            "--api-key",
            API_KEY_PLACEHOLDER,
            "--host-url",
            host_backend.base_url,
            "--image",
            "local/app:test",
            "--config",
            str(config),
        ]
    )

    assert result.exit_code == 0, output
    assert any(path.endswith("/push-token") for path in host_backend.paths)
    password = (state_dir / "login-password").read_text(encoding="utf-8")
    assert password == PUSH_TOKEN_PLACEHOLDER
    argv_log = (state_dir / "argv.log").read_text(encoding="utf-8")
    assert "--password-stdin" in argv_log
    assert _REGISTRY_HOST in argv_log
    # Verbose mode still prints the command line. That line is how this test
    # knows the flag reached `subp_exec`. The token must not be on it or after it.
    assert "+ " in output and "--password-stdin" in output
    _assert_secret_not_in_output(PUSH_TOKEN_PLACEHOLDER, output)


def test_deploy_without_verbose_does_not_echo_push_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, host_backend: _HostBackend
) -> None:
    bin_dir = tmp_path / "bin"
    state_dir = tmp_path / "docker-state"
    bin_dir.mkdir()
    state_dir.mkdir()
    _write_fake_docker(bin_dir, state_dir)
    _prepend_path(monkeypatch, bin_dir)
    config = _project(tmp_path)

    output, result = _invoke(
        [
            "deploy",
            "--json",
            "--no-input",
            "--no-wait",
            "--deployment-id",
            _DEPLOYMENT_ID,
            "--api-key",
            API_KEY_PLACEHOLDER,
            "--host-url",
            host_backend.base_url,
            "--image",
            "local/app:test",
            "--config",
            str(config),
        ]
    )

    assert result.exit_code == 0, output
    password = (state_dir / "login-password").read_text(encoding="utf-8")
    assert password == PUSH_TOKEN_PLACEHOLDER
    assert "--password-stdin" not in output
    _assert_secret_not_in_output(PUSH_TOKEN_PLACEHOLDER, output)


def test_up_verbose_does_not_echo_compose_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bin_dir = tmp_path / "bin"
    state_dir = tmp_path / "docker-state"
    bin_dir.mkdir()
    state_dir.mkdir()
    _write_fake_docker(bin_dir, state_dir)
    _prepend_path(monkeypatch, bin_dir)
    config_body = {
        "dependencies": ["."],
        "graphs": {"agent": "./agent.py:graph"},
        "env": {"APP_SECRET": ENV_SECRET_PLACEHOLDER},
    }
    (tmp_path / "agent.py").write_text("graph = None\n", encoding="utf-8")
    config = tmp_path / "langgraph.json"
    config.write_text(json.dumps(config_body), encoding="utf-8")
    postgres_uri = (
        f"postgres://app:{POSTGRES_PASSWORD_PLACEHOLDER}@db.example.test:5432/app"
    )

    output, result = _invoke(
        [
            "up",
            "--verbose",
            "--no-pull",
            "--image",
            "local/app:test",
            "--postgres-uri",
            postgres_uri,
            "--config",
            str(config),
        ]
    )

    assert result.exit_code == 0, output
    compose_stdin = (state_dir / "compose-stdin").read_text(encoding="utf-8")
    assert ENV_SECRET_PLACEHOLDER in compose_stdin
    assert POSTGRES_PASSWORD_PLACEHOLDER in compose_stdin
    assert "+ " in output
    _assert_secret_not_in_output(ENV_SECRET_PLACEHOLDER, output)
    _assert_secret_not_in_output(POSTGRES_PASSWORD_PLACEHOLDER, output)


def test_build_docker_image_verbose_does_not_echo_dockerfile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bin_dir = tmp_path / "bin"
    state_dir = tmp_path / "docker-state"
    bin_dir.mkdir()
    state_dir.mkdir()
    _write_fake_docker(bin_dir, state_dir)
    _prepend_path(monkeypatch, bin_dir)
    config_path = _project(tmp_path, dockerfile_secret=DOCKERFILE_SECRET_PLACEHOLDER)
    config_json = validate_config_file(config_path)

    captured: list[str] = []
    original = sys.stdout

    class _Buffer:
        def write(self, data: bytes) -> int:
            captured.append(data.decode())
            return len(data)

        def flush(self) -> None:
            return None

    class _Stream:
        def write(self, data: str) -> int:
            captured.append(data)
            return len(data)

        def flush(self) -> None:
            return None

        @property
        def buffer(self) -> _Buffer:
            return _Buffer()

    sys.stdout = _Stream()  # type: ignore[assignment]
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
                "local/app:test",
                verbose=True,
            )
    finally:
        sys.stdout = original

    output = "".join(captured)
    dockerfile = (state_dir / "build-stdin").read_text(encoding="utf-8")
    assert DOCKERFILE_SECRET_PLACEHOLDER in dockerfile
    assert "+ " in output and "build" in output
    _assert_secret_not_in_output(DOCKERFILE_SECRET_PLACEHOLDER, output)
