"""Env paths in langgraph.json must stay inside the project directory.

`langgraph deploy` loads the file named by `env` and sends every
non-reserved variable to LangSmith as deployment secrets. `langgraph up`
copies that same string into the compose `env_file` entry. These tests
drive those commands with an env path that resolves outside the project.

They fail while that path is still accepted. A fix has to refuse the path
before any secret is uploaded and before the outside file is modified.
Resolving the path, checking it, then resolving it again is not enough:
the symlink test retargets the link between lookups.
"""

from __future__ import annotations

import json
import pathlib
from collections.abc import Callable

import click
import httpx
import pytest
from click.testing import CliRunner

import langgraph_cli.deploy as deploy_module
from langgraph_cli.cli import cli, prepare_args_and_stdin
from langgraph_cli.config import validate_config_file
from langgraph_cli.docker import DockerCapabilities, Version
from langgraph_cli.host_backend import HostBackendClient

OUTSIDE_VALUE = "outside-placeholder"
INSIDE_VALUE = "inside-placeholder"
SECRET_NAME = "MODEL_PROVIDER_KEY"
DEPLOYMENT_NAME = "containment-test"
API_KEY_PLACEHOLDER = "placeholder-api-key"

CAPABILITIES = DockerCapabilities(
    version_docker=Version(26, 1, 1),
    version_compose=Version(2, 27, 0),
    healthcheck_start_interval=True,
)


def _config_dict(env: str | dict) -> dict:
    return {
        "python_version": "3.11",
        "dependencies": ["."],
        "graphs": {"agent": "./agent.py:graph"},
        "env": env,
    }


def _write_project(root: pathlib.Path, env: str | dict) -> pathlib.Path:
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "langgraph.json"
    config_path.write_text(json.dumps(_config_dict(env)), encoding="utf-8")
    (root / "agent.py").write_text("graph = None\n", encoding="utf-8")
    return config_path


def _write_env(path: pathlib.Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{SECRET_NAME}={value}\n", encoding="utf-8")


class _CapturedRequest:
    def __init__(self, method: str, path: str, body: bytes):
        self.method = method
        self.path = path
        self.body = body


def _install_host(
    monkeypatch: pytest.MonkeyPatch, existing: bool
) -> list[_CapturedRequest]:
    """Serve the host API locally and record every request body."""
    captured: list[_CapturedRequest] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(
            _CapturedRequest(request.method, request.url.path, request.content)
        )
        path = request.url.path
        if request.method == "GET" and path == "/v2/deployments":
            resources = (
                [{"id": "dep-existing", "name": DEPLOYMENT_NAME}] if existing else []
            )
            return httpx.Response(200, json={"resources": resources})
        if request.method == "POST" and path == "/v2/deployments":
            return httpx.Response(
                200, json={"id": "dep-created", "tenant_id": "tenant-1"}
            )
        if request.method == "POST" and path.endswith("/upload-url"):
            return httpx.Response(
                200,
                json={
                    "upload_url": "https://example.invalid/upload",
                    "object_path": "object",
                },
            )
        if request.method == "POST" and path.endswith("/push-token"):
            return httpx.Response(
                200,
                json={
                    "token": "placeholder-token",
                    "registry_url": "example.invalid",
                },
            )
        if request.method == "PATCH":
            return httpx.Response(
                200, json={"id": "dep-updated", "tenant_id": "tenant-1"}
            )
        return httpx.Response(200, json={"id": "dep-1", "status": "DEPLOYED"})

    real_init = HostBackendClient.__init__

    def init(self, base_url: str, api_key: str, tenant_id: str | None = None):
        real_init(self, base_url, api_key, tenant_id)
        self._client = httpx.Client(
            base_url=self._base_url,
            headers=dict(self._client.headers),
            transport=httpx.MockTransport(handler),
            timeout=30,
        )

    monkeypatch.setattr(HostBackendClient, "__init__", init)
    return captured


def _install_build_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip image build, registry push, and object upload.

    Secret bytes are not part of those steps. The deploy command still
    builds the secrets list and the host client still sends it.
    """

    async def fake_subp_exec(cmd, *args, input=None, collect=False, **kwargs):
        del cmd, input, kwargs
        if "inspect" in args:
            image = args[-1]
            repo = image.rsplit(":", 1)[0]
            digest = f'["{repo}@sha256:{"a" * 64}"]'
            return digest, ""
        if collect:
            return "", ""
        return None, None

    monkeypatch.setattr(deploy_module, "can_build_locally", lambda: (True, None))
    monkeypatch.setattr(deploy_module, "build_docker_image", lambda *a, **k: None)
    monkeypatch.setattr(deploy_module, "subp_exec", fake_subp_exec)
    monkeypatch.setattr(deploy_module, "_upload_to_gcs", lambda *a, **k: None)


def _invoke_deploy(
    project: pathlib.Path,
    *,
    remote: bool,
    no_input: bool,
) -> click.testing.Result:
    runner = CliRunner()
    args = [
        "deploy",
        "--config",
        str(project / "langgraph.json"),
        "--name",
        DEPLOYMENT_NAME,
        "--api-key",
        API_KEY_PLACEHOLDER,
        "--host-url",
        "https://api.example.com",
        "--no-wait",
        "--json",
    ]
    if remote:
        args.append("--remote")
    else:
        args.append("--no-remote")
    if no_input:
        args.append("--no-input")
    return runner.invoke(cli, args)


def _assert_refused(
    result: click.testing.Result,
    captured: list[_CapturedRequest],
    outside: pathlib.Path,
    before: bytes,
) -> None:
    assert outside.read_bytes() == before
    leaked = [item for item in captured if OUTSIDE_VALUE.encode() in item.body]
    assert leaked == []
    assert result.exit_code != 0, result.output


def _outside_paths(
    tmp_path: pathlib.Path, project: pathlib.Path
) -> dict[str, Callable[[], str]]:
    outside = tmp_path / "outside.env"

    def absolute() -> str:
        _write_env(outside, OUTSIDE_VALUE)
        return str(outside)

    def parent_relative() -> str:
        _write_env(outside, OUTSIDE_VALUE)
        return "../outside.env"

    def nested_traversal() -> str:
        _write_env(outside, OUTSIDE_VALUE)
        return "missing/../../outside.env"

    def symlink() -> str:
        _write_env(outside, OUTSIDE_VALUE)
        link = project / "link.env"
        link.symlink_to(outside)
        return "link.env"

    return {
        "absolute": absolute,
        "parent-relative": parent_relative,
        "nested-traversal": nested_traversal,
        "symlink": symlink,
    }


@pytest.fixture(autouse=True)
def _no_analytics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGGRAPH_CLI_NO_ANALYTICS", "1")


@pytest.mark.parametrize(
    "kind", ["absolute", "parent-relative", "nested-traversal", "symlink"]
)
@pytest.mark.parametrize("remote", [True, False], ids=["remote-build", "local-build"])
@pytest.mark.parametrize("existing", [False, True], ids=["create", "update"])
def test_deploy_refuses_env_path_outside_project(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    remote: bool,
    existing: bool,
) -> None:
    project = tmp_path / "project"
    _write_project(project, ".env")
    env_field = _outside_paths(tmp_path, project)[kind]()
    config_path = project / "langgraph.json"
    config_path.write_text(json.dumps(_config_dict(env_field)), encoding="utf-8")
    outside = tmp_path / "outside.env"
    before = outside.read_bytes()
    captured = _install_host(monkeypatch, existing=existing)
    _install_build_stubs(monkeypatch)
    monkeypatch.chdir(project)

    result = _invoke_deploy(project, remote=remote, no_input=True)

    _assert_refused(result, captured, outside, before)


def test_deploy_does_not_write_deployment_name_into_outside_file(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Interactive deploy writes the deployment name into the resolved env file.

    Refusal has to happen before that write. `--no-input` skips the write,
    so this case runs with prompts disabled only by supplying the name.
    """
    project = tmp_path / "project"
    outside = tmp_path / "outside.env"
    _write_env(outside, OUTSIDE_VALUE)
    _write_project(project, str(outside))
    before = outside.read_bytes()
    captured = _install_host(monkeypatch, existing=False)
    _install_build_stubs(monkeypatch)
    monkeypatch.chdir(project)

    result = _invoke_deploy(project, remote=True, no_input=False)

    _assert_refused(result, captured, outside, before)
    assert "LANGSMITH_DEPLOYMENT_NAME" not in outside.read_text(encoding="utf-8")


def test_deploy_uploads_env_file_that_stays_inside_project(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    _write_env(project / ".env", INSIDE_VALUE)
    _write_project(project, "nested/../.env")
    captured = _install_host(monkeypatch, existing=False)
    _install_build_stubs(monkeypatch)
    monkeypatch.chdir(project)

    result = _invoke_deploy(project, remote=True, no_input=True)

    assert result.exit_code == 0, result.output
    bodies = [item.body for item in captured if item.body]
    assert any(INSIDE_VALUE.encode() in body for body in bodies)
    create = next(
        item
        for item in captured
        if item.method == "POST" and item.path == "/v2/deployments"
    )
    update = next(item for item in captured if item.method == "PATCH")
    for item in (create, update):
        payload = json.loads(item.body)
        assert {"name": SECRET_NAME, "value": INSIDE_VALUE} in payload["secrets"]


def test_deploy_uploads_inside_env_on_local_build_update(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    _write_env(project / ".env", INSIDE_VALUE)
    _write_project(project, ".env")
    captured = _install_host(monkeypatch, existing=True)
    _install_build_stubs(monkeypatch)
    monkeypatch.chdir(project)

    result = _invoke_deploy(project, remote=False, no_input=True)

    assert result.exit_code == 0, result.output
    update = next(item for item in captured if item.method == "PATCH")
    payload = json.loads(update.body)
    assert payload["revision_source"] == "internal_docker"
    assert {"name": SECRET_NAME, "value": INSIDE_VALUE} in payload["secrets"]
    assert not any(
        item.method == "POST" and item.path == "/v2/deployments" for item in captured
    )


def test_symlink_retarget_between_resolves_is_not_uploaded(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retarget a project symlink after the first resolve of that path.

    The first lookup lands on a file inside the project. A later lookup of
    the same path follows the symlink to the outside file. Opening the path
    that was already resolved still sees the inside file.
    """
    project = tmp_path / "project"
    inside = project / "inside.env"
    outside = tmp_path / "outside.env"
    _write_env(inside, INSIDE_VALUE)
    _write_env(outside, OUTSIDE_VALUE)
    link = project / "link.env"
    link.symlink_to(inside)
    _write_project(project, "link.env")
    before = outside.read_bytes()

    real_resolve = pathlib.Path.resolve
    seen = {"n": 0}

    def resolve(self: pathlib.Path, *args, **kwargs):
        result = real_resolve(self, *args, **kwargs)
        if self.name == "link.env":
            seen["n"] += 1
            if seen["n"] == 1:
                link.unlink()
                link.symlink_to(outside)
        return result

    monkeypatch.setattr(pathlib.Path, "resolve", resolve)
    captured = _install_host(monkeypatch, existing=False)
    _install_build_stubs(monkeypatch)
    monkeypatch.chdir(project)

    result = _invoke_deploy(project, remote=True, no_input=True)

    assert outside.read_bytes() == before
    assert not any(OUTSIDE_VALUE.encode() in item.body for item in captured)
    if result.exit_code == 0:
        assert any(INSIDE_VALUE.encode() in item.body for item in captured)
        assert seen["n"] >= 1


@pytest.mark.parametrize(
    "env_field",
    ["../outside.env", "missing/../../outside.env"],
)
def test_up_compose_refuses_env_path_outside_project(
    tmp_path: pathlib.Path,
    env_field: str,
) -> None:
    project = tmp_path / "project"
    outside = tmp_path / "outside.env"
    _write_env(outside, OUTSIDE_VALUE)
    config_path = _write_project(project, env_field)
    before = outside.read_bytes()
    config = validate_config_file(config_path)

    with pytest.raises(click.ClickException):
        prepare_args_and_stdin(
            capabilities=CAPABILITIES,
            config_path=config_path,
            config=config,
            docker_compose=None,
            port=8123,
            watch=False,
            image="example.invalid/repo:placeholder",
        )

    assert outside.read_bytes() == before


def test_up_compose_refuses_absolute_env_path(
    tmp_path: pathlib.Path,
) -> None:
    project = tmp_path / "project"
    outside = tmp_path / "outside.env"
    _write_env(outside, OUTSIDE_VALUE)
    config_path = _write_project(project, str(outside))
    before = outside.read_bytes()
    config = validate_config_file(config_path)

    with pytest.raises(click.ClickException):
        prepare_args_and_stdin(
            capabilities=CAPABILITIES,
            config_path=config_path,
            config=config,
            docker_compose=None,
            port=8123,
            watch=False,
            image="example.invalid/repo:placeholder",
        )

    assert outside.read_bytes() == before


def test_up_compose_keeps_env_file_inside_project(tmp_path: pathlib.Path) -> None:
    project = tmp_path / "project"
    _write_env(project / ".env", INSIDE_VALUE)
    config_path = _write_project(project, "nested/../.env")
    config = validate_config_file(config_path)

    _args, stdin = prepare_args_and_stdin(
        capabilities=CAPABILITIES,
        config_path=config_path,
        config=config,
        docker_compose=None,
        port=8123,
        watch=False,
        image="example.invalid/repo:placeholder",
    )

    assert "env_file:" in stdin
    assert OUTSIDE_VALUE not in stdin
    env_line = next(line for line in stdin.splitlines() if "env_file:" in line)
    raw_env = env_line.split("env_file:", 1)[1].strip()
    resolved_env = (config_path.parent / raw_env).resolve()
    assert resolved_env == (project / ".env").resolve()
    assert resolved_env.is_relative_to(config_path.parent.resolve())
