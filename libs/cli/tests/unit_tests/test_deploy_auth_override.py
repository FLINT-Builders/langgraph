"""Cloud deploy must authenticate as the operator, not as the project.

A project ``langgraph.json`` env object or env file can name
``LANGGRAPH_HOST_API_KEY``, ``LANGSMITH_API_KEY``, ``LANGCHAIN_API_KEY``,
and ``LANGSMITH_TENANT_ID``. Those values are attacker-controlled whenever
someone else can edit the project the operator deploys. The process
environment is the operator's. These tests are the spec for that split:

* Resolve ``LANGGRAPH_HOST_API_KEY``, then ``LANGSMITH_API_KEY``, then
  ``LANGCHAIN_API_KEY`` from the process environment before any project
  env file or ``langgraph.json`` env object. Apply the same order to
  ``LANGSMITH_TENANT_ID``: the process environment wins when it is set.
* When a project file also contains one of those names, keep the process
  value and warn that the project value was ignored. The warning line
  names the variable and says it was ignored.
* Use a project API key only when the process environment has none and
  the operator has confirmed that source. ``--no-input`` cannot confirm,
  so it must fail. Declining the confirmation must fail. Either refusal
  happens before any host request, upload, or write to the env file.
* ``--api-key`` is an explicit operator value and wins the same way.

The commands below are the entry points that build a host client.
"""

from __future__ import annotations

import io
import json
import pathlib
import tarfile
from typing import Any

import httpx
import pytest
from click.testing import CliRunner

import langgraph_cli.deploy as deploy_module
from langgraph_cli.cli import cli

SHELL_HOST_API_KEY = "shell-langgraph-host-api-key"
SHELL_LANGSMITH_API_KEY = "shell-langsmith-api-key"
SHELL_LANGCHAIN_API_KEY = "shell-langchain-api-key"
SHELL_FLAG_API_KEY = "shell-flag-api-key"
SHELL_TENANT_ID = "shell-tenant-id"
PROJECT_HOST_API_KEY = "project-langgraph-host-api-key"
PROJECT_LANGSMITH_API_KEY = "project-langsmith-api-key"
PROJECT_TENANT_ID = "project-tenant-id"
PROJECT_APP_SECRET = "project-app-secret"
SOURCE_MARKER = "private-agent-source-marker"
DEPLOYMENT_NAME = "private-agent"
HOST = "api.host.langchain.com"
UPLOAD_URL = "https://uploads.invalid/source"
OBJECT_PATH = "objects/source.tar.gz"

_PROCESS_KEY_NAMES = (
    "LANGGRAPH_HOST_API_KEY",
    "LANGSMITH_API_KEY",
    "LANGCHAIN_API_KEY",
    "LANGSMITH_TENANT_ID",
)


@pytest.fixture(autouse=True)
def _reset_deploy_globals(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(deploy_module, "_no_input", False)
    monkeypatch.setattr(deploy_module, "_emitter", None)
    for name in _PROCESS_KEY_NAMES:
        monkeypatch.delenv(name, raising=False)


class HostCapture:
    """Record host requests and the signed-URL upload without leaving the machine.

    ``HostBackendClient`` still builds the httpx client and its headers.
    ``_upload_to_gcs`` still builds the PUT. Only the sockets are replaced.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.uploads: list[dict[str, Any]] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        transport = httpx.MockTransport(self._handle)
        real_client = httpx.Client

        def factory(*args: object, **kwargs: object) -> httpx.Client:
            kwargs["transport"] = transport
            return real_client(*args, **kwargs)

        monkeypatch.setattr("langgraph_cli.host_backend.httpx.Client", factory)

        def fake_urlopen(req: Any, timeout: float | None = None) -> io.BytesIO:
            data = req.data
            body = data.read() if hasattr(data, "read") else data
            if isinstance(body, str):
                body = body.encode()
            headers = {key.lower(): value for key, value in req.header_items()}
            self.uploads.append({"url": req.full_url, "body": body, "headers": headers})
            return io.BytesIO(b"")

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(
            {
                "method": request.method,
                "url": str(request.url),
                "host": request.url.host,
                "path": request.url.path,
                "api_key": request.headers.get("x-api-key"),
                "tenant_id": request.headers.get("x-tenant-id"),
                "body": request.content,
            }
        )
        path = request.url.path
        method = request.method
        if method == "GET" and path == "/v2/deployments":
            return httpx.Response(200, json={"resources": []})
        if method == "POST" and path == "/v2/deployments":
            return httpx.Response(
                200, json={"id": "dep-created", "tenant_id": SHELL_TENANT_ID}
            )
        if method == "GET" and path == "/v2/deployments/dep-existing":
            return httpx.Response(200, json={"id": "dep-existing", "source_config": {}})
        if method == "POST" and path == "/v2/deployments/dep-created/upload-url":
            return httpx.Response(
                200,
                json={"upload_url": UPLOAD_URL, "object_path": OBJECT_PATH},
            )
        if method == "PATCH" and path == "/v2/deployments/dep-created":
            return httpx.Response(
                200, json={"id": "dep-created", "tenant_id": SHELL_TENANT_ID}
            )
        if method == "DELETE" and path.startswith("/v2/deployments/"):
            return httpx.Response(200, json={})
        if method == "GET" and "/revisions" in path:
            return httpx.Response(
                200,
                json={
                    "resources": [
                        {
                            "id": "rev-1",
                            "status": "DEPLOYED",
                            "created_at": "2026-01-01T00:00:00Z",
                        }
                    ]
                },
            )
        if method == "POST" and (
            path.endswith("/deploy_logs") or path.endswith("/build_logs")
        ):
            return httpx.Response(200, json={"logs": []})
        return httpx.Response(500, json={"error": f"unexpected {method} {path}"})


def _write_project(
    directory: pathlib.Path,
    env: dict[str, str] | str | None,
    *,
    include_dotenv: dict[str, str] | None = None,
) -> pathlib.Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "agent.py").write_text(
        f"# Operator source that must stay in the operator account.\n"
        f"MARKER = {SOURCE_MARKER!r}\n",
        encoding="utf-8",
    )
    config: dict[str, Any] = {
        "python_version": "3.11",
        "dependencies": ["."],
        "graphs": {"agent": "./agent.py:graph"},
        "image_distro": "wolfi",
    }
    if env is not None:
        config["env"] = env
    config_path = directory / "langgraph.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    if include_dotenv is not None:
        lines = [f"{key}={value}" for key, value in include_dotenv.items()]
        (directory / ".env").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return config_path


def _invoke(
    args: list[str],
    *,
    cwd: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    shell: dict[str, str] | None = None,
    stdin: str | None = None,
) -> tuple[Any, HostCapture]:
    monkeypatch.chdir(cwd)
    host = HostCapture()
    host.install(monkeypatch)
    result = CliRunner().invoke(cli, args, input=stdin, env=shell or {})
    return result, host


def _assert_line_says_ignored(output: str, variable: str) -> None:
    for line in output.splitlines():
        if variable in line and "ignored" in line.lower():
            return
    raise AssertionError(
        f"expected a warning line naming {variable} and saying it was ignored\n"
        f"output:\n{output}"
    )


def _assert_operator_credentials(
    host: HostCapture, api_key: str, tenant_id: str
) -> None:
    assert host.calls, "deploy made no host request"
    for call in host.calls:
        assert call["host"] == HOST, call
        assert call["api_key"] == api_key, call
        assert call["tenant_id"] == tenant_id, call


def _secrets(call: dict[str, Any]) -> dict[str, str]:
    payload = json.loads(call["body"])
    return {item["name"]: item["value"] for item in payload["secrets"]}


def _signed_uploads(host: HostCapture) -> list[dict[str, Any]]:
    return [upload for upload in host.uploads if upload["url"] == UPLOAD_URL]


def _assert_deploy_uploaded_operator_project(
    host: HostCapture, api_key: str, tenant_id: str
) -> None:
    _assert_operator_credentials(host, api_key, tenant_id)
    by_path = {(call["method"], call["path"]): call for call in host.calls}
    create = by_path[("POST", "/v2/deployments")]
    upload = by_path[("POST", "/v2/deployments/dep-created/upload-url")]
    patch = by_path[("PATCH", "/v2/deployments/dep-created")]
    created_secrets = _secrets(create)
    patched_secrets = _secrets(patch)
    assert created_secrets["APP_SECRET"] == PROJECT_APP_SECRET
    assert patched_secrets["APP_SECRET"] == PROJECT_APP_SECRET
    # These names are reserved and must not be copied into the deployment.
    # LANGGRAPH_HOST_API_KEY is not reserved today; auth tests cover it as a
    # credential, not as a filtered secret.
    for reserved in (
        "LANGSMITH_API_KEY",
        "LANGCHAIN_API_KEY",
        "LANGSMITH_TENANT_ID",
    ):
        assert reserved not in created_secrets
        assert reserved not in patched_secrets
    assert upload["api_key"] == api_key
    patched = json.loads(patch["body"])
    assert patched["source_revision_config"]["source_tarball_path"] == OBJECT_PATH

    signed = _signed_uploads(host)
    assert len(signed) == 1
    uploaded = signed[0]
    assert uploaded["headers"]["content-type"] == "application/gzip"
    with tarfile.open(fileobj=io.BytesIO(uploaded["body"]), mode="r:gz") as archive:
        source = archive.extractfile("agent.py")
        assert source is not None
        assert SOURCE_MARKER.encode() in source.read()


def _assert_refused(
    result: Any,
    host: HostCapture,
    env_file: pathlib.Path | None = None,
    env_before: bytes | None = None,
) -> None:
    described = [
        (call["method"], call["path"], call["api_key"], call["tenant_id"])
        for call in host.calls
    ]
    assert host.calls == [], described
    assert _signed_uploads(host) == []
    assert result.exit_code != 0, result.output
    lowered = result.output.lower()
    assert "langsmith_api_key" in lowered or "api key" in lowered, result.output
    assert any(token in lowered for token in ("project", ".env", "langgraph.json"))
    if env_file is not None and env_before is not None:
        assert env_file.read_bytes() == env_before


def _project_env(key_name: str, key_value: str) -> dict[str, str]:
    return {
        key_name: key_value,
        "LANGSMITH_TENANT_ID": PROJECT_TENANT_ID,
        "APP_SECRET": PROJECT_APP_SECRET,
    }


def _deploy_args(config_path: pathlib.Path, *, no_input: bool = True) -> list[str]:
    args = [
        "deploy",
        "--remote",
        "--no-wait",
        "--name",
        DEPLOYMENT_NAME,
        "--config",
        str(config_path),
    ]
    if no_input:
        args.append("--no-input")
    return args


@pytest.mark.parametrize(
    ("shell_name", "shell_key", "project_key_name", "project_key"),
    [
        (
            "LANGSMITH_API_KEY",
            SHELL_LANGSMITH_API_KEY,
            "LANGSMITH_API_KEY",
            PROJECT_LANGSMITH_API_KEY,
        ),
        (
            "LANGCHAIN_API_KEY",
            SHELL_LANGCHAIN_API_KEY,
            "LANGSMITH_API_KEY",
            PROJECT_LANGSMITH_API_KEY,
        ),
        (
            "LANGSMITH_API_KEY",
            SHELL_LANGSMITH_API_KEY,
            "LANGGRAPH_HOST_API_KEY",
            PROJECT_HOST_API_KEY,
        ),
        (
            "LANGGRAPH_HOST_API_KEY",
            SHELL_HOST_API_KEY,
            "LANGSMITH_API_KEY",
            PROJECT_LANGSMITH_API_KEY,
        ),
    ],
    ids=[
        "shell-langsmith-vs-project-langsmith",
        "shell-langchain-vs-project-langsmith",
        "shell-langsmith-vs-project-host-key",
        "shell-host-key-vs-project-langsmith",
    ],
)
def test_deploy_keeps_process_env_over_config_env(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    shell_name: str,
    shell_key: str,
    project_key_name: str,
    project_key: str,
) -> None:
    config_path = _write_project(tmp_path, _project_env(project_key_name, project_key))
    result, host = _invoke(
        _deploy_args(config_path),
        cwd=tmp_path,
        monkeypatch=monkeypatch,
        shell={
            shell_name: shell_key,
            "LANGSMITH_TENANT_ID": SHELL_TENANT_ID,
        },
    )
    assert result.exit_code == 0, result.output
    _assert_deploy_uploaded_operator_project(host, shell_key, SHELL_TENANT_ID)
    _assert_line_says_ignored(result.output, project_key_name)
    _assert_line_says_ignored(result.output, "LANGSMITH_TENANT_ID")
    assert project_key not in {call["api_key"] for call in host.calls}


def test_process_env_prefers_langsmith_key_over_langchain_key(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_project(
        tmp_path,
        _project_env("LANGSMITH_API_KEY", PROJECT_LANGSMITH_API_KEY),
    )
    result, host = _invoke(
        _deploy_args(config_path),
        cwd=tmp_path,
        monkeypatch=monkeypatch,
        shell={
            "LANGCHAIN_API_KEY": SHELL_LANGCHAIN_API_KEY,
            "LANGSMITH_API_KEY": SHELL_LANGSMITH_API_KEY,
            "LANGSMITH_TENANT_ID": SHELL_TENANT_ID,
        },
    )
    assert result.exit_code == 0, result.output
    _assert_deploy_uploaded_operator_project(
        host, SHELL_LANGSMITH_API_KEY, SHELL_TENANT_ID
    )


def test_explicit_api_key_flag_beats_project_env(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_project(
        tmp_path,
        _project_env("LANGSMITH_API_KEY", PROJECT_LANGSMITH_API_KEY),
    )
    result, host = _invoke(
        [
            *_deploy_args(config_path),
            "--api-key",
            SHELL_FLAG_API_KEY,
        ],
        cwd=tmp_path,
        monkeypatch=monkeypatch,
        shell={"LANGSMITH_TENANT_ID": SHELL_TENANT_ID},
    )
    assert result.exit_code == 0, result.output
    _assert_deploy_uploaded_operator_project(host, SHELL_FLAG_API_KEY, SHELL_TENANT_ID)
    _assert_line_says_ignored(result.output, "LANGSMITH_API_KEY")
    _assert_line_says_ignored(result.output, "LANGSMITH_TENANT_ID")


@pytest.mark.parametrize(
    "env_field",
    ["named-file", "default-dotenv"],
    ids=["config-env-path", "default-dotenv"],
)
def test_deploy_keeps_process_env_over_dotenv_file(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    env_field: str,
) -> None:
    dotenv = _project_env("LANGSMITH_API_KEY", PROJECT_LANGSMITH_API_KEY)
    if env_field == "named-file":
        config_path = _write_project(tmp_path, "secrets.env")
        lines = [f"{key}={value}" for key, value in dotenv.items()]
        (tmp_path / "secrets.env").write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        config_path = _write_project(tmp_path, None, include_dotenv=dotenv)
    result, host = _invoke(
        _deploy_args(config_path),
        cwd=tmp_path,
        monkeypatch=monkeypatch,
        shell={
            "LANGSMITH_API_KEY": SHELL_LANGSMITH_API_KEY,
            "LANGSMITH_TENANT_ID": SHELL_TENANT_ID,
        },
    )
    assert result.exit_code == 0, result.output
    _assert_deploy_uploaded_operator_project(
        host, SHELL_LANGSMITH_API_KEY, SHELL_TENANT_ID
    )
    _assert_line_says_ignored(result.output, "LANGSMITH_API_KEY")
    _assert_line_says_ignored(result.output, "LANGSMITH_TENANT_ID")
    with tarfile.open(
        fileobj=io.BytesIO(_signed_uploads(host)[0]["body"]), mode="r:gz"
    ) as archive:
        names = archive.getnames()
        env_name = "secrets.env" if env_field == "named-file" else ".env"
        assert env_name in names
        extracted = archive.extractfile(env_name)
        assert extracted is not None
        assert PROJECT_APP_SECRET.encode() in extracted.read()


def test_no_input_rejects_unconfirmed_config_env_key(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_project(
        tmp_path,
        _project_env("LANGSMITH_API_KEY", PROJECT_LANGSMITH_API_KEY),
    )
    result, host = _invoke(
        _deploy_args(config_path),
        cwd=tmp_path,
        monkeypatch=monkeypatch,
    )
    _assert_refused(result, host)


def test_no_input_rejects_unconfirmed_dotenv_key(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_project(
        tmp_path,
        "secrets.env",
    )
    env_file = tmp_path / "secrets.env"
    env_file.write_text(
        "LANGSMITH_API_KEY="
        + PROJECT_LANGSMITH_API_KEY
        + "\nAPP_SECRET="
        + PROJECT_APP_SECRET
        + "\n",
        encoding="utf-8",
    )
    before = env_file.read_bytes()
    result, host = _invoke(
        _deploy_args(config_path),
        cwd=tmp_path,
        monkeypatch=monkeypatch,
    )
    _assert_refused(result, host, env_file, before)


def test_declining_project_key_writes_nothing_and_uploads_nothing(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_project(tmp_path, None)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                f"LANGSMITH_API_KEY={PROJECT_LANGSMITH_API_KEY}",
                f"LANGSMITH_TENANT_ID={PROJECT_TENANT_ID}",
                f"APP_SECRET={PROJECT_APP_SECRET}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    before = env_file.read_bytes()
    result, host = _invoke(
        _deploy_args(config_path, no_input=False),
        cwd=tmp_path,
        monkeypatch=monkeypatch,
        stdin="n\n",
    )
    _assert_refused(result, host, env_file, before)


_SUBCOMMANDS = [
    ["deploy", "list"],
    ["deploy", "revisions", "list", "dep-existing"],
    ["deploy", "delete", "--force", "dep-existing"],
    ["deploy", "logs", "--deployment-id", "dep-existing"],
]


@pytest.mark.parametrize("args", _SUBCOMMANDS)
def test_subcommand_keeps_process_env_over_dotenv(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    args: list[str],
) -> None:
    _write_project(
        tmp_path,
        None,
        include_dotenv=_project_env("LANGSMITH_API_KEY", PROJECT_LANGSMITH_API_KEY),
    )
    result, host = _invoke(
        args,
        cwd=tmp_path,
        monkeypatch=monkeypatch,
        shell={
            "LANGSMITH_API_KEY": SHELL_LANGSMITH_API_KEY,
            "LANGSMITH_TENANT_ID": SHELL_TENANT_ID,
        },
    )
    assert result.exit_code == 0, result.output
    _assert_operator_credentials(host, SHELL_LANGSMITH_API_KEY, SHELL_TENANT_ID)
    _assert_line_says_ignored(result.output, "LANGSMITH_API_KEY")
    _assert_line_says_ignored(result.output, "LANGSMITH_TENANT_ID")
    if args[1] == "delete":
        assert any(call["method"] == "DELETE" for call in host.calls)


@pytest.mark.parametrize("args", _SUBCOMMANDS)
def test_subcommand_rejects_unconfirmed_dotenv_key(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    args: list[str],
) -> None:
    _write_project(
        tmp_path,
        None,
        include_dotenv=_project_env("LANGSMITH_API_KEY", PROJECT_LANGSMITH_API_KEY),
    )
    env_file = tmp_path / ".env"
    before = env_file.read_bytes()
    result, host = _invoke(
        args,
        cwd=tmp_path,
        monkeypatch=monkeypatch,
        stdin="n\n",
    )
    _assert_refused(result, host, env_file, before)
