"""Env and credential files must stay out of remote-deploy source archives.

These tests exercise archive creation and `langgraph deploy` on the remote
build path. They fail while those files are copied into the tarball that is
uploaded for the remote builder.
"""

from __future__ import annotations

import io
import json
import tarfile
import textwrap
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from click.testing import CliRunner

from langgraph_cli.archive import create_archive
from langgraph_cli.config import config_to_docker, validate_config
from langgraph_cli.deploy import deploy

# Obvious stand-in. Not a credential.
_PLACEHOLDER = "ENVFILE_PLACEHOLDER"
_PLACEHOLDER_LINE = f"PROVIDER_KEY={_PLACEHOLDER}\n".encode()
_DEPLOYMENT_ID = "dep-under-test"
_OBJECT_PATH = "deploy-uploads/source.tar.gz"
_UPLOAD_PATH = "/gcs-upload/source.tar.gz"

# Return value of can_build_locally when the docker binary is absent.
_NO_DOCKER = (
    False,
    "Docker is required but not installed.\n"
    "Install Docker Desktop: https://docs.docker.com/get-docker/",
)


def _is_credential_member(name: str) -> bool:
    base = name.rsplit("/", 1)[-1]
    return base in {".env", ".npmrc", ".netrc"} or base.startswith(".env.")


def _read_archive(source: Path | bytes) -> tuple[list[str], bytes]:
    if isinstance(source, bytes):
        handle = tarfile.open(fileobj=io.BytesIO(source), mode="r:gz")
    else:
        handle = tarfile.open(source, mode="r:gz")
    with handle:
        names = handle.getnames()
        payload = b""
        for member in handle.getmembers():
            if not member.isfile():
                continue
            extracted = handle.extractfile(member)
            if extracted is not None:
                payload += extracted.read()
    return names, payload


def _assert_archive_safe(names: list[str], payload: bytes) -> None:
    leaked = [name for name in names if _is_credential_member(name)]
    assert leaked == [], f"credential files were packed: {leaked}"
    assert _PLACEHOLDER.encode() not in payload
    assert any(name.endswith("graphs/agent.py") for name in names)
    assert any(name.endswith("shared-lib/lib.py") for name in names)
    assert any(name.endswith("notes.txt") for name in names)
    assert not any(name.endswith(".langgraph-data") for name in names)


def _write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _write_project(root: Path, *, reinclude: bool) -> tuple[Path, dict[str, bytes]]:
    """Project shaped like the sample app: env is `.env`, gitignore is narrow."""
    project = root / "sampleapp"
    shared = root / "shared-lib"
    project.mkdir()
    shared.mkdir()

    config_path = project / "langgraph.json"
    config_path.write_text(
        json.dumps(
            {
                "python_version": "3.11",
                "dependencies": [".", "../shared-lib"],
                "graphs": {"agent": "./graphs/agent.py:graph"},
                "env": ".env",
            }
        )
    )
    _write_bytes(project / "graphs" / "agent.py", b"graph = object()\n")
    _write_bytes(project / "notes.txt", b"keep-me\n")
    _write_bytes(project / ".langgraph-data", b"local-state\n")
    _write_bytes(shared / "lib.py", b"value = 1\n")

    credential_paths = [
        project / ".env",
        project / ".env.local",
        project / ".env.production",
        project / ".npmrc",
        project / ".netrc",
        project / "graphs" / ".env",
        project / "graphs" / ".npmrc",
        project / "graphs" / ".env.local",
        shared / ".env",
        shared / ".env.production",
        shared / ".netrc",
        shared / ".npmrc",
    ]
    originals: dict[str, bytes] = {}
    for path in credential_paths:
        _write_bytes(path, _PLACEHOLDER_LINE)
        originals[str(path)] = _PLACEHOLDER_LINE

    gitignore = ".langgraph-data\n"
    if reinclude:
        gitignore += "!.env\n!.env.*\n!.npmrc\n!.netrc\n"
        (project / ".dockerignore").write_text("!.env\n!.env.*\n!.npmrc\n!.netrc\n")
        (shared / ".gitignore").write_text("!.env\n!.env.*\n!.npmrc\n!.netrc\n")
        (shared / ".dockerignore").write_text("!.env\n!.env.*\n!.npmrc\n!.netrc\n")
    (project / ".gitignore").write_text(gitignore)
    return config_path, originals


def _assert_sources_unchanged(originals: dict[str, bytes]) -> None:
    for name, content in originals.items():
        path = Path(name)
        assert path.is_file()
        assert path.read_bytes() == content


def _validated(config_path: Path) -> dict:
    return validate_config(json.loads(config_path.read_text()))


def test_create_archive_omits_env_files_the_sample_gitignore_keeps(tmp_path: Path):
    config_path, originals = _write_project(tmp_path, reinclude=False)

    with create_archive(config_path, _validated(config_path)) as (
        archive_path,
        file_size,
        config_rel,
    ):
        assert file_size > 0
        assert config_rel.endswith("langgraph.json")
        names, payload = _read_archive(Path(archive_path))

    _assert_archive_safe(names, payload)
    _assert_sources_unchanged(originals)


def test_create_archive_omits_credential_files_reincluded_by_ignore_files(
    tmp_path: Path,
):
    config_path, originals = _write_project(tmp_path, reinclude=True)

    with create_archive(config_path, _validated(config_path)) as (
        archive_path,
        _file_size,
        _config_rel,
    ):
        names, payload = _read_archive(Path(archive_path))

    _assert_archive_safe(names, payload)
    _assert_sources_unchanged(originals)


class _HostHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args) -> None:
        return

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return b""
        return self.rfile.read(length)

    def _record(self, body: bytes) -> None:
        self.server.requests.append(  # type: ignore[attr-defined]
            {
                "method": self.command,
                "path": self.path.split("?", 1)[0],
                "body": body,
            }
        )

    def _json(self, status: int, payload: dict) -> None:
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        body = self._read_body()
        self._record(body)
        if self.path.split("?", 1)[0] == "/v2/deployments":
            self._json(200, {"resources": []})
            return
        self._json(404, {"error": "unexpected"})

    def do_POST(self) -> None:
        body = self._read_body()
        self._record(body)
        path = self.path.split("?", 1)[0]
        if path == "/v2/deployments":
            self._json(200, {"id": _DEPLOYMENT_ID})
            return
        if path == f"/v2/deployments/{_DEPLOYMENT_ID}/upload-url":
            self._json(
                200,
                {
                    "upload_url": self.server.upload_url,  # type: ignore[attr-defined]
                    "object_path": _OBJECT_PATH,
                },
            )
            return
        self._json(404, {"error": "unexpected"})

    def do_PUT(self) -> None:
        body = self._read_body()
        self._record(body)
        self._json(200, {"ok": True})

    def do_PATCH(self) -> None:
        body = self._read_body()
        self._record(body)
        self._json(200, {"id": "rev-under-test", "tenant_id": "tenant-under-test"})


def _start_host() -> tuple[ThreadingHTTPServer, str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _HostHandler)
    server.requests = []  # type: ignore[attr-defined]
    port = server.server_address[1]
    host = f"http://127.0.0.1:{port}"
    server.upload_url = f"{host}{_UPLOAD_PATH}"  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, host


def _secrets_from(body: bytes) -> list[dict]:
    payload = json.loads(body.decode())
    secrets = payload.get("secrets")
    assert isinstance(secrets, list)
    return secrets


@pytest.mark.parametrize("remote", [True, False])
def test_deploy_remote_upload_omits_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, remote: bool
):
    config_path, originals = _write_project(tmp_path, reinclude=False)
    if not remote:
        # Auto mode selects the remote builder when local Docker cannot build.
        monkeypatch.setattr(
            "langgraph_cli.deploy.can_build_locally", lambda: _NO_DOCKER
        )
    monkeypatch.setenv("LANGGRAPH_CLI_NO_ANALYTICS", "1")
    server, host = _start_host()
    try:
        args = [
            "--no-wait",
            "--no-input",
            "--json",
            "--config",
            str(config_path),
            "--name",
            "sample-deploy",
            "--api-key",
            "placeholder-host-key",
            "--host-url",
            host,
        ]
        if remote:
            args.insert(0, "--remote")
        result = CliRunner().invoke(deploy, args)
    finally:
        server.shutdown()
        server.server_close()

    assert result.exit_code == 0, result.output
    requests = server.requests  # type: ignore[attr-defined]
    puts = [item for item in requests if item["method"] == "PUT"]
    assert len(puts) == 1
    assert puts[0]["path"] == _UPLOAD_PATH

    created = next(
        item
        for item in requests
        if item["method"] == "POST" and item["path"] == "/v2/deployments"
    )
    updated = next(item for item in requests if item["method"] == "PATCH")
    for item in (created, updated):
        secrets = _secrets_from(item["body"])
        assert secrets == [{"name": "PROVIDER_KEY", "value": _PLACEHOLDER}]
    update_payload = json.loads(updated["body"].decode())
    assert update_payload["source_revision_config"]["source_tarball_path"] == (
        _OBJECT_PATH
    )

    names, payload = _read_archive(puts[0]["body"])
    _assert_archive_safe(names, payload)
    _assert_sources_unchanged(originals)


def test_uv_lock_docker_plan_omits_root_credential_files(tmp_path: Path):
    """Top-level credential files must not be emitted as image copy lines.

    Directory copies such as `ADD src` are a separate path: Docker expands
    them using `.dockerignore` only, and this test does not cover children
    inside those directories.
    """
    project = tmp_path / "single"
    project.mkdir()
    (project / "uv.lock").write_text("# uv lock file\n")
    (project / "pyproject.toml").write_text(
        textwrap.dedent(
            """
            [project]
            name = "single-app"
            version = "0.1.0"
            dependencies = ["httpx>=0.28"]

            [build-system]
            requires = ["setuptools>=61"]
            build-backend = "setuptools.build_meta"
            """
        ).strip()
        + "\n"
    )
    (project / "langgraph.json").write_text("{}\n")
    (project / "src").mkdir()
    (project / "src" / "agent.py").write_text("graph = object()\n")
    (project / ".dockerignore").write_text("!.env\n!.env.*\n!.npmrc\n!.netrc\n")
    for name in (".env", ".env.local", ".env.production", ".npmrc", ".netrc"):
        (project / name).write_bytes(_PLACEHOLDER_LINE)

    config = validate_config(
        {
            "python_version": "3.11",
            "graphs": {"agent": "./src/agent.py:graph"},
            "source": {"kind": "uv"},
        }
    )
    docker, _contexts = config_to_docker(
        project / "langgraph.json",
        config,
        base_image="langchain/langgraph-api:0.2.47",
    )

    for leaked in (
        "ADD .env ",
        "ADD .env.local ",
        "ADD .env.production ",
        "ADD .npmrc ",
        "ADD .netrc ",
    ):
        assert leaked not in docker, docker
    assert _PLACEHOLDER not in docker
    assert "ADD src /deps/workspace/src" in docker
