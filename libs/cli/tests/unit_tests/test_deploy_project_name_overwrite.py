"""A project file must not choose which LangSmith deployment `langgraph deploy` replaces.

LANGSMITH_DEPLOYMENT_NAME in the process environment, `--name`, and
`--deployment-id` are the operator's choice. The same name in the project's
`.env` (or the `env` string/object in langgraph.json) may only be offered as
the default in a prompt. Replacing or creating a deployment still requires
the operator to accept that name and, when the name already exists, to
confirm the deployment name and id that will be changed. `--no-input` cannot
skip that confirmation.

The host API and the local Docker daemon are stood in for here. Name parsing,
normalization, lookup, and the create/update calls are the real CLI code.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import click
import httpx
import pytest
from click.testing import CliRunner

import langgraph_cli.deploy as deploy_mod
from langgraph_cli.cli import cli
from langgraph_cli.host_backend import HostBackendClient

OPERATOR_API_KEY = "operator-api-key-placeholder"
PUSH_TOKEN = "push-token-placeholder"
HOST_URL = "http://langsmith-host.test"
REPO_SETTING = "from-repo"
EXISTING_NAME = "production"
EXISTING_ID = "dep-existing"
SPARE_NAME = "production-east"
SPARE_ID = "dep-east"
RELOCK_ID = "dep-after-relock"
OTHER_NAME = "staging"
OTHER_ID = "dep-staging"
AUTHORIZED_ID = "dep-authorized"
NORMALIZED_NAME = "production-main"
UNNORMALIZED_NAME = "Production_Main"

MUTATING_METHODS = {"POST", "PATCH", "PUT", "DELETE"}
DOCKER_MUTATIONS = {"login", "tag", "push", "build", "buildx"}


class _Directory:
    """Scripted deployment list. The second exact-name lookup returns a new id."""

    def __init__(self, exact: dict[str, str], relock: dict[str, str] | None = None):
        self.exact = dict(exact)
        self.relock = dict(relock or {})
        self.list_calls = 0

    def resources(self, contains: str) -> list[dict[str, str]]:
        self.list_calls += 1
        rows: list[dict[str, str]] = []
        for name, dep_id in self.exact.items():
            if contains and contains not in name:
                continue
            shown = dep_id
            if self.list_calls > 1 and name == contains and name in self.relock:
                shown = self.relock[name]
            rows.append({"name": name, "id": shown})
        rows.sort(key=lambda row: row["name"] == contains)
        return rows


class _UploadServer:
    def __init__(self) -> None:
        self.uploads: list[str] = []
        parent = self

        class Handler(BaseHTTPRequestHandler):
            def do_PUT(self) -> None:
                length = int(self.headers.get("Content-Length", "0") or "0")
                if length:
                    self.rfile.read(length)
                parent.uploads.append(self.path)
                self.send_response(200)
                self.end_headers()

            def log_message(self, format: str, *args: object) -> None:
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        host, port = self._server.server_address[:2]
        self.url = f"http://{host}:{port}/upload"

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


class _World:
    def __init__(self, directory: _Directory, upload: _UploadServer) -> None:
        self.directory = directory
        self.upload = upload
        self.requests: list[httpx.Request] = []
        self.prompts: list[str] = []
        self.events: list[tuple[str, ...]] = []
        self.docker: list[tuple[str, ...]] = []
        self.before: dict[str, bytes] = {}
        self.root: Path | None = None

    def mutations(self) -> list[str]:
        described: list[str] = []
        for request in self.requests:
            if request.method not in MUTATING_METHODS:
                continue
            body = request.content.decode() if request.content else ""
            described.append(f"{request.method} {request.url.path} {body}")
        return described

    def docker_mutations(self) -> list[tuple[str, ...]]:
        return [
            args
            for args in self.docker
            if any(part in DOCKER_MUTATIONS for part in args)
        ]

    def files(self) -> dict[str, bytes]:
        assert self.root is not None
        found: dict[str, bytes] = {}
        for path in self.root.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                found[path.relative_to(self.root).as_posix()] = path.read_bytes()
        return found


def _snapshot(world: _World, root: Path) -> None:
    world.root = root
    world.before = world.files()


def _install_stubs(monkeypatch: pytest.MonkeyPatch, world: _World) -> None:
    directory = world.directory

    def handler(request: httpx.Request) -> httpx.Response:
        world.requests.append(request)
        world.events.append(("http", request.method, request.url.path))
        parts = request.url.path.strip("/").split("/")
        if request.method == "GET" and parts == ["v2", "deployments"]:
            contains = request.url.params.get("name_contains", "")
            return httpx.Response(
                200, json={"resources": directory.resources(contains)}
            )
        if (
            request.method == "GET"
            and parts[:2] == ["v2", "deployments"]
            and len(parts) == 3
        ):
            return httpx.Response(200, json={"id": parts[2], "name": EXISTING_NAME})
        if request.method == "POST" and parts == ["v2", "deployments"]:
            return httpx.Response(200, json={"id": "dep-created"})
        if (
            request.method == "POST"
            and len(parts) == 4
            and parts[:2] == ["v2", "deployments"]
            and parts[3] == "push-token"
        ):
            return httpx.Response(
                200,
                json={
                    "token": PUSH_TOKEN,
                    "registry_url": "registry.example.test/team",
                },
            )
        if (
            request.method == "POST"
            and len(parts) == 4
            and parts[:2] == ["v2", "deployments"]
            and parts[3] == "upload-url"
        ):
            return httpx.Response(
                200,
                json={
                    "upload_url": world.upload.url,
                    "object_path": "objects/source.tar.gz",
                },
            )
        if (
            request.method == "PATCH"
            and parts[:2] == ["v2", "deployments"]
            and len(parts) == 3
        ):
            return httpx.Response(200, json={"id": parts[2]})
        if request.method == "POST" and (
            request.url.path.endswith("/deploy_logs")
            or request.url.path.endswith("/build_logs")
        ):
            return httpx.Response(200, json={"logs": []})
        return httpx.Response(
            500, json={"unexpected": request.method, "path": request.url.path}
        )

    original_init = HostBackendClient.__init__

    def init(
        self: HostBackendClient,
        base_url: str,
        api_key: str,
        tenant_id: str | None = None,
    ) -> None:
        original_init(self, base_url, api_key, tenant_id=tenant_id)
        real = self._client
        self._client = httpx.Client(
            base_url=str(real.base_url),
            headers=dict(real.headers),
            transport=httpx.MockTransport(handler),
            timeout=30,
        )
        real.close()

    async def fake_subp_exec(
        cmd: str, *args: str, collect: bool = False, **kwargs: Any
    ) -> tuple[str, str] | None:
        world.docker.append((cmd, *args))
        if collect and "{{json .RepoDigests}}" in args:
            image = args[-1]
            repo = image.rsplit(":", 1)[0]
            return json.dumps([f"{repo}@sha256:abc123"]) + "\n", ""
        if collect:
            return "linux/amd64\n", ""
        return None

    real_prompt = click.prompt
    real_confirm = click.confirm

    def prompt(text: str, default: Any = None, **kwargs: Any) -> Any:
        world.prompts.append(str(text))
        world.events.append(("prompt", str(text)))
        return real_prompt(text, default=default, **kwargs)

    def confirm(text: str, default: bool = False, **kwargs: Any) -> bool:
        world.prompts.append(str(text))
        world.events.append(("prompt", str(text)))
        return real_confirm(text, default=default, **kwargs)

    monkeypatch.setattr(HostBackendClient, "__init__", init)
    monkeypatch.setattr(deploy_mod, "subp_exec", fake_subp_exec)
    monkeypatch.setattr(
        deploy_mod,
        "can_build_locally",
        lambda: (False, "Docker is not available in this test."),
    )
    monkeypatch.setattr(click, "prompt", prompt)
    monkeypatch.setattr(click, "confirm", confirm)
    monkeypatch.delenv("LANGSMITH_DEPLOYMENT_NAME", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    monkeypatch.delenv("LANGGRAPH_HOST_API_KEY", raising=False)
    monkeypatch.setenv("LANGGRAPH_CLI_NO_ANALYTICS", "1")


def _write_project(
    root: Path,
    *,
    source: str,
    deployment_name: str,
    extra_dotenv: dict[str, str] | None = None,
) -> None:
    env_lines = [
        f"APP_SETTING={REPO_SETTING}",
        f"LANGSMITH_DEPLOYMENT_NAME={deployment_name}",
    ]
    config: dict[str, Any] = {
        "dependencies": ["."],
        "graphs": {"agent": "./agent.py:graph"},
    }
    if source == "dotenv":
        (root / ".env").write_text("\n".join(env_lines) + "\n")
    elif source == "env_file":
        (root / "deploy.env").write_text("\n".join(env_lines) + "\n")
        config["env"] = "deploy.env"
    elif source == "env_object":
        config["env"] = {
            "APP_SETTING": REPO_SETTING,
            "LANGSMITH_DEPLOYMENT_NAME": deployment_name,
        }
        if extra_dotenv:
            lines = [f"{key}={value}" for key, value in extra_dotenv.items()]
            (root / ".env").write_text("\n".join(lines) + "\n")
    else:
        raise AssertionError(source)
    (root / "langgraph.json").write_text(json.dumps(config))
    (root / "agent.py").write_text("graph = None\n")


def _invoke(
    world: _World,
    root: Path,
    args: list[str],
    *,
    stdin: str | None = "",
    env: dict[str, str] | None = None,
) -> click.testing.Result:
    _snapshot(world, root)
    runner = CliRunner()
    return runner.invoke(cli, args, input=stdin, env=env, catch_exceptions=True)


def _deploy_args(*extra: str) -> list[str]:
    return [
        "deploy",
        "--api-key",
        OPERATOR_API_KEY,
        "--host-url",
        HOST_URL,
        "--no-wait",
        *extra,
    ]


def _assert_api_key(world: _World) -> None:
    for request in world.requests:
        assert request.headers.get("x-api-key") == OPERATOR_API_KEY, (
            "host request was not authenticated with the operator --api-key flag"
        )


def _refusal_problems(result: click.testing.Result, world: _World) -> list[str]:
    problems: list[str] = []
    mutations = world.mutations()
    if mutations:
        problems.append("host mutation:\n" + "\n".join(mutations))
    if world.docker_mutations():
        problems.append(f"docker mutation: {world.docker_mutations()}")
    if world.upload.uploads:
        problems.append(f"source upload: {world.upload.uploads}")
    if world.files() != world.before:
        problems.append("project files changed before the deploy was refused")
    if result.exit_code == 0:
        problems.append("deploy command succeeded")
    if problems:
        problems.append(f"prompts={world.prompts!r}\noutput={result.output}")
    return problems


def _patch_bodies(world: _World, deployment_id: str) -> list[dict[str, Any]]:
    bodies: list[dict[str, Any]] = []
    for request in world.requests:
        if (
            request.method == "PATCH"
            and request.url.path == f"/v2/deployments/{deployment_id}"
        ):
            bodies.append(json.loads(request.content.decode()))
    return bodies


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch) -> _World:
    directory = _Directory(
        {
            EXISTING_NAME: EXISTING_ID,
            SPARE_NAME: SPARE_ID,
            OTHER_NAME: OTHER_ID,
            NORMALIZED_NAME: EXISTING_ID,
        },
        relock={EXISTING_NAME: RELOCK_ID, NORMALIZED_NAME: RELOCK_ID},
    )
    upload = _UploadServer()
    built = _World(directory, upload)
    _install_stubs(monkeypatch, built)
    yield built
    upload.close()


@pytest.mark.parametrize("source", ["dotenv", "env_file", "env_object"])
def test_project_name_does_not_replace_existing_deployment(
    world: _World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    extra = (
        {"LANGSMITH_DEPLOYMENT_NAME": OTHER_NAME, "APP_SETTING": "from-dotenv"}
        if source == "env_object"
        else None
    )
    _write_project(
        tmp_path, source=source, deployment_name=EXISTING_NAME, extra_dotenv=extra
    )
    monkeypatch.chdir(tmp_path)

    result = _invoke(
        world,
        tmp_path,
        _deploy_args("--no-input", "--image", "local/app:test"),
    )

    problems = _refusal_problems(result, world)
    assert not problems, "\n".join(problems)
    _assert_api_key(world)


def test_normalized_project_name_does_not_replace_matching_deployment(
    world: _World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_project(tmp_path, source="dotenv", deployment_name=UNNORMALIZED_NAME)
    monkeypatch.chdir(tmp_path)

    result = _invoke(
        world,
        tmp_path,
        _deploy_args("--no-input", "--image", "local/app:test"),
    )

    problems = _refusal_problems(result, world)
    assert not problems, "\n".join(problems)


def test_project_name_does_not_create_deployment_or_select_substring_match(
    world: _World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_project(tmp_path, source="dotenv", deployment_name="brand-new-app")
    world.directory.exact["brand-new-app-live"] = "dep-live"
    monkeypatch.chdir(tmp_path)

    result = _invoke(
        world,
        tmp_path,
        _deploy_args("--no-input", "--image", "local/app:test"),
    )

    problems = _refusal_problems(result, world)
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("source", ["dotenv", "env_object"])
def test_remote_build_does_not_replace_deployment_from_project_name(
    world: _World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    _write_project(tmp_path, source=source, deployment_name=EXISTING_NAME)
    monkeypatch.chdir(tmp_path)

    result = _invoke(world, tmp_path, _deploy_args("--no-input", "--remote"))

    problems = _refusal_problems(result, world)
    assert not problems, "\n".join(problems)


def test_accepting_the_default_name_without_confirming_id_does_not_update(
    world: _World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Enter accepts the project name. A following 'n' declines the replacement."""
    _write_project(tmp_path, source="dotenv", deployment_name=EXISTING_NAME)
    monkeypatch.chdir(tmp_path)

    result = _invoke(
        world,
        tmp_path,
        _deploy_args("--image", "local/app:test"),
        stdin="\nn\n",
    )

    problems = _refusal_problems(result, world)
    assert not problems, "\n".join(problems)


def test_confirmed_id_stays_bound_if_the_name_lookup_changes(
    world: _World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Enter accepts the project name, then 'y' confirms the id from the first lookup.

    A later lookup of that same name returns a different id. The update has to
    use the id the operator confirmed.
    """
    _write_project(tmp_path, source="dotenv", deployment_name=EXISTING_NAME)
    monkeypatch.chdir(tmp_path)

    result = _invoke(
        world,
        tmp_path,
        _deploy_args("--image", "local/app:test"),
        stdin="\ny\n",
    )

    assert result.exit_code == 0, result.output
    confirmation = [
        text for text in world.prompts if EXISTING_ID in text and EXISTING_NAME in text
    ]
    assert confirmation, (
        "operator was not asked to confirm the existing deployment name and id; "
        f"prompts={world.prompts!r}\noutput={result.output}"
    )
    confirm_at = next(
        index
        for index, event in enumerate(world.events)
        if event[0] == "prompt"
        and EXISTING_ID in event[1]
        and EXISTING_NAME in event[1]
    )
    mutation_at = next(
        (
            index
            for index, event in enumerate(world.events)
            if event[0] == "http" and event[1] in MUTATING_METHODS
        ),
        None,
    )
    assert mutation_at is not None, "confirmed deploy did not update the deployment"
    assert confirm_at < mutation_at, world.events
    bodies = _patch_bodies(world, EXISTING_ID)
    assert len(bodies) == 1, world.mutations()
    assert _patch_bodies(world, RELOCK_ID) == []
    assert _patch_bodies(world, SPARE_ID) == []
    assert bodies[0]["source_revision_config"]["image_uri"].endswith("@sha256:abc123")
    secret_names = {item["name"]: item["value"] for item in bodies[0]["secrets"]}
    assert secret_names["APP_SETTING"] == REPO_SETTING
    assert "LANGSMITH_DEPLOYMENT_NAME" not in secret_names
    mutated_ids = [
        request.url.path.split("/")[3]
        for request in world.requests
        if request.method in MUTATING_METHODS
        and request.url.path.startswith("/v2/deployments/")
    ]
    assert mutated_ids
    assert set(mutated_ids) == {EXISTING_ID}
    _assert_api_key(world)


def test_explicit_name_updates_that_deployment_not_the_project_file(
    world: _World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_project(tmp_path, source="dotenv", deployment_name=OTHER_NAME)
    monkeypatch.chdir(tmp_path)

    result = _invoke(
        world,
        tmp_path,
        _deploy_args(
            "--no-input", "--name", EXISTING_NAME, "--image", "local/app:test"
        ),
    )

    assert result.exit_code == 0, result.output
    bodies = _patch_bodies(world, EXISTING_ID)
    assert len(bodies) == 1, world.mutations()
    assert _patch_bodies(world, OTHER_ID) == []
    assert _patch_bodies(world, SPARE_ID) == []
    assert bodies[0]["revision_source"] == "internal_docker"
    assert bodies[0]["source_revision_config"]["image_uri"].endswith("@sha256:abc123")
    secret_names = {item["name"]: item["value"] for item in bodies[0]["secrets"]}
    assert secret_names["APP_SETTING"] == REPO_SETTING
    assert set(
        request.url.path.split("/")[3]
        for request in world.requests
        if request.method in MUTATING_METHODS
        and request.url.path.startswith("/v2/deployments/")
    ) == {EXISTING_ID}
    listed = [
        request.url.params.get("name_contains")
        for request in world.requests
        if request.method == "GET" and request.url.path == "/v2/deployments"
    ]
    assert listed[0] == EXISTING_NAME
    _assert_api_key(world)


def test_explicit_name_remote_update_uses_that_deployment(
    world: _World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_project(tmp_path, source="dotenv", deployment_name=OTHER_NAME)
    monkeypatch.chdir(tmp_path)

    result = _invoke(
        world,
        tmp_path,
        _deploy_args("--no-input", "--name", EXISTING_NAME, "--remote"),
    )

    assert result.exit_code == 0, result.output
    bodies = _patch_bodies(world, EXISTING_ID)
    assert len(bodies) == 1, world.mutations()
    assert bodies[0]["revision_source"] == "internal_source"
    assert (
        bodies[0]["source_revision_config"]["source_tarball_path"]
        == "objects/source.tar.gz"
    )
    secret_names = {item["name"]: item["value"] for item in bodies[0]["secrets"]}
    assert secret_names["APP_SETTING"] == REPO_SETTING
    assert world.upload.uploads == ["/upload"]
    _assert_api_key(world)


def test_explicit_deployment_id_is_not_redirected_by_project_name(
    world: _World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_project(tmp_path, source="dotenv", deployment_name=EXISTING_NAME)
    monkeypatch.chdir(tmp_path)

    result = _invoke(
        world,
        tmp_path,
        _deploy_args(
            "--no-input", "--deployment-id", AUTHORIZED_ID, "--image", "local/app:test"
        ),
    )

    assert result.exit_code == 0, result.output
    assert len(_patch_bodies(world, AUTHORIZED_ID)) == 1
    assert _patch_bodies(world, EXISTING_ID) == []
    mutated_ids = {
        request.url.path.split("/")[3]
        for request in world.requests
        if request.method in MUTATING_METHODS
        and request.url.path.startswith("/v2/deployments/")
    }
    assert mutated_ids == {AUTHORIZED_ID}
    _assert_api_key(world)


def test_process_environment_name_wins_over_project_file(
    world: _World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_project(tmp_path, source="dotenv", deployment_name=OTHER_NAME)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LANGSMITH_DEPLOYMENT_NAME", EXISTING_NAME)

    result = _invoke(
        world,
        tmp_path,
        _deploy_args("--no-input", "--image", "local/app:test"),
    )

    assert result.exit_code == 0, result.output
    assert len(_patch_bodies(world, EXISTING_ID)) == 1, world.mutations()
    assert _patch_bodies(world, OTHER_ID) == []
    _assert_api_key(world)


def test_deploy_logs_does_not_follow_project_dotenv_name(
    world: _World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_project(tmp_path, source="dotenv", deployment_name=EXISTING_NAME)
    monkeypatch.chdir(tmp_path)

    result = _invoke(
        world,
        tmp_path,
        [
            "deploy",
            "logs",
            "--api-key",
            OPERATOR_API_KEY,
            "--host-url",
            HOST_URL,
        ],
        stdin="",
    )

    log_calls = [
        request.url.path
        for request in world.requests
        if request.url.path.endswith("/deploy_logs")
        or request.url.path.endswith("/build_logs")
    ]
    assert log_calls == [], (
        "deploy logs read a deployment selected by the project env file\n"
        f"paths={log_calls}\noutput={result.output}"
    )
    assert result.exit_code != 0
    assert world.files() == world.before
