"""Env files in a project must not be copied into a LangGraph image or deploy archive.

`langgraph build` and `langgraph up` generate a Dockerfile that ADDs each local
dependency directory, and pass that directory to Docker as the build context.
Docker includes every file the context's ignore rules do not exclude. The CLI
writes a `.dockerignore` that excludes `.env` and `.env.*` only for
`langgraph dockerfile --add-docker-compose`. These tests build the image
filesystem from the Dockerfile and context the CLI actually produces.

A file that is copied and then deleted by a later RUN is still present: the
oracle records every ADD/COPY and does not run RUN. Refusal before Docker is
invoked is also accepted. A successful image must still contain the graph
source.

The oracle implements the context rules of Docker CLI v29 / BuildKit v0.35.1 /
compose-go v2.9.1, using patternmatcher v0.6.1 (the version BuildKit v0.35.1
vendors) for ignore matching.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tarfile
import textwrap
from pathlib import Path

import click
import click.exceptions
import pytest
from click.testing import CliRunner

from langgraph_cli.archive import create_archive
from langgraph_cli.cli import cli, prepare_args_and_stdin
from langgraph_cli.config import validate_config_file
from langgraph_cli.docker import DockerCapabilities, Version, build_docker_image
from langgraph_cli.exec import Runner

ORACLE_DIR = Path(__file__).parent / "image_context_oracle"

DOTENV = b"placeholder-dotenv\n"
DOTENV_LOCAL = b"placeholder-dotenv-local\n"
KEYS_ENV = b"placeholder-keys\n"
SHARED_DOTENV = b"placeholder-shared\n"
GRAPH_SOURCE = b"GRAPH_SOURCE_MARKER\n"
SHARED_SOURCE = b"SHARED_SOURCE_MARKER\n"

_VALUE_FLAGS = {
    "-f",
    "--file",
    "-t",
    "--tag",
    "--target",
    "--platform",
    "--iidfile",
    "--network",
    "--label",
    "--ulimit",
    "--add-host",
    "--cache-from",
    "--secret",
    "--ssh",
    "--output",
    "--progress",
    "--builder",
    "-m",
    "--memory",
    "--memory-swap",
    "--shm-size",
    "--cgroup-parent",
    "--cpuset-cpus",
    "--cpuset-mems",
    "--cpu-shares",
    "--cpu-period",
    "--cpu-quota",
    "--build-arg",
    "--build-context",
    "-o",
    "--opt",
}

_REFUSALS = (
    click.ClickException,
    click.UsageError,
    click.exceptions.Exit,
    SystemExit,
)


@pytest.fixture(scope="session")
def oracle_bin(tmp_path_factory: pytest.TempPathFactory) -> Path:
    dest = tmp_path_factory.mktemp("oracle-bin") / "image-context-oracle"
    subprocess.run(
        ["go", "build", "-mod=vendor", "-o", str(dest), "."],
        cwd=ORACLE_DIR,
        check=True,
    )
    return dest


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _copied_files(
    oracle_bin: Path,
    *,
    dockerfile: str,
    dockerfile_path: str,
    context: str,
    named_contexts: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    payload = {
        "dockerfile": dockerfile,
        "dockerfile_path": dockerfile_path,
        "context": context,
        "named_contexts": named_contexts or {},
    }
    proc = subprocess.run(
        [str(oracle_bin)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=False,
    )
    data = json.loads(proc.stdout or "{}")
    if proc.returncode != 0 or data.get("error"):
        raise AssertionError(data.get("error") or proc.stderr or "oracle failed")
    return data["files"]


def _assert_image(
    files: list[dict[str, str]],
    *,
    required: set[bytes],
    forbidden: set[bytes],
) -> None:
    found = {item["sha256"] for item in files}
    leaked = [
        item["path"]
        for item in files
        if item["sha256"] in {_digest(secret) for secret in forbidden}
    ]
    assert not leaked, f"env file contents were copied into the image at {leaked}"
    missing = {_digest(blob) for blob in required} - found
    assert not missing, "application source was not copied into the image"


def _write_env_tree(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "secrets").mkdir()
    (root / ".env").write_bytes(DOTENV)
    (root / ".env.local").write_bytes(DOTENV_LOCAL)
    (root / "secrets" / "keys.env").write_bytes(KEYS_ENV)
    (root / "graph.py").write_bytes(GRAPH_SOURCE)


def _python_config(env: str = "secrets/keys.env") -> dict:
    return {
        "python_version": "3.11",
        "dependencies": ["."],
        "graphs": {"agent": "./graph.py:graph"},
        "env": env,
    }


def _write_real_package(root: Path, env: str = "secrets/keys.env") -> Path:
    _write_env_tree(root)
    (root / "pyproject.toml").write_text(
        "[project]\nname = 'myagent'\nversion = '0.1.0'\n",
        encoding="utf-8",
    )
    config_path = root / "langgraph.json"
    config_path.write_text(json.dumps(_python_config(env)), encoding="utf-8")
    return config_path


def _secret_blobs() -> set[bytes]:
    return {DOTENV, DOTENV_LOCAL, KEYS_ENV}


def _is_docker_build(cmd: str, args: tuple[str, ...]) -> bool:
    if cmd != "docker":
        return False
    seq = list(args)
    if seq and seq[0] == "buildx":
        seq = seq[1:]
    return bool(seq) and seq[0] == "build"


def _parse_docker_build(
    args: tuple[str, ...], stdin: str | None
) -> tuple[str, str, str, dict[str, str]]:
    seq = list(args)
    if seq and seq[0] == "buildx":
        seq = seq[1:]
    if seq and seq[0] == "build":
        seq = seq[1:]
    dockerfile_path = "-"
    named: dict[str, str] = {}
    positionals: list[str] = []
    index = 0
    while index < len(seq):
        token = seq[index]
        if token in _VALUE_FLAGS:
            if index + 1 >= len(seq):
                raise AssertionError(f"docker build flag {token} is missing a value")
            value = seq[index + 1]
            if token in {"-f", "--file"}:
                dockerfile_path = value
            elif token == "--build-context":
                name, path = value.split("=", 1)
                named[name] = path
            index += 2
            continue
        if token.startswith("--") and "=" in token:
            key, value = token.split("=", 1)
            if key == "--file":
                dockerfile_path = value
            elif key == "--build-context":
                name, path = value.split("=", 1)
                named[name] = path
            index += 1
            continue
        if token.startswith("-") and token != "-":
            index += 1
            continue
        positionals.append(token)
        index += 1
    if not positionals:
        raise AssertionError(f"docker build has no context argument: {args}")
    context = positionals[-1]
    if dockerfile_path in {"", "-"}:
        if stdin is None:
            raise AssertionError(
                "docker build read the Dockerfile from stdin but no stdin was passed"
            )
        return stdin, "-", context, named
    return (
        Path(dockerfile_path).read_text(encoding="utf-8"),
        dockerfile_path,
        context,
        named,
    )


def _run_build(
    monkeypatch: pytest.MonkeyPatch,
    config_path: Path,
    *,
    install_command: str | None = None,
    build_command: str | None = None,
) -> list[tuple[str, str, str, dict[str, str]]]:
    captured: list[tuple[str, tuple[str, ...], str | None]] = []

    async def fake_subp_exec(
        cmd: str, *args: str, input: str | None = None, **kwargs: object
    ):
        captured.append((cmd, args, input))
        return None, None

    monkeypatch.setattr("langgraph_cli.docker.subp_exec", fake_subp_exec)
    config_json = validate_config_file(config_path)
    refused = False
    with Runner() as runner:
        try:
            build_docker_image(
                runner,
                lambda _message: None,
                config_path,
                config_json,
                None,
                None,
                False,
                "local/agent:test",
                install_command=install_command,
                build_command=build_command,
            )
        except _REFUSALS:
            refused = True
    builds = [
        _parse_docker_build(args, stdin)
        for cmd, args, stdin in captured
        if _is_docker_build(cmd, args)
    ]
    if not builds:
        assert refused, "build neither invoked docker nor refused"
    return builds


def _assert_builds(
    oracle_bin: Path,
    builds: list[tuple[str, str, str, dict[str, str]]],
    *,
    required: set[bytes],
    forbidden: set[bytes],
) -> None:
    # An empty list means _run_build already required the CLI to refuse
    # before invoking Docker.
    if not builds:
        return
    for dockerfile, dockerfile_path, context, named in builds:
        files = _copied_files(
            oracle_bin,
            dockerfile=dockerfile,
            dockerfile_path=dockerfile_path,
            context=context,
            named_contexts=named,
        )
        _assert_image(files, required=required, forbidden=forbidden)


def _extract_compose_builds(
    compose_text: str, project_dir: Path
) -> list[tuple[str, str, dict[str, str]]]:
    lines = compose_text.splitlines()
    builds: list[tuple[str, str, dict[str, str]]] = []
    index = 0
    while index < len(lines):
        if lines[index].strip() != "dockerfile_inline: |":
            index += 1
            continue
        marker_indent = len(lines[index]) - len(lines[index].lstrip(" "))
        context = str(project_dir)
        named: dict[str, str] = {}
        for earlier in reversed(lines[:index]):
            stripped = earlier.strip()
            if stripped.startswith("context:"):
                raw = stripped.split(":", 1)[1].strip()
                context = raw if os.path.isabs(raw) else str(project_dir / raw)
                break
            if stripped.startswith("- ") and ":" in stripped:
                body = stripped[2:]
                name, path = body.split(":", 1)
                named[name.strip()] = path.strip()
        index += 1
        body: list[str] = []
        while index < len(lines):
            line = lines[index]
            if line.strip() and (len(line) - len(line.lstrip(" "))) <= marker_indent:
                break
            body.append(line)
            index += 1
        nonempty = [line for line in body if line.strip()]
        indent = min((len(line) - len(line.lstrip(" "))) for line in nonempty)
        dockerfile = "\n".join(line[indent:] if line.strip() else "" for line in body)
        builds.append((dockerfile, context, named))
    return builds


def test_oracle_includes_env_file_when_no_dockerignore(
    oracle_bin: Path, tmp_path: Path
) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    (root / ".env").write_bytes(DOTENV)
    (root / "graph.py").write_bytes(GRAPH_SOURCE)
    files = _copied_files(
        oracle_bin,
        dockerfile="FROM scratch\nADD . /deps/proj\n",
        dockerfile_path="-",
        context=str(root),
    )
    _assert_image(files, required={GRAPH_SOURCE}, forbidden=set())
    assert _digest(DOTENV) in {item["sha256"] for item in files}


def test_oracle_honors_context_dockerignore(oracle_bin: Path, tmp_path: Path) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    (root / ".env").write_bytes(DOTENV)
    (root / ".env.local").write_bytes(DOTENV_LOCAL)
    (root / "graph.py").write_bytes(GRAPH_SOURCE)
    (root / ".dockerignore").write_text(".env\n.env.*\n", encoding="utf-8")
    files = _copied_files(
        oracle_bin,
        dockerfile="FROM scratch\nADD . /deps/proj\n",
        dockerfile_path="-",
        context=str(root),
    )
    _assert_image(files, required={GRAPH_SOURCE}, forbidden={DOTENV, DOTENV_LOCAL})


def test_oracle_stdin_dockerfile_ignores_sidecar_in_the_context(
    oracle_bin: Path, tmp_path: Path
) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    (root / ".env").write_bytes(DOTENV)
    (root / "graph.py").write_bytes(GRAPH_SOURCE)
    (root / "Dockerfile.dockerignore").write_text(".env\n", encoding="utf-8")
    files = _copied_files(
        oracle_bin,
        dockerfile="FROM scratch\nADD . /deps/proj\n",
        dockerfile_path="-",
        context=str(root),
    )
    assert _digest(DOTENV) in {item["sha256"] for item in files}

    dockerfile_path = root / "Dockerfile"
    dockerfile_path.write_text("FROM scratch\nADD . /deps/proj\n", encoding="utf-8")
    files = _copied_files(
        oracle_bin,
        dockerfile=dockerfile_path.read_text(encoding="utf-8"),
        dockerfile_path=str(dockerfile_path),
        context=str(root),
    )
    _assert_image(files, required={GRAPH_SOURCE}, forbidden={DOTENV})


def test_build_python_package_copies_env_files(
    oracle_bin: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = _write_real_package(tmp_path / "myagent")
    builds = _run_build(monkeypatch, config_path)
    _assert_builds(
        oracle_bin,
        builds,
        required={GRAPH_SOURCE},
        forbidden=_secret_blobs(),
    )


def test_build_python_faux_package_copies_env_files(
    oracle_bin: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "myagent"
    _write_env_tree(root)
    (root / "__init__.py").write_text("", encoding="utf-8")
    config_path = root / "langgraph.json"
    config_path.write_text(json.dumps(_python_config()), encoding="utf-8")
    builds = _run_build(monkeypatch, config_path)
    _assert_builds(
        oracle_bin,
        builds,
        required={GRAPH_SOURCE},
        forbidden=_secret_blobs(),
    )


def test_build_gitignore_does_not_keep_env_files_out_of_the_image(
    oracle_bin: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "myagent"
    config_path = _write_real_package(root)
    (root / ".gitignore").write_text(
        ".env\n.env.*\nsecrets/keys.env\n", encoding="utf-8"
    )
    builds = _run_build(monkeypatch, config_path)
    _assert_builds(
        oracle_bin,
        builds,
        required={GRAPH_SOURCE},
        forbidden=_secret_blobs(),
    )


def test_build_unrelated_dockerignore_does_not_keep_env_files_out(
    oracle_bin: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "myagent"
    config_path = _write_real_package(root)
    (root / ".dockerignore").write_text("*.log\n__pycache__/\n", encoding="utf-8")
    builds = _run_build(monkeypatch, config_path)
    _assert_builds(
        oracle_bin,
        builds,
        required={GRAPH_SOURCE},
        forbidden=_secret_blobs(),
    )


def test_build_stdin_dockerfile_ignores_sidecar_dockerignore(
    oracle_bin: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "myagent"
    config_path = _write_real_package(root)
    (root / "Dockerfile.dockerignore").write_text(
        ".env\n.env.*\nsecrets/keys.env\n",
        encoding="utf-8",
    )
    builds = _run_build(monkeypatch, config_path)
    _assert_builds(
        oracle_bin,
        builds,
        required={GRAPH_SOURCE},
        forbidden=_secret_blobs(),
    )


def test_build_user_dockerignore_that_lists_env_files_excludes_them(
    oracle_bin: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "myagent"
    config_path = _write_real_package(root)
    (root / ".dockerignore").write_text(
        ".env\n.env.*\nsecrets/keys.env\n",
        encoding="utf-8",
    )
    builds = _run_build(monkeypatch, config_path)
    _assert_builds(
        oracle_bin,
        builds,
        required={GRAPH_SOURCE},
        forbidden=_secret_blobs(),
    )


def test_build_node_project_copies_env_files(
    oracle_bin: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "myagent"
    _write_env_tree(root)
    source = root / "src"
    source.mkdir()
    (source / "agent.ts").write_bytes(GRAPH_SOURCE)
    config_path = root / "langgraph.json"
    config_path.write_text(
        json.dumps(
            {
                "node_version": "20",
                "dependencies": ["."],
                "graphs": {"agent": "./src/agent.ts:graph"},
                "env": "secrets/keys.env",
            }
        ),
        encoding="utf-8",
    )
    builds = _run_build(monkeypatch, config_path)
    _assert_builds(
        oracle_bin,
        builds,
        required={GRAPH_SOURCE},
        forbidden=_secret_blobs(),
    )


def test_build_node_install_command_uses_cwd_and_copies_env_files(
    oracle_bin: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "myagent"
    _write_env_tree(root)
    (root / "src").mkdir()
    (root / "src" / "agent.ts").write_bytes(GRAPH_SOURCE)
    config_path = root / "langgraph.json"
    config_path.write_text(
        json.dumps(
            {
                "node_version": "20",
                "dependencies": ["."],
                "graphs": {"agent": "./src/agent.ts:graph"},
                "env": "secrets/keys.env",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(root)
    builds = _run_build(monkeypatch, config_path, install_command="npm ci")
    _assert_builds(
        oracle_bin,
        builds,
        required={GRAPH_SOURCE},
        forbidden=_secret_blobs(),
    )


def test_build_additional_context_copies_env_file(
    oracle_bin: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "myagent"
    shared = tmp_path / "sharedlib"
    config_path = _write_real_package(root)
    shared.mkdir()
    (shared / "pyproject.toml").write_text(
        "[project]\nname = 'sharedlib'\nversion = '0.1.0'\n",
        encoding="utf-8",
    )
    (shared / "lib.py").write_bytes(SHARED_SOURCE)
    (shared / ".env").write_bytes(SHARED_DOTENV)
    config = _python_config()
    config["dependencies"] = [".", "../sharedlib"]
    config_path.write_text(json.dumps(config), encoding="utf-8")
    builds = _run_build(monkeypatch, config_path)
    _assert_builds(
        oracle_bin,
        builds,
        required={GRAPH_SOURCE, SHARED_SOURCE},
        forbidden=_secret_blobs() | {SHARED_DOTENV},
    )


def test_build_uv_lock_copies_env_files(
    oracle_bin: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "myagent"
    _write_env_tree(root)
    (root / "uv.lock").write_text("# uv lock file\n", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        textwrap.dedent(
            """
            [project]
            name = "myagent"
            version = "0.1.0"
            dependencies = ["httpx>=0.28"]

            [build-system]
            requires = ["setuptools>=61"]
            build-backend = "setuptools.build_meta"
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    config_path = root / "langgraph.json"
    config_path.write_text(
        json.dumps(
            {
                "python_version": "3.11",
                "graphs": {"agent": "./graph.py:graph"},
                "env": "secrets/keys.env",
                "source": {"kind": "uv"},
            }
        ),
        encoding="utf-8",
    )
    builds = _run_build(monkeypatch, config_path)
    _assert_builds(
        oracle_bin,
        builds,
        required={GRAPH_SOURCE},
        forbidden=_secret_blobs(),
    )


def _assert_up(
    oracle_bin: Path,
    project_dir: Path,
    config_path: Path,
    *,
    engine_runtime_mode: str,
    required: set[bytes],
    forbidden: set[bytes],
) -> None:
    config = validate_config_file(config_path)
    capabilities = DockerCapabilities(
        version_docker=Version(26, 1, 1),
        version_compose=Version(2, 27, 0),
        healthcheck_start_interval=False,
    )
    try:
        _args, stdin = prepare_args_and_stdin(
            capabilities=capabilities,
            config_path=config_path,
            config=config,
            docker_compose=None,
            port=8123,
            watch=False,
            engine_runtime_mode=engine_runtime_mode,
        )
    except _REFUSALS:
        return
    builds = _extract_compose_builds(stdin, project_dir)
    assert builds, "langgraph up did not emit a build"
    for dockerfile, context, named in builds:
        # Compose passes the Dockerfile inline, so BuildKit stores it in a
        # temporary directory. A sidecar next to that temporary file is not
        # the project's Dockerfile.dockerignore.
        files = _copied_files(
            oracle_bin,
            dockerfile=dockerfile,
            dockerfile_path="-",
            context=context,
            named_contexts=named,
        )
        _assert_image(files, required=required, forbidden=forbidden)


def test_up_copies_env_files(oracle_bin: Path, tmp_path: Path) -> None:
    root = tmp_path / "myagent"
    config_path = _write_real_package(root)
    _assert_up(
        oracle_bin,
        root,
        config_path,
        engine_runtime_mode="combined_queue_worker",
        required={GRAPH_SOURCE},
        forbidden=_secret_blobs(),
    )


def test_up_distributed_executor_copies_env_files(
    oracle_bin: Path, tmp_path: Path
) -> None:
    root = tmp_path / "myagent"
    config_path = _write_real_package(root)
    _assert_up(
        oracle_bin,
        root,
        config_path,
        engine_runtime_mode="distributed",
        required={GRAPH_SOURCE},
        forbidden=_secret_blobs(),
    )


def test_dockerfile_command_copies_env_files(oracle_bin: Path, tmp_path: Path) -> None:
    root = tmp_path / "myagent"
    config_path = _write_real_package(root)
    dockerfile_path = root / "Dockerfile"
    result = CliRunner().invoke(
        cli,
        ["dockerfile", str(dockerfile_path), "--config", str(config_path)],
    )
    if not dockerfile_path.exists():
        assert result.exit_code != 0, result.output
        return
    files = _copied_files(
        oracle_bin,
        dockerfile=dockerfile_path.read_text(encoding="utf-8"),
        dockerfile_path=str(dockerfile_path),
        context=str(root),
    )
    _assert_image(files, required={GRAPH_SOURCE}, forbidden=_secret_blobs())


def test_dockerfile_add_docker_compose_excludes_default_env_names(
    oracle_bin: Path, tmp_path: Path
) -> None:
    root = tmp_path / "myagent"
    _write_env_tree(root)
    (root / "pyproject.toml").write_text(
        "[project]\nname = 'myagent'\nversion = '0.1.0'\n",
        encoding="utf-8",
    )
    config_path = root / "langgraph.json"
    config_path.write_text(json.dumps(_python_config(".env")), encoding="utf-8")
    dockerfile_path = root / "Dockerfile"
    result = CliRunner().invoke(
        cli,
        [
            "dockerfile",
            str(dockerfile_path),
            "--config",
            str(config_path),
            "--add-docker-compose",
        ],
    )
    if not dockerfile_path.exists():
        assert result.exit_code != 0, result.output
        return
    files = _copied_files(
        oracle_bin,
        dockerfile=dockerfile_path.read_text(encoding="utf-8"),
        dockerfile_path=str(dockerfile_path),
        context=str(root),
    )
    _assert_image(files, required={GRAPH_SOURCE}, forbidden={DOTENV, DOTENV_LOCAL})


def test_dockerfile_add_docker_compose_still_copies_configured_env_path(
    oracle_bin: Path, tmp_path: Path
) -> None:
    root = tmp_path / "myagent"
    config_path = _write_real_package(root)
    dockerfile_path = root / "Dockerfile"
    result = CliRunner().invoke(
        cli,
        [
            "dockerfile",
            str(dockerfile_path),
            "--config",
            str(config_path),
            "--add-docker-compose",
        ],
    )
    if not dockerfile_path.exists():
        assert result.exit_code != 0, result.output
        return
    files = _copied_files(
        oracle_bin,
        dockerfile=dockerfile_path.read_text(encoding="utf-8"),
        dockerfile_path=str(dockerfile_path),
        context=str(root),
    )
    _assert_image(files, required={GRAPH_SOURCE}, forbidden={KEYS_ENV})


def test_deploy_source_archive_copies_env_files(tmp_path: Path) -> None:
    root = tmp_path / "myagent"
    config_path = _write_real_package(root)
    config = validate_config_file(config_path)
    try:
        with create_archive(config_path, config) as (archive_path, _size, _rel):
            with tarfile.open(archive_path, "r:gz") as tar:
                payloads = [member for member in tar.getmembers() if member.isfile()]
                found: set[str] = set()
                leaked: list[str] = []
                forbidden = {_digest(blob) for blob in _secret_blobs()}
                for member in payloads:
                    extracted = tar.extractfile(member)
                    assert extracted is not None
                    digest = _digest(extracted.read())
                    found.add(digest)
                    if digest in forbidden:
                        leaked.append(member.name)
    except _REFUSALS:
        return
    assert not leaked, (
        f"env file contents were packed into the deploy archive at {leaked}"
    )
    assert _digest(GRAPH_SOURCE) in found, (
        "application source was not packed into the deploy archive"
    )
