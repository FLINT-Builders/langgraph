"""Pip config files must not be stored in exported image layers.

`pip_config_file` is copied into the image for pip and uv installs. These tests
build that Dockerfile (with the base image swapped so the install commands do
not need the LangGraph API image) and read every exported layer. A later `rm`
of the file still fails the test: the copy remains in the earlier layer.
"""

from __future__ import annotations

import copy
import gzip
import io
import os
import pathlib
import re
import subprocess
import tarfile
import textwrap
import uuid
from collections.abc import Sequence
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from langgraph_cli.cli import cli, prepare_args_and_stdin
from langgraph_cli.config import (
    config_to_compose,
    config_to_docker,
    validate_config,
    validate_config_file,
)
from langgraph_cli.docker import (
    DockerCapabilities,
    Version,
    build_docker_image,
)
from langgraph_cli.exec import Runner

SENTINEL = b"LANGGRAPH_PIP_CONFIG_SENTINEL"
SENTINEL_TEXT = SENTINEL.decode("ascii")
BUSYBOX = "busybox:1.36.1"
PIP_CONFIG_NAME = "pip.conf"

_FROM_RE = re.compile(
    r"^FROM\s+(?:--\S+\s+)*(\S+)(?:\s+[Aa][Ss]\s+(\S+))?\s*$",
)
_SECRET_ID_RE = re.compile(
    r"--mount=type=secret(?:=|,)(?:[^ \t]*?)\bid=([A-Za-z0-9_.-]+)"
)
_DOCKER: list[str] | None = None


def _docker() -> list[str]:
    global _DOCKER
    if _DOCKER is not None:
        return _DOCKER
    for prefix in (["docker"], ["sudo", "docker"]):
        probe = subprocess.run(
            [*prefix, "info"],
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode == 0:
            _DOCKER = prefix
            return prefix
    pytest.fail("docker is required to inspect exported image layers")


def _run(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture(scope="module", autouse=True)
def _busybox_present() -> None:
    docker = _docker()
    inspect = _run([*docker, "image", "inspect", BUSYBOX])
    if inspect.returncode == 0:
        return
    pull = _run([*docker, "pull", BUSYBOX])
    if pull.returncode != 0:
        pytest.fail(pull.stderr or pull.stdout)


def _write_pip_config(path: pathlib.Path) -> None:
    path.write_text(
        "[global]\n"
        "index-url = https://example.invalid/simple\n"
        f"marker = {SENTINEL_TEXT}\n",
        encoding="utf-8",
    )


def _write_real_package(root: pathlib.Path) -> None:
    package = root / "pkg"
    package.mkdir()
    (package / "graph.py").write_text("graph = None\n", encoding="utf-8")
    (package / "pyproject.toml").write_text(
        textwrap.dedent(
            """\
            [project]
            name = "pkg"
            version = "0.1.0"

            [build-system]
            requires = ["setuptools>=61"]
            build-backend = "setuptools.build_meta"
            """
        ),
        encoding="utf-8",
    )


def _sibling_config(root: pathlib.Path, *, pip_installer: str) -> dict:
    _write_pip_config(root / PIP_CONFIG_NAME)
    _write_real_package(root)
    (root / "langgraph.json").write_text("{}", encoding="utf-8")
    return validate_config(
        {
            "python_version": "3.11",
            "dependencies": ["./pkg", "example-private-pkg"],
            "graphs": {"agent": "./pkg/graph.py:graph"},
            "pip_config_file": PIP_CONFIG_NAME,
            "pip_installer": pip_installer,
        }
    )


def _project_root_config(root: pathlib.Path) -> dict:
    _write_pip_config(root / PIP_CONFIG_NAME)
    (root / "graph.py").write_text("graph = None\n", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        textwrap.dedent(
            """\
            [project]
            name = "app"
            version = "0.1.0"

            [build-system]
            requires = ["setuptools>=61"]
            build-backend = "setuptools.build_meta"
            """
        ),
        encoding="utf-8",
    )
    (root / "langgraph.json").write_text("{}", encoding="utf-8")
    return validate_config(
        {
            "python_version": "3.11",
            "dependencies": ["."],
            "graphs": {"agent": "./graph.py:graph"},
            "pip_config_file": PIP_CONFIG_NAME,
            "pip_installer": "pip",
        }
    )


def _requirements_config(root: pathlib.Path) -> dict:
    _write_pip_config(root / PIP_CONFIG_NAME)
    package = root / "pkg" / "appcode"
    package.mkdir(parents=True)
    (package / "graph.py").write_text("graph = None\n", encoding="utf-8")
    (root / "pkg" / "requirements.txt").write_text(
        "example-private-pkg\n",
        encoding="utf-8",
    )
    (root / "langgraph.json").write_text("{}", encoding="utf-8")
    return validate_config(
        {
            "python_version": "3.11",
            "dependencies": ["./pkg"],
            "graphs": {"agent": "./pkg/appcode/graph.py:graph"},
            "pip_config_file": PIP_CONFIG_NAME,
            "pip_installer": "pip",
        }
    )


def _write_uv_lock_workspace(tmpdir: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    project_root = tmpdir / "workspace"
    config_dir = project_root / "apps" / "agent"
    shared_dir = project_root / "libs" / "shared"
    deploy_dir = project_root / "deploy" / "agent"
    for path in (config_dir, shared_dir, deploy_dir):
        path.mkdir(parents=True)
    (project_root / "uv.lock").write_text("# uv lock file\n", encoding="utf-8")
    (project_root / "pyproject.toml").write_text(
        textwrap.dedent(
            """\
            [project]
            name = "workspace-root"
            version = "0.1.0"

            [tool.uv.workspace]
            members = ["apps/*", "libs/*"]

            [tool.uv.sources]
            shared = { workspace = true }

            [build-system]
            requires = ["setuptools>=61"]
            build-backend = "setuptools.build_meta"
            """
        ),
        encoding="utf-8",
    )
    (config_dir / "pyproject.toml").write_text(
        textwrap.dedent(
            """\
            [project]
            name = "agent"
            version = "0.1.0"
            dependencies = ["shared"]

            [tool.uv.sources]
            shared = { workspace = true }

            [build-system]
            requires = ["setuptools>=61"]
            build-backend = "setuptools.build_meta"
            """
        ),
        encoding="utf-8",
    )
    (shared_dir / "pyproject.toml").write_text(
        textwrap.dedent(
            """\
            [project]
            name = "shared"
            version = "0.1.0"

            [build-system]
            requires = ["setuptools>=61"]
            build-backend = "setuptools.build_meta"
            """
        ),
        encoding="utf-8",
    )
    graph_dir = config_dir / "src" / "agent"
    graph_dir.mkdir(parents=True)
    (graph_dir / "graph.py").write_text("graph = None\n", encoding="utf-8")
    (shared_dir / "src" / "shared").mkdir(parents=True)
    (shared_dir / "src" / "shared" / "__init__.py").write_text("", encoding="utf-8")
    config_path = deploy_dir / "langgraph.json"
    config_path.write_text("{}", encoding="utf-8")
    return project_root, config_path


def _uv_lock_config() -> dict:
    return validate_config(
        {
            "python_version": "3.11",
            "graphs": {"agent": "../../apps/agent/src/agent/graph.py:graph"},
            "source": {"kind": "uv", "root": "../..", "package": "agent"},
            "pip_config_file": PIP_CONFIG_NAME,
        }
    )


def _logical_lines(dockerfile: str) -> list[str]:
    logical: list[str] = []
    buffer = ""
    for line in dockerfile.splitlines():
        if buffer:
            buffer = buffer + " " + line.strip()
        elif not line.strip():
            continue
        else:
            buffer = line
        if buffer.rstrip().endswith("\\"):
            buffer = buffer.rstrip()[:-1]
            continue
        logical.append(buffer.strip())
        buffer = ""
    if buffer.strip():
        logical.append(buffer.strip())
    return logical


def _deletion_runs(run_line: str) -> list[str]:
    body = run_line.strip()[3:].strip()
    while body.startswith("--"):
        parts = body.split(None, 1)
        body = parts[1].strip() if len(parts) > 1 else ""
    names = (PIP_CONFIG_NAME, "pipconfig.txt", "pipconfig")
    kept: list[str] = []
    for segment in re.split(r"\s*(?:&&|\|\||;)\s*", body):
        command = segment.strip()
        if command.startswith("rm ") and any(name in command for name in names):
            kept.append("RUN " + command)
    return kept


def _project_dockerfile(dockerfile: str) -> str:
    """Keep copy and pip-config deletion instructions; swap the base image.

    Install commands from the generated Dockerfile need the LangGraph API image.
    Dropping them does not drop `ADD`/`COPY` layers, and a deletion of the pip
    config is kept so a later `rm` is not treated as removing earlier layers.
    """

    projected: list[str] = []
    for line in _logical_lines(dockerfile):
        if line.startswith("#"):
            if line.startswith("# syntax="):
                projected.append(line)
            continue
        head = line.split(None, 1)[0].upper()
        if head == "FROM":
            match = _FROM_RE.match(line)
            if match is None:
                raise AssertionError(f"unparsed FROM instruction: {line}")
            stage = f" AS {match.group(2)}" if match.group(2) else ""
            projected.append(f"FROM {BUSYBOX}{stage}")
        elif head in {"ADD", "COPY", "WORKDIR"}:
            projected.append(line)
        elif head == "RUN":
            projected.extend(_deletion_runs(line))
    if not any(line.startswith("FROM ") for line in projected):
        raise AssertionError("generated Dockerfile has no FROM instruction")
    return "\n".join(projected) + "\n"


def _secret_ids(dockerfile: str) -> list[str]:
    return list(dict.fromkeys(_SECRET_ID_RE.findall(dockerfile)))


def _build_args_from_argv(argv: Sequence[str]) -> list[str]:
    flags: list[str] = []
    index = 0
    items = list(argv)
    while index < len(items):
        item = items[index]
        if item in {"--build-context", "--secret", "--build-arg"}:
            flags.extend([item, items[index + 1]])
            index += 2
            continue
        if item.startswith(("--build-context=", "--secret=", "--build-arg=")):
            flags.append(item)
        index += 1
    return flags


def _assert_build_passes_pip_secret(
    argv: Sequence[str], dockerfile: str, pip_config: pathlib.Path
) -> None:
    secret_ids = _secret_ids(dockerfile)
    if not secret_ids:
        return
    passed: dict[str, str] = {}
    items = list(argv)
    index = 0
    while index < len(items):
        item = items[index]
        spec = None
        if item == "--secret":
            spec = items[index + 1]
            index += 2
        elif item.startswith("--secret="):
            spec = item.split("=", 1)[1]
            index += 1
        else:
            index += 1
            continue
        fields = {}
        for part in spec.split(","):
            if "=" in part:
                key, value = part.split("=", 1)
                fields[key] = value
        if "id" in fields and "src" in fields:
            passed[fields["id"]] = fields["src"]
    for secret_id in secret_ids:
        assert secret_id in passed, (
            "Dockerfile mounts a BuildKit secret but the image build does not "
            f"pass --secret for {secret_id}"
        )
        assert pathlib.Path(passed[secret_id]).resolve() == pip_config.resolve()


def _auto_secret_args(dockerfile: str, pip_config: pathlib.Path) -> list[str]:
    args: list[str] = []
    for secret_id in _secret_ids(dockerfile):
        args.extend(["--secret", f"id={secret_id},src={pip_config}"])
    return args


def _layer_hits(image_tar: pathlib.Path) -> list[str]:
    hits: list[str] = []
    with tarfile.open(image_tar) as outer:
        for member in outer.getmembers():
            if not member.isfile():
                continue
            payload = outer.extractfile(member).read()
            blobs = [payload]
            if payload[:2] == b"\x1f\x8b":
                blobs.append(gzip.decompress(payload))
            for blob in blobs:
                if len(blob) > 262 and blob[257:262] == b"ustar":
                    layer = tarfile.open(fileobj=io.BytesIO(blob))
                    for entry in layer.getmembers():
                        if not entry.isfile():
                            continue
                        content = layer.extractfile(entry).read()
                        if SENTINEL in content:
                            hits.append(f"{member.name}:{entry.name}")
                elif SENTINEL in blob and blob is not payload:
                    hits.append(member.name)
                elif (
                    SENTINEL in payload
                    and blob is payload
                    and payload[:2] != b"\x1f\x8b"
                ):
                    hits.append(member.name)
    return hits


def _assert_layers_omit_pip_config(
    dockerfile: str,
    context: pathlib.Path,
    pip_config: pathlib.Path,
    *,
    additional_contexts: dict[str, str] | None = None,
    extra_build_args: Sequence[str] = (),
    supply_missing_secrets: bool = True,
) -> None:
    projected = _project_dockerfile(dockerfile)
    dockerfile_path = context / "Dockerfile.pip-config-witness"
    dockerfile_path.write_text(projected, encoding="utf-8")
    tag = f"langgraph-pip-config-test:{uuid.uuid4().hex}"
    build_args = [
        *_docker(),
        "build",
        "-f",
        str(dockerfile_path),
        "-t",
        tag,
    ]
    for name, path in (additional_contexts or {}).items():
        build_args.extend(["--build-context", f"{name}={path}"])
    build_args.extend(extra_build_args)
    if supply_missing_secrets:
        already = " ".join(extra_build_args)
        for arg in _auto_secret_args(dockerfile, pip_config):
            if arg not in extra_build_args and arg not in already:
                build_args.append(arg)
    build_args.append(str(context))
    image_tar = context / "image.tar"
    try:
        built = _run(build_args)
        if built.returncode != 0:
            pytest.fail(
                "witness image build failed\n"
                f"dockerfile:\n{projected}\n"
                f"stdout:\n{built.stdout}\n"
                f"stderr:\n{built.stderr}"
            )
        saved = _run([*_docker(), "save", "-o", str(image_tar), tag])
        if saved.returncode != 0:
            pytest.fail(saved.stderr or saved.stdout)
        if not os.access(image_tar, os.R_OK):
            chmod = _run(["sudo", "chmod", "644", str(image_tar)])
            if chmod.returncode != 0:
                pytest.fail(chmod.stderr or "could not read saved image")
        hits = _layer_hits(image_tar)
        assert hits == [], (
            "pip config bytes were stored in exported image layers: " + ", ".join(hits)
        )
    finally:
        _run([*_docker(), "rmi", "-f", tag])


def _config_path(root: pathlib.Path) -> pathlib.Path:
    return root / "langgraph.json"


def _generate(root: pathlib.Path, config: dict, base_image: str):
    return config_to_docker(
        _config_path(root),
        config,
        base_image=base_image,
    )


@pytest.mark.parametrize(
    ("pip_installer", "base_image"),
    [
        ("pip", "langchain/langgraph-api:0.2.0"),
        ("uv", "langchain/langgraph-api:0.2.47"),
    ],
)
def test_package_install_omits_pip_config_from_image_layers(
    tmp_path: pathlib.Path, pip_installer: str, base_image: str
) -> None:
    config = _sibling_config(tmp_path, pip_installer=pip_installer)
    dockerfile, contexts = _generate(tmp_path, config, base_image)
    _assert_layers_omit_pip_config(
        dockerfile,
        tmp_path,
        tmp_path / PIP_CONFIG_NAME,
        additional_contexts=contexts,
    )


def test_local_requirements_install_omits_pip_config_from_image_layers(
    tmp_path: pathlib.Path,
) -> None:
    config = _requirements_config(tmp_path)
    dockerfile, contexts = _generate(tmp_path, config, "langchain/langgraph-api:0.2.0")
    _assert_layers_omit_pip_config(
        dockerfile,
        tmp_path,
        tmp_path / PIP_CONFIG_NAME,
        additional_contexts=contexts,
    )


def test_project_directory_dependency_omits_pip_config_from_image_layers(
    tmp_path: pathlib.Path,
) -> None:
    config = _project_root_config(tmp_path)
    dockerfile, contexts = _generate(tmp_path, config, "langchain/langgraph-api:0.2.0")
    _assert_layers_omit_pip_config(
        dockerfile,
        tmp_path,
        tmp_path / PIP_CONFIG_NAME,
        additional_contexts=contexts,
    )


def test_uv_lock_omits_pip_config_from_image_layers(tmp_path: pathlib.Path) -> None:
    _project_root, config_path = _write_uv_lock_workspace(tmp_path)
    _write_pip_config(config_path.parent / PIP_CONFIG_NAME)
    dockerfile, contexts = config_to_docker(
        config_path,
        _uv_lock_config(),
        base_image="langchain/langgraph-api:0.2.47",
    )
    _assert_layers_omit_pip_config(
        dockerfile,
        config_path.parent,
        config_path.parent / PIP_CONFIG_NAME,
        additional_contexts=contexts,
    )


def test_uv_lock_package_directory_omits_pip_config_from_image_layers(
    tmp_path: pathlib.Path,
) -> None:
    project_root, _deploy_config = _write_uv_lock_workspace(tmp_path)
    config_path = project_root / "apps" / "agent" / "langgraph.json"
    config_path.write_text("{}", encoding="utf-8")
    _write_pip_config(config_path.parent / PIP_CONFIG_NAME)
    config = validate_config(
        {
            "python_version": "3.11",
            "graphs": {"agent": "src/agent/graph.py:graph"},
            "source": {"kind": "uv", "root": "../..", "package": "agent"},
            "pip_config_file": PIP_CONFIG_NAME,
        }
    )
    dockerfile, contexts = config_to_docker(
        config_path,
        config,
        base_image="langchain/langgraph-api:0.2.47",
    )
    _assert_layers_omit_pip_config(
        dockerfile,
        config_path.parent,
        config_path.parent / PIP_CONFIG_NAME,
        additional_contexts=contexts,
    )


def _inline_dockerfiles(compose: str) -> list[str]:
    lines = compose.splitlines()
    dockerfiles: list[str] = []
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if stripped.startswith("dockerfile_inline:") and stripped.endswith("|"):
            key_indent = len(lines[index]) - len(lines[index].lstrip(" "))
            index += 1
            block: list[str] = []
            content_indent: int | None = None
            while index < len(lines):
                line = lines[index]
                if line.strip() == "":
                    block.append("")
                    index += 1
                    continue
                indent = len(line) - len(line.lstrip(" "))
                if indent <= key_indent:
                    break
                if content_indent is None:
                    content_indent = indent
                block.append(line[content_indent:])
                index += 1
            dockerfiles.append("\n".join(block))
            continue
        index += 1
    return dockerfiles


def test_compose_and_up_stdin_omit_pip_config_from_image_layers(
    tmp_path: pathlib.Path,
) -> None:
    config = _project_root_config(tmp_path)
    config_path = _config_path(tmp_path)
    pip_config = tmp_path / PIP_CONFIG_NAME
    for mode in ("combined_queue_worker", "distributed"):
        compose = config_to_compose(
            config_path,
            copy.deepcopy(config),
            base_image="langchain/langgraph-api",
            engine_runtime_mode=mode,
        )
        dockerfiles = _inline_dockerfiles(compose)
        assert dockerfiles, f"{mode} compose did not inline a Dockerfile"
        for dockerfile in dockerfiles:
            _assert_layers_omit_pip_config(dockerfile, tmp_path, pip_config)

    args, stdin = prepare_args_and_stdin(
        capabilities=DockerCapabilities(
            version_docker=Version(29, 1, 3),
            version_compose=Version(2, 27, 0),
            healthcheck_start_interval=True,
        ),
        config_path=config_path,
        config=copy.deepcopy(config),
        docker_compose=None,
        port=8123,
        watch=False,
    )
    assert args
    for dockerfile in _inline_dockerfiles(stdin):
        _assert_layers_omit_pip_config(dockerfile, tmp_path, pip_config)


def test_dockerfile_command_omits_pip_config_from_image_layers(
    tmp_path: pathlib.Path,
) -> None:
    config = _project_root_config(tmp_path)
    config_path = _config_path(tmp_path)
    config_path.write_text(
        textwrap.dedent(
            """\
            {
              "python_version": "3.11",
              "dependencies": ["."],
              "graphs": {"agent": "./graph.py:graph"},
              "pip_config_file": "pip.conf",
              "pip_installer": "pip"
            }
            """
        ),
        encoding="utf-8",
    )
    save_path = tmp_path / "Dockerfile"
    result = CliRunner().invoke(
        cli,
        ["dockerfile", str(save_path), "--config", str(config_path)],
    )
    assert result.exit_code == 0, result.output
    dockerfile = save_path.read_text(encoding="utf-8")
    _assert_layers_omit_pip_config(
        dockerfile,
        tmp_path,
        tmp_path / PIP_CONFIG_NAME,
    )
    # The validated dict was not used after writing the file; keep the
    # generator and the command on the same project.
    assert config["pip_config_file"] == PIP_CONFIG_NAME


def test_build_command_omits_pip_config_from_image_layers(
    tmp_path: pathlib.Path,
) -> None:
    _project_root_config(tmp_path)
    config_path = _config_path(tmp_path)
    config_path.write_text(
        textwrap.dedent(
            """\
            {
              "python_version": "3.11",
              "dependencies": ["."],
              "graphs": {"agent": "./graph.py:graph"},
              "pip_config_file": "pip.conf",
              "pip_installer": "uv"
            }
            """
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    async def fake_subp_exec(cmd, *args, input=None, **kwargs):
        captured["argv"] = (cmd, *args)
        captured["input"] = input
        return None, None

    with patch("langgraph_cli.docker.subp_exec", fake_subp_exec):
        result = CliRunner().invoke(
            cli,
            [
                "build",
                "--tag",
                "langgraph-pip-config-test",
                "--no-pull",
                "--config",
                str(config_path),
            ],
        )
    assert result.exit_code == 0, result.output
    dockerfile = str(captured["input"])
    argv = tuple(captured["argv"])
    _assert_build_passes_pip_secret(
        argv, dockerfile, config_path.parent / PIP_CONFIG_NAME
    )
    _assert_layers_omit_pip_config(
        dockerfile,
        config_path.parent,
        config_path.parent / PIP_CONFIG_NAME,
        extra_build_args=_build_args_from_argv(argv),
        supply_missing_secrets=False,
    )


def test_deploy_local_build_omits_pip_config_from_image_layers(
    tmp_path: pathlib.Path,
) -> None:
    """`langgraph deploy` builds through `build_docker_image` before it pushes."""

    _sibling_config(tmp_path, pip_installer="uv")
    config_path = _config_path(tmp_path)
    config_path.write_text(
        textwrap.dedent(
            """\
            {
              "python_version": "3.11",
              "dependencies": ["./pkg", "example-private-pkg"],
              "graphs": {"agent": "./pkg/graph.py:graph"},
              "pip_config_file": "pip.conf",
              "pip_installer": "uv"
            }
            """
        ),
        encoding="utf-8",
    )
    config_json = validate_config_file(config_path)
    captured: dict[str, object] = {}

    async def fake_subp_exec(cmd, *args, input=None, **kwargs):
        captured["argv"] = (cmd, *args)
        captured["input"] = input
        return None, None

    with patch("langgraph_cli.docker.subp_exec", fake_subp_exec):
        with Runner() as runner:
            build_docker_image(
                runner,
                lambda _message: None,
                config_path,
                config_json,
                "langchain/langgraph-api:0.2.47",
                None,
                False,
                "langgraph-deploy-tmp:test",
                (),
                None,
                None,
                docker_command=("docker", "buildx", "build"),
                extra_flags=("--platform", "linux/amd64", "--load"),
                verbose=False,
            )
    dockerfile = str(captured["input"])
    argv = tuple(captured["argv"])
    assert argv[:3] == ("docker", "buildx", "build")
    _assert_build_passes_pip_secret(argv, dockerfile, tmp_path / PIP_CONFIG_NAME)
    _assert_layers_omit_pip_config(
        dockerfile,
        tmp_path,
        tmp_path / PIP_CONFIG_NAME,
        extra_build_args=_build_args_from_argv(argv),
        supply_missing_secrets=False,
    )
