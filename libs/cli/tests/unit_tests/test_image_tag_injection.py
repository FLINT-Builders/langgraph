"""Dockerfile instruction injection through image fields in langgraph.json.

`python_version`, `base_image`, `_INTERNAL_docker_tag`, and `pip_config_file`
are copied into the generated Dockerfile. A newline in those fields becomes
another Dockerfile instruction, which `docker build` and `docker compose`
execute as root while building the image `langgraph build` and `langgraph up`
use.

These tests are the expected behavior. They fail while the fields are copied
through unchanged, including when a hyphen suffix makes the Python version
check look only at the major.minor prefix and the full string is still written
into the Dockerfile.
"""

import copy
import json
import textwrap
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from langgraph_cli.cli import cli
from langgraph_cli.config import (
    config_to_compose,
    config_to_docker,
    validate_config_file,
)
from langgraph_cli.docker import DockerCapabilities, Version

INJECTED = "INJECTED_DOCKERFILE_INSTRUCTION"
PY_VERSION_PAYLOAD = f"3.11-x\nRUN echo {INJECTED}"
PY_VERSION_CRLF_PAYLOAD = f"3.11-x\r\nRUN echo {INJECTED}"
BASE_IMAGE_PAYLOAD = (
    f"alpine:3.19\nRUN echo {INJECTED}\nRUN --mount=type=bind,source=.,target=/ctx "
    "cat /ctx/.env\nRUN true"
)
INTERNAL_TAG_PAYLOAD = (
    f"3.19\nRUN echo {INJECTED}\nRUN --mount=type=bind,source=.,target=/ctx "
    "cat /ctx/.env"
)
PIP_CONFIG_PAYLOAD = (
    f"marker.txt /tmp/pipdest\nRUN echo {INJECTED}\n"
    "RUN --mount=type=bind,source=.,target=/ctx cat /ctx/.env\nADD marker.txt"
)

_CAPS = DockerCapabilities(
    version_docker=Version(29, 1, 3),
    version_compose=Version(2, 40, 3),
    healthcheck_start_interval=True,
    compose_type="plugin",
)


def _write_config(directory: Path, config: dict, name: str = "langgraph.json") -> Path:
    path = directory / name
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _python_project(directory: Path, config: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "agent.py").write_text("graph = None\n", encoding="utf-8")
    config = {
        "dependencies": ["."],
        "graphs": {"agent": "agent.py:graph"},
        "pip_installer": "pip",
        **config,
    }
    return _write_config(directory, config)


def _node_project(directory: Path, config: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "agent.js").write_text(
        "export const graph = null;\n", encoding="utf-8"
    )
    config = {
        "node_version": "20",
        "graphs": {"agent": "./agent.js:graph"},
        **config,
    }
    return _write_config(directory, config)


def _uv_project(directory: Path, config: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "uv.lock").write_text("# uv lock file\n", encoding="utf-8")
    (directory / "pyproject.toml").write_text(
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
        + "\n",
        encoding="utf-8",
    )
    source = directory / "src"
    source.mkdir()
    (source / "agent.py").write_text("graph = object()\n", encoding="utf-8")
    config = {
        "graphs": {"agent": "./src/agent.py:graph"},
        "source": {"kind": "uv"},
        **config,
    }
    return _write_config(directory, config)


def _assert_config_rejected(config_path: Path) -> None:
    """Run the validate-then-generate sequence used by build, up, and deploy.

    Rejection has to happen on the value that is written into the Dockerfile.
    A check that accepts the major.minor prefix and still emits the original
    string fails this test. So does generation that succeeds and returns the
    injected instruction.
    """
    try:
        validated = validate_config_file(config_path)
        # Each generator is a separate use of the validated value. Copying keeps
        # one generator from rewriting the value the next one reads.
        dockerfile, _contexts = config_to_docker(config_path, copy.deepcopy(validated))
        compose = config_to_compose(config_path, copy.deepcopy(validated))
        distributed = config_to_compose(
            config_path,
            copy.deepcopy(validated),
            engine_runtime_mode="distributed",
        )
    except click.UsageError:
        return
    rendered = "\n".join((dockerfile, compose, distributed))
    assert INJECTED in rendered
    pytest.fail("image fields were accepted and copied into the build:\n" + dockerfile)


@pytest.mark.parametrize(
    "config",
    [
        {"python_version": PY_VERSION_PAYLOAD},
        {"python_version": PY_VERSION_CRLF_PAYLOAD},
        {"base_image": BASE_IMAGE_PAYLOAD},
        {
            "base_image": "alpine",
            "_INTERNAL_docker_tag": INTERNAL_TAG_PAYLOAD,
        },
        {"pip_config_file": PIP_CONFIG_PAYLOAD},
    ],
)
def test_python_build_rejects_image_field_injection(
    tmp_path: Path, config: dict
) -> None:
    _assert_config_rejected(_python_project(tmp_path, config))


def test_node_build_rejects_internal_tag_injection(tmp_path: Path) -> None:
    config_path = _node_project(
        tmp_path,
        {"base_image": "alpine", "_INTERNAL_docker_tag": INTERNAL_TAG_PAYLOAD},
    )
    _assert_config_rejected(config_path)


def test_node_build_rejects_base_image_injection(tmp_path: Path) -> None:
    _assert_config_rejected(_node_project(tmp_path, {"base_image": BASE_IMAGE_PAYLOAD}))


@pytest.mark.parametrize(
    "config",
    [
        {"python_version": PY_VERSION_PAYLOAD},
        {"base_image": BASE_IMAGE_PAYLOAD},
        {
            "python_version": "3.11",
            "base_image": "alpine",
            "_INTERNAL_docker_tag": INTERNAL_TAG_PAYLOAD,
        },
        {"python_version": "3.11", "pip_config_file": PIP_CONFIG_PAYLOAD},
    ],
)
def test_uv_build_rejects_image_field_injection(tmp_path: Path, config: dict) -> None:
    _assert_config_rejected(_uv_project(tmp_path, config))


def test_dockerfile_command_does_not_write_injected_file(tmp_path: Path) -> None:
    config_path = _python_project(tmp_path, {"python_version": PY_VERSION_PAYLOAD})
    save_path = tmp_path / "Dockerfile"
    result = CliRunner().invoke(
        cli,
        ["dockerfile", str(save_path), "--config", str(config_path)],
    )
    assert isinstance(result.exception, click.UsageError)
    assert result.exit_code != 0
    assert not save_path.exists()
    assert not any(path.name == "Dockerfile" for path in tmp_path.iterdir())


@pytest.mark.parametrize(
    "extra_config",
    [
        {"base_image": BASE_IMAGE_PAYLOAD},
        {"base_image": "alpine", "_INTERNAL_docker_tag": INTERNAL_TAG_PAYLOAD},
        {"pip_config_file": PIP_CONFIG_PAYLOAD},
    ],
)
def test_dockerfile_command_does_not_write_other_injected_fields(
    tmp_path: Path, extra_config: dict
) -> None:
    config_path = _python_project(tmp_path, {"python_version": "3.11", **extra_config})
    save_path = tmp_path / "Dockerfile"
    result = CliRunner().invoke(
        cli,
        ["dockerfile", str(save_path), "--config", str(config_path)],
    )
    assert isinstance(result.exception, click.UsageError)
    assert result.exit_code != 0
    assert not save_path.exists()


def test_build_command_rejects_before_docker_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _python_project(tmp_path, {"python_version": PY_VERSION_PAYLOAD})
    called = {"build": False}

    def fail_build(*_args, **_kwargs):
        called["build"] = True
        raise AssertionError("docker build started before the config was rejected")

    monkeypatch.setattr(
        "langgraph_cli.cli.shutil.which", lambda _name: "/usr/bin/docker"
    )
    monkeypatch.setattr("langgraph_cli.cli.build_docker_image", fail_build)

    result = CliRunner().invoke(
        cli,
        ["build", "--no-pull", "--tag", "injected-test", "--config", str(config_path)],
    )
    assert called["build"] is False
    assert isinstance(result.exception, click.UsageError)
    assert result.exit_code != 0


def test_up_command_rejects_before_compose_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _python_project(
        tmp_path,
        {
            "python_version": "3.11",
            "base_image": "alpine",
            "_INTERNAL_docker_tag": INTERNAL_TAG_PAYLOAD,
        },
    )
    called = {"subprocess": False}

    async def fail_exec(*_args, **_kwargs):
        called["subprocess"] = True
        raise AssertionError("docker compose started before the config was rejected")

    monkeypatch.setattr(
        "langgraph_cli.docker.check_capabilities",
        lambda _runner: _CAPS,
    )
    monkeypatch.setattr("langgraph_cli.exec.asyncio.create_subprocess_exec", fail_exec)

    result = CliRunner().invoke(
        cli,
        ["up", "--no-pull", "--config", str(config_path), "--port", "8123"],
    )
    assert called["subprocess"] is False
    assert isinstance(result.exception, click.UsageError)
    assert result.exit_code != 0
    # Compose generation must not leave a Dockerfile beside the project.
    assert not (tmp_path / "Dockerfile").exists()


def test_local_deploy_build_uses_the_same_rejected_config(tmp_path: Path) -> None:
    """Local deploy calls the same generator as `langgraph build` after validation."""
    config_path = _python_project(tmp_path, {"pip_config_file": PIP_CONFIG_PAYLOAD})
    _assert_config_rejected(config_path)


def test_rejection_leaves_no_dockerfile_beside_the_project(tmp_path: Path) -> None:
    project = tmp_path / "project"
    before = set(project.glob("**/*")) if project.exists() else set()
    config_path = _python_project(project, {"python_version": PY_VERSION_PAYLOAD})
    with pytest.raises(click.UsageError):
        validate_config_file(config_path)
    after = set(project.iterdir())
    assert after == {config_path, project / "agent.py"}
    assert before == set() or config_path in after
    assert not (project / "Dockerfile").exists()
