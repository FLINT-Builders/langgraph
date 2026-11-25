"""Dockerfile instructions must not be smuggled in through dependency paths.

`langgraph build` and `langgraph up` turn local dependency paths and directory
names into Dockerfile comments, ADD, COPY, and WORKDIR lines. A newline in
that text is a new Dockerfile line. These tests fail while that text is
emitted, and pass once generation rejects the path or emits a single
instruction that does not start with the injected command.
"""

import json
import pathlib
from unittest.mock import patch

import click

from langgraph_cli.config import config_to_compose, config_to_docker, validate_config
from langgraph_cli.docker import build_docker_image
from langgraph_cli.exec import Runner

INJECTED_RUN = "RUN echo INJECTED_MARKER"
BASE_IMAGE = "example.invalid/langgraph-base:1"


def _payload(prefix: str) -> str:
    return f"{prefix}\n{INJECTED_RUN}\n#"


def _assert_no_injected_instruction(text: str) -> None:
    """A Dockerfile line whose leading whitespace is stripped is one instruction."""
    for line in text.splitlines():
        assert not line.lstrip().startswith(INJECTED_RUN), line


def _accepts_rejection(exc: BaseException) -> bool:
    # Missing-path errors count only when generation itself raises them.
    # FileExistsError is deliberately excluded so a fixture mistake cannot pass.
    return isinstance(
        exc, (ValueError, FileNotFoundError, NotADirectoryError, click.ClickException)
    )


def _generated_text(produced: object) -> str:
    if isinstance(produced, tuple):
        return produced[0]
    return produced


def _require_safe(produce) -> None:
    try:
        produced = produce()
    except (
        ValueError,
        FileNotFoundError,
        NotADirectoryError,
        click.ClickException,
    ) as exc:
        if _accepts_rejection(exc):
            return
        raise
    _assert_no_injected_instruction(_generated_text(produced))


def _python_config(dependencies: list[str]) -> dict:
    return {
        "dependencies": dependencies,
        "graphs": {"agent": "agent:graph"},
        "python_version": "3.11",
        "dockerfile_lines": [],
    }


def _write_real_package(directory: pathlib.Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "pyproject.toml").write_text(
        '[project]\nname = "localpkg"\nversion = "0.1"\n'
    )
    (directory / "app.py").write_text("graph = None\n")


def _config_path(directory: pathlib.Path, config: dict) -> pathlib.Path:
    path = directory / "langgraph.json"
    path.write_text(json.dumps(config))
    return path


def test_dependency_string_with_newline_is_not_a_dockerfile_instruction(tmp_path):
    """The path listed in dependencies is copied into the comment and ADD line."""
    dep_name = _payload("pkg")
    package = tmp_path / dep_name
    _write_real_package(package)
    config_path = _config_path(tmp_path, _python_config(["./" + dep_name]))

    def produce():
        return config_to_docker(
            config_path,
            validate_config(json.loads(config_path.read_text())),
            base_image=BASE_IMAGE,
        )

    _require_safe(produce)


def test_project_directory_name_with_newline_is_not_a_dockerfile_instruction(tmp_path):
    """dependencies: ["."] still interpolates the resolved directory name."""
    project = tmp_path / _payload("proj")
    _write_real_package(project)
    config_path = _config_path(project, _python_config(["."]))

    def produce():
        return config_to_docker(
            config_path,
            validate_config(json.loads(config_path.read_text())),
            base_image=BASE_IMAGE,
        )

    _require_safe(produce)


def test_symlink_target_directory_name_is_not_a_dockerfile_instruction(tmp_path):
    """The config path can be clean while resolve().name carries the newline."""
    target = tmp_path / _payload("evil")
    _write_real_package(target)
    (tmp_path / "safe").symlink_to(target, target_is_directory=True)
    config_path = _config_path(tmp_path, _python_config(["./safe"]))

    def produce():
        return config_to_docker(
            config_path,
            validate_config(json.loads(config_path.read_text())),
            base_image=BASE_IMAGE,
        )

    _require_safe(produce)


def test_faux_package_directory_name_is_not_a_dockerfile_instruction(tmp_path):
    project = tmp_path / _payload("faux")
    project.mkdir()
    (project / "app.py").write_text("graph = None\n")
    config_path = _config_path(project, _python_config(["."]))

    def produce():
        return config_to_docker(
            config_path,
            validate_config(json.loads(config_path.read_text())),
            base_image=BASE_IMAGE,
        )

    _require_safe(produce)


def test_requirements_path_with_newline_is_not_a_dockerfile_instruction(tmp_path):
    package = tmp_path / _payload("reqs")
    package.mkdir()
    (package / "requirements.txt").write_text("anyio\n")
    (package / "app.py").write_text("graph = None\n")
    config_path = _config_path(tmp_path, _python_config(["./" + package.name]))

    def produce():
        return config_to_docker(
            config_path,
            validate_config(json.loads(config_path.read_text())),
            base_image=BASE_IMAGE,
        )

    _require_safe(produce)


def test_parent_dependency_context_name_is_not_a_dockerfile_instruction(tmp_path):
    outer = tmp_path / _payload("outer")
    _write_real_package(outer)
    inner = outer / "inner"
    inner.mkdir()
    config_path = _config_path(inner, _python_config([".."]))

    def produce():
        return config_to_docker(
            config_path,
            validate_config(json.loads(config_path.read_text())),
            base_image=BASE_IMAGE,
        )

    _require_safe(produce)


def test_non_local_dependency_with_newline_is_not_a_dockerfile_instruction(tmp_path):
    _write_real_package(tmp_path)
    config_path = _config_path(tmp_path, _python_config([".", _payload("requests")]))

    def produce():
        return config_to_docker(
            config_path,
            validate_config(json.loads(config_path.read_text())),
            base_image=BASE_IMAGE,
        )

    _require_safe(produce)


def test_pip_config_file_with_newline_is_not_a_dockerfile_instruction(tmp_path):
    _write_real_package(tmp_path)
    (tmp_path / "pip.conf").write_text("[global]\n")
    config = _python_config(["."])
    config["pip_config_file"] = "pip.conf /pipconfig.txt\n" + INJECTED_RUN + "\n#"
    config_path = _config_path(tmp_path, config)

    def produce():
        return config_to_docker(
            config_path,
            validate_config(json.loads(config_path.read_text())),
            base_image=BASE_IMAGE,
        )

    _require_safe(produce)


def test_node_directory_name_is_not_a_dockerfile_instruction(tmp_path):
    project = tmp_path / _payload("nodeproj")
    project.mkdir()
    (project / "package.json").write_text('{"name": "demo"}\n')
    (project / "agent.js").write_text("export const graph = {};\n")
    config = {
        "node_version": "20",
        "graphs": {"agent": "./agent.js:graph"},
        "dockerfile_lines": [],
    }
    config_path = _config_path(project, config)

    def produce_without_context():
        return config_to_docker(
            config_path,
            validate_config(json.loads(config_path.read_text())),
            base_image=BASE_IMAGE,
        )

    def produce_with_build_context():
        return config_to_docker(
            config_path,
            validate_config(json.loads(config_path.read_text())),
            base_image=BASE_IMAGE,
            build_context=str(project),
        )

    _require_safe(produce_without_context)
    _require_safe(produce_with_build_context)


def test_uv_workspace_member_path_is_not_a_dockerfile_instruction(tmp_path):
    member_name = _payload("pkg")
    member = tmp_path / member_name
    member.mkdir()
    (tmp_path / "uv.lock").write_text("# lock\n")
    (tmp_path / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project]",
                'name = "workspace-root"',
                'version = "0.1.0"',
                "[tool.uv.workspace]",
                f"members = {json.dumps([member_name])}",
                "[build-system]",
                'requires = ["setuptools>=61"]',
                'build-backend = "setuptools.build_meta"',
                "",
            ]
        )
    )
    (member / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project]",
                'name = "member"',
                'version = "0.1.0"',
                "[build-system]",
                'requires = ["setuptools>=61"]',
                'build-backend = "setuptools.build_meta"',
                "",
            ]
        )
    )
    (member / "app.py").write_text("graph = None\n")
    config = {
        "python_version": "3.11",
        "dependencies": [],
        "source": {"kind": "uv", "root": "..", "package": "member"},
        "graphs": {"agent": "./app.py:graph"},
        "dockerfile_lines": [],
    }
    config_path = _config_path(member, config)

    def produce():
        return config_to_docker(
            config_path,
            validate_config(json.loads(config_path.read_text())),
            base_image=BASE_IMAGE,
        )

    _require_safe(produce)


def test_compose_for_up_does_not_embed_injected_instruction(tmp_path):
    """langgraph up inlines this Dockerfile in the compose file it sends to Docker."""
    project = tmp_path / _payload("proj")
    _write_real_package(project)
    config_path = _config_path(project, _python_config(["."]))

    def produce():
        return config_to_compose(
            config_path,
            validate_config(json.loads(config_path.read_text())),
            base_image=BASE_IMAGE,
        )

    _require_safe(produce)


def test_build_docker_image_does_not_send_injected_instruction(tmp_path):
    """langgraph build and local deploy both call build_docker_image."""
    project = tmp_path / _payload("proj")
    _write_real_package(project)
    config_path = _config_path(project, _python_config(["."]))
    config = validate_config(json.loads(config_path.read_text()))
    captured: list[str] = []

    async def fake_exec(*_args, input=None, **_kwargs):
        captured.append(input or "")
        return None, None

    with patch("langgraph_cli.docker.subp_exec", fake_exec):
        try:
            with Runner() as runner:
                build_docker_image(
                    runner,
                    lambda _message: None,
                    config_path,
                    config,
                    BASE_IMAGE,
                    None,
                    False,
                    "example.invalid/app:test",
                    verbose=False,
                )
        except (
            ValueError,
            FileNotFoundError,
            NotADirectoryError,
            click.ClickException,
        ) as exc:
            if not _accepts_rejection(exc):
                raise
            assert captured == []
            return

    assert captured, "build_docker_image did not invoke the docker build"
    for stdin in captured:
        _assert_no_injected_instruction(stdin)


def test_clean_local_dependency_still_generates_add(tmp_path):
    """A path without Dockerfile metacharacters still produces an image recipe."""
    package = tmp_path / "pkg"
    _write_real_package(package)
    config_path = _config_path(tmp_path, _python_config(["./pkg"]))
    dockerfile, _contexts = config_to_docker(
        config_path,
        validate_config(json.loads(config_path.read_text())),
        base_image=BASE_IMAGE,
    )
    assert "ADD ./pkg /deps/pkg" in dockerfile
    _assert_no_injected_instruction(dockerfile)
