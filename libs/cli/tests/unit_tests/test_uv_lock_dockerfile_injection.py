"""Package names from pyproject.toml must not become extra Dockerfile instructions.

`source.kind: uv` reads `[project].name` and places it in a shell-form `RUN uv
export --package ...` line. TOML basic strings can decode a newline into that
name. BuildKit's Dockerfile parser splits instructions on newlines and does not
treat quotes as line-continuations, so the newline starts a new instruction.
A following `FROM` becomes the default (last) stage, and its `RUN` is what
`docker build` executes.

These tests fail while that happens. They pass when generation rejects the name
before writing anything, or when the rendered Dockerfile does not contain the
injected instruction as its own parsed command. Callers of `config_to_docker`
(`langgraph build`, `langgraph dockerfile`, `langgraph deploy`) and
`config_to_compose` (`langgraph up`, including the distributed executor
Dockerfile) all go through the same rendering.
"""

from __future__ import annotations

import pathlib
import re
import tempfile

import click
import pytest

from langgraph_cli.config import config_to_compose, config_to_docker, validate_config

# A name that closes the shlex single-quote sequence, starts a new stage, and
# comments out the rest of the original `uv export` line. The marker command is
# not a substring of any legitimate instruction this generator emits.
_INJECTED_FROM = "FROM busybox"
_INJECTED_RUN = "RUN echo INJECTED_STAGE_RAN"
_MALICIOUS_NAME_LITERAL = "legit'\\nFROM busybox\\nRUN echo INJECTED_STAGE_RAN\\n#"
_BASE_IMAGE = "langchain/langgraph-api:0.2.47"

# BuildKit v0.35.1 frontend/dockerfile/parser/parser.go: a line continues only
# when it ends in an unescaped escape token. Quotes are not consulted.
_CONTINUATION = re.compile(r"([^\\])\\[ \t]*$|^\\[ \t]*$")


def _dockerfile_instructions(dockerfile: str) -> list[str]:
    instructions: list[str] = []
    buffer = ""
    continuing = False
    for raw_line in dockerfile.split("\n"):
        line = raw_line.rstrip("\r")
        if not continuing:
            stripped = line.lstrip()
            if not stripped or stripped.startswith("#"):
                continue
            line = stripped
        if match := _CONTINUATION.search(line):
            buffer += line[: match.start()] + (match.group(1) or "")
            continuing = True
            continue
        buffer += line
        instructions.append(buffer)
        buffer = ""
        continuing = False
    if buffer:
        instructions.append(buffer)
    return instructions


def _assert_name_cannot_inject(text: str) -> None:
    instructions = _dockerfile_instructions(text)
    assert _INJECTED_FROM not in instructions
    assert _INJECTED_RUN not in instructions
    assert not any(
        instruction.startswith("RUN ") and "INJECTED_STAGE_RAN" in instruction
        for instruction in instructions
    )


def _project_files(root: pathlib.Path, *, name_literal: str) -> None:
    (root / "uv.lock").write_text("# uv lock file\n")
    (root / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project]",
                f'name = "{name_literal}"',
                'version = "0.1.0"',
                "dependencies = []",
                "",
                "[build-system]",
                'requires = ["setuptools>=61"]',
                'build-backend = "setuptools.build_meta"',
                "",
            ]
        )
    )
    (root / "src").mkdir()
    (root / "src" / "agent.py").write_text("graph = object()\n")


def _uv_config(config_path: pathlib.Path, *, source: dict) -> dict:
    return validate_config(
        {
            "python_version": "3.11",
            "graphs": {"agent": "./src/agent.py:graph"},
            "source": source,
        }
    )


def _render(config_path: pathlib.Path, config: dict) -> str:
    dockerfile, _contexts = config_to_docker(
        config_path,
        config,
        base_image=_BASE_IMAGE,
    )
    return dockerfile


def _assert_rejected_or_safe(config_path: pathlib.Path, config: dict) -> None:
    project_root = config_path.parent
    before = {path.relative_to(project_root) for path in project_root.rglob("*")}
    try:
        dockerfile = _render(config_path, config)
    except click.UsageError:
        after = {path.relative_to(project_root) for path in project_root.rglob("*")}
        assert after == before
        return
    _assert_name_cannot_inject(dockerfile)


def test_newline_in_root_package_name_cannot_inject_dockerfile_instruction():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = pathlib.Path(tmpdir) / "single"
        root.mkdir()
        _project_files(root, name_literal=_MALICIOUS_NAME_LITERAL)
        config_path = root / "langgraph.json"
        config_path.write_text("{}\n")
        config = _uv_config(config_path, source={"kind": "uv"})
        _assert_rejected_or_safe(config_path, config)


def test_newline_in_workspace_member_name_cannot_inject_dockerfile_instruction():
    """The emitted name is the selected member, not the workspace root."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = pathlib.Path(tmpdir) / "workspace"
        member = root / "apps" / "agent"
        member.mkdir(parents=True)
        (root / "uv.lock").write_text("# uv lock file\n")
        (root / "pyproject.toml").write_text(
            "\n".join(
                [
                    "[project]",
                    'name = "workspace-root"',
                    'version = "0.1.0"',
                    "",
                    "[tool.uv.workspace]",
                    'members = ["apps/*"]',
                    "",
                    "[build-system]",
                    'requires = ["setuptools>=61"]',
                    'build-backend = "setuptools.build_meta"',
                    "",
                ]
            )
        )
        _project_files(member, name_literal=_MALICIOUS_NAME_LITERAL)
        config_path = member / "langgraph.json"
        config_path.write_text("{}\n")
        config = _uv_config(
            config_path,
            source={"kind": "uv", "root": "../.."},
        )
        _assert_rejected_or_safe(config_path, config)


def test_newline_in_package_name_cannot_inject_compose_inline_dockerfile():
    """`langgraph up` embeds the same Dockerfile in compose's dockerfile_inline."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = pathlib.Path(tmpdir) / "single"
        root.mkdir()
        _project_files(root, name_literal=_MALICIOUS_NAME_LITERAL)
        config_path = root / "langgraph.json"
        config_path.write_text("{}\n")
        config = _uv_config(config_path, source={"kind": "uv"})
        project_root = config_path.parent
        before = {path.relative_to(project_root) for path in project_root.rglob("*")}
        try:
            compose = config_to_compose(
                config_path,
                config,
                base_image=_BASE_IMAGE,
            )
        except click.UsageError:
            after = {path.relative_to(project_root) for path in project_root.rglob("*")}
            assert after == before
            return
        _assert_name_cannot_inject(compose)
        try:
            distributed = config_to_compose(
                config_path,
                config,
                base_image=_BASE_IMAGE,
                engine_runtime_mode="distributed",
            )
        except click.UsageError:
            after = {path.relative_to(project_root) for path in project_root.rglob("*")}
            assert after == before
            return
        _assert_name_cannot_inject(distributed)


def test_ordinary_uv_package_name_is_still_exported():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = pathlib.Path(tmpdir) / "single"
        root.mkdir()
        _project_files(root, name_literal="single-app")
        config_path = root / "langgraph.json"
        config_path.write_text("{}\n")
        config = _uv_config(config_path, source={"kind": "uv"})
        dockerfile = _render(config_path, config)
        assert (
            "uv export --package single-app --frozen --no-hashes "
            "--no-emit-project --no-emit-workspace" in dockerfile
        )


@pytest.mark.parametrize(
    "literal",
    [
        "legit\\nRUN echo INJECTED_STAGE_RAN\\n#",
        "legit\\r\\nRUN echo INJECTED_STAGE_RAN\\n#",
    ],
)
def test_newline_forms_cannot_start_a_run_instruction(literal: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        root = pathlib.Path(tmpdir) / "single"
        root.mkdir()
        _project_files(root, name_literal=literal)
        config_path = root / "langgraph.json"
        config_path.write_text("{}\n")
        config = _uv_config(config_path, source={"kind": "uv"})
        _assert_rejected_or_safe(config_path, config)
