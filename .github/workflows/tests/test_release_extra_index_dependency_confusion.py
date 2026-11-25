#!/usr/bin/env python3
"""Release pre-checks must not run code from an extra package index.

The pre-release-checks job installs the package it is about to publish with
`uv run pip install --extra-index-url https://test.pypi.org/simple/`. Pip
collects every index it was given and installs the highest matching version.
A dependency with only a lower bound, the same shape as `xxhash>=3.5.0` in
`libs/langgraph/pyproject.toml`, can therefore be satisfied by a higher
version that exists only on the extra index. Publishing that version is
unauthenticated. Its build backend runs on the release runner.

These tests execute the install scripts from `.github/workflows/release.yml`.
The extra index URL is pointed at a local directory in the same simple-index
layout pip reads, so the run does not contact a public index. They fail while
those scripts still accept the extra-index build, and they pass when the
package under test and its dependencies are installed without running that
build.

Not covered, because those commands do not add an extra index:

- the `uv run pip install langgraph` line that runs only when the package
  name contains `prebuilt`
- `uv sync --group test`
- the publish job, which uploads the dist artifact and does not pip-install
"""

from __future__ import annotations

import hashlib
import io
import os
import shutil
import subprocess
import sys
import tarfile
import textwrap
import zipfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "release.yml"
TEST_PYPI = "https://test.pypi.org/simple/"

PACKAGE = "lg-release-probe"
PACKAGE_VERSION = "1.2.3"
DEP = "lg-confuse-dep"
# Published only on the stand-in for the primary index (PyPI).
BENIGN_DEP_VERSION = "1.0.0"
# Published only on the stand-in for the extra index (Test PyPI).
ATTACKER_DEP_VERSION = "99.0.0"

STEP_NAMES = (
    "Import published package",
    "Import published package (again)",
)


def _uv_executable() -> str:
    found = shutil.which("uv")
    if found:
        return found
    candidate = Path.home() / ".local" / "bin" / "uv"
    if candidate.is_file():
        return str(candidate)
    raise SystemExit("uv is not on PATH; the release workflow invokes uv")


def _wheel_bytes(name: str, version: str, requires: list[str], module_body: str) -> bytes:
    dist_name = name.replace("-", "_")
    info = f"{dist_name}-{version}.dist-info"
    module = f"{dist_name}/__init__.py"
    metadata = "\n".join(
        [
            "Metadata-Version: 2.1",
            f"Name: {name}",
            f"Version: {version}",
            "Requires-Python: >=3.8",
            *[f"Requires-Dist: {req}" for req in requires],
            "",
        ]
    )
    wheel = "Wheel-Version: 1.0\nGenerator: release-index-test\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    payloads = {
        module: module_body,
        f"{info}/METADATA": metadata,
        f"{info}/WHEEL": wheel,
    }
    buffer = io.BytesIO()
    record_lines = []
    with zipfile.ZipFile(buffer, "w") as archive:
        for item, text in payloads.items():
            data = text.encode()
            archive.writestr(item, data)
            digest = hashlib.sha256(data).hexdigest()
            record_lines.append(f"{item},sha256={digest},{len(data)}")
        record_lines.append(f"{info}/RECORD,,")
        archive.writestr(f"{info}/RECORD", ("\n".join(record_lines) + "\n").encode())
    return buffer.getvalue()


def _write_index(simple: Path, project: str, files: dict[str, bytes]) -> None:
    project_dir = simple / project
    project_dir.mkdir(parents=True)
    links = []
    for filename, payload in files.items():
        (project_dir / filename).write_bytes(payload)
        links.append(f'<a href="{filename}">{filename}</a>')
    (project_dir / "index.html").write_text(
        "<!DOCTYPE html><html><body>\n" + "\n".join(links) + "\n</body></html>\n",
        encoding="utf-8",
    )


def _install_invocations(script: str) -> list[str]:
    """Every `uv run pip install` in a step, including the retry behind `||`.

    The shell skips the retry when the first install succeeds, so running only
    the whole script would miss a retry that still adds the extra index.
    """
    logical: list[str] = []
    pending = ""
    for line in script.splitlines():
        if line.rstrip().endswith("\\"):
            pending += line.rstrip()[:-1] + " "
            continue
        pending += line
        logical.append(pending)
        pending = ""
    if pending:
        logical.append(pending)
    commands = []
    needle = "uv run pip install"
    for line in logical:
        start = 0
        while True:
            index = line.find(needle, start)
            if index < 0:
                break
            rest = line[index:]
            cut = len(rest)
            for token in ("||", ")"):
                position = rest.find(token)
                if position != -1:
                    cut = min(cut, position)
            command = " ".join(rest[:cut].split())
            start = index + len(needle)
            if "extra-index-url" in command or "test.pypi.org" in command:
                commands.append(command)
                continue
            if "PKG_NAME" in command or "dist/" in command:
                commands.append(command)
    return commands


def _attacker_sdist(marker: Path) -> bytes:
    backend = textwrap.dedent(
        f"""\
        import os
        import zipfile

        MARKER = {str(marker)!r}

        def _ran():
            os.makedirs(os.path.dirname(MARKER), exist_ok=True)
            with open(MARKER, "a", encoding="utf-8") as handle:
                handle.write("executed\\n")

        _ran()

        def get_requires_for_build_wheel(config_settings):
            _ran()
            return []

        def prepare_metadata_for_build_wheel(metadata_directory, config_settings):
            _ran()
            info = "lg_confuse_dep-{ATTACKER_DEP_VERSION}.dist-info"
            dest = os.path.join(metadata_directory, info)
            os.makedirs(dest, exist_ok=True)
            with open(os.path.join(dest, "METADATA"), "w", encoding="utf-8") as handle:
                handle.write(
                    "Metadata-Version: 2.1\\n"
                    "Name: {DEP}\\n"
                    "Version: {ATTACKER_DEP_VERSION}\\n"
                )
            with open(os.path.join(dest, "WHEEL"), "w", encoding="utf-8") as handle:
                handle.write(
                    "Wheel-Version: 1.0\\n"
                    "Generator: release-index-test\\n"
                    "Root-Is-Purelib: true\\n"
                    "Tag: py3-none-any\\n"
                )
            return info

        def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
            _ran()
            filename = "lg_confuse_dep-{ATTACKER_DEP_VERSION}-py3-none-any.whl"
            path = os.path.join(wheel_directory, filename)
            info = "lg_confuse_dep-{ATTACKER_DEP_VERSION}.dist-info"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr(
                    "lg_confuse_dep/__init__.py",
                    "version = '{ATTACKER_DEP_VERSION}'\\n",
                )
                archive.writestr(
                    info + "/METADATA",
                    "Metadata-Version: 2.1\\nName: {DEP}\\nVersion: {ATTACKER_DEP_VERSION}\\n",
                )
                archive.writestr(
                    info + "/WHEEL",
                    "Wheel-Version: 1.0\\nGenerator: release-index-test\\n"
                    "Root-Is-Purelib: true\\nTag: py3-none-any\\n",
                )
                archive.writestr(info + "/RECORD", "")
            return filename
        """
    )
    project = textwrap.dedent(
        f"""\
        [build-system]
        requires = []
        build-backend = "malicious_backend"
        backend-path = ["."]

        [project]
        name = "{DEP}"
        version = "{ATTACKER_DEP_VERSION}"
        """
    )
    root = f"{DEP}-{ATTACKER_DEP_VERSION}"
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for filename, text in (
            ("pyproject.toml", project),
            ("malicious_backend.py", backend),
        ):
            payload = text.encode()
            info = tarfile.TarInfo(f"{root}/{filename}")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def _release_wheel() -> bytes:
    # Lower bound and no upper bound, matching xxhash>=3.5.0 and pydantic>=2.7.4.
    return _wheel_bytes(
        PACKAGE,
        PACKAGE_VERSION,
        requires=[f"{DEP}>={BENIGN_DEP_VERSION}"],
        module_body=f"__version__ = '{PACKAGE_VERSION}'\n",
    )


def _benign_wheel() -> bytes:
    return _wheel_bytes(
        DEP,
        BENIGN_DEP_VERSION,
        requires=[],
        module_body=f"__version__ = '{BENIGN_DEP_VERSION}'\n",
    )


def _file_index(simple: Path) -> str:
    uri = simple.resolve().as_uri()
    if not uri.endswith("/"):
        uri += "/"
    return uri


def _scripts(workflow: Path) -> list[tuple[str, str]]:
    document = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    steps = document["jobs"]["pre-release-checks"]["steps"]
    found = []
    for step in steps:
        name = step.get("name")
        if name in STEP_NAMES:
            run = step.get("run")
            if not isinstance(run, str) or not run.strip():
                raise AssertionError(f"{name} has no run script in {workflow}")
            found.append((name, run))
    missing = [name for name in STEP_NAMES if name not in {item[0] for item in found}]
    if missing:
        raise AssertionError(f"workflow is missing steps: {', '.join(missing)}")
    return found


def _installed_versions(root: Path, dist_prefix: str) -> set[str]:
    versions = set()
    if not root.exists():
        return versions
    for metadata in root.rglob("METADATA"):
        info = metadata.parent
        if not info.name.endswith(".dist-info"):
            continue
        if not info.name.startswith(dist_prefix + "-"):
            continue
        for line in metadata.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("Version:"):
                versions.add(line.split(":", 1)[1].strip())
                break
    return versions


def _run_step(name: str, script: str, uv: str) -> None:
    sandbox = Path(
        subprocess.check_output(["mktemp", "-d", "/tmp/release-index-XXXXXX"], text=True).strip()
    )
    try:
        primary = sandbox / "primary" / "simple"
        extra = sandbox / "extra" / "simple"
        marker = sandbox / "attacker-ran" / "marker"
        home = sandbox / "home"
        project = sandbox / "project"
        dist = project / "dist"
        for path in (home, project, dist):
            path.mkdir(parents=True)
        _write_index(
            primary,
            DEP,
            {f"lg_confuse_dep-{BENIGN_DEP_VERSION}-py3-none-any.whl": _benign_wheel()},
        )
        release_wheel = _release_wheel()
        _write_index(
            extra,
            PACKAGE,
            {f"lg_release_probe-{PACKAGE_VERSION}-py3-none-any.whl": release_wheel},
        )
        _write_index(
            extra,
            DEP,
            {f"{DEP}-{ATTACKER_DEP_VERSION}.tar.gz": _attacker_sdist(marker)},
        )
        (dist / f"lg_release_probe-{PACKAGE_VERSION}-py3-none-any.whl").write_bytes(
            release_wheel
        )
        (project / "pyproject.toml").write_text(
            textwrap.dedent(
                """\
                [project]
                name = "probe"
                version = "0.0.1"
                requires-python = ">=3.11"
                dependencies = []
                """
            ),
            encoding="utf-8",
        )
        # Same setting the GitHub-hosted runner writes to /etc/pip.conf
        # (actions/runner-images images/ubuntu/scripts/build/install-python.sh):
        # break-system-packages lets the system pip install proceed.
        pip_conf = sandbox / "pip.conf"
        pip_conf.write_text(
            "[global]\nbreak-system-packages = true\n",
            encoding="utf-8",
        )
        extra_url = _file_index(extra)
        primary_url = _file_index(primary)
        ran = script.replace(TEST_PYPI, extra_url).replace(
            TEST_PYPI.rstrip("/"), extra_url
        )
        env = os.environ.copy()
        for key in (
            "VIRTUAL_ENV",
            "PYTHONPATH",
            "PIP_INDEX_URL",
            "PIP_EXTRA_INDEX_URL",
            "UV_INDEX",
            "UV_INDEX_URL",
            "UV_EXTRA_INDEX_URL",
            "UV_DEFAULT_INDEX",
        ):
            env.pop(key, None)
        env.update(
            {
                "HOME": str(home),
                "PKG_NAME": PACKAGE,
                "VERSION": PACKAGE_VERSION,
                "PIP_INDEX_URL": primary_url,
                "UV_DEFAULT_INDEX": primary_url,
                "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                "PIP_NO_CACHE_DIR": "1",
                "PIP_NO_INPUT": "1",
                "PIP_CONFIG_FILE": str(pip_conf),
                "UV_PYTHON": "3.11",
                "UV_CACHE_DIR": str(sandbox / "uv-cache"),
                "PATH": str(Path(uv).parent) + os.pathsep + env.get("PATH", ""),
            }
        )
        completed = subprocess.run(
            ["bash", "-eo", "pipefail", "-c", ran],
            cwd=project,
            env=env,
            text=True,
            capture_output=True,
            timeout=180,
        )
        output = (completed.stdout + "\n" + completed.stderr)[-4000:]
        dep_prefix = DEP.replace("-", "_")
        package_prefix = PACKAGE.replace("-", "_")
        dep_versions = _installed_versions(sandbox, dep_prefix)
        package_versions = _installed_versions(sandbox, package_prefix)
        problems = []
        if marker.exists():
            problems.append(
                "the extra-index dependency build backend ran "
                f"({marker.read_text(encoding='utf-8').strip() or 'marker created'})"
            )
        if ATTACKER_DEP_VERSION in dep_versions:
            problems.append(
                f"{DEP} {ATTACKER_DEP_VERSION} from the extra index was installed"
            )
        if PACKAGE_VERSION not in package_versions:
            problems.append(
                f"{PACKAGE} {PACKAGE_VERSION} was not installed "
                f"(found {sorted(package_versions) or 'nothing'})"
            )
        if BENIGN_DEP_VERSION not in dep_versions or dep_versions != {BENIGN_DEP_VERSION}:
            problems.append(
                f"{DEP} should be exactly {BENIGN_DEP_VERSION} from the primary "
                f"index, found {sorted(dep_versions) or 'nothing'}"
            )
        if problems:
            raise AssertionError(
                f"{name} is not safe:\n- "
                + "\n- ".join(problems)
                + f"\nexit {completed.returncode}\n{output}"
            )
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)


def test_pre_release_checks_do_not_install_extra_index_dependencies() -> None:
    uv = _uv_executable()
    failures = []
    cases = []
    for name, script in _scripts(WORKFLOW):
        cases.append((name, script))
        for number, command in enumerate(_install_invocations(script), start=1):
            cases.append((f"{name} command {number}", command))
    for name, script in cases:
        try:
            _run_step(name, script, uv)
        except AssertionError as exc:
            failures.append(str(exc))
    if failures:
        raise AssertionError("\n\n".join(failures))


if __name__ == "__main__":
    try:
        test_pre_release_checks_do_not_install_extra_index_dependencies()
    except AssertionError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
    print("pre-release install scripts ignored the extra-index dependency")
