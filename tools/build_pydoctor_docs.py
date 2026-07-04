#!/usr/bin/env python3
"""Build pydoctor API documentation for GitHub Pages."""

from __future__ import annotations

import argparse
import ast
import html
import inspect
import json
import logging
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

PACKAGE = "luxonis_ml"
PROJECT_NAME = "LuxonisML"
PROJECT_URL = "https://github.com/luxonis/luxonis-ml"
PAGES_URL = "https://luxonis.github.io/luxonis-ml"
DEFAULT_RELEASE_BASELINE = "v0.8.6-beta"
EXTRACTALL_SUPPORTS_FILTER = (
    "filter" in inspect.signature(tarfile.TarFile.extractall).parameters
)
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class VersionEntry:
    label: str
    path: str
    version: str
    kind: str


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    args = parse_args()
    repo_root = Path.cwd().resolve()
    output = args.output.resolve()

    clean_output(output, repo_root)

    latest_version = read_project_version(repo_root)
    build_pydoctor(
        source_root=repo_root,
        destination=output / "latest",
        version=latest_version,
        base_url=f"{PAGES_URL}/latest/",
    )

    releases: list[VersionEntry] = []
    if args.mode == "site":
        for tag in release_tags_after(args.release_baseline):
            tag_path = safe_release_path(tag)
            with export_git_tag(tag) as source_root:
                version = read_project_version(source_root)
                build_pydoctor(
                    source_root=source_root,
                    destination=output / "releases" / tag_path,
                    version=version,
                    base_url=f"{PAGES_URL}/releases/{tag_path}/",
                )
            releases.append(
                VersionEntry(
                    label=tag,
                    path=f"releases/{tag_path}/",
                    version=tag.removeprefix("v"),
                    kind="release",
                )
            )

    latest = VersionEntry(
        label="latest",
        path="latest/",
        version=latest_version,
        kind="latest",
    )
    write_site_index(output, latest=latest, releases=releases)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build pydoctor docs for LuxonisML."
    )
    parser.add_argument(
        "--mode",
        choices=["current", "site"],
        default="current",
        help=(
            "Build only the current checkout, or build the full Pages site "
            "with future release versions."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("apidocs"),
        help="Directory where the generated static site will be written.",
    )
    parser.add_argument(
        "--release-baseline",
        default=DEFAULT_RELEASE_BASELINE,
        help=(
            "Release tag used as the lower bound for versioned docs. Only "
            "tags newer than this baseline are built in site mode."
        ),
    )
    return parser.parse_args()


def clean_output(output: Path, repo_root: Path) -> None:
    if output == repo_root:
        raise SystemExit("Refusing to use the repository root as docs output.")
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)


def build_pydoctor(
    *,
    source_root: Path,
    destination: Path,
    version: str,
    base_url: str,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        shutil.rmtree(destination)

    command = [
        executable("pydoctor"),
        "--docformat=google",
        "--project-name",
        PROJECT_NAME,
        "--project-url",
        PROJECT_URL,
        "--project-version",
        version,
        "--project-base-dir",
        str(source_root),
        "--html-base-url",
        base_url,
        "--html-output",
        str(destination),
        str(source_root / PACKAGE),
        "--theme",
        "readthedocs",
    ]
    LOGGER.info("Building %s from %s", destination, source_root)
    completed = subprocess.run(command, check=False)
    if completed.returncode == 0:
        return
    if completed.returncode == 2 and (destination / "index.html").is_file():
        LOGGER.warning(
            "pydoctor exited with non-fatal documentation warnings; "
            "continuing because HTML was generated."
        )
        return
    raise subprocess.CalledProcessError(completed.returncode, command)


def read_project_version(source_root: Path) -> str:
    init_path = source_root / PACKAGE / "__init__.py"
    module = ast.parse(init_path.read_text(encoding="utf-8"))
    for node in module.body:
        if (
            isinstance(node, ast.AnnAssign)
            and is_name(node.target, "__version__")
            and isinstance(node.value, ast.Constant)
        ):
            return str(node.value.value)
        if (
            isinstance(node, ast.Assign)
            and any(is_name(target, "__version__") for target in node.targets)
            and isinstance(node.value, ast.Constant)
        ):
            return str(node.value.value)
    return ""


def is_name(node: ast.AST, value: str) -> bool:
    return isinstance(node, ast.Name) and node.id == value


def release_tags_after(baseline: str) -> list[str]:
    baseline_version = parse_release_tag(baseline)
    if baseline_version is None:
        raise SystemExit(f"Invalid release baseline tag: {baseline}")

    result = subprocess.run(
        [executable("git"), "tag", "--list", "v*"],
        check=True,
        capture_output=True,
        text=True,
    )
    tags: list[tuple[tuple[int, int, int, int, int], str]] = []
    for tag in result.stdout.splitlines():
        version = parse_release_tag(tag)
        if version is not None and version > baseline_version:
            tags.append((version, tag))
    return [tag for _, tag in sorted(tags)]


def parse_release_tag(tag: str) -> tuple[int, int, int, int, int] | None:
    version = tag.removeprefix("v")
    core, separator, prerelease = version.partition("-")
    parts = core.split(".")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        return None

    major, minor, patch = (int(part) for part in parts)
    if not separator:
        return major, minor, patch, 3, 0

    name = "".join(ch for ch in prerelease.lower() if ch.isalpha())
    number = "".join(ch for ch in prerelease if ch.isdigit())
    prerelease_rank = {
        "alpha": 0,
        "a": 0,
        "beta": 1,
        "b": 1,
        "rc": 2,
    }.get(name, -1)
    return major, minor, patch, prerelease_rank, int(number or 0)


def safe_release_path(tag: str) -> str:
    if tag in {"", ".", ".."} or "/" in tag or "\\" in tag:
        raise SystemExit(f"Release tag cannot be used as a Pages path: {tag}")
    return tag


class export_git_tag:
    def __init__(self, tag: str) -> None:
        self.tag = tag
        self._tmpdir: tempfile.TemporaryDirectory[str] | None = None

    def __enter__(self) -> Path:
        self._tmpdir = tempfile.TemporaryDirectory()
        tmp_path = Path(self._tmpdir.name)
        archive_path = tmp_path / "source.tar"
        source_root = tmp_path / "source"
        source_root.mkdir()

        subprocess.run(
            [
                executable("git"),
                "archive",
                "--format=tar",
                "--output",
                str(archive_path),
                self.tag,
            ],
            check=True,
        )
        with tarfile.open(archive_path) as archive:
            if EXTRACTALL_SUPPORTS_FILTER:
                archive.extractall(source_root, filter="data")
            else:
                archive.extractall(source_root)  # noqa: S202
        return source_root

    def __exit__(self, *_: object) -> None:
        if self._tmpdir is not None:
            self._tmpdir.cleanup()


def write_site_index(
    output: Path, *, latest: VersionEntry, releases: list[VersionEntry]
) -> None:
    versions = {
        "latest": asdict(latest),
        "releases": [asdict(release) for release in releases],
    }
    (output / "versions.json").write_text(
        json.dumps(versions, indent=2) + "\n", encoding="utf-8"
    )
    (output / ".nojekyll").write_text("", encoding="utf-8")

    release_items = "\n".join(
        f'<li><a href="{html.escape(release.path)}">'
        f"{html.escape(release.label)}</a></li>"
        for release in releases
    )
    if not release_items:
        release_items = "<li>No versioned releases published yet.</li>"

    index = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{PROJECT_NAME} documentation</title>
  <style>
    body {{
      color: #1f2933;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.5;
      margin: 0 auto;
      max-width: 720px;
      padding: 48px 24px;
    }}
    a {{ color: #005f9e; }}
  </style>
</head>
<body>
  <h1>{PROJECT_NAME} documentation</h1>
  <h2>Current</h2>
  <ul>
    <li><a href="{html.escape(latest.path)}">latest</a></li>
  </ul>
  <h2>Releases</h2>
  <ul>
    {release_items}
  </ul>
</body>
</html>
"""
    (output / "index.html").write_text(index, encoding="utf-8")


def executable(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise SystemExit(f"Required executable not found on PATH: {name}")
    return path


if __name__ == "__main__":
    main()
