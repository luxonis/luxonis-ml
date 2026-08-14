#!/usr/bin/env python3
"""Read or change the version in ``luxonis_ml/__init__.py``.

Run it from the repository root.
"""

import argparse
import ast
import re
import sys
from pathlib import Path

INIT_PATH = Path("luxonis_ml/__init__.py")
VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")


def main() -> None:
    args = parse_args()
    current = read_version()

    if args.new_version is None:
        sys.stdout.write(f"{current}\n")
        return

    new_version = resolve_version(current, args.new_version)
    write_version(new_version)
    sys.stdout.write(f"{new_version}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read or change the LuxonisML version."
    )
    parser.add_argument(
        "--set",
        dest="new_version",
        metavar="VERSION",
        help=(
            "Increase one part of the version with 'major', 'minor', or "
            "'patch', or give an explicit version such as '1.2.3'. Without "
            "this option the tool only reports the current version."
        ),
    )
    return parser.parse_args()


def read_version() -> str:
    return str(_version_node(INIT_PATH.read_text(encoding="utf-8")).value)


def resolve_version(current: str, request: str) -> str:
    major, minor, patch = parse_version(current)
    if request == "major":
        new_version = (major + 1, 0, 0)
    elif request == "minor":
        new_version = (major, minor + 1, 0)
    elif request == "patch":
        new_version = (major, minor, patch + 1)
    else:
        new_version = parse_version(request)

    if new_version <= (major, minor, patch):
        raise SystemExit(
            f"Version {_format_version(new_version)} is not above the "
            f"current version {current}."
        )
    return _format_version(new_version)


def parse_version(version: str) -> tuple[int, int, int]:
    if not VERSION_PATTERN.match(version):
        raise SystemExit(
            f"Expected 'major', 'minor', 'patch', or a version of the form "
            f"'1.2.3'. Got {version!r}."
        )
    major, minor, patch = (int(part) for part in version.split("."))
    return major, minor, patch


def write_version(new_version: str) -> None:
    text = INIT_PATH.read_text(encoding="utf-8")
    node = _version_node(text)
    lines = text.splitlines(keepends=True)
    line = lines[node.lineno - 1]
    prefix = line[: node.col_offset]
    suffix = line[node.end_col_offset :]
    lines[node.lineno - 1] = f'{prefix}"{new_version}"{suffix}'
    INIT_PATH.write_text("".join(lines), encoding="utf-8")


def _format_version(version: tuple[int, int, int]) -> str:
    return ".".join(str(part) for part in version)


def _version_node(text: str) -> ast.Constant:
    for node in ast.parse(text).body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "__version__"
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            return node.value
    raise SystemExit(f"Found no `__version__` string in {INIT_PATH}.")


if __name__ == "__main__":
    main()
