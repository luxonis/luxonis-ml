from __future__ import annotations

import json
import os
import platform
from pathlib import Path
from typing import Any
from uuid import uuid4


def default_install_id_path() -> Path:
    """Return the default path for storing the install id."""
    if os.name == "nt" and "APPDATA" in os.environ:
        base = Path(os.environ["APPDATA"])
    elif "XDG_CONFIG_HOME" in os.environ:
        base = Path(os.environ["XDG_CONFIG_HOME"])
    else:
        base = Path.home() / ".config"
    return base / "luxonis" / "telemetry.json"


def load_install_id(path: Path) -> str | None:
    """Load or create a persistent anonymous install id.

    If the file cannot be read or written, returns None.

    @type path: L{Path}
    @param path: Path where the install id is stored.
    """
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            value = data.get("install_id")
            if isinstance(value, str) and value:
                return value
        path.parent.mkdir(parents=True, exist_ok=True)
        install_id = str(uuid4())
        path.write_text(
            json.dumps({"install_id": install_id}, sort_keys=True),
            encoding="utf-8",
        )
    except OSError:
        return None
    else:
        return install_id


def is_ci() -> bool:
    """Best-effort detection of CI environments."""
    markers = [
        "CI",
        "GITHUB_ACTIONS",
        "GITLAB_CI",
        "BUILDKITE",
        "JENKINS_URL",
        "TEAMCITY_VERSION",
    ]
    return any(os.environ.get(marker) for marker in markers)


def base_context(
    *,
    library_name: str,
    library_version: str | None,
    install_id: str | None,
    session_id: str,
) -> dict[str, Any]:
    """Create base context for all events.

    @type library_name: str
    @param library_name: Name of the emitting library.
    @type library_version: Optional[str]
    @param library_version: Version string for the library.
    @type install_id: Optional[str]
    @param install_id: Anonymous install id (distinct id).
    @type session_id: str
    @param session_id: Random per-process session id.
    """
    return {
        "os": platform.system(),
        "os_version": platform.release(),
        "platform": platform.platform(),
        "arch": platform.machine(),
        "python_version": platform.python_version(),
        "library": library_name,
        "library_version": library_version,
        "session_id": session_id,
        "install_id": install_id,
        "ci": is_ci(),
    }


def system_context() -> dict[str, Any]:
    """Extended system metadata for telemetry events."""
    return {
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "is_docker": Path("/.dockerenv").exists(),
    }
