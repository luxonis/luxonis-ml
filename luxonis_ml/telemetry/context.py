from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Any

from luxonis_ml.utils.environ import environ


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


def is_luxonis_cloud() -> bool:
    """Return True when telemetry runs inside the Luxonis cloud."""
    return bool(environ.LUXONIS_TELEMETRY_IS_LUXONIS_CLOUD)


def normalized_processor() -> str:
    """Return a coarse processor family for telemetry."""
    raw_values = [platform.machine().lower(), platform.processor().lower()]
    for value in raw_values:
        if not value:
            continue
        if any(token in value for token in ("aarch64", "arm64")):
            return "arm64"
        if "arm" in value:
            return "arm"
        if any(token in value for token in ("x86_64", "amd64")):
            return "x86_64"
        if any(token in value for token in ("x86", "i386", "i686")):
            return "x86"
        if "ppc" in value or "powerpc" in value:
            return "powerpc"
        if "riscv" in value:
            return "riscv"
    return "unknown"


def base_context(
    *,
    library_name: str,
    library_version: str | None,
    session_id: str,
) -> dict[str, Any]:
    """Create base context for all events.

    @type library_name: str
    @param library_name: Name of the emitting library.
    @type library_version: Optional[str]
    @param library_version: Version string for the library.
    @type session_id: str
    @param session_id: Random per-process session id.
    """
    return {
        "os": platform.system(),
        "os_version": platform.release(),
        "arch": platform.machine(),
        "python_version": platform.python_version(),
        "library": library_name,
        "library_version": library_version,
        "session_id": session_id,
        "is_luxonis_cloud": is_luxonis_cloud(),
        "ci": is_ci(),
    }


def system_context() -> dict[str, Any]:
    """Extended system metadata for telemetry events."""
    return {
        "processor": normalized_processor(),
        "cpu_count": os.cpu_count(),
        "is_docker": Path("/.dockerenv").exists(),
    }
