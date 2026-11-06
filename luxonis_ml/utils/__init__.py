"""Shared utility entry point for Luxonis ML internals and integrations.

The `luxonis_ml.utils` package gathers cross-cutting helpers used throughout
the data, tracker, telemetry, and archive modules. It includes configuration
loading, environment-backed settings, a local/remote filesystem abstraction,
graph traversal helpers, logging/deprecation utilities, and registry
primitives for plugin-style extension points.

Example:
    Access the shared filesystem abstraction from the package import.

    .. code-block:: python

        from luxonis_ml.utils import LuxonisFileSystem

        fs = LuxonisFileSystem("file:///tmp/dataset")
        files = fs.walk_dir(".")

Note:
    Some utilities rely on optional dependencies. Install
    ``luxonis-ml[utils]`` when using filesystem, config, and logging helpers
    outside environments that already provide those packages.

See:
    `luxonis_ml.utils.config`, `luxonis_ml.utils.environ`,
    `luxonis_ml.utils.filesystem`, and `luxonis_ml.utils.registry` for the
    implementation-level contracts.

"""

from luxonis_ml.guard_extras import guard_missing_extra

with guard_missing_extra("utils"):
    from luxonis_ml.typing import BaseModelExtraForbid

    from .color import Color
    from .config import LuxonisConfig
    from .environ import Environ, environ
    from .filesystem import PUT_FILE_REGISTRY, LuxonisFileSystem
    from .graph import is_acyclic, traverse_graph
    from .logging import deprecated, log_once, setup_logging
    from .registry import AutoRegisterMeta, Registry
    from .validation import (
        ValidationProblem,
        format_validation_error,
        render_validation_error,
    )


__all__ = [
    "PUT_FILE_REGISTRY",
    "AutoRegisterMeta",
    "BaseModelExtraForbid",
    "Color",
    "Environ",
    "LuxonisConfig",
    "LuxonisFileSystem",
    "Registry",
    "ValidationProblem",
    "deprecated",
    "environ",
    "format_validation_error",
    "is_acyclic",
    "log_once",
    "render_validation_error",
    "setup_logging",
    "traverse_graph",
]
