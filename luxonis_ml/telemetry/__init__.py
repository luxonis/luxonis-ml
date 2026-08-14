"""Public telemetry API for Luxonis packages.

The `luxonis_ml.telemetry` package exposes a lightweight event-capture
client, environment-driven configuration, singleton helpers, context
providers, and suppression tools. It is designed so consuming packages can
emit usage events without coupling directly to a specific analytics backend.
PostHog support is optional through ``luxonis-ml[telemetry]``; disabled,
misconfigured, and debug configurations can use no-op or stdout backends.

Example:
    Emit an event from a component while keeping backend configuration
    separate from the call site.

    .. code-block:: python

        from luxonis_ml.telemetry import Telemetry

        telemetry = Telemetry("luxonis_ml", source_component="data")
        telemetry.capture(
            "dataset_export_started",
            {"dataset_type": "coco"},
        )

Note:
    Telemetry failures are intentionally non-fatal for host applications. If a
    backend cannot be initialized or capture fails, the telemetry layer falls
    back or suppresses the error rather than interrupting the caller.


Environment Variables
=====================

The package reads these variables. `TelemetryConfig.from_environ` resolves
them first, then the values of `TelemetryDefaults`, then the base
`TelemetryConfig` defaults.

    - ``LUXONIS_TELEMETRY_ENABLED``: enables telemetry. It is enabled by
      default. Set a falsy value to turn telemetry off.
    - ``LUXONIS_TELEMETRY_BACKEND``: the backend name, such as ``posthog``,
      ``stdout``, or a custom registered name.
    - ``LUXONIS_TELEMETRY_API_KEY``: the API key of the backend.
    - ``LUXONIS_TELEMETRY_ENDPOINT``: the backend host URL.
    - ``LUXONIS_TELEMETRY_DEBUG``: when truthy, the backend falls back to
      ``stdout`` instead of ``posthog``.

`luxonis_ml.telemetry.context` reads one more variable,
``LUXONIS_TELEMETRY_IS_LUXONIS_CLOUD``. It marks events that a Luxonis cloud
job emits.

To turn telemetry off for one block of code instead, use
`suppress_telemetry`.

See:
    `luxonis_ml.telemetry.client` for the capture client,
    `luxonis_ml.telemetry.config` for configuration,
    `luxonis_ml.telemetry.context` for the exact context keys that each
    event carries, `luxonis_ml.telemetry.cli` for command-line
    instrumentation, and `luxonis_ml.telemetry.backends` for backend
    implementations.

"""

from luxonis_ml.telemetry.client import Telemetry
from luxonis_ml.telemetry.config import TelemetryConfig, TelemetryDefaults
from luxonis_ml.telemetry.context import (
    host_context,
    host_context_provider,
    system_context,
    system_context_provider,
)
from luxonis_ml.telemetry.singleton import (
    get_or_init,
    get_telemetry,
    initialize_telemetry,
    shutdown_on_exit,
)
from luxonis_ml.telemetry.suppression import suppress_telemetry

__all__ = [
    "Telemetry",
    "TelemetryConfig",
    "TelemetryDefaults",
    "get_or_init",
    "get_telemetry",
    "host_context",
    "host_context_provider",
    "initialize_telemetry",
    "shutdown_on_exit",
    "suppress_telemetry",
    "system_context",
    "system_context_provider",
]
