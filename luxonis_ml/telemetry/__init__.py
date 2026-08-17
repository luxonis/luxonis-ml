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
    Telemetry failures are intentionally non-fatal for host applications. A
    backend that cannot be initialized falls back to `NoopBackend`, and a
    failed capture, flush, or shutdown is suppressed rather than raised.

.. contents:: Table of Contents
   :depth: 2


Configuration
=============

`Telemetry` reads the environment when you pass no config. Pass a
`TelemetryConfig` to control the backend from the call site instead:

.. code-block:: python

    from luxonis_ml.telemetry import Telemetry, TelemetryConfig

    config = TelemetryConfig(
        enabled=True,
        backend="posthog",
        api_key="phc_xxx",
        endpoint="https://us.i.posthog.com",
    )

    telemetry = Telemetry("luxonis_ml", config=config)
    telemetry.capture("dataset_parse_started", {"dataset_format": "coco"})

Three fields change what leaves the machine.
``include_base_context=False`` drops the shared base context, for a library
that builds its own. ``disable_geoip=True`` stops PostHog from deriving
coarse location data from the IP address; the default leaves that
enrichment on. ``allow_reserved_overrides=True`` lets a caller or a context
provider overwrite the base-context fields below, which are otherwise
protected.

Use `TelemetryConfig.from_environ` with `TelemetryDefaults` for
environment-based configuration that keeps library-specific fallbacks:

.. code-block:: python

    from luxonis_ml.telemetry import TelemetryConfig, TelemetryDefaults

    config = TelemetryConfig.from_environ(
        defaults=TelemetryDefaults(
            backend="stdout",
            include_system_metadata=True,
        )
    )

Give ``source_component`` when one library emits from several surfaces and
those emitters need separate base context:

.. code-block:: python

    from luxonis_ml.telemetry import Telemetry

    data_telemetry = Telemetry("luxonis_ml", source_component="data")
    archive_telemetry = Telemetry("luxonis_ml", source_component="nn_archive")


CLI Instrumentation
===================

`instrument_typer` and `instrument_cyclopts` wrap the commands of an
application. Each instrumented command emits one ``cli_command`` event:

.. code-block:: python

    import typer

    from luxonis_ml.telemetry import Telemetry
    from luxonis_ml.telemetry.cli import instrument_typer

    app = typer.Typer()
    telemetry = Telemetry("luxonis_ml")

    instrument_typer(app, telemetry)

The event holds the command name, a success flag, and the duration. The
argument values stay local. Name the ones you allow to leave the machine,
and one allowlist then covers every command of the application:

.. code-block:: python

    instrument_typer(app, telemetry, allowlist={"name"})

`instrument_cyclopts` takes the same arguments and wraps a
``cyclopts.App`` the same way.

See:
    `luxonis_ml.telemetry.cli` for the excluded callbacks, the
    ``exclude_commands`` argument, and the `skip_telemetry` decorator.


Custom Backends
===============

The package ships `PostHogBackend`, `StdoutBackend`, and `NoopBackend`.
For anything else, subclass `TelemetryBackend`, register the class, and
select it by name:

.. code-block:: python

    from luxonis_ml.telemetry import Telemetry, TelemetryConfig
    from luxonis_ml.telemetry.backends.base import TelemetryBackend


    class MyBackend(TelemetryBackend):
        def capture(self, event):
            print("event", event)


    Telemetry.register_backend("my_backend", MyBackend)

    config = TelemetryConfig(enabled=True, backend="my_backend")
    telemetry = Telemetry("luxonis_ml", config=config)
    telemetry.capture("custom_event")

The backend names ignore case, so ``"My_Backend"`` and ``"my_backend"``
resolve to the same backend.


Context and Metadata
====================

Each event carries a base context with ``$process_person_profile``,
``$session_id``, ``source_product``, ``source_component``, and
``sdk_version``. The ``$session_id`` value stays the same for the lifetime
of one `Telemetry` instance. The PostHog backend also uses it as the
default ``distinct_id``, which you can override for a single event:

.. code-block:: python

    from luxonis_ml.telemetry import Telemetry

    telemetry = Telemetry("luxonis_ml")

    telemetry.capture("dataset_info_requested", {"name": "beans"})
    telemetry.capture(
        "dataset_export_finished",
        {"format": "coco"},
        distinct_id="export-run-123",
    )

`system_context_provider` adds the host and runtime fields ``os``,
``os_version``, ``arch``, ``python_version``, ``ci``, ``is_luxonis_cloud``,
``processor``, ``cpu_count``, and ``is_docker``. Use
`host_context_provider` for the coarse host fields alone:

.. code-block:: python

    from luxonis_ml.telemetry import Telemetry, system_context_provider

    telemetry = Telemetry(
        "luxonis_ml",
        system_context_providers=[system_context_provider],
    )

    telemetry.capture("dataset_export_finished", include_system_metadata=True)

`host_context` and `system_context` return the same values directly. A
custom provider adds your own fields to every event:

.. code-block:: python

    from luxonis_ml.telemetry import Telemetry


    def dataset_context(_telemetry):
        return {"dataset_plugin": "internal"}


    telemetry = Telemetry("luxonis_ml", context_providers=[dataset_context])


Singleton Usage
===============

`get_or_init` and `get_telemetry` keep one instance for each
``(library_name, source_component)`` pair:

.. code-block:: python

    from luxonis_ml.telemetry import get_or_init

    telemetry = get_or_init("luxonis_ml")
    telemetry.capture("dataset_ls_invoked")

Initialize each component of a library separately:

.. code-block:: python

    from luxonis_ml.telemetry import get_or_init, get_telemetry

    get_or_init("luxonis_ml", source_component="data")
    get_or_init("luxonis_ml", source_component="nn_archive")

    data_telemetry = get_telemetry("luxonis_ml", source_component="data")
    archive_telemetry = get_telemetry(
        "luxonis_ml", source_component="nn_archive"
    )

If you omit ``source_component``, it defaults to the library name. A second
`get_or_init` call for the same pair reuses the instance. It ignores a
different ``config`` or ``library_version`` with a warning, and it merges
new context providers into the instance. `get_telemetry` returns ``None``
for an ambiguous lookup, such as ``get_telemetry("luxonis_ml")`` while both
``data`` and ``nn_archive`` are registered.


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
