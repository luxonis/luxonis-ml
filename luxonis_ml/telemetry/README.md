# LuxonisML Telemetry

## Introduction

LuxonisML Telemetry provides a small, reusable telemetry layer that can be
shared across Luxonis libraries. It offers a single client API, consistent
context, and pluggable backends (PostHog by default).

The primary goal is to make it easy to emit lightweight usage events without
coupling your library to any specific analytics vendor.

## Table of Contents

- [LuxonisML Telemetry](#luxonisml-telemetry)
  - [Introduction](#introduction)
  - [Installation](#installation)
  - [Core Components](#core-components)
  - [Quickstart](#quickstart)
  - [Configuration](#configuration)
  - [CLI Instrumentation](#cli-instrumentation)
- [Custom Backends](#custom-backends)
- [Context and Metadata](#context-and-metadata)
- [Singleton Usage](#singleton-usage)
- [Environment Variables](#environment-variables)

## Installation

Install telemetry package:

```bash
pip install luxonis-ml[telemetry]
```

## Core Components

- **Telemetry**
  - Main client that captures events with context and sends them via a backend.
- **TelemetryConfig**
  - Configuration object used to control backend selection and behavior.
- **Backends**
  - `PostHogBackend`, `StdoutBackend`, `NoopBackend` (and custom backends).
- **CLI Instrumentation**
  - `instrument_typer(...)` and `instrument_cyclopts(...)` to log CLI commands.

## Quickstart

```python
from luxonis_ml.telemetry import Telemetry

telemetry = Telemetry("luxonis_ml")
telemetry.capture("train_started", {"epochs": 20})
```

## Configuration

You can configure telemetry explicitly via `TelemetryConfig`:

```python
from luxonis_ml.telemetry import Telemetry, TelemetryConfig

config = TelemetryConfig(
    enabled=True,
    backend="posthog",
    api_key="phc_xxx",
    endpoint="https://us.i.posthog.com",
    include_base_context=True,
)

telemetry = Telemetry("luxonis_ml", config=config)
telemetry.capture("train_started", {"epochs": 20})
```

Set `include_base_context=False` if a consuming library wants to build
its own default event context instead of always attaching the shared
LuxonisML base metadata.

## CLI Instrumentation

```python
import typer
from luxonis_ml.telemetry import Telemetry
from luxonis_ml.telemetry.cli import instrument_typer

app = typer.Typer()
telemetry = Telemetry("luxonis_ml")

instrument_typer(app, telemetry)

@app.command()
def train(epochs: int = 10):
    telemetry.capture("train_invoked", {"epochs": epochs})
```

By default, CLI telemetry logs only the command name, success flag, and
duration. Command arguments are omitted unless you explicitly pass an
`allowlist`:

```python
instrument_typer(app, telemetry, allowlist={"epochs"})
```

Use the Cyclopts adapter for Cyclopts apps:

```python
from cyclopts import App
from luxonis_ml.telemetry import Telemetry
from luxonis_ml.telemetry.cli import instrument_cyclopts

app = App(name="demo")
telemetry = Telemetry("luxonis_ml")

instrument_cyclopts(app, telemetry)

@app.command
def train(epochs: int = 10):
    return epochs
```

## Custom Backends

```python
from luxonis_ml.telemetry import Telemetry, TelemetryConfig
from luxonis_ml.telemetry.backends.base import TelemetryBackend

class MyBackend(TelemetryBackend):
    def __init__(self, config):
        super().__init__(config)
    def capture(self, event):
        print("event", event)

Telemetry.register_backend("my_backend", MyBackend)

config = TelemetryConfig(enabled=True, backend="my_backend")
telemetry = Telemetry("luxonis_ml", config=config)
telemetry.capture("custom_event")
```

Backend names are matched case-insensitively, so registering
`"My_Backend"` and configuring `backend="my_backend"` will resolve to
the same backend.

## Context and Metadata

Each event includes a base context with:

- `$process_person_profile` set to `False` to keep PostHog captures anonymous and
  avoid person profile creation.
- `$session_id` for PostHog session correlation using an ephemeral
  per-process ID
- `source_product`
- `source_component`
- `sdk_version`

Host and runtime metadata are available through utility functions and
ready-to-use context providers:

```python
from luxonis_ml.telemetry import (
    Telemetry,
    system_context_provider,
)

telemetry = Telemetry(
    "luxonis_ml",
    system_context_providers=[system_context_provider],
)

telemetry.capture("train_started", include_system_metadata=True)
```

`system_context_provider` adds:

- `os`
- `os_version`
- `arch`
- `python_version`
- `ci`
- `is_luxonis_cloud`
- normalized `processor` family (`x86_64`, `x86`, `arm64`, `arm`, etc.)
- `cpu_count`
- `is_docker`

If you only want the coarse host metadata, use `host_context_provider`.

You can also call the utility functions directly:

```python
from luxonis_ml.telemetry import host_context, system_context

host = host_context()
system = system_context()
```

You can add custom context providers:

```python
from luxonis_ml.telemetry import Telemetry

def my_context(_telemetry):
    return {"team": "vision"}

telemetry = Telemetry("luxonis_ml", context_providers=[my_context])
```

## Singleton Usage

For a global instance per library name:

```python
from luxonis_ml.telemetry import get_or_init, get_telemetry

get_or_init(library_name="luxonis_ml")
telemetry = get_telemetry("luxonis_ml")
if telemetry:
    telemetry.capture("init")
```

Use different library names to keep multiple singleton instances.

For the same library name, `get_or_init(...)` reuses the existing
telemetry instance. Conflicting config or version arguments are ignored
with a warning, while new `context_providers` and
`system_context_providers` are merged into the existing instance. This
lets multiple integration points contribute telemetry context without
rebuilding the singleton.

## Failure Behavior

Telemetry is designed to fail open inside consuming libraries:

- If the configured backend is unavailable or misconfigured, telemetry
  falls back to `NoopBackend`.
- Capture, flush, and shutdown failures are swallowed so they
  do not break the host application.

## Environment Variables

The telemetry module reads the following environment variables:

- `LUXONIS_TELEMETRY_ENABLED`\
  Enables telemetry (default). Set to a falsy value to disable.
- `LUXONIS_TELEMETRY_BACKEND`\
  Backend name to use (e.g., `posthog`, `stdout`, or a custom registered backend).
- `LUXONIS_TELEMETRY_API_KEY`\
  API key for the backend (used by PostHog).
- `LUXONIS_TELEMETRY_ENDPOINT`\
  Backend endpoint/host URL (e.g., PostHog host).
- `LUXONIS_TELEMETRY_DEBUG`\
  When truthy, defaults backend to `stdout` unless explicitly overridden.
