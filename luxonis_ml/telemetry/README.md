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
  - `instrument_typer(...)` to automatically log Typer commands.

## Quickstart

```python
from luxonis_ml.telemetry import Telemetry

telemetry = Telemetry("luxonis_ml")
telemetry.capture("train.start", {"epochs": 20})
```

## Configuration

You can configure telemetry explicitly via `TelemetryConfig`:

```python
from luxonis_ml.telemetry import Telemetry
from luxonis_ml.telemetry.config import TelemetryConfig

config = TelemetryConfig(
    enabled=True,
    backend="posthog",
    api_key="phc_xxx",
    endpoint="https://us.i.posthog.com",
)

telemetry = Telemetry("luxonis_ml", config=config)
telemetry.capture("train.start", {"epochs": 20})
```

## CLI Instrumentation

```python
import typer
from luxonis_ml.telemetry import Telemetry, instrument_typer

app = typer.Typer()
telemetry = Telemetry("luxonis_ml")

instrument_typer(app, telemetry)

@app.command()
def train(epochs: int = 10):
    telemetry.capture("train.invoked", {"epochs": epochs})
```

## Custom Backends

```python
from luxonis_ml.telemetry import Telemetry
from luxonis_ml.telemetry.config import TelemetryConfig

class MyBackend:
    def capture(self, event):
        print("event", event)
    def identify(self, user_id, traits):
        pass
    def flush(self):
        pass
    def shutdown(self):
        pass

Telemetry.register_backend("my_backend", lambda cfg: MyBackend())

config = TelemetryConfig(enabled=True, backend="my_backend")
telemetry = Telemetry("luxonis_ml", config=config)
telemetry.capture("custom.event")
```

## Context and Metadata

Each event includes a base context (OS, Python version, library name/version,
session id, install id, and CI indicator). You can optionally include extended
system metadata per event:

```python
telemetry.capture("train.start", include_system_metadata=True)
```

You can also add custom context providers:

```python
from luxonis_ml.telemetry import Telemetry

def my_context(_telemetry):
    return {"team": "vision"}

telemetry = Telemetry("luxonis_ml", context_providers=[my_context])
```

## Singleton Usage

For a global instance per library name:

```python
from luxonis_ml.telemetry import initialize_telemetry, get_telemetry

initialize_telemetry(library_name="luxonis_ml")
telemetry = get_telemetry("luxonis_ml")
if telemetry:
    telemetry.capture("init")
```

Multiple singletons are supported by using different library names.

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
- `LUXONIS_TELEMETRY_INSTALL_ID_PATH`\
  Custom path to store the anonymous install ID.
- `LUXONIS_TELEMETRY_ID`\
  Override the anonymous distinct ID.
