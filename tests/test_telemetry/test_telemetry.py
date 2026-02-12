import json
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
import typer

from luxonis_ml.telemetry import (
    Telemetry,
    TelemetryConfig,
    get_telemetry,
    initialize_telemetry,
    suppress_telemetry,
)
from luxonis_ml.telemetry.redaction import sanitize_properties
from luxonis_ml.telemetry.singleton import _telemetry_by_name


class DummyBackend:
    def __init__(self) -> None:
        self.events: list[Any] = []
        self.identify_calls: list[tuple[str, dict[str, Any]]] = []
        self.flush_count = 0
        self.shutdown_count = 0

    def capture(self, event: Any) -> None:
        self.events.append(event)

    def identify(self, user_id: str, traits: dict[str, Any]) -> None:
        self.identify_calls.append((user_id, traits))

    def flush(self) -> None:
        self.flush_count += 1

    def shutdown(self) -> None:
        self.shutdown_count += 1


@pytest.fixture
def reset_backend_registry() -> Generator[None, None, None]:
    original = dict(Telemetry._backend_factories)
    Telemetry._backend_factories = {}
    yield
    Telemetry._backend_factories = original


@pytest.fixture
def dummy_backend(
    reset_backend_registry: Generator[None, None, None],
) -> DummyBackend:
    backend = DummyBackend()
    Telemetry.register_backend("dummy", lambda cfg: backend)
    return backend


@pytest.fixture(autouse=True)
def reset_singletons() -> Generator[None, None, None]:
    _telemetry_by_name.clear()
    yield
    _telemetry_by_name.clear()


def test_config_from_environ(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("LUXONIS_TELEMETRY_ENABLED", "1")
    monkeypatch.setenv("LUXONIS_TELEMETRY_BACKEND", "stdout")
    monkeypatch.setenv("LUXONIS_TELEMETRY_API_KEY", "secret")
    monkeypatch.setenv("LUXONIS_TELEMETRY_ENDPOINT", "https://example")
    monkeypatch.setenv("LUXONIS_TELEMETRY_DEBUG", "1")
    monkeypatch.setenv("LUXONIS_TELEMETRY_ID", "override")
    monkeypatch.setenv(
        "LUXONIS_TELEMETRY_INSTALL_ID_PATH",
        str(tmp_path / "telemetry.json"),
    )

    cfg = TelemetryConfig.from_environ()
    assert cfg.enabled is True
    assert cfg.backend == "stdout"
    assert cfg.api_key == "secret"
    assert cfg.endpoint == "https://example"
    assert cfg.debug is True
    assert cfg.distinct_id == "override"
    assert cfg.install_id_path == tmp_path / "telemetry.json"


def test_capture_includes_context(dummy_backend: DummyBackend) -> None:
    config = TelemetryConfig(
        enabled=True,
        backend="dummy",
        distinct_id="abc",
        include_system_metadata=True,
    )

    def my_context(_telemetry: Telemetry) -> dict[str, str]:
        return {"custom": "value"}

    telemetry = Telemetry(
        "luxonis_ml",
        config=config,
        context_providers=[my_context],
    )

    telemetry.capture("event.test", {"foo": "bar"})

    assert len(dummy_backend.events) == 1
    event = dummy_backend.events[0]
    assert event.name == "event.test"
    assert event.properties["foo"] == "bar"
    assert event.context["custom"] == "value"
    assert "cpu_count" in event.context
    assert event.distinct_id == "abc"


def test_include_system_metadata_flag(dummy_backend: DummyBackend) -> None:
    config = TelemetryConfig(enabled=True, backend="dummy")
    telemetry = Telemetry("luxonis_ml", config=config)

    telemetry.capture("event.no_system")
    event = dummy_backend.events[-1]
    assert "cpu_count" not in event.context

    telemetry.capture("event.with_system", include_system_metadata=True)
    event = dummy_backend.events[-1]
    assert "cpu_count" in event.context


def test_suppression_skips_capture(dummy_backend: DummyBackend) -> None:
    config = TelemetryConfig(enabled=True, backend="dummy")
    telemetry = Telemetry("luxonis_ml", config=config)

    telemetry.capture("event.one")
    assert len(dummy_backend.events) == 1

    with suppress_telemetry():
        telemetry.capture("event.suppressed")

    assert len(dummy_backend.events) == 1


def test_sanitize_properties_allowlist() -> None:
    props = {"token": "abc", "keep": 1, "drop": 2}
    out = sanitize_properties(props, allowlist={"token", "keep"})
    assert out["token"] == "<redacted>"  # noqa: S105
    assert out["keep"] == 1
    assert "drop" not in out


def test_install_id_created(
    tmp_path: Path, reset_backend_registry: Generator[None, None, None]
) -> None:
    install_path = tmp_path / "telemetry.json"
    config = TelemetryConfig(
        enabled=True,
        backend="noop",
        install_id_path=install_path,
    )
    telemetry = Telemetry("luxonis_ml", config=config)
    assert install_path.exists()
    data = json.loads(install_path.read_text(encoding="utf-8"))
    assert telemetry._distinct_id == data["install_id"]


def test_distinct_id_override_avoids_file(
    tmp_path: Path, reset_backend_registry: Generator[None, None, None]
) -> None:
    install_path = tmp_path / "telemetry.json"
    config = TelemetryConfig(
        enabled=True,
        backend="noop",
        install_id_path=install_path,
        distinct_id="override",
    )
    telemetry = Telemetry("luxonis_ml", config=config)
    assert telemetry._distinct_id == "override"
    assert not install_path.exists()


def test_singleton_registry_multiple(dummy_backend: DummyBackend) -> None:
    config = TelemetryConfig(enabled=True, backend="dummy")
    t1 = initialize_telemetry(
        library_name="lib_a",
        config=config,
        register_exit_handler=False,
    )
    t2 = initialize_telemetry(
        library_name="lib_b",
        config=config,
        register_exit_handler=False,
    )
    assert t1 is not t2
    assert get_telemetry("lib_a") is t1
    assert get_telemetry("lib_b") is t2
    assert get_telemetry() is None


def test_singleton_registry_single(dummy_backend: DummyBackend) -> None:
    config = TelemetryConfig(enabled=True, backend="dummy")
    t1 = initialize_telemetry(
        library_name="only_lib",
        config=config,
        register_exit_handler=False,
    )
    assert get_telemetry() is t1


def test_instrument_typer_emits_event(dummy_backend: DummyBackend) -> None:
    config = TelemetryConfig(enabled=True, backend="dummy")
    telemetry = Telemetry("luxonis_ml", config=config)

    app = typer.Typer()

    @app.command()
    def train(epochs: int = 10) -> int:
        return epochs

    from luxonis_ml.telemetry.cli import instrument_typer

    instrument_typer(app, telemetry)

    cmd = app.registered_commands[0].callback
    result = cmd(epochs=5)
    assert result == 5
    assert any(event.name == "cli.command" for event in dummy_backend.events)
