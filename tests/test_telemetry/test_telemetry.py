import json
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
import typer

from luxonis_ml.telemetry import (
    Telemetry,
    TelemetryConfig,
    get_or_init,
    get_telemetry,
    initialize_telemetry,
    skip_telemetry,
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
    monkeypatch.setenv("LUXONIS_TELEMETRY_IS_LUXONIS_CLOUD", "1")
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


def test_capture_includes_context(
    dummy_backend: DummyBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LUXONIS_TELEMETRY_IS_LUXONIS_CLOUD", "1")
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
    assert event.context["is_luxonis_cloud"] is True


def test_include_system_metadata_flag(dummy_backend: DummyBackend) -> None:
    config = TelemetryConfig(enabled=True, backend="dummy")
    telemetry = Telemetry("luxonis_ml", config=config)

    telemetry.capture("event.no_system")
    event = dummy_backend.events[-1]
    assert "cpu_count" not in event.context

    telemetry.capture("event.with_system", include_system_metadata=True)
    event = dummy_backend.events[-1]
    assert "cpu_count" in event.context


def test_base_context_can_be_disabled(dummy_backend: DummyBackend) -> None:
    config = TelemetryConfig(
        enabled=True,
        backend="dummy",
        include_base_context=False,
    )
    telemetry = Telemetry("luxonis_ml", config=config)

    telemetry.capture("event.no_base")

    event = dummy_backend.events[-1]
    assert event.context == {}


def test_system_context_providers_only_apply_when_requested(
    dummy_backend: DummyBackend,
) -> None:
    config = TelemetryConfig(enabled=True, backend="dummy")

    def my_system_context(_telemetry: Telemetry) -> dict[str, str]:
        return {"runtime": "docker"}

    telemetry = Telemetry(
        "luxonis_ml",
        config=config,
        system_context_providers=[my_system_context],
    )

    telemetry.capture("event.no_system")
    event = dummy_backend.events[-1]
    assert "runtime" not in event.context

    telemetry.capture("event.with_system", include_system_metadata=True)
    event = dummy_backend.events[-1]
    assert event.context["runtime"] == "docker"


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


def test_sanitize_properties_redacts_nested_mappings() -> None:
    props = {
        "config": {
            "api_key": "secret",
            "nested": {"token": "abc"},
        }
    }

    out = sanitize_properties(props)

    assert out["config"]["api_key"] == "<redacted>"
    assert out["config"]["nested"]["token"] == "<redacted>"  # noqa: S105


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


def test_install_id_recovers_from_invalid_file(
    tmp_path: Path, reset_backend_registry: Generator[None, None, None]
) -> None:
    install_path = tmp_path / "telemetry.json"
    install_path.write_text("{invalid", encoding="utf-8")
    config = TelemetryConfig(
        enabled=True,
        backend="noop",
        install_id_path=install_path,
    )

    telemetry = Telemetry("luxonis_ml", config=config)

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


def test_get_or_init_reuses_existing_instance(
    dummy_backend: DummyBackend,
) -> None:
    config = TelemetryConfig(enabled=True, backend="dummy")
    t1 = initialize_telemetry(
        library_name="lib_a",
        config=config,
        register_exit_handler=False,
    )
    t2 = get_or_init(
        library_name="lib_a",
        config=TelemetryConfig(enabled=True, backend="noop"),
        register_exit_handler=False,
    )
    assert t1 is t2
    assert t2.config.backend == "dummy"


def test_get_or_init_merges_new_context_providers(
    dummy_backend: DummyBackend,
) -> None:
    config = TelemetryConfig(enabled=True, backend="dummy")

    def first_context(_telemetry: Telemetry) -> dict[str, str]:
        return {"first": "value"}

    def second_context(_telemetry: Telemetry) -> dict[str, str]:
        return {"second": "value"}

    telemetry = initialize_telemetry(
        library_name="lib_a",
        config=config,
        context_providers=[first_context],
        register_exit_handler=False,
    )
    reused = get_or_init(
        library_name="lib_a",
        context_providers=[second_context],
        register_exit_handler=False,
    )

    reused.capture("event.test")

    assert reused is telemetry
    event = dummy_backend.events[-1]
    assert event.context["first"] == "value"
    assert event.context["second"] == "value"


def test_registering_custom_backend_keeps_builtin_backends(
    reset_backend_registry: Generator[None, None, None],
) -> None:
    Telemetry.register_backend("dummy", lambda cfg: DummyBackend())

    telemetry = Telemetry(
        "luxonis_ml",
        config=TelemetryConfig(enabled=True, backend="noop"),
    )

    assert telemetry._backend.__class__.__name__ == "NoopBackend"


def test_register_backend_is_case_insensitive(
    reset_backend_registry: Generator[None, None, None],
) -> None:
    backend = DummyBackend()
    Telemetry.register_backend("Dummy", lambda cfg: backend)

    telemetry = Telemetry(
        "luxonis_ml",
        config=TelemetryConfig(enabled=True, backend="dummy"),
    )
    telemetry.capture("event.test")

    assert telemetry._backend is backend
    assert len(backend.events) == 1


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


def test_instrument_typer_exclude_commands(
    dummy_backend: DummyBackend,
) -> None:
    config = TelemetryConfig(enabled=True, backend="dummy")
    telemetry = Telemetry("luxonis_ml", config=config)

    app = typer.Typer()

    @app.command()
    def train(epochs: int = 10) -> int:
        return epochs

    @app.command()
    def evaluate() -> int:
        return 1

    from luxonis_ml.telemetry.cli import instrument_typer

    instrument_typer(app, telemetry, exclude_commands={"train"})

    cmd_by_name = {cmd.name: cmd.callback for cmd in app.registered_commands}
    cmd_by_name["train"](epochs=5)
    cmd_by_name["evaluate"]()

    assert len(dummy_backend.events) == 1
    event = dummy_backend.events[0]
    assert event.name == "cli.command"
    assert event.properties["command"] == "evaluate"


def test_instrument_typer_skip_decorator(
    dummy_backend: DummyBackend,
) -> None:
    config = TelemetryConfig(enabled=True, backend="dummy")
    telemetry = Telemetry("luxonis_ml", config=config)

    app = typer.Typer()

    @skip_telemetry
    @app.command()
    def secret() -> int:
        return 0

    @app.command()
    def visible() -> int:
        return 1

    from luxonis_ml.telemetry.cli import instrument_typer

    instrument_typer(app, telemetry)

    cmd_by_name = {cmd.name: cmd.callback for cmd in app.registered_commands}
    cmd_by_name["secret"]()
    cmd_by_name["visible"]()

    assert len(dummy_backend.events) == 1
    event = dummy_backend.events[0]
    assert event.name == "cli.command"
    assert event.properties["command"] == "visible"
