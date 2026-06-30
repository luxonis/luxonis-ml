import threading
import time
from collections.abc import Generator
from importlib import import_module
from pathlib import Path
from typing import Any

import pytest
from cyclopts import App

import luxonis_ml.telemetry.singleton as telemetry_singleton
from luxonis_ml.telemetry import (
    Telemetry,
    TelemetryConfig,
    TelemetryDefaults,
    get_or_init,
    get_telemetry,
    initialize_telemetry,
    suppress_telemetry,
)
from luxonis_ml.telemetry import context as telemetry_context
from luxonis_ml.telemetry.backends.base import (
    TELEMETRY_BACKENDS,
    TelemetryBackend,
)
from luxonis_ml.telemetry.cli import skip_telemetry
from luxonis_ml.telemetry.events import TelemetryEvent
from luxonis_ml.telemetry.redaction import sanitize_properties
from luxonis_ml.telemetry.singleton import _singleton_state, _telemetry_by_key


class DummyBackend(TelemetryBackend):
    def __init__(self) -> None:
        super().__init__(TelemetryConfig(enabled=True, backend="dummy"))
        self.events: list[Any] = []
        self.flush_count = 0
        self.shutdown_count = 0

    def capture(self, event: Any) -> None:
        self.events.append(event)

    def flush(self) -> None:
        self.flush_count += 1

    def shutdown(self) -> None:
        self.shutdown_count += 1


@pytest.fixture
def reset_backend_registry() -> Generator[None, None, None]:
    original = dict(TELEMETRY_BACKENDS._module_dict)
    yield
    TELEMETRY_BACKENDS._module_dict = original


@pytest.fixture
def dummy_backend(
    reset_backend_registry: Generator[None, None, None],
) -> DummyBackend:
    backend = DummyBackend()

    class RegisteredDummyBackend(TelemetryBackend):
        def __init__(self, config: TelemetryConfig) -> None:
            super().__init__(config)

        def capture(self, event: Any) -> None:
            backend.capture(event)

        def flush(self) -> None:
            backend.flush()

        def shutdown(self) -> None:
            backend.shutdown()

    Telemetry.register_backend("dummy", RegisteredDummyBackend)
    return backend


@pytest.fixture(autouse=True)
def reset_singletons() -> Generator[None, None, None]:
    _telemetry_by_key.clear()
    _singleton_state["exit_handler_registered"] = False
    yield
    _telemetry_by_key.clear()
    _singleton_state["exit_handler_registered"] = False


def _make_typer_app() -> Any:
    typer_module = import_module("typer")
    return typer_module.Typer()


def _invoke_typer(app: Any, args: list[str]) -> Any:
    testing_module = import_module("typer.testing")
    runner = testing_module.CliRunner()
    return runner.invoke(app, args)


def _invoke_cyclopts(app: App, args: list[str]) -> Any:
    try:
        return app(args, exit_on_error=False)
    except SystemExit as exc:
        # Newer Cyclopts releases can treat integer return values as exit
        # codes and raise SystemExit even with exit_on_error=False.
        if isinstance(exc.code, int):
            return exc.code
        raise


def test_config_from_environ(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LUXONIS_TELEMETRY_ENABLED", "1")
    monkeypatch.setenv("LUXONIS_TELEMETRY_BACKEND", "stdout")
    monkeypatch.setenv("LUXONIS_TELEMETRY_API_KEY", "secret")
    monkeypatch.setenv("LUXONIS_TELEMETRY_ENDPOINT", "https://example")
    monkeypatch.setenv("LUXONIS_TELEMETRY_DEBUG", "1")
    monkeypatch.setenv("LUXONIS_TELEMETRY_IS_LUXONIS_CLOUD", "1")

    cfg = TelemetryConfig.from_environ()
    assert cfg.enabled is True
    assert cfg.backend == "stdout"
    assert cfg.api_key == "secret"
    assert cfg.endpoint == "https://example"
    assert cfg.debug is True


def test_config_from_environ_uses_product_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LUXONIS_TELEMETRY_ENABLED", raising=False)
    monkeypatch.delenv("LUXONIS_TELEMETRY_BACKEND", raising=False)
    monkeypatch.delenv("LUXONIS_TELEMETRY_API_KEY", raising=False)
    monkeypatch.delenv("LUXONIS_TELEMETRY_ENDPOINT", raising=False)
    monkeypatch.delenv("LUXONIS_TELEMETRY_DEBUG", raising=False)

    cfg = TelemetryConfig.from_environ(
        defaults=TelemetryDefaults(
            backend="dummy",
            api_key="default-secret",
            endpoint="https://default.example",
            include_system_metadata=True,
        )
    )

    assert cfg.enabled is True
    assert cfg.backend == "dummy"
    assert cfg.api_key == "default-secret"
    assert cfg.endpoint == "https://default.example"
    assert cfg.debug is False
    assert cfg.include_system_metadata is True


def test_config_from_environ_env_overrides_product_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LUXONIS_TELEMETRY_ENABLED", "0")
    monkeypatch.setenv("LUXONIS_TELEMETRY_BACKEND", "stdout")
    monkeypatch.setenv("LUXONIS_TELEMETRY_API_KEY", "env-secret")
    monkeypatch.setenv("LUXONIS_TELEMETRY_ENDPOINT", "https://env.example")
    monkeypatch.setenv("LUXONIS_TELEMETRY_DEBUG", "1")

    cfg = TelemetryConfig.from_environ(
        defaults=TelemetryDefaults(
            enabled=True,
            backend="dummy",
            api_key="default-secret",
            endpoint="https://default.example",
            debug=False,
        )
    )

    assert cfg.enabled is False
    assert cfg.backend == "stdout"
    assert cfg.api_key == "env-secret"
    assert cfg.endpoint == "https://env.example"
    assert cfg.debug is True


def test_config_from_environ_uses_debug_default_for_backend() -> None:
    cfg = TelemetryConfig.from_environ(defaults=TelemetryDefaults(debug=True))

    assert cfg.debug is True
    assert cfg.backend == "stdout"


def test_config_from_environ_reads_dotenv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LUXONIS_TELEMETRY_API_KEY", raising=False)
    monkeypatch.delenv("LUXONIS_TELEMETRY_BACKEND", raising=False)
    monkeypatch.delenv("LUXONIS_TELEMETRY_DEBUG", raising=False)
    (tmp_path / ".env").write_text(
        "LUXONIS_TELEMETRY_DEBUG=true\n"
        "LUXONIS_TELEMETRY_API_KEY=dotenv-secret\n",
        encoding="utf-8",
    )

    cfg = TelemetryConfig.from_environ()

    assert cfg.debug is True
    assert cfg.backend == "stdout"
    assert cfg.api_key == "dotenv-secret"


def test_capture_includes_context(
    dummy_backend: DummyBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LUXONIS_TELEMETRY_IS_LUXONIS_CLOUD", "1")
    config = TelemetryConfig(
        enabled=True,
        backend="dummy",
        include_system_metadata=True,
    )

    def my_context(_telemetry: Telemetry) -> dict[str, str]:
        return {"custom": "value"}

    telemetry = Telemetry(
        "luxonis_ml",
        config=config,
        context_providers=[my_context],
        system_context_providers=[telemetry_context.system_context_provider],
    )

    telemetry.capture("event_test", {"foo": "bar"})

    assert len(dummy_backend.events) == 1
    event = dummy_backend.events[0]
    assert event.name == "event_test"
    assert event.properties["foo"] == "bar"
    assert event.context["custom"] == "value"
    assert "cpu_count" in event.context
    assert (
        event.context["$session_id"] == telemetry._base_context["$session_id"]
    )
    assert event.context["$process_person_profile"] is False
    assert event.context["source_product"] == "luxonis_ml"
    assert event.context["source_component"] == "luxonis_ml"
    assert event.context["is_luxonis_cloud"] is True
    assert event.distinct_id == telemetry._session_id


def test_capture_supports_distinct_id_override(
    dummy_backend: DummyBackend,
) -> None:
    config = TelemetryConfig(enabled=True, backend="dummy")
    telemetry = Telemetry("luxonis_ml", config=config)

    telemetry.capture(
        "event_test",
        {"foo": "bar"},
        distinct_id="conversion-run-123",
    )

    event = dummy_backend.events[-1]
    assert event.distinct_id == "conversion-run-123"
    assert (
        event.context["$session_id"] == telemetry._base_context["$session_id"]
    )


def test_include_system_metadata_flag(dummy_backend: DummyBackend) -> None:
    config = TelemetryConfig(enabled=True, backend="dummy")
    telemetry = Telemetry(
        "luxonis_ml",
        config=config,
        system_context_providers=[telemetry_context.system_context_provider],
    )

    telemetry.capture("event_no_system")
    event = dummy_backend.events[-1]
    assert "cpu_count" not in event.context

    telemetry.capture("event_with_system", include_system_metadata=True)
    event = dummy_backend.events[-1]
    assert "cpu_count" in event.context


def test_system_metadata_normalizes_processor_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        telemetry_context.platform,
        "machine",
        lambda: "AMD64",
    )
    monkeypatch.setattr(
        telemetry_context.platform,
        "processor",
        lambda: "Intel(R) Core(TM) i7-1185G7 CPU @ 3.00GHz",
    )

    system = telemetry_context.system_context()

    assert system["processor"] == "x86_64"


def test_base_context_can_be_disabled(dummy_backend: DummyBackend) -> None:
    config = TelemetryConfig(
        enabled=True,
        backend="dummy",
        include_base_context=False,
    )
    telemetry = Telemetry("luxonis_ml", config=config)

    telemetry.capture("event_no_base")

    event = dummy_backend.events[-1]
    assert event.context == {}
    assert event.distinct_id == telemetry._session_id


def test_host_context_is_opt_in(dummy_backend: DummyBackend) -> None:
    config = TelemetryConfig(enabled=True, backend="dummy")
    telemetry = Telemetry("luxonis_ml", config=config)

    telemetry.capture("event_without_host")

    event = dummy_backend.events[-1]
    assert "os" not in event.context
    assert "python_version" not in event.context


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

    telemetry.capture("event_no_system")
    event = dummy_backend.events[-1]
    assert "runtime" not in event.context

    telemetry.capture("event_with_system", include_system_metadata=True)
    event = dummy_backend.events[-1]
    assert event.context["runtime"] == "docker"


def test_suppression_skips_capture(dummy_backend: DummyBackend) -> None:
    config = TelemetryConfig(enabled=True, backend="dummy")
    telemetry = Telemetry("luxonis_ml", config=config)

    telemetry.capture("event_one")
    assert len(dummy_backend.events) == 1

    with suppress_telemetry():
        telemetry.capture("event_suppressed")

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


def test_sanitize_properties_blocks_nested_identifier_keys() -> None:
    props = {
        "config": {
            "user_id": "user-123",
            "hostname": "devbox",
            "nested": {"team_id": "team-123"},
        }
    }

    out = sanitize_properties(props)

    assert out["config"]["user_id"] == "<redacted>"
    assert out["config"]["hostname"] == "<redacted>"
    assert out["config"]["nested"]["team_id"] == "<redacted>"


def test_sanitize_properties_redacts_paths_urls_and_free_text() -> None:
    props = {
        "path": "../private/file.txt",
        "homepage": "https://example.com/model",
        "note": "this is arbitrary user text",
        "command": "data ls",
    }

    out = sanitize_properties(props)

    assert out["path"] == "<redacted>"
    assert out["homepage"] == "<redacted>"
    assert out["note"] == "<string>"
    assert out["command"] == "data ls"


def test_default_telemetry_uses_ephemeral_session_id(
    reset_backend_registry: Generator[None, None, None],
) -> None:
    config = TelemetryConfig(
        enabled=True,
        backend="noop",
    )
    telemetry = Telemetry("luxonis_ml", config=config)
    assert "install_id" not in telemetry._base_context
    assert telemetry._base_context["$session_id"] == telemetry._session_id


def test_telemetry_event_payload_includes_distinct_id_field() -> None:
    payload = TelemetryEvent.create(
        name="event_test",
        properties={"foo": "bar"},
        context={"$session_id": "session-1"},
        distinct_id="distinct-1",
    ).to_payload()

    assert payload["distinct_id"] == "distinct-1"


def test_source_component_can_be_overridden(
    reset_backend_registry: Generator[None, None, None],
) -> None:
    config = TelemetryConfig(
        enabled=True,
        backend="noop",
    )
    telemetry = Telemetry(
        "luxonis_ml",
        source_component="luxonis_ml_cli",
        config=config,
    )
    assert telemetry._base_context["source_component"] == "luxonis_ml_cli"


def test_context_utilities_expose_ready_to_use_metadata() -> None:
    host = telemetry_context.host_context()
    system = telemetry_context.system_context()

    assert "os" in host
    assert "python_version" in host
    assert "cpu_count" in system
    assert "is_docker" in system


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


def test_singleton_registry_multiple_components_same_library(
    dummy_backend: DummyBackend,
) -> None:
    config = TelemetryConfig(enabled=True, backend="dummy")
    cli = initialize_telemetry(
        library_name="lib_a",
        source_component="lib_a_cli",
        config=config,
        register_exit_handler=False,
    )
    runtime = initialize_telemetry(
        library_name="lib_a",
        source_component="lib_a_runtime",
        config=config,
        register_exit_handler=False,
    )

    assert cli is not runtime
    assert get_telemetry("lib_a", source_component="lib_a_cli") is cli
    assert get_telemetry("lib_a", source_component="lib_a_runtime") is runtime
    assert get_telemetry("lib_a") is None


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


def test_get_or_init_reuses_existing_component_instance(
    dummy_backend: DummyBackend,
) -> None:
    config = TelemetryConfig(enabled=True, backend="dummy")
    t1 = initialize_telemetry(
        library_name="lib_a",
        source_component="lib_a_cli",
        config=config,
        register_exit_handler=False,
    )
    t2 = get_or_init(
        library_name="lib_a",
        source_component="lib_a_cli",
        config=TelemetryConfig(enabled=True, backend="noop"),
        register_exit_handler=False,
    )

    assert t1 is t2
    assert t2.source_component == "lib_a_cli"
    assert t2.config.backend == "dummy"


def test_get_or_init_is_thread_safe() -> None:
    original_telemetry = telemetry_singleton.Telemetry
    created_count = 0
    created_count_lock = threading.Lock()
    start_event = threading.Event()
    config = TelemetryConfig(enabled=False, backend="noop")

    class SlowTelemetry(original_telemetry):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            nonlocal created_count
            time.sleep(0.05)
            with created_count_lock:
                created_count += 1
            super().__init__(*args, **kwargs)

    telemetry_singleton.Telemetry = SlowTelemetry
    try:
        results: list[int] = []

        def worker() -> None:
            start_event.wait()
            telemetry = get_or_init(
                "threaded_lib",
                config=config,
                register_exit_handler=False,
            )
            results.append(id(telemetry))

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        start_event.set()
        for thread in threads:
            thread.join()
    finally:
        telemetry_singleton.Telemetry = original_telemetry

    assert created_count == 1
    assert len(set(results)) == 1
    assert len(_telemetry_by_key) == 1


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

    reused.capture("event_test")

    assert reused is telemetry
    event = dummy_backend.events[-1]
    assert event.context["first"] == "value"
    assert event.context["second"] == "value"


def test_registering_custom_backend_keeps_builtin_backends(
    reset_backend_registry: Generator[None, None, None],
) -> None:
    class RegisteredDummyBackend(TelemetryBackend):
        def capture(self, event: Any) -> None:
            return

    Telemetry.register_backend("dummy", RegisteredDummyBackend)

    telemetry = Telemetry(
        "luxonis_ml",
        config=TelemetryConfig(enabled=True, backend="noop"),
    )

    assert telemetry._backend.__class__.__name__ == "NoopBackend"


def test_register_backend_is_case_insensitive(
    reset_backend_registry: Generator[None, None, None],
) -> None:
    backend = DummyBackend()

    class RegisteredDummyBackend(TelemetryBackend):
        def __init__(self, config: TelemetryConfig) -> None:
            super().__init__(config)

        def capture(self, event: Any) -> None:
            backend.capture(event)

    Telemetry.register_backend("Dummy", RegisteredDummyBackend)

    telemetry = Telemetry(
        "luxonis_ml",
        config=TelemetryConfig(enabled=True, backend="dummy"),
    )
    telemetry.capture("event_test")

    assert telemetry._backend.__class__ is RegisteredDummyBackend
    assert len(backend.events) == 1


def test_instrument_typer_emits_event(dummy_backend: DummyBackend) -> None:
    config = TelemetryConfig(enabled=True, backend="dummy")
    telemetry = Telemetry("luxonis_ml", config=config)

    app = _make_typer_app()

    @app.command()
    def train(epochs: int = 10) -> int:
        return epochs

    from luxonis_ml.telemetry.cli import instrument_typer

    instrument_typer(app, telemetry)

    cmd = app.registered_commands[0].callback
    result = cmd(epochs=5)
    assert result == 5
    event = dummy_backend.events[-1]
    assert event.name == "cli_command"
    assert event.properties["command"] == "train"
    assert "epochs" not in event.properties


def test_instrument_typer_uses_cli_visible_command_name(
    dummy_backend: DummyBackend,
) -> None:
    config = TelemetryConfig(enabled=True, backend="dummy")
    telemetry = Telemetry("luxonis_ml", config=config)

    app = _make_typer_app()

    @app.command()
    def my_command() -> int:
        return 1

    @app.command()
    def other_command() -> int:
        return 2

    from luxonis_ml.telemetry.cli import instrument_typer

    instrument_typer(app, telemetry)

    result = _invoke_typer(app, ["my-command"])

    assert result.exit_code == 0
    event = dummy_backend.events[-1]
    assert event.properties["command"] == "my-command"


def test_instrument_typer_allowlist_keeps_core_fields(
    dummy_backend: DummyBackend,
) -> None:
    config = TelemetryConfig(enabled=True, backend="dummy")
    telemetry = Telemetry("luxonis_ml", config=config)

    app = _make_typer_app()

    @app.command()
    def train(epochs: int = 10, dataset_name: str = "private") -> int:
        return epochs

    from luxonis_ml.telemetry.cli import instrument_typer

    instrument_typer(app, telemetry, allowlist={"epochs"})

    cmd = app.registered_commands[0].callback
    cmd(epochs=5, dataset_name="secret")

    event = dummy_backend.events[-1]
    assert event.properties["command"] == "train"
    assert event.properties["epochs"] == 5
    assert "dataset_name" not in event.properties
    assert "success" in event.properties
    assert "duration_ms" in event.properties


def test_instrument_typer_exclude_commands(
    dummy_backend: DummyBackend,
) -> None:
    config = TelemetryConfig(enabled=True, backend="dummy")
    telemetry = Telemetry("luxonis_ml", config=config)

    app = _make_typer_app()

    @app.command()
    def train(epochs: int = 10) -> int:
        return epochs

    @app.command()
    def evaluate() -> int:
        return 1

    from luxonis_ml.telemetry.cli import instrument_typer

    instrument_typer(app, telemetry, exclude_commands={"train"})

    train_cmd = app.registered_commands[0].callback
    evaluate_cmd = app.registered_commands[1].callback
    train_cmd(epochs=5)
    evaluate_cmd()

    assert len(dummy_backend.events) == 1
    event = dummy_backend.events[0]
    assert event.name == "cli_command"
    assert event.properties["command"] == "evaluate"


def test_instrument_typer_exclude_commands_uses_cli_name(
    dummy_backend: DummyBackend,
) -> None:
    config = TelemetryConfig(enabled=True, backend="dummy")
    telemetry = Telemetry("luxonis_ml", config=config)

    app = _make_typer_app()

    @app.command()
    def my_command() -> int:
        return 1

    @app.command()
    def visible_command() -> int:
        return 2

    from luxonis_ml.telemetry.cli import instrument_typer

    instrument_typer(app, telemetry, exclude_commands={"my-command"})

    hidden = _invoke_typer(app, ["my-command"])
    visible = _invoke_typer(app, ["visible-command"])

    assert hidden.exit_code == 0
    assert visible.exit_code == 0
    assert len(dummy_backend.events) == 1
    assert dummy_backend.events[0].properties["command"] == "visible-command"


def test_instrument_typer_skip_decorator(
    dummy_backend: DummyBackend,
) -> None:
    config = TelemetryConfig(enabled=True, backend="dummy")
    telemetry = Telemetry("luxonis_ml", config=config)

    app = _make_typer_app()

    @skip_telemetry
    @app.command()
    def secret() -> int:
        return 0

    @app.command()
    def visible() -> int:
        return 1

    from luxonis_ml.telemetry.cli import instrument_typer

    instrument_typer(app, telemetry)

    secret_cmd = app.registered_commands[0].callback
    visible_cmd = app.registered_commands[1].callback
    secret_cmd()
    visible_cmd()

    assert len(dummy_backend.events) == 1
    event = dummy_backend.events[0]
    assert event.name == "cli_command"
    assert event.properties["command"] == "visible"


def test_instrument_cyclopts_emits_event(dummy_backend: DummyBackend) -> None:
    config = TelemetryConfig(enabled=True, backend="dummy")
    telemetry = Telemetry("luxonis_ml", config=config)

    app = App(name="demo")

    @app.command
    def train(epochs: int = 10) -> int:
        return epochs

    from luxonis_ml.telemetry.cli import instrument_cyclopts

    instrument_cyclopts(app, telemetry)

    result = _invoke_cyclopts(app, ["train", "--epochs", "5"])
    assert result == 5
    event = dummy_backend.events[-1]
    assert event.name == "cli_command"
    assert event.properties["command"] == "train"
    assert "epochs" not in event.properties


def test_instrument_cyclopts_allowlist_keeps_core_fields(
    dummy_backend: DummyBackend,
) -> None:
    config = TelemetryConfig(enabled=True, backend="dummy")
    telemetry = Telemetry("luxonis_ml", config=config)

    app = App(name="demo")

    @app.command
    def train(epochs: int = 10, dataset_name: str = "private") -> int:
        return epochs

    from luxonis_ml.telemetry.cli import instrument_cyclopts

    instrument_cyclopts(app, telemetry, allowlist={"epochs"})

    _invoke_cyclopts(
        app,
        ["train", "--epochs", "5", "--dataset-name", "secret"],
    )

    event = dummy_backend.events[-1]
    assert event.properties["command"] == "train"
    assert event.properties["epochs"] == 5
    assert "dataset_name" not in event.properties
    assert "success" in event.properties
    assert "duration_ms" in event.properties


def test_instrument_cyclopts_exclude_commands(
    dummy_backend: DummyBackend,
) -> None:
    config = TelemetryConfig(enabled=True, backend="dummy")
    telemetry = Telemetry("luxonis_ml", config=config)

    app = App(name="demo")

    @app.command
    def train(epochs: int = 10) -> int:
        return epochs

    @app.command
    def evaluate() -> int:
        return 1

    from luxonis_ml.telemetry.cli import instrument_cyclopts

    instrument_cyclopts(app, telemetry, exclude_commands={"train"})

    _invoke_cyclopts(app, ["train", "--epochs", "5"])
    _invoke_cyclopts(app, ["evaluate"])

    assert len(dummy_backend.events) == 1
    event = dummy_backend.events[0]
    assert event.name == "cli_command"
    assert event.properties["command"] == "evaluate"


def test_instrument_cyclopts_skip_decorator(
    dummy_backend: DummyBackend,
) -> None:
    config = TelemetryConfig(enabled=True, backend="dummy")
    telemetry = Telemetry("luxonis_ml", config=config)

    app = App(name="demo")

    @app.command
    @skip_telemetry
    def secret() -> int:
        return 0

    @app.command
    def visible() -> int:
        return 1

    from luxonis_ml.telemetry.cli import instrument_cyclopts

    instrument_cyclopts(app, telemetry)

    _invoke_cyclopts(app, ["secret"])
    _invoke_cyclopts(app, ["visible"])

    assert len(dummy_backend.events) == 1
    event = dummy_backend.events[0]
    assert event.name == "cli_command"
    assert event.properties["command"] == "visible"


def test_instrument_cyclopts_nested_subapps(
    dummy_backend: DummyBackend,
) -> None:
    config = TelemetryConfig(enabled=True, backend="dummy")
    telemetry = Telemetry("luxonis_ml", config=config)

    app = App(name="demo")
    data_app = App(name="data")

    @data_app.command
    def ls(full: bool = False) -> bool:
        return full

    app.command(data_app, name="data")

    from luxonis_ml.telemetry.cli import instrument_cyclopts

    instrument_cyclopts(app, telemetry, allowlist={"full"})

    _invoke_cyclopts(app, ["data", "ls", "--full"])

    event = dummy_backend.events[-1]
    assert event.name == "cli_command"
    assert event.properties["command"] == "data ls"
    assert event.properties["full"] is True


def test_capture_preserves_explicit_empty_allowlist(
    dummy_backend: DummyBackend,
) -> None:
    config = TelemetryConfig(
        enabled=True,
        backend="dummy",
        allowlist={"keep"},
    )
    telemetry = Telemetry("luxonis_ml", config=config)

    telemetry.capture(
        "event_test",
        {"keep": 1, "drop": 2},
        allowlist=set(),
    )

    event = dummy_backend.events[-1]
    assert event.properties == {}
