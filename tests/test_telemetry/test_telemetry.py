import inspect
import sys
import threading
import time
from collections.abc import Generator
from importlib import import_module
from pathlib import Path
from typing import Any, cast

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
from luxonis_ml.telemetry import client as telemetry_client
from luxonis_ml.telemetry import context as telemetry_context
from luxonis_ml.telemetry.backends.base import (
    TELEMETRY_BACKENDS,
    TelemetryBackend,
)
from luxonis_ml.telemetry.backends.noop import NoopBackend
from luxonis_ml.telemetry.backends.posthog import (
    PostHogBackend,
    _merge_properties,
)
from luxonis_ml.telemetry.backends.stdout import StdoutBackend
from luxonis_ml.telemetry.cli import skip_telemetry
from luxonis_ml.telemetry.cli.cyclopts import (
    _is_builtin_cyclopts_command,
    _iter_unique_subapps,
    _primary_name,
)
from luxonis_ml.telemetry.cli.shared import (
    extract_params,
    is_click_context,
    join_command,
    wrap_command_callback,
)
from luxonis_ml.telemetry.cli.typer import _resolve_command_name
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
    assert cfg.disable_geoip is False
    assert cfg.allow_reserved_overrides is False


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
            disable_geoip=True,
            allow_reserved_overrides=True,
        )
    )

    assert cfg.enabled is True
    assert cfg.backend == "dummy"
    assert cfg.api_key == "default-secret"
    assert cfg.endpoint == "https://default.example"
    assert cfg.debug is False
    assert cfg.include_system_metadata is True
    assert cfg.disable_geoip is True
    assert cfg.allow_reserved_overrides is True


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
            disable_geoip=True,
            allow_reserved_overrides=True,
        )
    )

    assert cfg.enabled is False
    assert cfg.backend == "stdout"
    assert cfg.api_key == "env-secret"
    assert cfg.endpoint == "https://env.example"
    assert cfg.debug is True
    assert cfg.disable_geoip is True
    assert cfg.allow_reserved_overrides is True


def test_config_from_environ_uses_debug_default_for_backend() -> None:
    cfg = TelemetryConfig.from_environ(defaults=TelemetryDefaults(debug=True))

    assert cfg.debug is True
    assert cfg.backend == "stdout"
    assert cfg.disable_geoip is False
    assert cfg.allow_reserved_overrides is False


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


def test_context_providers_cannot_override_protected_base_context(
    dummy_backend: DummyBackend,
) -> None:
    config = TelemetryConfig(enabled=True, backend="dummy")

    def overriding_context(_telemetry: Telemetry) -> dict[str, Any]:
        return {
            "$process_person_profile": True,
            "$session_id": "override-session",
            "source_product": "other_product",
            "source_component": "other_component",
            "sdk_version": "0.0.0",
            "custom": "value",
        }

    telemetry = Telemetry(
        "luxonis_ml",
        source_component="cli",
        library_version="1.2.3",
        config=config,
        context_providers=[overriding_context],
    )

    telemetry.capture("event_test")

    event = dummy_backend.events[-1]
    assert event.context["$process_person_profile"] is False
    assert event.context["$session_id"] == telemetry._session_id
    assert event.context["source_product"] == "luxonis_ml"
    assert event.context["source_component"] == "cli"
    assert event.context["sdk_version"] == "1.2.3"
    assert event.context["custom"] == "value"


def test_context_providers_can_override_reserved_fields_when_enabled(
    dummy_backend: DummyBackend,
) -> None:
    config = TelemetryConfig(
        enabled=True,
        backend="dummy",
        allow_reserved_overrides=True,
    )

    def overriding_context(_telemetry: Telemetry) -> dict[str, Any]:
        return {
            "$process_person_profile": True,
            "$session_id": "override-session",
            "source_product": "other_product",
            "source_component": "other_component",
            "sdk_version": "0.0.0",
        }

    telemetry = Telemetry(
        "luxonis_ml",
        source_component="cli",
        library_version="1.2.3",
        config=config,
        context_providers=[overriding_context],
    )

    telemetry.capture("event_test")

    event = dummy_backend.events[-1]
    assert event.context["$process_person_profile"] is True
    assert event.context["$session_id"] == "override-session"
    assert event.context["source_product"] == "other_product"
    assert event.context["source_component"] == "other_component"
    assert event.context["sdk_version"] == "0.0.0"


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


@pytest.mark.parametrize(
    ("machine", "processor", "expected"),
    [
        ("aarch64", "", "arm64"),
        ("armv7l", "", "arm"),
        ("", "i686", "x86"),
        ("powerpc64", "", "powerpc"),
        ("", "riscv64", "riscv"),
        ("mips", "", "unknown"),
    ],
)
def test_normalized_processor_variants(
    monkeypatch: pytest.MonkeyPatch,
    machine: str,
    processor: str,
    expected: str,
) -> None:
    monkeypatch.setattr(telemetry_context.platform, "machine", lambda: machine)
    monkeypatch.setattr(
        telemetry_context.platform, "processor", lambda: processor
    )

    assert telemetry_context.normalized_processor() == expected


def test_is_ci_detects_known_markers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in [
        "CI",
        "GITHUB_ACTIONS",
        "GITLAB_CI",
        "BUILDKITE",
        "JENKINS_URL",
        "TEAMCITY_VERSION",
    ]:
        monkeypatch.delenv(key, raising=False)

    assert telemetry_context.is_ci() is False

    monkeypatch.setenv("GITHUB_ACTIONS", "1")
    assert telemetry_context.is_ci() is True


def test_base_context_uses_library_name_when_component_missing() -> None:
    context = telemetry_context.base_context(
        library_name="luxonis_ml",
        library_version="1.2.3",
        session_id="session-1",
        source_component=None,
    )

    assert context["source_component"] == "luxonis_ml"


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


def test_telemetry_accessors_and_missing_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(telemetry_client, "_safe_version", lambda _: None)
    telemetry = Telemetry(
        "luxonis_ml",
        source_component="cli",
        config=TelemetryConfig(enabled=False, backend="noop"),
    )

    assert telemetry.config.backend == "noop"
    assert telemetry.library_name == "luxonis_ml"
    assert telemetry.library_version is None
    assert telemetry.source_component == "cli"
    assert telemetry.is_enabled is False


def test_capture_noops_when_disabled(dummy_backend: DummyBackend) -> None:
    telemetry = Telemetry(
        "luxonis_ml",
        config=TelemetryConfig(enabled=False, backend="dummy"),
    )

    telemetry.capture("event_test")

    assert dummy_backend.events == []


def test_capture_handles_backend_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    telemetry = Telemetry(
        "luxonis_ml",
        config=TelemetryConfig(enabled=True, backend="noop"),
    )
    backend = telemetry._backend
    monkeypatch.setattr(
        backend,
        "capture",
        lambda event: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    telemetry.capture("event_test")


def test_flush_and_shutdown_handle_disabled_and_backend_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    disabled = Telemetry(
        "luxonis_ml",
        config=TelemetryConfig(enabled=False, backend="noop"),
    )
    disabled.flush()
    disabled.shutdown()

    telemetry = Telemetry(
        "luxonis_ml",
        config=TelemetryConfig(enabled=True, backend="noop"),
    )
    backend = telemetry._backend
    monkeypatch.setattr(
        backend, "flush", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    monkeypatch.setattr(
        backend,
        "shutdown",
        lambda: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    telemetry.flush()
    telemetry.shutdown()


def test_init_backend_falls_back_to_noop_for_unknown_backend() -> None:
    telemetry = Telemetry(
        "luxonis_ml",
        config=TelemetryConfig(enabled=True, backend="missing"),
    )

    assert isinstance(telemetry._backend, NoopBackend)


def test_init_backend_falls_back_to_noop_when_backend_init_fails(
    reset_backend_registry: Generator[None, None, None],
) -> None:
    class BrokenBackend(TelemetryBackend):
        def __init__(self, config: TelemetryConfig) -> None:
            super().__init__(config)
            raise RuntimeError("broken")

        def capture(self, event: Any) -> None:
            return

    Telemetry.register_backend("broken", BrokenBackend)

    telemetry = Telemetry(
        "luxonis_ml",
        config=TelemetryConfig(enabled=True, backend="broken"),
    )

    assert isinstance(telemetry._backend, NoopBackend)


def test_context_providers_skip_failures_none_and_non_mappings(
    dummy_backend: DummyBackend,
) -> None:
    telemetry = Telemetry(
        "luxonis_ml",
        config=TelemetryConfig(enabled=True, backend="dummy"),
    )

    def broken_provider(_telemetry: Telemetry) -> dict[str, str]:
        raise RuntimeError("broken")

    def none_provider(_telemetry: Telemetry) -> None:
        return None

    def non_mapping_provider(_telemetry: Telemetry) -> str:
        return "invalid"

    def valid_provider(_telemetry: Telemetry) -> dict[str, str]:
        return {"kept": "value"}

    telemetry.add_context_provider(broken_provider)
    telemetry.add_context_provider(cast(Any, none_provider))
    telemetry.add_context_provider(cast(Any, non_mapping_provider))
    telemetry.add_context_provider(valid_provider)
    telemetry.add_context_provider(valid_provider)

    telemetry.capture("event_test")

    event = dummy_backend.events[-1]
    assert event.context["kept"] == "value"


def test_safe_version_returns_none_for_expected_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        telemetry_client,
        "version",
        lambda _: (_ for _ in ()).throw(telemetry_client.PackageNotFoundError),
    )
    assert telemetry_client._safe_version("luxonis_ml") is None

    monkeypatch.setattr(
        telemetry_client,
        "version",
        lambda _: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert telemetry_client._safe_version("luxonis_ml") is None


def test_telemetry_backend_base_flush_and_shutdown() -> None:
    backend = DummyBackend()
    TelemetryBackend.flush(backend)
    TelemetryBackend.shutdown(backend)

    assert backend.flush_count == 1


def test_stdout_backend_capture_outputs_payload(capsys: Any) -> None:
    backend = StdoutBackend(TelemetryConfig(enabled=True, backend="stdout"))
    event = TelemetryEvent.create(
        name="event_test",
        properties={"value": 1},
        context={"$session_id": "session-1"},
    )

    backend.capture(event)
    backend.flush()

    out = capsys.readouterr().out
    assert "event_test" in out
    assert '"value": 1' in out


def test_posthog_backend_requires_api_key() -> None:
    with pytest.raises(ValueError, match="API key"):
        PostHogBackend(TelemetryConfig(enabled=True, backend="posthog"))


def test_posthog_backend_raises_when_package_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "posthog", None)

    with pytest.raises(ImportError, match="requires the 'posthog' package"):
        PostHogBackend(
            TelemetryConfig(
                enabled=True,
                backend="posthog",
                api_key="secret",
            )
        )

    monkeypatch.delitem(sys.modules, "posthog", raising=False)


def test_posthog_backend_capture_and_flush(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}

    class FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            calls["kwargs"] = kwargs

        def capture(self, **kwargs: Any) -> None:
            calls["capture"] = kwargs

        def flush(self) -> None:
            calls["flushed"] = True

    monkeypatch.setitem(
        sys.modules,
        "posthog",
        cast(Any, type("FakePosthogModule", (), {"Posthog": FakeClient})()),
    )

    backend = PostHogBackend(
        TelemetryConfig(
            enabled=True,
            backend="posthog",
            api_key="secret",
            endpoint="https://ph.example",
        )
    )
    event = TelemetryEvent.create(
        name="event_test",
        properties={"value": 1},
        context={"$session_id": "session-1", "source_product": "luxonis_ml"},
        distinct_id=None,
    )

    backend.capture(event)
    backend.flush()

    assert calls["kwargs"]["project_api_key"] == "secret"
    assert calls["kwargs"]["disable_geoip"] is False
    assert calls["kwargs"]["host"] == "https://ph.example"
    assert calls["capture"]["distinct_id"] == "session-1"
    assert calls["capture"]["event"] == "event_test"
    assert calls["capture"]["properties"]["schema_version"] == 1
    assert calls["capture"]["properties"]["value"] == 1
    assert calls["flushed"] is True


def test_merge_properties_merges_schema_context_and_properties() -> None:
    event = TelemetryEvent.create(
        name="event_test",
        properties={"value": 1},
        context={"$session_id": "session-1", "value": 2},
    )

    merged = _merge_properties(event)

    assert merged["schema_version"] == 1
    assert merged["$session_id"] == "session-1"
    assert merged["value"] == 1


def test_merge_properties_preserves_reserved_posthog_keys() -> None:
    event = TelemetryEvent.create(
        name="event_test",
        properties={
            "$process_person_profile": True,
            "$session_id": "override-session",
            "schema_version": 99,
        },
        context={
            "$process_person_profile": False,
            "$session_id": "session-1",
        },
    )

    merged = _merge_properties(event)

    assert merged["$process_person_profile"] is False
    assert merged["$session_id"] == "session-1"
    assert merged["schema_version"] == 1


def test_merge_properties_allows_reserved_overrides_when_enabled() -> None:
    event = TelemetryEvent.create(
        name="event_test",
        properties={
            "$process_person_profile": True,
            "$session_id": "override-session",
            "schema_version": 99,
        },
        context={
            "$process_person_profile": False,
            "$session_id": "session-1",
        },
    )

    merged = _merge_properties(event, allow_reserved_overrides=True)

    assert merged["$process_person_profile"] is True
    assert merged["$session_id"] == "override-session"
    assert merged["schema_version"] == 99


def test_wrap_command_callback_records_failures(
    dummy_backend: DummyBackend,
) -> None:
    telemetry = Telemetry(
        "luxonis_ml",
        config=TelemetryConfig(enabled=True, backend="dummy"),
    )

    def failing(epochs: int) -> None:
        raise RuntimeError("boom")

    wrapped = wrap_command_callback(
        failing,
        telemetry,
        "train",
        allowlist={"epochs"},
        include_system_metadata=None,
    )

    with pytest.raises(RuntimeError, match="boom"):
        wrapped(epochs=5)

    event = dummy_backend.events[-1]
    assert event.properties["command"] == "train"
    assert event.properties["epochs"] == 5
    assert event.properties["success"] is False


def test_extract_params_type_error_and_click_context() -> None:
    def command(ctx: Any, epochs: int, ignored: int) -> None:
        return

    signature = inspect.signature(command)
    fake_ctx = type(
        "FakeContext",
        (),
        {"info_name": "train", "command": object()},
    )()

    assert extract_params(signature, (), {}, allowlist=None) == {}
    assert extract_params(signature, (), {}, allowlist={"epochs"}) == {}
    assert extract_params(
        signature,
        (fake_ctx, 5, 7),
        {},
        allowlist={"ctx", "epochs", "ignored"},
    ) == {"epochs": 5, "ignored": 7}
    assert is_click_context(fake_ctx) is True
    assert is_click_context(object()) is False


def test_join_command_and_name_resolvers() -> None:
    assert join_command("", "train") == "train"
    assert join_command("data", "") == "data"
    assert join_command("data", "export") == "data export"
    assert _resolve_command_name("visible", lambda: None) == "visible"


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


def test_instrument_typer_skips_missing_callbacks_and_groups(
    dummy_backend: DummyBackend,
) -> None:
    config = TelemetryConfig(enabled=True, backend="dummy")
    telemetry = Telemetry("luxonis_ml", config=config)
    app = _make_typer_app()
    sub = _make_typer_app()

    @app.command()
    def root() -> int:
        return 1

    @sub.command()
    def nested() -> int:
        return 2

    app.add_typer(sub, name="admin")
    app.registered_commands.append(
        type("Command", (), {"callback": None, "name": "skip"})()
    )
    app.registered_groups.append(
        type("Group", (), {"name": "ghost", "typer_instance": None})()
    )

    from luxonis_ml.telemetry.cli import instrument_typer

    instrument_typer(app, telemetry, exclude_commands={"admin nested"})

    assert app.registered_commands[0].callback() == 1
    assert sub.registered_commands[0].callback() == 2
    assert len(dummy_backend.events) == 1
    assert dummy_backend.events[0].properties["command"] == "root"


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


def test_cyclopts_helpers_cover_builtin_and_alias_paths() -> None:
    app = App(name=("demo", "alias"))
    subapp = App(name=None)
    app.command(subapp, name="child")

    unique = _iter_unique_subapps([subapp, subapp, object()])

    assert unique == [subapp]
    assert _primary_name(("demo", "alias")) == "demo"
    assert _primary_name(None) == ""
    assert _is_builtin_cyclopts_command(app, app.help_print) is True
    assert _is_builtin_cyclopts_command(app, object()) is False
