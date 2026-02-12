from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

DEFAULT_REDACT_KEYS = {
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "api-key",
    "auth",
    "credential",
    "access_key",
    "access-key",
    "private_key",
    "private-key",
}


def sanitize_properties(
    properties: Mapping[str, Any] | None,
    *,
    allowlist: set[str] | None = None,
    redact_keys: set[str] | None = None,
) -> dict[str, Any]:
    """Sanitize event properties before emitting telemetry.

    This applies an allowlist (if provided), redacts secret-like keys,
    and converts values to JSON-friendly types.

    @type properties: Optional[Mapping[str, Any]]
    @param properties: Input properties to sanitize.
    @type allowlist: Optional[set]
    @param allowlist: If set, only these keys are included.
    @type redact_keys: Optional[set]
    @param redact_keys: Key substrings that should be redacted.
    """
    if not properties:
        return {}
    redact_keys = redact_keys or DEFAULT_REDACT_KEYS
    output: dict[str, Any] = {}
    for key, value in properties.items():
        if allowlist is not None and key not in allowlist:
            continue
        if _should_redact(key, redact_keys):
            output[key] = "<redacted>"
            continue
        output[key] = _safe_serialize(value)
    return output


def _should_redact(key: str, redact_keys: set[str]) -> bool:
    """Return True if the key should be redacted."""
    lowered = key.lower()
    return any(token in lowered for token in redact_keys)


def _safe_serialize(value: Any) -> Any:
    """Convert values into a safe, JSON-friendly representation."""
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, str):
            return _truncate(value)
        return value
    if isinstance(value, Path):
        return _truncate(str(value))
    if isinstance(value, bytes):
        return "<bytes>"
    if isinstance(value, (list, tuple, set)):
        return [_safe_serialize(item) for item in list(value)[:20]]
    if isinstance(value, dict):
        return {
            _truncate(str(k)): _safe_serialize(v)
            for k, v in list(value.items())[:50]
        }
    if hasattr(value, "value"):
        return _safe_serialize(value.value)
    return f"<{type(value).__name__}>"


def _truncate(value: str, limit: int = 200) -> str:
    """Truncate long strings to keep payload size reasonable."""
    if len(value) <= limit:
        return value
    return f"{value[:limit]}..."
