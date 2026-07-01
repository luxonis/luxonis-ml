import re
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

DEFAULT_BLOCKED_KEYS = {
    "account_id",
    "anonymous_analytics_id",
    "app_id",
    "device_id",
    "email",
    "file_path",
    "hostname",
    "installation_id",
    "machine_id",
    "mac_address",
    "output_path",
    "remote_url",
    "serial_number",
    "team_id",
    "url",
    "user_id",
    "user_path",
}

_URL_PATTERN = re.compile(r"^(?:[a-z][a-z0-9+.-]*://|www\.)", re.IGNORECASE)
_EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PATH_PATTERN = re.compile(r"^(?:~?/|\.{1,2}/|[a-zA-Z]:[\\/]|/|\\\\)")

# Bound collection sizes so telemetry payloads stay coarse and small even when
# callers accidentally pass large nested structures.
MAX_SEQUENCE_ITEMS = 20
MAX_MAPPING_ITEMS = 50


def sanitize_properties(
    properties: Mapping[str, Any] | None,
    *,
    allowlist: set[str] | None = None,
    redact_keys: set[str] | None = None,
) -> dict[str, Any]:
    """Sanitize event properties before emitting telemetry.

    Applies an allowlist, redacts secret-like keys, and converts values
    to JSON-friendly types.

    Args:
        properties: Input properties to sanitize.
        allowlist: If set, only these keys are included.
        redact_keys: Key substrings that should be redacted.

    Returns:
        The sanitized property mapping.
    """
    if not properties:
        return {}
    redact_keys = redact_keys or DEFAULT_REDACT_KEYS
    output: dict[str, Any] = {}
    for key, value in properties.items():
        if allowlist is not None and key not in allowlist:
            continue
        safe_key = _truncate(str(key))
        if _should_block_key(safe_key):
            output[safe_key] = "<redacted>"
            continue
        if _should_redact(safe_key, redact_keys):
            output[safe_key] = "<redacted>"
            continue
        output[safe_key] = _safe_serialize(value, redact_keys=redact_keys)
    return output


def _should_block_key(key: str) -> bool:
    """Return True for keys that are forbidden in coarse telemetry."""
    lowered = key.lower()
    return lowered in DEFAULT_BLOCKED_KEYS


def _should_redact(key: str, redact_keys: set[str]) -> bool:
    """Return True if the key should be redacted."""
    lowered = key.lower()
    return any(token in lowered for token in redact_keys)


def _safe_serialize(value: Any, *, redact_keys: set[str]) -> Any:
    """Convert values into a safe, JSON-friendly representation."""
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, str):
            return _safe_string(value)
        return value
    if isinstance(value, Path):
        return "<redacted>"
    if isinstance(value, bytes):
        return "<bytes>"
    if isinstance(value, (list, tuple, set)):
        return [
            _safe_serialize(item, redact_keys=redact_keys)
            for item in list(value)[:MAX_SEQUENCE_ITEMS]
        ]
    if isinstance(value, Mapping):
        return _safe_mapping(value, redact_keys=redact_keys)
    if hasattr(value, "value"):
        return _safe_serialize(value.value, redact_keys=redact_keys)
    return f"<{type(value).__name__}>"


def _safe_mapping(
    value: Mapping[Any, Any], *, redact_keys: set[str]
) -> dict[str, Any]:
    """Serialize a mapping while redacting nested secrets."""
    output: dict[str, Any] = {}
    for key, nested_value in list(value.items())[:MAX_MAPPING_ITEMS]:
        safe_key = _truncate(str(key))
        if _should_block_key(safe_key):
            output[safe_key] = "<redacted>"
            continue
        if _should_redact(safe_key, redact_keys):
            output[safe_key] = "<redacted>"
            continue
        output[safe_key] = _safe_serialize(
            nested_value, redact_keys=redact_keys
        )
    return output


def _truncate(value: str, limit: int = 200) -> str:
    """Truncate long strings to keep payload size reasonable."""
    if len(value) <= limit:
        return value
    return f"{value[:limit]}..."


def _safe_string(value: str) -> str:
    """Keep enum-like strings while redacting risky free-form values."""
    stripped = value.strip()
    if not stripped:
        return ""
    if _URL_PATTERN.match(stripped):
        return "<redacted>"
    if _EMAIL_PATTERN.match(stripped):
        return "<redacted>"
    if _PATH_PATTERN.match(stripped):
        return "<redacted>"
    if "\\" in stripped or "/" in stripped:
        return "<redacted>"
    if "\n" in stripped or "\r" in stripped or "\t" in stripped:
        return "<string>"
    if len(stripped.split()) > 3:
        return "<string>"
    return _truncate(stripped)
