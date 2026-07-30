import pytest
from pydantic import SecretStr

import luxonis_ml.tracker
from luxonis_ml.tracker import LuxonisRequestHeaderProvider
from luxonis_ml.utils import environ


def test_request_header_provider_is_exported_lazily() -> None:
    # importing it must not require MLflow to be installed
    assert (
        luxonis_ml.tracker.LuxonisRequestHeaderProvider
        is LuxonisRequestHeaderProvider
    )

    with pytest.raises(AttributeError, match="no attribute 'nonexistent'"):
        luxonis_ml.tracker.nonexistent  # noqa: B018


def test_headers_without_cloudflare_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(environ, "MLFLOW_CLOUDFLARE_ID", None)
    monkeypatch.setattr(environ, "MLFLOW_CLOUDFLARE_SECRET", None)

    provider = LuxonisRequestHeaderProvider()
    headers = provider.request_headers()

    assert provider.in_context() is True
    assert "User-Agent" in headers
    assert "CF-Access-Client-Id" not in headers
    assert "CF-Access-Client-Secret" not in headers


def test_headers_with_cloudflare_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(environ, "MLFLOW_CLOUDFLARE_ID", "client-id")
    monkeypatch.setattr(
        environ, "MLFLOW_CLOUDFLARE_SECRET", SecretStr("client-secret")
    )

    headers = LuxonisRequestHeaderProvider().request_headers()

    assert headers["CF-Access-Client-Id"] == "client-id"
    # the secret has to be unwrapped, `str(SecretStr(...))` is masked
    assert headers["CF-Access-Client-Secret"] == "client-secret"
