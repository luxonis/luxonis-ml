from mlflow import __version__
from mlflow.tracking.request_header.abstract_request_header_provider import (
    RequestHeaderProvider,
)

from luxonis_ml.utils import environ

_USER_AGENT = "User-Agent"
_DEFAULT_HEADERS = {_USER_AGENT: f"mlflow-python-client/{__version__}"}


class LuxonisRequestHeaderProvider(RequestHeaderProvider):
    """Adds Cloudflare Access credentials to every MLflow request.

    The credentials are read from the ``MLFLOW_CLOUDFLARE_ID`` and
    ``MLFLOW_CLOUDFLARE_SECRET`` environment variables. Headers for
    credentials that are not set are omitted.
    """

    def in_context(self) -> bool:
        return True

    def request_headers(self) -> dict[str, str]:
        headers = dict(_DEFAULT_HEADERS)
        if environ.MLFLOW_CLOUDFLARE_ID is not None:
            headers["CF-Access-Client-Id"] = environ.MLFLOW_CLOUDFLARE_ID
        if environ.MLFLOW_CLOUDFLARE_SECRET is not None:
            headers["CF-Access-Client-Secret"] = (
                environ.MLFLOW_CLOUDFLARE_SECRET.get_secret_value()
            )
        return headers
