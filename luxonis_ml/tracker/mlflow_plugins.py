"""MLflow request headers for servers behind Cloudflare Access.

`LuxonisRequestHeaderProvider` adds the Cloudflare Access credentials to
every MLflow request, so a tracker can reach an MLflow server that
Cloudflare protects.

See:
    `luxonis_ml.tracker.tracker` for the tracker that logs to MLflow.

"""

from mlflow import __version__
from mlflow.tracking.request_header.abstract_request_header_provider import (
    RequestHeaderProvider,
)

from luxonis_ml.utils import environ

_USER_AGENT = "User-Agent"
_DEFAULT_HEADERS = {_USER_AGENT: f"mlflow-python-client/{__version__}"}


class LuxonisRequestHeaderProvider(RequestHeaderProvider):
    """Add Cloudflare Access headers to every MLflow request.

    The provider reports itself as always in context, so it contributes
    its headers to each MLflow request. It reads the credentials from
    the environment:

        - ``MLFLOW_CLOUDFLARE_ID`` becomes the ``CF-Access-Client-Id``
          header;
        - ``MLFLOW_CLOUDFLARE_SECRET`` becomes the
          ``CF-Access-Client-Secret`` header.

    Set both variables when your MLflow endpoint sits behind Cloudflare
    Access. The provider keeps the default MLflow user agent as well.

    Note:
        MLflow finds request header providers through the
        ``mlflow.request_header_provider`` entry point group. LuxonisML
        declares no such entry point, so the two environment variables
        alone do not change MLflow requests. Register the provider in
        your own application first.

    """

    def in_context(self) -> bool:
        """Return True, because the provider always applies."""
        return True

    def request_headers(self) -> dict:
        """Return the default headers with the Cloudflare credentials."""
        headers = dict(**_DEFAULT_HEADERS)
        headers["CF-Access-Client-Id"] = environ.MLFLOW_CLOUDFLARE_ID
        headers["CF-Access-Client-Secret"] = environ.MLFLOW_CLOUDFLARE_SECRET
        return headers
