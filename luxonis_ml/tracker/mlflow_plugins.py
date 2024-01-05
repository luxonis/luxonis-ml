from mlflow import __version__
from mlflow.tracking.request_header.abstract_request_header_provider import (
    RequestHeaderProvider,
)

from luxonis_ml.utils import environ

_USER_AGENT = "User-Agent"
_DEFAULT_HEADERS = {_USER_AGENT: "mlflow-python-client/%s" % __version__}


class LuxonisRequestHeaderProvider(RequestHeaderProvider):
    def in_context(self):
        return True

    def request_headers(self):
        headers = dict(**_DEFAULT_HEADERS)
        headers["CF-Access-Client-Id"] = environ.MLFLOW_CLOUDFLARE_ID
        headers["CF-Access-Client-Secret"] = environ.MLFLOW_CLOUDFLARE_SECRET
        return headers
