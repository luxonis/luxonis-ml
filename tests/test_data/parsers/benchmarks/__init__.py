"""Parser performance benchmarks.

Every test in this package is marked ``benchmark`` and is therefore
deselected by the default ``addopts``. Run them explicitly::

    pytest -m benchmark tests/test_data/parsers/benchmarks

The benchmarks build large synthetic datasets covering every feature a
parser supports, so they need no credentials and no network access.
"""
