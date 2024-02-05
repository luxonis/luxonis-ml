from contextlib import contextmanager


@contextmanager
def guard_missing_extra(name: str):
    try:
        yield
    except ImportError as e:
        raise ImportError(
            f"Some of the depndencies of `luxonis-ml[{name}]` are not installed. "
            f"Ensure you installed the package with the `[{name}]` or `[all]` extra specified. "
            f"Use `pip install luxonis-ml[{name}]` to install dependencies for the `{name}` submodule."
        ) from e


__all__ = ["guard_missing_extra"]
