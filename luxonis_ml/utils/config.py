"""Base model for configs loaded from YAML with dotted-key overrides."""

import ast
from pathlib import Path, PurePath
from typing import Any, TypeVar

import yaml
from typing_extensions import Self, deprecated, override

from luxonis_ml.typing import BaseModelExtraForbid, Params, PathType

from .filesystem import LuxonisFileSystem

T = TypeVar("T", bound="LuxonisConfig")


class LuxonisConfig(BaseModelExtraForbid):
    """Class for storing configuration."""

    @classmethod
    def get_config(
        cls,
        cfg: PathType | Params | None = None,
        overrides: Params
        | dict[str, Any]
        | list[str]
        | tuple[str, ...]
        | None = None,
    ) -> Self:
        """Load config from a yaml file or a dictionary.

        Args:
            cfg: Path to a config file or a dictionary.
            overrides: CLI overrides. Can be a dictionary mapping
                "dotted" keys to unparsed string or Python values, or a
                list or a tuple of alternating key-value pairs.

        Returns:
            Instance of the config class.

        Raises:
            ValueError: If neither ``cfg`` nor ``overrides`` is provided,
                or if ``overrides`` is a list or a tuple with an odd number
                of items.

        """
        if cfg is None and overrides is None:
            raise ValueError(
                "At least one of `cfg` or `overrides` must be set."
            )

        if isinstance(overrides, list | tuple):
            if len(overrides) % 2 != 0:
                raise ValueError(
                    "Override options should be a list of key-value pairs "
                    "but it's length is not divisible by 2."
                )

            overrides = dict(zip(overrides[::2], overrides[1::2], strict=True))

        overrides = overrides or {}
        cfg = cfg or {}

        if isinstance(cfg, Path):
            data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
        elif isinstance(cfg, str):
            fs = LuxonisFileSystem(cfg)
            buffer = fs.read_to_byte_buffer()
            data = yaml.safe_load(buffer)
        else:
            data = cfg

        cls._merge_overrides(data, overrides)
        return cls(**data)

    @override
    def __str__(self) -> str:
        return self.model_dump_json(indent=4)

    @override
    def __repr__(self) -> str:
        return self.__str__()

    @deprecated("Use `model_json_schema(mode='validation')` instead.")
    def get_json_schema(self) -> Params:
        """Return the JSON schema of the config.

        .. deprecated:: 0.9.0
           Use ``model_json_schema(mode='validation')`` instead.

        Returns:
            Dictionary with the JSON schema.

        """
        return self.model_json_schema(mode="validation")

    def save_data(self, path: PathType) -> None:
        """Save the config to a YAML file.

        Args:
            path: Path to the file where the config should be saved.

        """

        def path_representer(
            dumper: yaml.SafeDumper, data: PurePath
        ) -> yaml.ScalarNode:
            return dumper.represent_scalar("tag:yaml.org,2002:str", str(data))

        yaml.SafeDumper.add_multi_representer(PurePath, path_representer)

        with Path(path).open("w+") as f:
            yaml.safe_dump(self.model_dump(), f, default_flow_style=False)

    def get(self, key_merged: str, default: Any = None) -> Any:
        """Return a value from the config based on the given key.

        If the key doesn't exist, the default value is returned.

        Args:
            key_merged: Key as a string with levels separated by dots.
            default: Default value to return if the key doesn't exist.

        Returns:
            Value of the key, or the default value.

        Raises:
            ValueError: If a list is accessed with a non-integer key.

        """
        value = self
        for key in key_merged.split("."):
            value = _child(value, key)
            if value is _MISSING:
                return default
        return value

    @staticmethod
    def _merge_overrides(data: Params, overrides: Params) -> None:
        """Merge the config dictionary with the CLI overrides.

        The overrides are a dictionary mapping "dotted" keys to either
        final or unparsed values.

        Args:
            data: Dictionary with config data.
            overrides: Dictionary with CLI overrides.

        Raises:
            ValueError: If the overrides contain an invalid option.

        """
        dot_name = ""
        try:
            for dot_name, value in overrides.items():
                _merge_override(data, dot_name, value)
        except Exception as e:
            raise ValueError(f"Invalid option `{dot_name}`: {e}") from e


_MISSING = object()


def _child(value: Any, key: str) -> Any:
    """Return the child of a config node at ``key``, or `_MISSING`."""
    if isinstance(value, list):
        try:
            index = int(key)
        except ValueError:
            raise ValueError(
                f"Can't access list with non-int key `{key}`."
            ) from None
        return value[index] if index < len(value) else _MISSING
    if isinstance(value, dict):
        return value.get(key, _MISSING)
    return getattr(value, key, _MISSING)


def _parse_override_value(value: Any) -> Any:
    """Parse a string override as a Python literal, if it is one."""
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        # keep as string and hope for the best
        return value


def _merge_override(data: dict | list, dot_name: str, value: Any) -> None:
    """Set ``value`` at the dotted path ``dot_name`` of ``data``.

    A decimal key indexes a list, ``+`` appends to one, and any other
    key names a dict entry. Missing containers on the way are created.
    """
    key, *tail = dot_name.split(".")
    if not tail:
        _set_leaf(data, key, _parse_override_value(value))
        return

    key_tail = ".".join(tail)
    if key.isdecimal() or key == "+":
        if not isinstance(data, list):
            raise ValueError("int keys are not allowed for non-list values")
        _merge_into_list(data, key, key_tail, value)
    elif isinstance(data, dict):
        _merge_into_dict(data, key, key_tail, value)
    else:
        raise ValueError("Only dict values can be accessed with string keys")


def _set_leaf(data: dict | list, key: str, value: Any) -> None:
    """Store a parsed value under the last key of an override path."""
    if key.isdecimal():
        if not isinstance(data, list):
            raise ValueError("int keys are not allowed for non-list values")
        index = int(key)
        if index >= len(data):
            data.append(value)
        else:
            data[index] = value
    elif isinstance(data, list):
        if key != "+":
            raise ValueError("Only int keys are allowed for list values")
        data.append(value)
    else:
        data[key] = value


def _merge_into_list(data: list, key: str, key_tail: str, value: Any) -> None:
    """Descend into the list item at ``key``.

    The key ``+`` appends an empty item of the type of the first one.
    """
    if key == "+":
        data.append(type(data[0])())
        _merge_override(data[-1], key_tail, value)
        return

    index = int(key)
    if index < len(data):
        _merge_override(data[index], key_tail, value)
    elif data:
        data.append(type(data[0])())
        _merge_override(data[-1], key_tail, value)
    else:
        data.append([])
        _merge_into_new(data, 0, key_tail, value)


def _merge_into_dict(data: dict, key: str, key_tail: str, value: Any) -> None:
    """Descend into the dict entry at ``key``, creating it if missing."""
    if key in data:
        _merge_override(data[key], key_tail, value)
        return
    data[key] = []
    _merge_into_new(data, key, key_tail, value)


def _merge_into_new(
    container: dict | list, key: Any, key_tail: str, value: Any
) -> None:
    """Merge into the new empty list at ``key``; a dict if that fails."""
    # Try to guess type, backtrack if fails
    try:
        _merge_override(container[key], key_tail, value)
    except (ValueError, IndexError):
        container[key] = {}
        _merge_override(container[key], key_tail, value)
