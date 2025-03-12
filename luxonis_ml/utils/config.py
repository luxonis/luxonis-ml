import ast
from pathlib import PurePath
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import yaml

from luxonis_ml.typing import Params, PathType

from .filesystem import LuxonisFileSystem
from .pydantic_utils import BaseModelExtraForbid

T = TypeVar("T", bound="LuxonisConfig")


class LuxonisConfig(BaseModelExtraForbid):
    """Class for storing configuration."""

    @classmethod
    def get_config(
        cls: Type[T],
        cfg: Optional[Union[str, Params]] = None,
        overrides: Optional[Union[Params, List[str], Tuple[str, ...]]] = None,
    ) -> T:
        """Loads config from a yaml file or a dictionary.

        @type cfg: Optional[Union[str, dict]]
        @param cfg: Path to config file or a dictionary.
        @type overrides: Optional[Union[dict, list[str], tuple[str, ...]]]
        @param overrides: List of CLI overrides in a form of a dictionary mapping
            "dotted" keys to unparsed string or python values.
        @rtype: LuxonisConfig
        @return: Instance of the config class.
        @raise ValueError: If neither C{cfg} nor C{overrides} are provided.
        """
        if cfg is None and overrides is None:
            raise ValueError(
                "At least one of `cfg` or `overrides` must be set."
            )

        if isinstance(overrides, (list, tuple)):
            if len(overrides) % 2 != 0:
                raise ValueError(
                    "Override options should be a list of key-value pairs "
                    "but it's length is not divisible by 2."
                )

            overrides = dict(zip(overrides[::2], overrides[1::2]))

        overrides = overrides or {}
        cfg = cfg or {}

        if isinstance(cfg, str):
            fs = LuxonisFileSystem(cfg)
            buffer = fs.read_to_byte_buffer()
            data = yaml.safe_load(buffer)
        else:
            data = cfg

        cls._merge_overrides(data, overrides)
        return cls(**data)  # type: ignore

    def __str__(self) -> str:
        return self.model_dump_json(indent=4)

    def __repr__(self) -> str:
        return self.__str__()

    def get_json_schema(self) -> Params:
        """Retuns dict representation of the config json schema.

        @rtype: dict
        @return: Dictionary with config json schema.
        """
        return self.model_json_schema(mode="validation")

    def save_data(self, path: PathType) -> None:
        """Saves config to a yaml file.

        @type path: str
        @param path: Path to output yaml file.
        """

        def path_representer(
            dumper: yaml.SafeDumper, data: PurePath
        ) -> yaml.ScalarNode:
            return dumper.represent_scalar("tag:yaml.org,2002:str", str(data))

        yaml.SafeDumper.add_multi_representer(PurePath, path_representer)

        with open(path, "w+") as f:
            yaml.safe_dump(self.model_dump(), f, default_flow_style=False)

    def get(self, key_merged: str, default: Any = None) -> Any:
        """Returns a value from L{Config} based on the given key.

        If the key doesn't exist, the default value is returned.

        @type key_merged: str
        @param key_merged: Key in a form of a string with levels
            separated by dots.
        @type default: Any
        @param default: Default value to return if the key doesn't
            exist.
        @rtype: Any
        @return: Value of the key or default value.
        """
        value = self
        for key in key_merged.split("."):
            if isinstance(value, list):
                try:
                    index = int(key)
                except ValueError:
                    raise ValueError(
                        f"Can't access list with non-int key `{key}`."
                    ) from None
                if index >= len(value):
                    return default
                value = value[index]
            elif isinstance(value, dict):
                if key not in value:
                    return default
                value = value[key]
            else:
                if not hasattr(value, key):
                    return default
                value = getattr(value, key)
        return value

    @staticmethod
    def _merge_overrides(data: Params, overrides: Params) -> None:
        """Merges the config dictionary with the CLI overrides.

        The overrides are a dictionary mapping "dotted" keys to either
        final or unparsed values.

        @type data: dict
        @param data: Dictionary with config data.
        @type overrides: dict
        @param overrides: Dictionary with CLI overrides.
        @raise ValueError: If the overrides contain an invalid option.
        """

        def _parse_value(value: Any) -> Any:
            if not isinstance(value, str):
                return value

            try:
                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                # keep as string and hope for the best
                return value

        def _merge_recursive(
            data: Union[Dict, List], dot_name: str, value: Any
        ) -> None:
            key, *tail = dot_name.split(".")
            if not tail:
                parsed_value = _parse_value(value)
                if key.isdecimal():
                    index = int(key)
                    if not isinstance(data, list):
                        raise ValueError(
                            "int keys are not allowed for non-list values"
                        )
                    if index >= len(data):
                        data.append(parsed_value)
                    else:
                        data[index] = parsed_value
                elif isinstance(data, list):
                    raise ValueError(
                        "Only int keys are allowed for list values"
                    )
                else:
                    data[key] = parsed_value

                return

            key_tail = ".".join(tail)

            if key.isdecimal():
                index = int(key)
                if not isinstance(data, list):
                    raise ValueError(
                        "int keys are not allowed for non-list values"
                    )
                if index >= len(data):
                    index = len(data)
                    if data:
                        data.append(type(data[0])())
                        _merge_recursive(data[index], key_tail, value)
                    else:
                        # Try to guess type, backtrack if fails
                        data.append([])
                        try:
                            _merge_recursive(data[index], key_tail, value)
                        except Exception:
                            data[index] = {}
                            _merge_recursive(data[index], key_tail, value)
                else:
                    _merge_recursive(data[index], key_tail, value)
            else:
                if not isinstance(data, dict):
                    raise ValueError(
                        "Only dict values can be accessed with string keys"
                    )
                if key not in data:
                    # Try to guess type, backtrack if fails
                    data[key] = []
                    try:
                        _merge_recursive(data[key], key_tail, value)
                    except Exception:
                        data[key] = {}
                        _merge_recursive(data[key], key_tail, value)
                else:
                    _merge_recursive(data[key], key_tail, value)

        for dot_name, value in overrides.items():
            try:
                _merge_recursive(data, dot_name, value)  # type: ignore
            except Exception as e:
                raise ValueError(f"Invalid option `{dot_name}`: {e}") from e
