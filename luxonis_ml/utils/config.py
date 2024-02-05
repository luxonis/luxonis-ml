import ast
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

import yaml
from pydantic import BaseModel, ConfigDict

from .filesystem import LuxonisFileSystem

T = TypeVar("T", bound="LuxonisConfig")


class LuxonisConfig(BaseModel):
    """Class for storing configuration.

    Singleton class which checks and merges user config with a default one and provides
    access to its values.

    @warning: Only the L{get_config} method can be used to instantiate this class. Using
        C{__init__} directly will raise a L{NotImplementedError}.
    """

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def get_config(
        cls: Type[T],
        cfg: Optional[Union[str, Dict[str, Any]]] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> T:
        """Loads config from a yaml file or a dictionary.

        If config was already loaded before, it returns the same instance.

        @type cfg: str or dict
        @param cfg: Path to config or config dictionary.
        @type overrides: dict
        @param overrides: List of CLI overrides in a form of a dictionary mapping
            "dotted" keys to unparsed string or python values.
        @rtype: LuxonisConfig
        @return: Singleton instance of the config class.
        @raise ValueError: If the config is instantiated for the first time and neither
            C{cfg} nor C{overrides} are provided.
        """
        if getattr(cls, "_instance", None) is None:
            if cfg is None and overrides is None:
                raise ValueError(
                    "At least one of `cfg` or `overrides` must be set"
                    f"when calling `{cls.__name__}.get_config` for the first time."
                )
            cls._from_get_config = True
            cls._instance = cls(cfg, overrides)
        return cls._instance

    def __init__(
        self,
        cfg: Optional[Union[str, Dict[str, Any]]] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ):
        """Loads config from a yaml file or a dictionary.

        @warning: Only the L{get_config} method can be used to instantiate this class.
        @type cfg: str or dict
        @param cfg: Path to config or config dictionary.
        @type overrides: dict
        @param overrides: List of CLI overrides in a form of a dictionary mapping
            "dotted" keys to unparsed string or python values.
        @raise NotImplementedError: When C{__init__} is called directly.
            L{LuxonisConfig} can only be instantiated using the L{get_config}
            classmethod. Using C{__init__} directly will raise this error.
        """
        if not getattr(self, "_from_get_config", False):
            raise NotImplementedError(
                "You cannot use `__init__` on the `LuxonisConfig` class"
                f" directly. Use `{self.__class__.__name__}.get_config` instead."
            )
        overrides = overrides or {}
        cfg = cfg or {}

        if isinstance(cfg, str):
            fs = LuxonisFileSystem(cfg)
            buffer = fs.read_to_byte_buffer()
            data = yaml.load(buffer, Loader=yaml.SafeLoader)
        else:
            data = cfg

        self._merge_overrides(data, overrides)
        super().__init__(**data)

    def __str__(self) -> str:
        return self.model_dump_json(indent=4)

    def __repr__(self) -> str:
        return self.__str__()

    @classmethod
    def clear_instance(cls) -> None:
        """Clears all singleton instances, should be only used for unit-testing."""
        cls._instance = None
        cls._from_get_config = False

    def get_json_schema(self) -> Dict[str, Any]:
        """Retuns dict representation of the config json schema.

        @rtype: dict
        @return: Dictionary with config json schema.
        """
        return self.model_json_schema(mode="validation")

    def save_data(self, path: str) -> None:
        """Saves config to a yaml file.

        @type path: str
        @param path: Path to output yaml file.
        """
        with open(path, "w+") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)

    def get(self, key_merged: str, default: Any = None) -> Any:
        """Returns a value from L{Config} based on the given key.

        If the key doesn't exist, the default value is returned.

        @type key_merged: str
        @param key_merged: Key in a form of a string with levels separated by dots.
        @type default: Any
        @param default: Default value to return if the key doesn't exist.
        @rtype: Any
        @return: Value of the key or default value.
        """
        value = self
        for key in key_merged.split("."):
            if isinstance(value, list):
                if not key.isdecimal():
                    raise ValueError(f"Can't access list with non-int key `{key}`.")
                index = int(key)
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
    def _merge_overrides(data: Dict[str, Any], overrides: Dict[str, Any]) -> None:
        """Merges the config dictionary with the CLI overrides.

        The overrides are a dictionary mapping "dotted" keys to either final or unparsed
        values.

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

        def _merge_recursive(data: Union[Dict, List], dot_name: str, value: Any):
            key, *tail = dot_name.split(".")
            if not tail:
                parsed_value = _parse_value(value)
                if key.isdecimal():
                    index = int(key)
                    if not isinstance(data, list):
                        raise ValueError("int keys are not allowed for non-list values")
                    if index >= len(data):
                        data.append(parsed_value)
                    else:
                        data[index] = parsed_value
                elif isinstance(data, list):
                    raise ValueError("Only int keys are allowed for list values")
                else:
                    data[key] = parsed_value

                return

            key_tail = ".".join(tail)

            if key.isdecimal():
                index = int(key)
                if not isinstance(data, list):
                    raise ValueError("int keys are not allowed for non-list values")
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
                _merge_recursive(data, dot_name, value)
            except Exception as e:
                raise ValueError(f"Invalid option `{dot_name}`: {e}") from e
