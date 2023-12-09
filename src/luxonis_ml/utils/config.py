import yaml
import ast
from typing import (
    Optional,
    Union,
    Dict,
    Any,
)
from pydantic import BaseModel

from .filesystem import LuxonisFileSystem


class Config(BaseModel):
    """Class to store configuration.

    Singleton class which checks and merges user config with a default one
    and provides access to its values.

    """

    def __new__(cls, cfg: Optional[Union[str, Dict[str, Any]]] = None) -> "Config":
        """
        If needed creates new singleton instance of the Config,
        otherwise returns already created one.

        Args:
            cfg (Optional[Union[str, Dict[str, Any]]], optional): Path to config or
                config dictionary. Defaults to None.

        Returns:
            Config: Singleton instance
        """
        if not hasattr(cls, "instance"):
            if cfg is None:
                raise ValueError("Provide either config path or config dictionary.")

            cls.instance = super().__new__(cls)
            cls._fs = None
            if isinstance(cfg, str):
                cls._fs = LuxonisFileSystem(cfg)

        return cls.instance

    def __init__(
        self, cfg: Union[str, Dict[str, Any]], opts: Optional[Dict[str, Any]] = None
    ):
        """Loads cfg data in cfg class

        Args:
            cfg (Union[str, Dict[str, Any]]): Path to config or config dictionary
            cfg_cls (type): Class to use as internal config structure representation. This
            should be a Pydantic BaseModel class.
        """
        opts = opts or {}

        if self._fs is not None:
            buffer = self._fs.read_to_byte_buffer()
            data = yaml.load(buffer, Loader=yaml.SafeLoader)
        elif isinstance(cfg, dict):
            data = cfg
        else:
            raise ValueError("Provided cfg is neither path(string) or dictionary.")
        data = self._merge_from_list(data, opts)
        super().__init__(**data)

    def __repr__(self) -> str:
        return self.model_dump_json(indent=4)

    @classmethod
    def clear_instance(cls) -> None:
        """Clears all singleton instances, should be only used for unit-testing"""
        if hasattr(cls, "instance"):
            del cls.instance

    def get_data(self) -> Dict[str, Any]:
        """Returns dict reperesentation of the config"""
        return self.model_dump()

    def get_json_schema(self) -> Dict[str, Any]:
        """Retuns dict representation of config json schema"""
        return self.model_json_schema(mode="validation")

    def save_data(self, path: str) -> None:
        """Saves config to yaml file

        Args:
            path (str): Path to output yaml file
        """
        with open(path, "w+") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)

    def get(self, key_merged: str, default: Any = None) -> Any:
        """Returns value from Config based on key

        Args:
            key_merged (str): Merged key in format key1.key2.key3 where each key
            goes one level deeper
            default (Any, optional): Default returned value if key doesn't exist. Defaults to None.

        Returns:
            Any: Value of the key
        """
        value = self
        for key in key_merged.split("."):
            if not hasattr(value, key):
                return default
            value = getattr(value, key)
        return value

    @staticmethod
    def _merge_from_list(
        data: Dict[str, Any], overrides: Dict[str, str]
    ) -> Dict[str, Any]:
        """Merges the dictionary with the list of overrides.

        Args:
            _dict (dict): Data loaded from the config file
            overrides (List[str]): List of CLI overrides

        Raises:
            ConfigError: If out of range index is used for overriding a list

        Returns:
            dict: The innitial dictionary merged with the overrides
        """
        for dot_name, value in overrides.items():
            cfg = data
            *name_path, key = dot_name.split(".")
            for name in name_path:
                if isinstance(cfg, list):
                    if not name.isdecimal():
                        raise ValueError(
                            f"Can't access list with non-int key `{name}`."
                        )
                    index = int(name)
                    if index >= len(cfg) and cfg:
                        cfg.append(type(cfg[0])())
                    cfg = cfg[index]
                else:
                    if name not in cfg:
                        if name.isdecimal():
                            cfg[int(name)] = []
                        else:
                            cfg[name] = {}
                    cfg = cfg[name]
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                # load as string and hope for the best
                pass
            cfg[key] = value
        return data
