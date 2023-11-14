import yaml
import warnings
from typing import (
    Optional,
    Union,
    Dict,
    Any,
    Tuple,
    List,
    get_type_hints,
    get_args,
    get_origin,
)
from pydantic import BaseModel, TypeAdapter

from .filesystem import LuxonisFileSystem


class ConfigHandler:
    """Singleton class which checks and merges user config with default one and provides access to its values"""

    def __new__(
        cls,
        cfg: Optional[Union[str, Dict[str, Any]]] = None,
        cfg_cls: Optional[type] = None,
    ):
        """If needed creates new singleton instance of the Config, otherwise returns already created one

        Args:
            cfg (Optional[Union[str, Dict[str, Any]]], optional): Path to config or config dictionary. Defaults to None.
            cfg_cls (Optional[type], optional): Class to use as internal config structure representation. This should be
            a Pydantic BaseModel class. Defaults to None.

        Returns:
            _type_: Singleton instance
        """
        if not hasattr(cls, "instance"):
            if cfg is None:
                raise ValueError("Provide either config path or config dictionary.")

            if cfg_cls is None:
                raise ValueError("Provide class for internal config object.")

            cls.instance = super(ConfigHandler, cls).__new__(cls)
            cls.instance._load(cfg, cfg_cls)

        return cls.instance

    def __repr__(self) -> str:
        return self._data.model_dump_json(indent=4)

    @classmethod
    def clear_instance(cls) -> None:
        """Clears all singleton instances, should be only used for unit-testing"""
        if hasattr(cls, "instance"):
            del cls.instance

    def get_data(self) -> Dict[str, Any]:
        """Returns dict reperesentation of the config"""
        return self._data.model_dump()

    def get_json_schema(self) -> Dict[str, Any]:
        """Retuns dict representation of config json schema"""
        return self._data.model_json_schema(model="validation")

    def save_data(self, path: str) -> None:
        """Saves config to yaml file

        Args:
            path (str): Path to output yaml file
        """
        with open(path, "w+") as f:
            yaml.dump(self._data.model_dump(), f, default_flow_style=False)

    def get(self, key_merged: str, default: Any = None) -> Any:
        """Returns value from Config based on key

        Args:
            key_merged (str): Merged key in format key1.key2.key3 where each key
            goes one level deeper
            default (Any, optional): Default returned value if key doesn't exist. Defaults to None.

        Returns:
            Any: Value of the key
        """
        last_obj, last_key = self._iterate_config(key_merged.split("."), obj=self._data)

        if last_obj is None:
            warnings.warn(
                f"Itaration for key `{key_merged}` failed, returning default."
            )
            return default

        if isinstance(last_obj, list):
            if 0 <= last_key < len(last_obj):
                return last_obj[last_key]
            else:
                warnings.warn(
                    f"Last key of `{key_merged}` out of range, returning default."
                )
                return default
        elif isinstance(last_obj, dict):
            if last_key not in last_obj:
                warnings.warn(
                    f"Last key of `{key_merged}` not present, returning default."
                )
            return last_obj.get(last_key, default)
        else:
            if not hasattr(last_obj, last_key):
                warnings.warn(
                    f"Last key of `{key_merged}` not present, returning default."
                )
            return getattr(last_obj, last_key, default)

    def override_config(self, args: Dict[str, Any]) -> None:
        """Performs config override based on input dict key-value pairs. Keys in
        form: key1.key2.key3 where each key refers to one layer deeper. If last key
        doesn't exist then add it (Note: adding new keys this way is not encouraged, rather
        specify them in the config itself)

        Args:
            args (Dict[str, Any]): Dict of key-value pairs for override
        """
        for key_merged, value in args.items():
            keys = key_merged.split(".")
            last_obj, last_key = self._iterate_config(keys, obj=self._data)

            # iterate failed
            if last_obj is None:
                warnings.warn(f"Can't override key `{'.'.join(keys)}`, skipping.")
                continue

            # transform value into correct type
            if isinstance(last_obj, list):
                if 0 <= last_key < len(last_obj):
                    target_type = type(last_obj[last_key])
                    value_typed = TypeAdapter(target_type).validate_python(value)
                else:
                    warnings.warn(
                        f"Last key of `{'.'.join(keys)}` out of range, "
                        f"adding element to the end of the list."
                    )
                    # infer correct type
                    types = self._trace_types(keys[:-1], self._data)
                    type_hint = get_type_hints(types[-1]).get(keys[-2])
                    type_args = get_args(type_hint)
                    if get_origin(type_hint) == Union:  # if it's Optional or Union
                        type_args = get_args(type_args[0])
                    target_type = type_args[0]
                    value_typed = TypeAdapter(target_type).validate_python(value)
                    last_obj.append(value_typed)
                    continue
            elif isinstance(last_obj, dict):
                attr = last_obj.get(last_key, None)
                if attr != None:
                    value_typed = TypeAdapter(type(attr)).validate_python(value)
                else:
                    # infer correct type
                    warnings.warn(
                        f"Last key of `{'.'.join(keys)}` not in dict, "
                        f"adding new key-value pair."
                    )
                    types = self._trace_types(keys[:-1], self._data)
                    type_hint = get_type_hints(types[-1]).get(keys[-2])
                    type_args = get_args(type_hint)
                    if get_origin(type_hint) == Union:  # if it's Optional or Union
                        type_args = get_args(type_args[0])
                    key_type, target_type = type_args
                    value_typed = TypeAdapter(target_type).validate_python(value)
            else:
                attr = getattr(last_obj, last_key, None)
                all_types = get_type_hints(last_obj)
                target_type = all_types.get(last_key, None)
                if target_type != None:
                    value_typed = TypeAdapter(target_type).validate_python(value)
                else:
                    warnings.warn(
                        f"Last key of `{'.'.join(keys)}` not present, "
                        f"adding new class attribute."
                    )
                    value_typed = value  # if new attribute leave type as is

            if isinstance(last_obj, list) or isinstance(last_obj, dict):
                last_obj[last_key] = value_typed
            else:
                setattr(last_obj, last_key, value_typed)

        if len(args) == 0:
            return

    def _load(self, cfg: Union[str, Dict[str, Any]], cfg_cls: type) -> None:
        """Loads cfg data in cfg class

        Args:
            cfg (Union[str, Dict[str, Any]]): Path to config or config dictionary
            cfg_cls (type): Class to use as internal config structure representation. This
            should be a Pydantic BaseModel class.
        """
        if isinstance(cfg, str):
            from dotenv import load_dotenv

            load_dotenv()  # load environment variables needed for authorization

            fs = LuxonisFileSystem(cfg)
            buffer = fs.read_to_byte_buffer()
            cfg_data = yaml.load(buffer, Loader=yaml.SafeLoader)
            self._data = cfg_cls(**cfg_data)

            if fs.is_mlflow:
                warnings.warn(
                    "Setting `project_id` and `run_id` to config's MLFlow run"
                )
                # set logger parameters to continue run
                self._data.logger.project_id = fs.experiment_id
                self._data.logger.run_id = fs.run_id

        elif isinstance(cfg, dict):
            self._data = cfg_cls(**cfg)
        else:
            raise ValueError("Provided cfg is neither path(string) or dictionary.")

        # perform validation on config object
        self._validate()

    def _iterate_config(
        self, keys: List[str], obj: Any
    ) -> Tuple[Optional[BaseModel | List[Any] | Dict[str, Any]], Optional[str | int]]:
        """Iterates over config object and returns last object and key encoutered.
        If a key in between isn't matched then it returns (None, None)

        Args:
            keys (List[str]): List of keys for current level and all levels below
            obj (Any): Object at current level

        Returns:
            Tuple[Optional[BaseModel | List[Any] | Dict[str, Any]], Optional[str | int]]:
                Last matched object and last key. If it fails before that than Tuple[None, None]
        """
        if len(keys) == 1:
            # try to convert last key to int if obj is list
            if isinstance(obj, list):
                try:
                    keys[0] = int(keys[0])
                except (ValueError, IndexError):
                    warnings.warn(
                        f"Key `{keys[0]}` can't be converted to list index, skipping."
                    )
                    return None, None
            return obj, keys[0]
        else:
            curr_key, *rest_keys = keys

            if isinstance(obj, list):
                try:
                    index = int(curr_key)
                except (ValueError, IndexError):
                    warnings.warn(
                        f"Key `{curr_key}` can't be converted to list index, skipping."
                    )
                    return None, None
                if len(rest_keys) == 0:
                    return obj, index
                try:
                    return self._iterate_config(rest_keys, obj[index])
                except IndexError:
                    warnings.warn(f"Index `{index}` out of range, skipping.")
                    return None, None
            elif isinstance(obj, dict):
                try:
                    if len(rest_keys) == 0:
                        return obj, curr_key

                    return self._iterate_config(rest_keys, obj[curr_key])
                except KeyError:
                    warnings.warn(f"Key {curr_key} not matched, skipping.")
                    return None, None
            elif isinstance(obj, BaseModel):
                return self._iterate_config(rest_keys, getattr(obj, curr_key, None))
            else:
                warnings.warn(f"Key `{curr_key}` not matched, skipping.")
                return None, None

    def _trace_types(self, keys: List[str], obj: Any) -> List[Any]:
        """Iterates over base object and returns all object types along the way.
        Note that this function assumes that all keys match base object.

        Args:
            keys (List[str]): List of keys for current level and all levels below
            obj (Any): Object at current level

        Returns:
            List[Any]: List of object types from every level
        """
        types = []
        curr_obj = obj
        for key in keys:
            types.append(type(curr_obj))
            if isinstance(curr_obj, list):
                curr_obj = curr_obj[int(key)]
            elif isinstance(curr_obj, dict):
                curr_obj = curr_obj[key]
            else:
                curr_obj = getattr(curr_obj, key)
        return types

    def _validate(self) -> None:
        """Performs custom validation of the config"""
        pass
