import yaml
import warnings
import ast
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
from pydantic import BaseModel, TypeAdapter, ValidationError

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

    def __init__(self, cfg: Union[str, Dict[str, Any]]):
        """Loads cfg data in cfg class

        Args:
            cfg (Union[str, Dict[str, Any]]): Path to config or config dictionary
            cfg_cls (type): Class to use as internal config structure representation. This
            should be a Pydantic BaseModel class.
        """
        if self._fs is not None:
            buffer = self._fs.read_to_byte_buffer()
            data = yaml.load(buffer, Loader=yaml.SafeLoader)
        elif isinstance(cfg, dict):
            data = cfg
        else:
            raise ValueError("Provided cfg is neither path(string) or dictionary.")
        super().__init__(**data)
        self._validate()

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
        last_obj, last_key = self._iterate_config(key_merged.split("."), obj=self)

        if last_obj is None and last_key is None:
            warnings.warn(
                f"Itaration for key `{key_merged}` failed, returning default."
            )
            return default

        if isinstance(last_obj, list):
            if not isinstance(last_key, int):
                raise ValueError(
                    f"Attemped to access list with non-int key `{last_key}`."
                )
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

        def _cast_type(target_type: type, value: Any) -> Any:
            """Casts a value to the target type.

            If the value is a string and casting using `pydantic.TypeAdapter` fails,
            then the value is parsed as a python literal and casted again.

            Args:
                target_type (type): Target type to cast to.
                value (Any): Value to cast.

            Returns:
                Any: Casted value.

            Raises:
                ValidationError: If the value can't be casted to the target type.
            """
            adapter = TypeAdapter(target_type)
            try:
                return adapter.validate_python(value)
            except ValidationError as e:
                if not isinstance(value, str):
                    raise e
                # try to evaluate the string to python literal
                try:
                    literal = ast.literal_eval(value)
                except Exception:
                    raise ValueError(f"Can't parse string `{value}`.")

                return adapter.validate_python(literal)

        for key_merged, value in args.items():
            keys = key_merged.split(".")
            last_obj, last_key = self._iterate_config(keys, obj=self)

            # iterate failed
            if last_obj is None:
                warnings.warn(f"Can't override key `{'.'.join(keys)}`, skipping.")
                continue

            # transform value into correct type
            if isinstance(last_obj, list):
                if 0 <= last_key < len(last_obj):
                    target_type = type(last_obj[last_key])
                    value_typed = _cast_type(target_type, value)
                else:
                    warnings.warn(
                        f"Last key of `{'.'.join(keys)}` out of range, "
                        f"adding element to the end of the list."
                    )
                    # infer correct type
                    types = self._trace_types(keys[:-1], self)
                    type_hint = get_type_hints(types[-1]).get(keys[-2])
                    type_args = get_args(type_hint)
                    if get_origin(type_hint) == Union:  # if it's Optional or Union
                        type_args = get_args(type_args[0])
                    target_type = type_args[0]
                    value_typed = _cast_type(target_type, value)
                    last_obj.append(value_typed)
                    continue
            elif isinstance(last_obj, dict):
                attr = last_obj.get(last_key, None)
                if attr is not None:
                    value_typed = _cast_type(type(attr), value)
                else:
                    # infer correct type
                    warnings.warn(
                        f"Last key of `{'.'.join(keys)}` not in dict, "
                        f"adding new key-value pair."
                    )
                    types = self._trace_types(keys[:-1], self)
                    type_hint = get_type_hints(types[-1]).get(keys[-2])
                    type_args = get_args(type_hint)
                    if get_origin(type_hint) == Union:  # if it's Optional or Union
                        type_args = get_args(type_args[0])
                    key_type, target_type = type_args
                    value_typed = _cast_type(target_type, value)
            else:
                attr = getattr(last_obj, last_key, None)
                all_types = get_type_hints(last_obj)
                target_type = all_types.get(last_key, None)
                if target_type is not None:
                    value_typed = _cast_type(target_type, value)
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
