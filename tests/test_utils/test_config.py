import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest
import yaml
from pydantic import BaseModel

from luxonis_ml.typing import Params
from luxonis_ml.utils import environ
from luxonis_ml.utils.config import LuxonisConfig

CONFIG_DATA = {
    "sub_config": {
        "str_sub_param": "sub_param",
        "int_sub_param": 42,
        "float_sub_param": 1.0,
        "list_sub_param": [],
        "dict_sub_param": {},
    },
    "sub_config_default": {
        "str_sub_param": "sub_param_default",
    },
}


@pytest.fixture
def config_file() -> str:
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(yaml.dump(CONFIG_DATA).encode())
    return f.name


class SubConfigDefault(BaseModel):
    str_sub_param: str = "sub_param"
    int_sub_param: int = 42
    float_sub_param: float = 1.0

    list_sub_param: List[int] = []
    dict_sub_param: Dict[str, int] = {}


class SubConfig(BaseModel):
    str_sub_param: str
    int_sub_param: int
    float_sub_param: float

    list_sub_param: List[int]
    dict_sub_param: Dict[str, int]


class ListConfig(BaseModel):
    int_list_param: int
    float_list_param: float = 1.0
    str_list_param: Optional[str] = None


class UnsafeConfig(BaseModel):
    tuple_param: Tuple[int, int] = (1, 2)
    path_param: Path = Path.cwd()


class Config(LuxonisConfig):
    sub_config: SubConfig
    sub_config_default: SubConfigDefault = SubConfigDefault()

    optional_int: Optional[int] = None
    list_config: List[ListConfig] = []
    nested_list_param: List[List[int]] = []
    nested_dict_param: Dict[str, Dict[str, int]] = {}
    unsafe_config: UnsafeConfig = UnsafeConfig()


def test_invalid_config_path():
    with pytest.raises(FileNotFoundError):
        Config.get_config("invalid_path")

    with pytest.raises(
        ValueError, match="At least one of `cfg` or `overrides`"
    ):
        Config.get_config(None)


def test_config_simple(config_file: str):
    cfg = Config.get_config(config_file)
    assert (
        cfg.sub_config.str_sub_param
        == CONFIG_DATA["sub_config"]["str_sub_param"]
    )
    assert (
        cfg.sub_config.int_sub_param
        == CONFIG_DATA["sub_config"]["int_sub_param"]
    )
    assert (
        cfg.sub_config.float_sub_param
        == CONFIG_DATA["sub_config"]["float_sub_param"]
    )


def test_config_simple_override(config_file: str):
    overrides: Params = {
        "sub_config.str_sub_param": "sub_param_override",
    }
    cfg = Config.get_config(config_file, overrides)
    assert (
        cfg.sub_config.str_sub_param == overrides["sub_config.str_sub_param"]
    )
    assert (
        cfg.sub_config.int_sub_param
        == CONFIG_DATA["sub_config"]["int_sub_param"]
    )
    assert (
        cfg.sub_config.float_sub_param
        == CONFIG_DATA["sub_config"]["float_sub_param"]
    )


def test_config_list_override(config_file: str):
    overrides = ["sub_config.str_sub_param", "sub_param_override"]
    cfg = Config.get_config(config_file, overrides)
    assert cfg.sub_config.str_sub_param == overrides[1]
    assert (
        cfg.sub_config.int_sub_param
        == CONFIG_DATA["sub_config"]["int_sub_param"]
    )
    assert (
        cfg.sub_config.float_sub_param
        == CONFIG_DATA["sub_config"]["float_sub_param"]
    )
    with pytest.raises(ValueError, match="length is not divisible by 2"):
        Config.get_config(config_file, ["sub_config.str_sub_param"])


def test_config_nested_structure(config_file: str):
    cfg = Config.get_config(config_file)
    # Asserting nested configurations
    assert isinstance(cfg.sub_config.dict_sub_param, dict)
    assert isinstance(cfg.sub_config.list_sub_param, list)
    assert cfg.sub_config_default.str_sub_param == "sub_param_default"
    assert cfg.sub_config_default.int_sub_param == 42


def test_config_optional_params(config_file: str):
    cfg = Config.get_config(config_file)
    # Testing optional parameters
    assert cfg.optional_int is None
    assert isinstance(cfg.list_config, list)
    assert len(cfg.list_config) == 0


def test_config_override_list(config_file: str):
    overrides = {
        "list_config.0.int_list_param": 10,
        "list_config.0.float_list_param": 2.5,
        "list_config.0.str_list_param": "test",
        "list_config.1.int_list_param": 20,
        "nested_list_param.0": [30],
        "nested_list_param.0.1": 40,
    }
    cfg = Config.get_config(config_file, overrides)
    # Testing list configurations
    assert len(cfg.list_config) == 2
    assert cfg.list_config[0].int_list_param == 10
    assert cfg.list_config[0].float_list_param == 2.5
    assert cfg.list_config[0].str_list_param == "test"
    assert cfg.list_config[1].int_list_param == 20
    assert cfg.list_config[1].float_list_param == 1.0
    assert cfg.list_config[1].str_list_param is None
    assert cfg.nested_list_param[0][0] == 30
    assert cfg.nested_list_param[0][1] == 40


def test_config_list_override_json(config_file: str):
    overrides = {
        "list_config": '[{"int_list_param": 10, "float_list_param": 2.5, '
        '"str_list_param": "test"}, {"int_list_param": 20} ]',
        "list_config.2.int_list_param": 30,
    }

    cfg = Config.get_config(config_file, overrides)

    assert len(cfg.list_config) == 3
    assert cfg.list_config[0].int_list_param == 10
    assert cfg.list_config[0].float_list_param == 2.5
    assert cfg.list_config[0].str_list_param == "test"
    assert cfg.list_config[1].int_list_param == 20
    assert cfg.list_config[1].float_list_param == 1.0
    assert cfg.list_config[1].str_list_param is None
    assert cfg.list_config[2].int_list_param == 30


def test_invalid_config(config_file: str):
    overrides = {
        "list_config.-1.int_list_param": 10,
        "list_config.0.float_list_param": [],
        "list_config.0.str_list_param": "test",
        "sub_config.str_sub_param": 10,
        "list_config.2": 30,
        "sub_config.5": 30,
        "5.sub_config": 30,
        "nested_list_param.non": [30],
        "nested_list_param.0.non": 40,
        "sub_config.str_sub_param.non": "sub_param_override",
        "list_config": '[{"int_list_param": 10, "float_list_param": 2.5, '
        '"str_list_param": "test"}, {"int_list_param": 20.5} ]',
        "extra_param": "test",
    }
    for key, value in overrides.items():
        with pytest.raises(ValueError):  # noqa: PT011
            Config.get_config(config_file, {key: value})


def test_from_dict():
    config_dict = deepcopy(CONFIG_DATA)
    config_dict["nested_list_param"] = [[1, 2]]
    config_dict["nested_dict_param"] = {"a": {"b": 1}}
    cfg = Config.get_config(
        config_dict,
        {
            "nested_list_param.0.1": 3,
            "nested_dict_param.a.b": 2,
        },
    )
    assert (
        cfg.sub_config.str_sub_param
        == CONFIG_DATA["sub_config"]["str_sub_param"]
    )
    assert (
        cfg.sub_config.int_sub_param
        == CONFIG_DATA["sub_config"]["int_sub_param"]
    )
    assert (
        cfg.sub_config.float_sub_param
        == CONFIG_DATA["sub_config"]["float_sub_param"]
    )
    assert cfg.nested_list_param[0][1] == 3
    assert cfg.nested_dict_param["a"]["b"] == 2


def test_save(config_file: str):
    cfg = Config.get_config(config_file)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        cfg.save_data(f.name)
    cfg2 = Config.get_config(f.name)
    assert cfg.__repr__() == cfg2.__repr__()
    assert cfg.get_json_schema() == cfg2.get_json_schema()


def test_get(config_file: str):
    cfg = Config.get_config(
        config_file,
        {"nested_list_param": [[1, 2]], "nested_dict_param": {"a": {"b": 1}}},
    )
    assert (
        cfg.get("sub_config.str_sub_param")
        == CONFIG_DATA["sub_config"]["str_sub_param"]
    )
    assert (
        cfg.get("sub_config.int_sub_param")
        == CONFIG_DATA["sub_config"]["int_sub_param"]
    )
    assert cfg.get("list_config.2.int_list_param", -56) == -56
    assert cfg.get("list_config.0.int_list_param") is None
    assert cfg.get("nested_list_param.0.1") == 2
    assert cfg.get("nested_dict_param.a.b") == 1
    assert cfg.get("nested_dict_param.a.c") is None

    assert cfg.get("sub_config.str_sub_param.non", "default") == "default"

    with pytest.raises(ValueError, match="Can't access list with non-int key"):
        cfg.get("list_config.index.int_list_param")


def test_environ():
    assert environ.model_dump() == {}


def test_safe_load(config_file: str):
    cfg = Config.get_config(config_file)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        cfg.save_data(f.name)

    cfg2 = Config.get_config(f.name)
    assert cfg == cfg2
