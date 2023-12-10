from luxonis_ml.utils.config import LuxonisConfig
from pydantic import BaseModel
from typing import List, Dict, Optional
import tempfile
import pytest
import yaml

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


@pytest.fixture(scope="function")
def config_file():
    Config.clear_instance()
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(yaml.dump(CONFIG_DATA).encode())
    yield f.name


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


class Config(LuxonisConfig):
    sub_config: SubConfig
    sub_config_default: SubConfigDefault = SubConfigDefault()

    optional_int: Optional[int] = None
    list_config: List[ListConfig] = []


def test_invalid_config_path():
    Config.clear_instance()
    with pytest.raises(FileNotFoundError):
        Config.load_config("invalid_path")
    Config.clear_instance()

    with pytest.raises(ValueError):
        Config.load_config(None)

    Config.clear_instance()


def test_persisted_config(config_file: str):
    cfg = Config.load_config(config_file)
    assert cfg.sub_config.str_sub_param == CONFIG_DATA["sub_config"]["str_sub_param"]
    assert cfg.sub_config.int_sub_param == CONFIG_DATA["sub_config"]["int_sub_param"]
    assert (
        cfg.sub_config.float_sub_param == CONFIG_DATA["sub_config"]["float_sub_param"]
    )
    cfg.sub_config.str_sub_param = "sub_param_test_persistent"
    cfg.sub_config.int_sub_param = 43
    cfg.sub_config.float_sub_param = -2.0

    assert cfg._instance is not None
    cfg = Config.load_config()

    assert cfg.sub_config.str_sub_param == "sub_param_test_persistent"
    assert cfg.sub_config.int_sub_param == 43
    assert cfg.sub_config.float_sub_param == -2.0

    Config.clear_instance()
    cfg = Config.load_config(config_file)

    assert cfg.sub_config.str_sub_param == CONFIG_DATA["sub_config"]["str_sub_param"]
    assert cfg.sub_config.int_sub_param == CONFIG_DATA["sub_config"]["int_sub_param"]
    assert (
        cfg.sub_config.float_sub_param == CONFIG_DATA["sub_config"]["float_sub_param"]
    )


def test_disabled_init(config_file: str):
    with pytest.raises(NotImplementedError):
        Config(config_file)  # type: ignore


def test_config_simple(config_file: str):
    cfg = Config.load_config(config_file)
    assert cfg.sub_config.str_sub_param == CONFIG_DATA["sub_config"]["str_sub_param"]
    assert cfg.sub_config.int_sub_param == CONFIG_DATA["sub_config"]["int_sub_param"]
    assert (
        cfg.sub_config.float_sub_param == CONFIG_DATA["sub_config"]["float_sub_param"]
    )


def test_config_simple_override(config_file: str):
    overrides = {
        "sub_config.str_sub_param": "sub_param_override",
    }
    cfg = Config.load_config(config_file, overrides)
    assert cfg.sub_config.str_sub_param == overrides["sub_config.str_sub_param"]
    assert cfg.sub_config.int_sub_param == CONFIG_DATA["sub_config"]["int_sub_param"]
    assert (
        cfg.sub_config.float_sub_param == CONFIG_DATA["sub_config"]["float_sub_param"]
    )


def test_config_nested_structure(config_file: str):
    cfg = Config.load_config(config_file)
    # Asserting nested configurations
    assert isinstance(cfg.sub_config.dict_sub_param, dict)
    assert isinstance(cfg.sub_config.list_sub_param, list)
    assert cfg.sub_config_default.str_sub_param == "sub_param_default"
    assert cfg.sub_config_default.int_sub_param == 42


def test_config_optional_params(config_file: str):
    cfg = Config.load_config(config_file)
    # Testing optional parameters
    assert cfg.optional_int is None
    assert isinstance(cfg.list_config, list)
    assert len(cfg.list_config) == 0


def test_config_list_override(config_file: str):
    overrides = {
        "list_config.0.int_list_param": 10,
        "list_config.0.float_list_param": 2.5,
        "list_config.0.str_list_param": "test",
        "list_config.1.int_list_param": 20,
    }
    cfg = Config.load_config(config_file, overrides)
    # Testing list configurations
    assert len(cfg.list_config) == 2
    assert cfg.list_config[0].int_list_param == 10
    assert cfg.list_config[0].float_list_param == 2.5
    assert cfg.list_config[0].str_list_param == "test"
    assert cfg.list_config[1].int_list_param == 20
    assert cfg.list_config[1].float_list_param == 1.0  # Default value
    assert cfg.list_config[1].str_list_param is None  # Default value


def test_config_list_override_json(config_file: str):
    overrides = {
        "list_config": '[{"int_list_param": 10, "float_list_param": 2.5, '
        '"str_list_param": "test"}, {"int_list_param": 20} ]',
        "list_config.2.int_list_param": 30,
    }

    cfg = Config.load_config(config_file, overrides)

    assert len(cfg.list_config) == 3
    assert cfg.list_config[0].int_list_param == 10
    assert cfg.list_config[0].float_list_param == 2.5
    assert cfg.list_config[0].str_list_param == "test"
    assert cfg.list_config[1].int_list_param == 20
    assert cfg.list_config[1].float_list_param == 1.0  # Default value
    assert cfg.list_config[1].str_list_param is None  # Default value
    assert cfg.list_config[2].int_list_param == 30


def test_invalid_config(config_file: str):
    overrides = {
        "list_config.-1.int_list_param": 10,
        "list_config.0.float_list_param": [],
        "list_config.0.str_list_param": "test",
        "sub_config.str_sub_param": 10,
        "list_config.2": 30,
        "sub_config.str_sub_param.non": "sub_param_override",
        "list_config": '[{"int_list_param": 10, "float_list_param": 2.5, '
        '"str_list_param": "test"}, {"int_list_param": 20.5} ]',
    }
    for key, value in overrides.items():
        with pytest.raises(ValueError):
            Config.load_config(config_file, {key: value})
        Config.clear_instance()


def test_get(config_file: str):
    cfg = Config.load_config(config_file)
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

    assert cfg.get("sub_config.str_sub_param.non", "default") == "default"

