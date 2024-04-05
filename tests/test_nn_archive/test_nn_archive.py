import shutil
import tarfile
from pathlib import Path
from typing import Literal

import onnx
import pytest
from onnx import checker, helper
from onnx.onnx_pb import TensorProto

from luxonis_ml.nn_archive import ArchiveGenerator, is_nn_archive

DATA_DIR = Path("tests/data/test_nn_archive")


def create_onnx_model():
    input0 = helper.make_tensor_value_info("input0", TensorProto.FLOAT, [1, 3, 64, 64])
    input1 = helper.make_tensor_value_info(
        "input1", TensorProto.FLOAT, [1, 3, 128, 128]
    )

    output0 = helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 10])
    output1 = helper.make_tensor_value_info("output1", TensorProto.FLOAT, [1, 5, 5, 5])
    graph = helper.make_graph([], "DummyModel", [input0, input1], [output0, output1])

    model = helper.make_model(graph, producer_name="DummyModelProducer")
    checker.check_model(model)
    onnx.save(model, str(DATA_DIR / "test_model.onnx"))


@pytest.fixture(autouse=True, scope="session")
def setup():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    create_onnx_model()

    yield

    shutil.rmtree(DATA_DIR)


@pytest.mark.parametrize("compression", ["xz", "gz", "bz2"])
@pytest.mark.dependency(name="test_archive_generator")
def test_archive_generator(compression: Literal["xz", "gz", "bz2"]):
    generator = ArchiveGenerator(
        archive_name="test_archive",
        save_path="tests/data/test_nn_archive",
        cfg_dict={
            "config_version": "1.0",
            "model": {
                "metadata": {
                    "name": "test_model",
                    "path": "test_model.onnx",
                },
                "inputs": [
                    {
                        "name": "input",
                        "shape": [1, 3, 224, 224],
                        "input_type": "image",
                        "dtype": "float32",
                        "preprocessing": {
                            "mean": [0.485, 0.456, 0.406],
                            "scale": [0.229, 0.224, 0.225],
                            "reverse_channels": False,
                            "interleaved_to_planar": False,
                        },
                    }
                ],
                "outputs": [
                    {
                        "name": "output",
                        "dtype": "float32",
                    }
                ],
                "heads": [
                    {
                        "family": "Classification",
                        "outputs": {"predictions": "191"},
                        "classes": [
                            "tench, Tinca tinca",
                            "goldfish, Carassius auratus",
                        ],
                        "n_classes": 2,
                        "is_softmax": True,
                    }
                ],
            },
        },
        executables_paths=[str(DATA_DIR / "test_model.onnx")],
        compression=compression,
    )
    generator.make_archive()
    assert (DATA_DIR / "test_archive.tar.xz").exists()
    assert tarfile.is_tarfile(DATA_DIR / f"test_archive.tar.{compression}")
    with tarfile.open(DATA_DIR / f"test_archive.tar.{compression}") as tar:
        assert "test_model.onnx" in tar.getnames()
        assert "config.json" in tar.getnames()


@pytest.mark.dependency(depends=["test_archive_generator"])
def test_is_nn_archive():
    assert is_nn_archive(DATA_DIR / "test_archive.tar.xz")
    assert not is_nn_archive(DATA_DIR)
    assert not is_nn_archive(DATA_DIR / "test_model.onnx")
