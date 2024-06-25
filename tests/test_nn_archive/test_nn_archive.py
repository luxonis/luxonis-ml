import shutil
import tarfile
from pathlib import Path
from typing import Any, Dict, Literal

import onnx
import pytest
from heads import (
    classification_head,
    ssd_object_detection_head,
    yolo_instance_seg_kpts_head,
    yolo_instance_segmentation_head,
    yolo_keypoint_detection_head,
    yolo_obb_detection_head,
    yolo_object_detection_head,
)
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
@pytest.mark.parametrize(
    "head, archive_name",
    [
        (classification_head, "classification"),
        (ssd_object_detection_head, "ssd_detection"),
        (yolo_object_detection_head, "yolo_detection"),
        (yolo_instance_segmentation_head, "yolo_instance_segmentation"),
        (yolo_keypoint_detection_head, "yolo_keypoint_detection"),
        (yolo_obb_detection_head, "yolo_obb_detection"),
        (yolo_instance_seg_kpts_head, "yolo_instance_seg_kpts"),
    ],
)
@pytest.mark.dependency(name="test_archive_generator")
def test_archive_generator(
    compression: Literal["xz", "gz", "bz2"],
    head: Dict[str, Any],
    archive_name: Literal[
        "classification",
        "ssd_detection",
        "yolo_detection",
        "yolo_instance_segmentation",
        "yolo_keypoint_detection",
        "yolo_obb_detection",
        "yolo_instance_seg_kpts",
    ],
):
    generator = ArchiveGenerator(
        archive_name=archive_name,
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
                "heads": [head],
            },
        },
        executables_paths=[str(DATA_DIR / "test_model.onnx")],
        compression=compression,
    )
    generator.make_archive()
    assert (DATA_DIR / "{}.tar.xz".format(archive_name)).exists()
    assert tarfile.is_tarfile(DATA_DIR / f"{archive_name}.tar.{compression}")
    with tarfile.open(DATA_DIR / f"{archive_name}.tar.{compression}") as tar:
        assert "test_model.onnx" in tar.getnames()
        assert "config.json" in tar.getnames()


@pytest.mark.parametrize(
    "archive_name",
    [
        "classification",
        "ssd_detection",
        "yolo_detection",
        "yolo_instance_segmentation",
        "yolo_keypoint_detection",
        "yolo_obb_detection",
        "yolo_instance_seg_kpts",
    ],
)
@pytest.mark.dependency(depends=["test_archive_generator"])
def test_is_nn_archive(
    archive_name: Literal[
        "classification",
        "ssd_detection",
        "yolo_detection",
        "yolo_instance_segmentation",
        "yolo_keypoint_detection",
        "yolo_obb_detection",
        "yolo_instance_seg_kpts",
    ],
):
    assert is_nn_archive(DATA_DIR / f"{archive_name}.tar.xz")
    assert not is_nn_archive(DATA_DIR)
    assert not is_nn_archive(DATA_DIR / "test_model.onnx")
