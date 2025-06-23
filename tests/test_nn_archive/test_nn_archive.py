import tarfile
from functools import lru_cache
from pathlib import Path
from typing import Literal

import onnx
import pytest
from onnx import checker, helper
from onnx.onnx_pb import TensorProto
from pydantic import ValidationError

from luxonis_ml.nn_archive import ArchiveGenerator, is_nn_archive
from luxonis_ml.nn_archive.config_building_blocks import HeadMetadata
from luxonis_ml.nn_archive.config_building_blocks.enums.data_type import (
    DataType,
)
from luxonis_ml.nn_archive.model import HeadType, Input, Output
from luxonis_ml.typing import Params

from .heads import (
    classification_head,
    custom_segmentation_head,
    ssd_object_detection_head,
    yolo_instance_seg_kpts_head,
    yolo_instance_segmentation_head,
    yolo_keypoint_detection_head,
    yolo_obb_detection_head,
    yolo_object_detection_head,
)


@lru_cache
def create_onnx_model():
    input0 = helper.make_tensor_value_info(
        "input0", TensorProto.FLOAT, [1, 3, 64, 64]
    )
    input1 = helper.make_tensor_value_info(
        "input1", TensorProto.FLOAT, [1, 3, 128, 128]
    )

    output0 = helper.make_tensor_value_info(
        "output0", TensorProto.FLOAT, [1, 10]
    )
    output1 = helper.make_tensor_value_info(
        "output1", TensorProto.FLOAT, [1, 5, 5, 5]
    )
    graph = helper.make_graph(
        [], "DummyModel", [input0, input1], [output0, output1]
    )

    model = helper.make_model(graph, producer_name="DummyModelProducer")
    checker.check_model(model)
    return model


@pytest.fixture
def onnx_path(tempdir: Path):
    onnx.save(create_onnx_model(), str(tempdir / "test_model.onnx"))
    return tempdir / "test_model.onnx"


@pytest.mark.parametrize("compression", ["xz", "gz", "bz2"])
@pytest.mark.parametrize(
    ("head", "archive_name"),
    [
        (classification_head, "classification"),
        (ssd_object_detection_head, "ssd_detection"),
        (yolo_object_detection_head, "yolo_detection"),
        (yolo_instance_segmentation_head, "yolo_instance_segmentation"),
        (yolo_keypoint_detection_head, "yolo_keypoint_detection"),
        (yolo_obb_detection_head, "yolo_obb_detection"),
        (yolo_instance_seg_kpts_head, "yolo_instance_seg_kpts"),
        (custom_segmentation_head, "custom_segmentation"),
    ],
)
@pytest.mark.dependency(name="test_archive_generator")
def test_archive_generator(
    compression: Literal["xz", "gz", "bz2"],
    head: Params,
    archive_name: Literal[
        "classification",
        "ssd_detection",
        "yolo_detection",
        "yolo_instance_segmentation",
        "yolo_keypoint_detection",
        "yolo_obb_detection",
        "yolo_instance_seg_kpts",
        "custom_segmentation",
    ],
    onnx_path: Path,
):
    tempdir = onnx_path.parent
    generator = ArchiveGenerator(
        archive_name=archive_name,
        save_path=str(tempdir),
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
                        "layout": "nchw",
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
        executables_paths=[str(onnx_path)],
        compression=compression,
    )
    generator.make_archive()
    archive_path = tempdir / f"{archive_name}.tar.{compression}"
    assert archive_path.exists()
    assert tarfile.is_tarfile(archive_path)
    with tarfile.open(archive_path) as tar:
        assert "test_model.onnx" in tar.getnames()
        assert "config.json" in tar.getnames()

    assert is_nn_archive(archive_path)
    assert not is_nn_archive(onnx_path)
    assert not is_nn_archive(tempdir)


def test_config_version():
    from luxonis_ml.nn_archive import Config

    cfg_dict = {
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
                    "layout": "nchw",
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
            "heads": [],
        },
    }
    Config(**cfg_dict)
    cfg_dict["config_version"] = "1.2"
    Config(**cfg_dict)
    cfg_dict["config_version"] = "1.2.2"
    with pytest.raises(ValidationError):
        Config(**cfg_dict)
    cfg_dict["config_version"] = "1.a"
    with pytest.raises(ValidationError):
        Config(**cfg_dict)


def test_optional_head_name():
    # without head name
    HeadType(
        parser="Parser",
        metadata=HeadMetadata(),  # type: ignore
        outputs=["output"],
    )
    # with head name
    HeadType(
        parser="Parser",
        name="HeadName",
        metadata=HeadMetadata(),  # type: ignore
        outputs=["output"],
    )


def test_layout():
    default = {
        "name": "input",
        "dtype": "float32",
        "input_type": "image",
        "preprocessing": {},
    }
    inp = Input(
        **{
            **default,
            "shape": [1, 3, 224, 224],
            "layout": "nchw",
        }
    )
    assert inp.layout == "NCHW"
    inp = Input(
        **{
            **default,
            "shape": [3, 256, 256, 16],
            "layout": "chwd",
        }
    )
    assert inp.layout == "CHWD"
    out = Output(
        name="output",
        dtype=DataType.FLOAT32,
        shape=[1, 10],
        layout="nc",
    )
    assert out.layout == "NC"

    with pytest.raises(ValidationError):
        Input(
            **{
                **default,
                "shape": [3, 256, 256, 16],
                "layout": "1chwc2",
            }
        )

    with pytest.raises(ValidationError):
        Input(
            **{
                **default,
                "shape": [1, 3, 256, 256],
                "layout": "nch",
            }
        )

    with pytest.raises(ValidationError):
        Input(
            **{
                **default,
                "shape": [1, 3, 256, 256],
                "layout": "nchh",
            }
        )

    with pytest.raises(ValidationError):
        Output(
            name="output", dtype=DataType.FLOAT32, shape=[1, 10], layout="ncn"
        )

    with pytest.raises(ValidationError):
        Output(
            name="output",
            dtype=DataType.FLOAT32,
            layout=list("nc"),
        )  # type: ignore
