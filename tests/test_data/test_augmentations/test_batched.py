from copy import deepcopy

import numpy as np
import pytest

from luxonis_ml.data import AlbumentationsEngine
from luxonis_ml.data.augmentations.batch_compose import (
    CONTRIBUTOR_INDICES_KEY,
)
from luxonis_ml.typing import Annotations, Params


@pytest.fixture(
    params=[
        {"image": np.zeros((320, 320, 3), dtype=np.uint8)},
        {
            "rgb_image": np.zeros((320, 320, 3), dtype=np.uint8),
            "ir_image": np.zeros((320, 320, 1), dtype=np.uint8),
        },
        {
            "left_img": np.zeros((320, 320, 1), dtype=np.uint8),
            "right_img": np.zeros((320, 320, 1), dtype=np.uint8),
            "middle_img": np.zeros((320, 320, 3), dtype=np.uint8),
        },
    ]
)
def images_dict(request: pytest.FixtureRequest) -> dict[str, np.ndarray]:
    return request.param


@pytest.fixture
def annotations() -> Annotations:
    return {
        "task/classification": np.array([1.0]),
        "task/boundingbox": np.array(
            [
                [0.0, 0.57, 0.30, 0.17, 0.25],
                [0.0, 0.39, 0.27, 0.20, 0.10],
            ]
        ),
        "task/keypoints": np.array(
            [
                [0.69, 0.37, 2.0, 0.0, 0.0, 0.0, 0.68, 0.36, 2.0],
                [0.51, 0.33, 2.0, 0.0, 0.0, 0.0, 0.50, 0.32, 2.0],
            ]
        ),
        "task/segmentation": np.zeros((1, 320, 320)),
    }


@pytest.fixture
def targets() -> dict[str, str]:
    return {
        "task/boundingbox": "boundingbox",
        "task/keypoints": "keypoints",
        "task/segmentation": "segmentation",
    }


@pytest.fixture
def n_classes() -> dict[str, int]:
    return {
        "task/boundingbox": 1,
        "task/keypoints": 1,
        "task/segmentation": 1,
    }


def test_contributor_indices_key_exported() -> None:
    assert CONTRIBUTOR_INDICES_KEY == "_luxonis_contributor_indices"


def test_mosaic4(
    images_dict: dict[str, np.ndarray],
    annotations: Annotations,
    targets: dict[str, str],
    n_classes: dict[str, int],
) -> None:
    config = [
        {
            "name": "Mosaic4",
            "params": {"p": 1.0, "out_width": 640, "out_height": 640},
        }
    ]
    source_names = list(images_dict.keys())
    augmentations = AlbumentationsEngine(
        256, 256, targets, n_classes, source_names, config
    )
    augmentations.apply(
        [(images_dict, deepcopy(annotations), {}) for _ in range(4)]
    )


def test_mixup(
    images_dict: dict[str, np.ndarray],
    annotations: Annotations,
    targets: dict[str, str],
    n_classes: dict[str, int],
) -> None:
    config = [{"name": "MixUp", "params": {"p": 1.0}}]
    source_names = list(images_dict.keys())
    augmentations = AlbumentationsEngine(
        256, 256, targets, n_classes, source_names, config
    )
    augmentations.apply(
        [(images_dict, deepcopy(annotations), {}) for _ in range(2)]
    )


def test_mixup_record_metadata(
    images_dict: dict[str, np.ndarray],
    annotations: Annotations,
    targets: dict[str, str],
    n_classes: dict[str, int],
) -> None:
    config = [{"name": "MixUp", "params": {"p": 1.0}}]
    source_names = list(images_dict.keys())
    augmentations = AlbumentationsEngine(
        256, 256, targets, n_classes, source_names, config
    )
    metadata_batch = [
        {"file_name": "anchor.jpg"},
        {"file_name": "support.jpg"},
    ]

    _, _, metadata = augmentations.apply(
        [
            (images_dict, deepcopy(annotations), metadata_batch[0]),
            (images_dict, deepcopy(annotations), metadata_batch[1]),
        ]  # type: ignore
    )

    assert metadata["file_name"] == "anchor.jpg"
    assert metadata["augmentation_sources"] == [
        {
            "role": "anchor",
            "input_index": 0,
            "metadata": {"file_name": "anchor.jpg"},
        },
        {
            "role": "support",
            "input_index": 1,
            "metadata": {"file_name": "support.jpg"},
        },
    ]


def test_batch_record_metadata_deep_copied(
    images_dict: dict[str, np.ndarray],
    annotations: Annotations,
    targets: dict[str, str],
    n_classes: dict[str, int],
) -> None:
    config = [{"name": "MixUp", "params": {"p": 1.0}}]
    source_names = list(images_dict.keys())
    augmentations = AlbumentationsEngine(
        256, 256, targets, n_classes, source_names, config
    )
    anchor_metadata = {
        "file_name": "anchor.jpg",
        "nested": {"value": "original"},
    }
    support_metadata: Params = {"file_name": "support.jpg"}

    _, _, metadata = augmentations.apply(
        [
            (images_dict, deepcopy(annotations), anchor_metadata),
            (images_dict, deepcopy(annotations), support_metadata),
        ]
    )

    anchor_metadata["nested"]["value"] = "input-mutated"
    metadata["augmentation_sources"][0]["metadata"]["nested"]["value"] = (  # type: ignore
        "source-mutated"
    )

    assert metadata["nested"]["value"] == "original"  # type: ignore


def test_augmentation_sources_metadata_key_collision_warns(
    images_dict: dict[str, np.ndarray],
    annotations: Annotations,
    targets: dict[str, str],
    n_classes: dict[str, int],
) -> None:
    config = [{"name": "MixUp", "params": {"p": 1.0}}]
    source_names = list(images_dict.keys())
    augmentations = AlbumentationsEngine(
        256, 256, targets, n_classes, source_names, config
    )

    with pytest.warns(UserWarning, match="augmentation_sources"):
        _, _, metadata = augmentations.apply(
            [
                (
                    images_dict,
                    deepcopy(annotations),
                    {
                        "file_name": "anchor.jpg",
                        "augmentation_sources": "user-value",
                    },
                ),
                (
                    images_dict,
                    deepcopy(annotations),
                    {"file_name": "support.jpg"},
                ),
            ]
        )

    assert (
        metadata["augmentation_sources"][0]["metadata"]["augmentation_sources"]  # type: ignore
        == "user-value"
    )


def test_nested_batched_record_metadata_applies_after_skip(
    images_dict: dict[str, np.ndarray],
    annotations: Annotations,
    targets: dict[str, str],
    n_classes: dict[str, int],
) -> None:
    config = [
        {
            "name": "Mosaic4",
            "params": {"p": 0, "out_width": 640, "out_height": 640},
        },
        {"name": "MixUp", "params": {"p": 1.0}},
    ]
    source_names = list(images_dict.keys())
    augmentations = AlbumentationsEngine(
        256, 256, targets, n_classes, source_names, config
    )
    metadata_batch = [
        {"file_name": f"image_{i}.jpg"}
        for i in range(augmentations.batch_size)
    ]
    annotations_batch = []
    expected_keypoints = []
    for i in range(augmentations.batch_size):
        sample_annotations = deepcopy(annotations)
        keypoints = sample_annotations["task/keypoints"].copy()
        keypoints[:, 0::3] += i * 0.01
        sample_annotations["task/keypoints"] = keypoints
        expected_keypoints.append(keypoints.copy())
        annotations_batch.append(sample_annotations)

    _, out_annotations, metadata = augmentations.apply(
        [
            (images_dict, annotations_batch[i], metadata)
            for i, metadata in enumerate(metadata_batch)
        ]  # type: ignore
    )

    assert [
        source["metadata"]["file_name"]
        for source in metadata["augmentation_sources"]  # type: ignore
    ] == ["image_0.jpg", "image_4.jpg"]
    assert [
        source["input_index"]
        for source in metadata["augmentation_sources"]  # type: ignore
    ] == [0, 4]
    np.testing.assert_allclose(
        out_annotations["task/keypoints"][:2],
        expected_keypoints[0],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        out_annotations["task/keypoints"][2:],
        expected_keypoints[4],
        atol=1e-6,
    )


def test_duplicate_batch_record_metadata_preserved(
    images_dict: dict[str, np.ndarray],
    annotations: Annotations,
    targets: dict[str, str],
    n_classes: dict[str, int],
) -> None:
    config = [
        {
            "name": "Mosaic4",
            "params": {"p": 1.0, "out_width": 640, "out_height": 640},
        }
    ]
    source_names = list(images_dict.keys())
    augmentations = AlbumentationsEngine(
        256, 256, targets, n_classes, source_names, config
    )
    metadata_batch = [
        {"file_name": "anchor.jpg"},
        {"file_name": "support.jpg"},
        {"file_name": "support.jpg"},
        {"file_name": "support.jpg"},
    ]

    _, _, metadata = augmentations.apply(
        [
            (images_dict, deepcopy(annotations), metadata)
            for metadata in metadata_batch  # type: ignore
        ]
    )

    assert [
        source["metadata"]["file_name"]
        for source in metadata["augmentation_sources"]  # type: ignore
    ] == ["anchor.jpg", "support.jpg", "support.jpg", "support.jpg"]
    assert [
        source["input_index"]
        for source in metadata["augmentation_sources"]  # type: ignore
    ] == [0, 1, 2, 3]


def test_at_least_one_bbox_random_crop() -> None:
    """Test that AtLeastOneBBoxRandomCrop guarantees at least one bbox.

    This is a test for ensuring that the correct "bboxes" key is passed
    to Albumentations transforms that read data["bboxes"] directly.
    """
    image = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
    images_dict = {"image": image}
    annotations: Annotations = {
        "task/boundingbox": np.array(
            [
                [0.0, 0.5, 0.5, 0.1, 0.1],
            ]
        ),
    }
    targets = {"task/boundingbox": "boundingbox"}
    n_classes = {"task/boundingbox": 1}
    config = [
        {
            "name": "AtLeastOneBBoxRandomCrop",
            "params": {
                "height": 40,
                "width": 40,
                "erosion_factor": 0.0,
                "p": 1.0,
            },
        }
    ]
    engine = AlbumentationsEngine(
        256, 256, targets, n_classes, ["image"], config
    )
    for _ in range(10):
        _, out_annotations, _ = engine.apply(
            [(images_dict, deepcopy(annotations), {})]
        )
        bboxes = out_annotations.get("task/boundingbox")
        assert bboxes is not None, (
            "AtLeastOneBBoxRandomCrop should produce bounding box output"
        )
        assert len(bboxes) > 0, (
            "AtLeastOneBBoxRandomCrop should guarantee at least one "
            "bounding box per crop"
        )


def test_batched_p_0(
    images_dict: dict[str, np.ndarray],
    annotations: Annotations,
    targets: dict[str, str],
    n_classes: dict[str, int],
) -> None:
    config = [
        {
            "name": "Mosaic4",
            "params": {"p": 0, "out_width": 640, "out_height": 640},
        },
        {"name": "MixUp", "params": {"p": 0}},
    ]
    source_names = list(images_dict.keys())
    augmentations = AlbumentationsEngine(
        256, 256, targets, n_classes, source_names, config
    )
    metadata_batch = [
        {"file_name": f"image_{i}.jpg"}
        for i in range(augmentations.batch_size)
    ]

    _, _, metadata = augmentations.apply(
        [
            (images_dict, deepcopy(annotations), metadata)
            for metadata in metadata_batch  # type: ignore
        ]
    )
    assert metadata == {"file_name": "image_0.jpg"}


def test_nested_batched_record_metadata_uses_actual_contributors(
    images_dict: dict[str, np.ndarray],
    annotations: Annotations,
    targets: dict[str, str],
    n_classes: dict[str, int],
) -> None:
    config = [
        {
            "name": "Mosaic4",
            "params": {"p": 1.0, "out_width": 640, "out_height": 640},
        },
        {"name": "MixUp", "params": {"p": 0}},
    ]
    source_names = list(images_dict.keys())
    augmentations = AlbumentationsEngine(
        256, 256, targets, n_classes, source_names, config
    )
    metadata_batch = [
        {"file_name": f"image_{i}.jpg"}
        for i in range(augmentations.batch_size)
    ]

    _, _, metadata = augmentations.apply(
        [
            (images_dict, deepcopy(annotations), metadata)
            for metadata in metadata_batch  # type: ignore
        ]
    )

    assert metadata["file_name"] == "image_0.jpg"
    assert metadata["augmentation_sources"] == [
        {
            "role": "anchor",
            "input_index": 0,
            "metadata": {"file_name": "image_0.jpg"},
        },
        {
            "role": "support",
            "input_index": 1,
            "metadata": {"file_name": "image_1.jpg"},
        },
        {
            "role": "support",
            "input_index": 2,
            "metadata": {"file_name": "image_2.jpg"},
        },
        {
            "role": "support",
            "input_index": 3,
            "metadata": {"file_name": "image_3.jpg"},
        },
    ]
