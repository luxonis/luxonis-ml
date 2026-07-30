from copy import deepcopy
from typing import Any, cast

import albumentations as A
import numpy as np
import pytest

from luxonis_ml.data import AlbumentationsEngine
from luxonis_ml.data.augmentations import BatchCompose, BatchTransform, MixUp
from luxonis_ml.typing import Labels


def assert_boxes_match_masks(
    boxes: np.ndarray, masks: np.ndarray, tol: float = 0.05
) -> None:
    """Assert every instance mask is paired with the correct bounding box.

    Box and instance mask undergo the same geometric transforms, so a
    correctly paired mask always has its nonzero pixels inside its box.
    A reindexing bug that scrambles the box-to-mask mapping (e.g. pairing a
    box with a mask blob located elsewhere in the image) is caught here even
    when the row counts still match.
    """
    assert boxes.shape[0] == masks.shape[0]
    _, height, width = masks.shape
    for i, (box, mask) in enumerate(zip(boxes, masks, strict=True)):
        ys, xs = np.nonzero(mask)
        assert ys.size > 0, f"instance {i} has an empty mask"
        cx = xs.mean() / width
        cy = ys.mean() / height
        x, y, w, h = box[1:5]
        assert x - tol <= cx <= x + w + tol, (
            f"instance {i}: mask centroid x={cx:.3f} outside box "
            f"[{x:.3f}, {x + w:.3f}] — box paired with the wrong mask"
        )
        assert y - tol <= cy <= y + h + tol, (
            f"instance {i}: mask centroid y={cy:.3f} outside box "
            f"[{y:.3f}, {y + h:.3f}] — box paired with the wrong mask"
        )


def assert_boxes_match_keypoints(
    boxes: np.ndarray, keypoints: np.ndarray, tol: float = 0.05
) -> None:
    """Assert visible keypoints remain paired with their bounding boxes."""
    assert boxes.shape[0] == keypoints.shape[0]
    visible = keypoints[:, 2] > 0
    assert np.any(visible)
    for box, keypoint in zip(boxes[visible], keypoints[visible], strict=True):
        x, y, w, h = box[1:5]
        px, py = keypoint[:2]
        assert x - tol <= px <= x + w + tol
        assert y - tol <= py <= y + h + tol


def assert_masks_match_values(masks: np.ndarray, values: np.ndarray) -> None:
    """Assert each mask channel carries its associated scalar value."""
    assert masks.shape[0] == values.shape[0]
    for mask, value in zip(masks, values, strict=True):
        assert np.unique(mask[mask != 0]).tolist() == [value]


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
def labels() -> Labels:
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


def test_mosaic4(
    images_dict: dict[str, np.ndarray],
    labels: Labels,
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
    augmentations.apply([(images_dict, deepcopy(labels)) for _ in range(4)])


def test_mixup(
    images_dict: dict[str, np.ndarray],
    labels: Labels,
    targets: dict[str, str],
    n_classes: dict[str, int],
) -> None:
    config = [{"name": "MixUp", "params": {"p": 1.0}}]
    source_names = list(images_dict.keys())
    augmentations = AlbumentationsEngine(
        256, 256, targets, n_classes, source_names, config
    )
    augmentations.apply([(images_dict, deepcopy(labels)) for _ in range(2)])


def test_at_least_one_bbox_random_crop() -> None:
    """Test that AtLeastOneBBoxRandomCrop guarantees at least one bbox.

    This is a test for ensuring that the correct "bboxes" key is passed
    to Albumentations transforms that read data["bboxes"] directly.
    """
    image = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
    images_dict = {"image": image}
    labels: Labels = {
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
        _, out_labels = engine.apply([(images_dict, deepcopy(labels))])
        bboxes = out_labels.get("task/boundingbox")
        assert bboxes is not None, (
            "AtLeastOneBBoxRandomCrop should produce bounding box output"
        )
        assert len(bboxes) > 0, (
            "AtLeastOneBBoxRandomCrop should guarantee at least one "
            "bounding box per crop"
        )


def test_batched_p_0(
    images_dict: dict[str, np.ndarray],
    labels: Labels,
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
    augmentations.apply([(images_dict, deepcopy(labels)) for _ in range(8)])


def test_batch_compose_without_bbox_params() -> None:
    """BatchCompose remains usable for image-only batch transformations."""
    first = np.zeros((8, 8, 3), dtype=np.uint8)
    second = np.ones((8, 8, 3), dtype=np.uint8)
    compose = BatchCompose([MixUp(p=0)])

    output = compose([{"image": first}, {"image": second}])

    assert np.array_equal(output["image"], first)


@pytest.mark.parametrize(
    ("mixup_probability", "expected_instances"), [(0.0, 2), (1.0, 4)]
)
def test_skipped_mosaic_before_mixup_reindexes_instance_masks(
    mixup_probability: float, expected_instances: int
) -> None:
    image = np.zeros((320, 320, 3), dtype=np.uint8)
    # Two instances whose masks are centered inside their boxes, so a
    # box-to-mask mispairing is detectable geometrically.
    instance_mask = np.zeros((2, 320, 320), dtype=np.uint8)
    instance_mask[0, 48:112, 48:112] = 1  # box 0 spans [0.15, 0.35]
    instance_mask[1, 192:256, 192:256] = 1  # box 1 spans [0.60, 0.80]
    labels: Labels = {
        "task/boundingbox": np.array(
            [
                [0.0, 0.15, 0.15, 0.20, 0.20],
                [0.0, 0.60, 0.60, 0.20, 0.20],
            ]
        ),
        "task/instance_segmentation": instance_mask,
    }
    targets = {
        "task/boundingbox": "boundingbox",
        "task/instance_segmentation": "instance_segmentation",
    }
    config = [
        {
            "name": "Mosaic4",
            "params": {"p": 0, "out_width": 640, "out_height": 640},
        },
        {"name": "MixUp", "params": {"p": mixup_probability}},
    ]
    augmentations = AlbumentationsEngine(
        256,
        256,
        targets,
        dict.fromkeys(targets, 1),
        ["image"],
        config,
    )

    other_instance_mask = np.zeros((2, 320, 320), dtype=np.uint8)
    other_instance_mask[0, 48:112, 192:256] = 1
    other_instance_mask[1, 192:256, 48:112] = 1
    other_labels: Labels = {
        "task/boundingbox": np.array(
            [
                [0.0, 0.60, 0.15, 0.20, 0.20],
                [0.0, 0.15, 0.60, 0.20, 0.20],
            ]
        ),
        "task/instance_segmentation": other_instance_mask,
    }
    _, out_labels = augmentations.apply(
        [
            ({"image": image}, deepcopy(labels if i < 4 else other_labels))
            for i in range(8)
        ]
    )

    boxes = out_labels["task/boundingbox"]
    masks = out_labels["task/instance_segmentation"]
    assert boxes.shape[0] == masks.shape[0] == expected_instances
    assert_boxes_match_masks(boxes, masks)


@pytest.mark.parametrize("mixup_probability", [None, 0.0, 1.0])
def test_mosaic_drops_boxes_keeps_masks_aligned(
    mixup_probability: float | None,
) -> None:
    """Boxes dropped by ``min_bbox_visibility`` must not scramble masks.

    When a batch transform (here Mosaic4) crops instances so some boxes fall
    below the visibility threshold, ``check_data_post_transform`` drops those
    boxes while every instance-mask channel remains. The surviving boxes must
    still be paired with their own masks; reindexing them to a contiguous
    range after the drop would silently pair them with the wrong masks.
    """
    image = np.zeros((320, 320, 3), dtype=np.uint8)
    # Four instances in the four corners, each mask centered in its box.
    corners = [(0.05, 0.05), (0.80, 0.05), (0.05, 0.80), (0.80, 0.80)]
    size = 0.15

    def make_labels(sample: int) -> Labels:
        """Build one sample whose per-instance ids are unique batch-wide.

        Repeating the same ids in every batch member would make the value
        oracles below hold under any permutation that maps instance *i* of
        one sample onto instance *i* of another, which is exactly the kind
        of wrong pairing compaction can introduce.
        """
        instance_mask = np.zeros((4, 320, 320), dtype=np.uint8)
        boxes_in = []
        keypoints_in = []
        values = []
        for i, (x, y) in enumerate(corners):
            r0, r1 = int(y * 320), int((y + size) * 320)
            c0, c1 = int(x * 320), int((x + size) * 320)
            value = sample * len(corners) + i + 1
            instance_mask[i, r0:r1, c0:c1] = value
            boxes_in.append([0.0, x, y, size, size])
            keypoints_in.append([x + size / 2, y + size / 2, 2.0])
            values.append(value)
        ids = np.array(values)
        return {
            "task/boundingbox": np.array(boxes_in),
            "task/instance_segmentation": instance_mask,
            "task/keypoints": np.array(keypoints_in),
            "task/array": ids,
            "task/metadata/id": ids + 100,
        }

    targets = {
        "task/boundingbox": "boundingbox",
        "task/instance_segmentation": "instance_segmentation",
        "task/keypoints": "keypoints",
        "task/array": "array",
        "task/metadata/id": "metadata",
    }
    config = [
        {
            "name": "Mosaic4",
            "params": {"p": 1.0, "out_width": 640, "out_height": 640},
        }
    ]
    batch_size = 4
    if mixup_probability is not None:
        config.append({"name": "MixUp", "params": {"p": mixup_probability}})
        batch_size *= 2

    augmentations = AlbumentationsEngine(
        256,
        256,
        targets,
        dict.fromkeys(targets, 1),
        ["image"],
        config,
        min_bbox_visibility=0.1,
        seed=42,
    )

    _, out_labels = augmentations.apply(
        [
            ({"image": image}, make_labels(sample))
            for sample in range(batch_size)
        ]
    )

    boxes = out_labels["task/boundingbox"]
    masks = out_labels["task/instance_segmentation"]
    keypoints = out_labels["task/keypoints"]
    arrays = out_labels["task/array"]
    metadata = out_labels["task/metadata/id"]
    # Boxes occupy every corner of every quadrant, so every possible Mosaic
    # crop keeps some candidates and drops others.
    assert 0 < boxes.shape[0] < 4 * batch_size
    assert_boxes_match_masks(boxes, masks)
    assert_boxes_match_keypoints(boxes, keypoints)
    assert arrays.shape == metadata.shape == (boxes.shape[0],)
    assert_masks_match_values(masks, arrays)
    assert np.array_equal(metadata, arrays + 100)


def test_batch_transforms_keep_nested_task_groups_independent() -> None:
    """Associated labels from nested task groups must not be compacted together."""

    def make_mask(
        instances: list[tuple[int, float, float]], size: int = 64
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        boxes = []
        masks = np.zeros((len(instances), size, size), dtype=np.uint8)
        values = []
        for i, (value, x, y) in enumerate(instances):
            x0, y0 = int(x * size), int(y * size)
            masks[i, y0 : y0 + 8, x0 : x0 + 8] = value
            boxes.append([0.0, x, y, 0.125, 0.125])
            values.append(value)
        return np.array(boxes), masks, np.array(values)

    def make_labels(
        root_instances: list[tuple[int, float, float]],
        nested_instances: list[tuple[int, float, float]],
    ) -> Labels:
        root_boxes, root_masks, root_values = make_mask(root_instances)
        nested_boxes, nested_masks, nested_values = make_mask(nested_instances)
        return {
            "/boundingbox": root_boxes,
            "/instance_segmentation": root_masks,
            "/metadata/id": root_values,
            "/nested/boundingbox": nested_boxes,
            "/nested/instance_segmentation": nested_masks,
            "/nested/metadata/id": nested_values,
        }

    targets = {
        "/boundingbox": "boundingbox",
        "/instance_segmentation": "instance_segmentation",
        "/metadata/id": "metadata",
        "/nested/boundingbox": "boundingbox",
        "/nested/instance_segmentation": "instance_segmentation",
        "/nested/metadata/id": "metadata",
    }
    augmentations = AlbumentationsEngine(
        64,
        64,
        targets,
        dict.fromkeys(targets, 1),
        ["image"],
        [
            {
                "name": "Mosaic4",
                "params": {"p": 0.0, "height": 64, "width": 64},
            },
            {"name": "MixUp", "params": {"p": 1.0}},
        ],
    )
    first = make_labels(
        [(1, 0.10, 0.10), (2, 0.70, 0.70)],
        [(101, 0.40, 0.40)],
    )
    second = make_labels(
        [(3, 0.70, 0.10), (4, 0.10, 0.70)],
        [(102, 0.20, 0.40)],
    )
    image = np.zeros((64, 64, 3), dtype=np.uint8)

    _, output = augmentations.apply(
        [
            ({"image": image}, deepcopy(first if i < 4 else second))
            for i in range(8)
        ]
    )

    for task_group, expected_values in [
        ("", np.array([1, 2, 3, 4])),
        ("/nested", np.array([101, 102])),
    ]:
        boxes = output[f"{task_group}/boundingbox"]
        masks = output[f"{task_group}/instance_segmentation"]
        values = output[f"{task_group}/metadata/id"]
        assert np.array_equal(values, expected_values)
        assert_boxes_match_masks(boxes, masks)
        assert_masks_match_values(masks, values)


def test_standalone_labels_survive_bboxes_from_another_task_group() -> None:
    """A bbox task must not filter standalone labels from another task group."""
    targets = {
        "detection/boundingbox": "boundingbox",
        "pose/keypoints": "keypoints",
        "pose/array": "array",
        "pose/metadata/id": "metadata",
    }
    augmentations = AlbumentationsEngine(
        64,
        64,
        targets,
        dict.fromkeys(targets, 1),
        ["image"],
        [{"name": "MixUp", "params": {"p": 1.0}}],
    )
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    batch = []
    for i in range(2):
        batch.append(
            (
                {"image": image},
                {
                    "detection/boundingbox": np.array(
                        [[0.0, 0.1 + i * 0.5, 0.1, 0.2, 0.2]]
                    ),
                    "pose/keypoints": np.array([[0.2 + i * 0.5, 0.2, 2.0]]),
                    "pose/array": np.array([[i, i + 1]]),
                    "pose/metadata/id": np.array([10 + i]),
                },
            )
        )

    _, output = augmentations.apply(batch)

    assert output["pose/keypoints"].shape == (2, 3)
    assert np.array_equal(output["pose/array"], [[0, 1], [1, 2]])
    assert np.array_equal(output["pose/metadata/id"], [10, 11])


def test_compaction_handles_all_filtered_bboxes() -> None:
    """Associated labels stay present but empty when no bbox survives.

    Driven end to end rather than by calling the compaction helper, so it
    covers how ``bbox_counts`` is derived and how the emptied labels travel
    back out through postprocessing. Dropping the task keys entirely would
    let ``LuxonisLoader._add_empty_annotations`` refill metadata with
    ``n_classes`` phantom rows.
    """
    targets = {
        "task/boundingbox": "boundingbox",
        "task/instance_segmentation": "instance_segmentation",
        "task/keypoints": "keypoints",
        "task/array": "array",
        "task/metadata/id": "metadata",
    }
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    # One instance filling the whole image: any Mosaic crop that is not
    # exactly quadrant-aligned clips it, and min_bbox_visibility=1.0 then
    # drops it.
    labels: Labels = {
        "task/boundingbox": np.array([[0.0, 0.0, 0.0, 1.0, 1.0]]),
        "task/instance_segmentation": np.ones((1, 64, 64), dtype=np.uint8),
        "task/keypoints": np.array([[0.5, 0.5, 2.0]]),
        "task/array": np.array([[1, 2]]),
        "task/metadata/id": np.array([7]),
    }

    all_filtered_seen = False
    for seed in range(10):
        augmentations = AlbumentationsEngine(
            64,
            64,
            targets,
            dict.fromkeys(targets, 1),
            ["image"],
            [
                {
                    "name": "Mosaic4",
                    "params": {"p": 1.0, "height": 64, "width": 64},
                },
                {"name": "MixUp", "params": {"p": 1.0}},
            ],
            min_bbox_visibility=1.0,
            seed=seed,
        )
        _, output = augmentations.apply(
            [({"image": image}, deepcopy(labels)) for _ in range(8)]
        )

        if output.get("task/boundingbox", np.zeros((0, 5))).shape[0]:
            continue

        all_filtered_seen = True
        # The bbox task itself is dropped, as it always has been; the loader
        # refills it with a correctly shaped empty array. The labels hanging
        # off it are the ones that must not disappear.
        assert output["task/instance_segmentation"].shape == (0, 64, 64)
        assert output["task/keypoints"].shape == (0, 3)
        assert output["task/array"].shape == (0, 2)
        assert output["task/metadata/id"].shape == (0,)

    assert all_filtered_seen, "no seed produced the all-filtered case"


def test_instance_masks_without_bboxes_in_their_task_group() -> None:
    """An instance-mask task needs no bbox task in its own group.

    Sizing the pass-through ordering by the mask's leading axis reads the
    image height instead of the instance count and raises ``IndexError``.
    """
    targets = {
        "det/boundingbox": "boundingbox",
        "seg/instance_segmentation": "instance_segmentation",
    }
    instance_mask = np.zeros((2, 64, 64), dtype=np.uint8)
    instance_mask[0, 8:24, 8:24] = 1
    instance_mask[1, 40:56, 40:56] = 2
    labels: Labels = {
        "det/boundingbox": np.array([[0.0, 0.1, 0.1, 0.2, 0.2]]),
        "seg/instance_segmentation": instance_mask,
    }
    augmentations = AlbumentationsEngine(
        64, 64, targets, dict.fromkeys(targets, 1), ["image"], []
    )

    _, output = augmentations.apply(
        [({"image": np.zeros((64, 64, 3), dtype=np.uint8)}, labels)]
    )

    assert output["seg/instance_segmentation"].shape == (2, 64, 64)


def test_bbox_label_column_is_left_alone_without_associations() -> None:
    """Plain Albumentations use keeps its class column.

    Without ``bbox_associations`` the trailing bbox column is the class
    label, not an index, so the composition must not restamp it.
    """
    compose = BatchCompose(
        [MixUp(p=1.0)],
        bbox_params=A.BboxParams(format="albumentations"),
    )
    boxes = np.array([[0.1, 0.1, 0.3, 0.3, 7.0], [0.5, 0.5, 0.9, 0.9, 9.0]])

    output = compose(
        [
            {
                "image": np.zeros((16, 16, 3), dtype=np.uint8),
                "bboxes": boxes.copy(),
            }
            for _ in range(2)
        ]
    )

    assert sorted(output["bboxes"][:, -1].tolist()) == [7.0, 7.0, 9.0, 9.0]


def test_bbox_params_cannot_be_passed_positionally() -> None:
    """``bbox_associations`` must not occupy ``A.Compose``'s second slot."""
    untyped_compose = cast(Any, BatchCompose)
    with pytest.raises(TypeError):
        untyped_compose([MixUp(p=1.0)], A.BboxParams(format="albumentations"))


class KeepFirstSample(BatchTransform):
    """Batch transform that keeps the first sample's targets untouched.

    Lets a test observe what `BatchCompose` itself does to the data,
    without a real transform rewriting the targets on the way through.
    """

    def __init__(self):
        super().__init__(batch_size=2, p=1.0)

    def apply(self, image_batch: list[np.ndarray], **_) -> np.ndarray:
        return image_batch[0]

    def apply_to_mask(self, masks_batch: list[np.ndarray], **_) -> np.ndarray:
        return masks_batch[0]

    def apply_to_instance_mask(
        self, masks_batch: list[np.ndarray], **_
    ) -> np.ndarray:
        return masks_batch[0]

    def apply_to_bboxes(
        self, bboxes_batch: list[np.ndarray], **_
    ) -> np.ndarray:
        return bboxes_batch[0]

    def apply_to_keypoints(
        self, keypoints_batch: list[np.ndarray], **_
    ) -> np.ndarray:
        return keypoints_batch[0]


def test_read_only_bboxes_are_not_mutated_in_place() -> None:
    """Reindexing must not write into a caller's read-only buffer."""
    boxes = np.array([[0.1, 0.1, 0.3, 0.3, 0.0, 5.0]])
    boxes.setflags(write=False)
    compose = BatchCompose(
        [KeepFirstSample()],
        bbox_params=A.BboxParams(format="albumentations"),
        bbox_associations={"bboxes": {"metadata": "metadata"}},
    )

    output = compose(
        [
            {
                "image": np.zeros((16, 16, 3), dtype=np.uint8),
                "bboxes": boxes,
                "metadata": np.array([3]),
            }
            for _ in range(2)
        ]
    )

    assert output["bboxes"].shape[0] == 1
    assert boxes[0, -1] == 5.0


def test_associations_are_kept_when_bboxes_are_not_processed() -> None:
    """A missing bbox processor must not be read as "every bbox was cut".

    Without ``bbox_params`` there is no bbox count to compare against, which
    is not the same thing as a count of zero.
    """
    compose = BatchCompose(
        [MixUp(p=1.0)],
        bbox_associations={"bboxes": {"metadata": "metadata"}},
    )

    output = compose(
        [
            {
                "image": np.zeros((16, 16, 3), dtype=np.uint8),
                "bboxes": np.array([[0.1, 0.1, 0.3, 0.3, 0.0, 0.0]]),
                "metadata": np.array([7]),
            }
            for _ in range(2)
        ]
    )

    assert output["metadata"].tolist() == [7, 7]


def test_unmatched_label_counts_are_left_as_is() -> None:
    """Labels that cannot be matched to bboxes must not abort the batch.

    Instance counts and bbox counts can legitimately disagree, for example
    when some annotations carry metadata but no bounding box.
    """
    targets = {
        "task/boundingbox": "boundingbox",
        "task/metadata/id": "metadata",
    }
    augmentations = AlbumentationsEngine(
        64,
        64,
        targets,
        dict.fromkeys(targets, 1),
        ["image"],
        [{"name": "MixUp", "params": {"p": 1.0}}],
    )
    labels: Labels = {
        "task/boundingbox": np.array([[0.0, 0.1, 0.1, 0.2, 0.2]]),
        "task/metadata/id": np.array([11, 12, 13]),
    }

    _, output = augmentations.apply(
        [
            (
                {"image": np.zeros((64, 64, 3), dtype=np.uint8)},
                deepcopy(labels),
            )
            for _ in range(2)
        ]
    )

    assert output["task/metadata/id"].size > 0


@pytest.mark.parametrize("target", ["keypoints", "bboxes"])
def test_mixup_pads_empty_targets_to_the_sibling_width(target: str) -> None:
    """Emptied targets must keep the column count of their sibling.

    Compaction can empty one half of a MixUp pair while the other half keeps
    its rows, and a placeholder of the wrong width then fails to concatenate.
    """
    mixup = MixUp(p=1.0)
    batch = [np.zeros((0, 6)), np.ones((2, 6))]
    apply_to_target = getattr(mixup, f"apply_to_{target}")

    output = apply_to_target(batch, image_shapes=[(16, 16), (16, 16)])

    assert output.shape == (2, 6)


def test_multiple_bbox_targets_in_one_task_group_are_rejected() -> None:
    targets = {
        "task/first_boxes": "boundingbox",
        "task/second_boxes": "boundingbox",
    }

    with pytest.raises(
        ValueError,
        match="Multiple bounding-box targets belong to task group 'task'",
    ):
        AlbumentationsEngine(
            32,
            32,
            targets,
            dict.fromkeys(targets, 1),
            ["image"],
            [],
        )
