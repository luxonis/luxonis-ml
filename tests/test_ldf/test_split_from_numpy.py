"""`split_from_numpy` is the inverse of `combine_to_numpy`."""

from pathlib import Path

import numpy as np

from luxonis_ml.ldf import (
    ArrayAnnotation,
    BBoxAnnotation,
    ClassificationAnnotation,
    InstanceSegmentationAnnotation,
    KeypointAnnotation,
    SegmentationAnnotation,
)


def test_bounding_boxes_keep_their_coordinates_and_class():
    boxes = [
        BBoxAnnotation(x=0.1, y=0.2, w=0.3, h=0.4),
        BBoxAnnotation(x=0.5, y=0.5, w=0.2, h=0.2),
    ]
    combined = BBoxAnnotation.combine_to_numpy(boxes, [2, 0])

    split = BBoxAnnotation.split_from_numpy(combined)

    assert [class_id for _, class_id in split] == [2, 0]
    assert [annotation for annotation, _ in split] == boxes


def test_keypoints_keep_their_visibility():
    keypoints = [
        KeypointAnnotation.model_validate(
            {"keypoints": [(0.1, 0.2, 2), (0.3, 0.4, 0)]}
        ),
        KeypointAnnotation.model_validate(
            {"keypoints": [(0.5, 0.6, 1), (0.7, 0.8, 2)]}
        ),
    ]
    combined = KeypointAnnotation.combine_to_numpy(keypoints)

    split = KeypointAnnotation.split_from_numpy(combined)

    assert [annotation for annotation, _ in split] == keypoints
    assert all(class_id is None for _, class_id in split)


def test_semantic_masks_come_back_per_class():
    road = np.zeros((4, 4), dtype=np.uint8)
    road[0:2] = 1
    sky = np.zeros((4, 4), dtype=np.uint8)
    sky[3] = 1
    masks = [
        SegmentationAnnotation.model_validate({"mask": road}),
        SegmentationAnnotation.model_validate({"mask": sky}),
    ]
    combined = SegmentationAnnotation.combine_to_numpy(masks, [1, 2], 3)

    split = SegmentationAnnotation.split_from_numpy(combined)

    assert [class_id for _, class_id in split] == [1, 2]
    assert np.array_equal(split[0][0].to_numpy(), road)
    assert np.array_equal(split[1][0].to_numpy(), sky)


def test_an_empty_class_has_no_annotation():
    mask = np.ones((2, 2), dtype=np.uint8)
    combined = SegmentationAnnotation.combine_to_numpy(
        [SegmentationAnnotation.model_validate({"mask": mask})], [0], 3
    )

    assert len(SegmentationAnnotation.split_from_numpy(combined)) == 1


def test_instance_masks_come_back_per_instance():
    first = np.zeros((3, 3), dtype=np.uint8)
    first[0] = 1
    second = np.zeros((3, 3), dtype=np.uint8)
    second[2] = 1
    masks = [
        InstanceSegmentationAnnotation.model_validate({"mask": first}),
        InstanceSegmentationAnnotation.model_validate({"mask": second}),
    ]
    combined = InstanceSegmentationAnnotation.combine_to_numpy(masks)

    split = InstanceSegmentationAnnotation.split_from_numpy(combined)

    assert np.array_equal(split[0][0].to_numpy(), first)
    assert np.array_equal(split[1][0].to_numpy(), second)
    assert all(class_id is None for _, class_id in split)


def test_a_multi_hot_vector_becomes_one_annotation_per_class():
    combined = ClassificationAnnotation.combine_to_numpy(
        [ClassificationAnnotation(), ClassificationAnnotation()], [0, 3], 4
    )

    split = ClassificationAnnotation.split_from_numpy(combined)

    assert [class_id for _, class_id in split] == [0, 3]


def test_arrays_come_back_from_their_class_slot(tempdir: Path):
    first = tempdir / "first.npy"
    second = tempdir / "second.npy"
    np.save(first, np.array([1.0, 2.0]))
    np.save(second, np.array([3.0, 4.0]))
    arrays = [
        ArrayAnnotation(data=first),  # type: ignore[call-arg]
        ArrayAnnotation(data=second),  # type: ignore[call-arg]
    ]
    combined = ArrayAnnotation.combine_to_numpy(arrays, [2, 0], 3)

    split = ArrayAnnotation.split_from_numpy(combined)

    assert [class_id for _, class_id in split] == [2, 0]
    assert split[0][0].to_numpy().tolist() == [1.0, 2.0]
    assert split[1][0].to_numpy().tolist() == [3.0, 4.0]


def test_an_all_zero_array_reports_no_class():
    combined = np.zeros((1, 3, 2))

    (annotation, class_id) = ArrayAnnotation.split_from_numpy(combined)[0]

    assert class_id is None
    assert annotation.to_numpy().tolist() == [0.0, 0.0]


def test_visibility_survives_a_float_round_trip():
    """The arrays are float, so the visibility comes back as one too."""
    keypoints = KeypointAnnotation.model_validate(
        {"keypoints": [(0.1, 0.2, 1)]}
    )
    combined = KeypointAnnotation.combine_to_numpy([keypoints])

    (annotation, _) = KeypointAnnotation.split_from_numpy(combined)[0]

    # A record that carries a plain list keys its keypoints by position.
    visibility = annotation.keypoints["0"].visibility
    assert visibility == 1
    assert isinstance(visibility, int)
