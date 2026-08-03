import numpy as np
import pytest

from luxonis_ml.data import AlbumentationsEngine
from luxonis_ml.data.augmentations.custom.cutmix import CutMix
from luxonis_ml.typing import Labels

from .test_batched import assert_boxes_match_masks


def test_compute_patch() -> None:
    assert CutMix._compute_patch(0.75, 8, 8, 4, 4) == (2, 2, 6, 6)
    assert CutMix._compute_patch(0.75, 8, 8, 0, 0) == (0, 0, 2, 2)


def test_cutmix_image() -> None:
    cutmix = CutMix(p=1.0, alpha=1.0)
    image1 = np.zeros((6, 8, 3), dtype=np.uint8)
    image2 = np.full((6, 8, 3), 255, dtype=np.uint8)

    image = cutmix.apply(
        [image1, image2],
        image_shapes=[(6, 8), (6, 8)],
        x1=2,
        y1=1,
        x2=5,
        y2=4,
    )

    assert np.all(image[1:4, 2:5] == 255)
    assert np.all(image[:1] == 0)
    assert np.all(image[4:] == 0)
    assert np.all(image[:, :2] == 0)
    assert np.all(image[:, 5:] == 0)


def test_cutmix_image_zero_area_patch() -> None:
    cutmix = CutMix(p=1.0, alpha=1.0)
    image1 = np.zeros((6, 8, 3), dtype=np.uint8)
    image2 = np.full((6, 8, 3), 255, dtype=np.uint8)

    image = cutmix.apply(
        [image1, image2],
        image_shapes=[(6, 8), (6, 8)],
        x1=2,
        y1=1,
        x2=2,
        y2=4,
    )

    np.testing.assert_array_equal(image, image1)


def test_cutmix_without_aspect_ratio_resize() -> None:
    cutmix = CutMix(p=1.0, alpha=1.0, keep_aspect_ratio=False)
    image1 = np.zeros((6, 8, 3), dtype=np.uint8)
    image2 = np.full((3, 4, 3), 255, dtype=np.uint8)

    image = cutmix.apply(
        [image1, image2],
        image_shapes=[(6, 8), (3, 4)],
        x1=2,
        y1=1,
        x2=5,
        y2=4,
    )

    assert np.all(image[1:4, 2:5] == 255)


def test_cutmix_semantic_mask() -> None:
    cutmix = CutMix(p=1.0, alpha=1.0)
    mask1 = np.ones((6, 8, 1), dtype=np.uint8)
    mask2 = np.full((6, 8, 1), 2, dtype=np.uint8)

    mask = cutmix.apply_to_mask(
        [mask1, mask2],
        image_shapes=[(6, 8), (6, 8)],
        x1=2,
        y1=1,
        x2=5,
        y2=4,
    )

    assert np.all(mask[1:4, 2:5] == 2)
    assert np.all(mask[:1] == 1)
    assert np.all(mask[4:] == 1)
    assert np.all(mask[:, :2] == 1)
    assert np.all(mask[:, 5:] == 1)


def test_cutmix_semantic_mask_empty_base() -> None:
    cutmix = CutMix(p=1.0, alpha=1.0)
    mask2 = np.full((6, 8), 2, dtype=np.uint8)

    mask = cutmix.apply_to_mask(
        [np.array([]), mask2],
        image_shapes=[(6, 8), (6, 8)],
        x1=2,
        y1=1,
        x2=5,
        y2=4,
    )

    assert mask.shape == (6, 8, 1)
    assert np.all(mask[1:4, 2:5] == 2)
    assert np.all(mask[:1] == 0)
    assert np.all(mask[4:] == 0)


def test_cutmix_semantic_mask_2d_patch_source_after_resize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cutmix = CutMix(p=1.0, alpha=1.0)
    mask1 = np.ones((6, 8, 1), dtype=np.uint8)
    mask2 = np.full((6, 8), 2, dtype=np.uint8)
    monkeypatch.setattr(cutmix, "_resize", lambda *_, **__: mask2)

    mask = cutmix.apply_to_mask(
        [mask1, mask2],
        image_shapes=[(6, 8), (6, 8)],
        x1=2,
        y1=1,
        x2=5,
        y2=4,
    )

    assert mask.shape == (6, 8, 1)
    assert np.all(mask[1:4, 2:5] == 2)
    assert np.all(mask[:1] == 1)
    assert np.all(mask[4:] == 1)


def test_cutmix_semantic_mask_empty_inputs() -> None:
    cutmix = CutMix(p=1.0, alpha=1.0)

    mask = cutmix.apply_to_mask(
        [np.array([]), np.array([])],
        image_shapes=[(6, 8), (6, 8)],
        x1=2,
        y1=1,
        x2=5,
        y2=4,
    )

    assert mask.size == 0


def test_cutmix_semantic_mask_zero_area_patch() -> None:
    cutmix = CutMix(p=1.0, alpha=1.0)
    mask1 = np.ones((6, 8, 1), dtype=np.uint8)
    mask2 = np.full((6, 8, 1), 2, dtype=np.uint8)

    mask = cutmix.apply_to_mask(
        [mask1, mask2],
        image_shapes=[(6, 8), (6, 8)],
        x1=2,
        y1=1,
        x2=2,
        y2=4,
    )

    np.testing.assert_array_equal(mask, mask1)


def test_cutmix_semantic_mask_empty_patch_source() -> None:
    cutmix = CutMix(p=1.0, alpha=1.0)
    mask1 = np.ones((6, 8, 1), dtype=np.uint8)

    mask = cutmix.apply_to_mask(
        [mask1, np.array([])],
        image_shapes=[(6, 8), (6, 8)],
        x1=2,
        y1=1,
        x2=5,
        y2=4,
    )

    assert np.all(mask[1:4, 2:5] == 0)
    assert np.all(mask[:1] == 1)
    assert np.all(mask[4:] == 1)


def test_cutmix_instance_masks() -> None:
    cutmix = CutMix(p=1.0, alpha=1.0)
    mask1 = np.ones((6, 8, 1), dtype=np.uint8)
    mask2 = np.full((6, 8, 1), 2, dtype=np.uint8)

    masks = cutmix.apply_to_instance_mask(
        [mask1, mask2],
        image_shapes=[(6, 8), (6, 8)],
        x1=2,
        y1=1,
        x2=5,
        y2=4,
    )

    assert masks.shape == (6, 8, 2)
    assert np.all(masks[1:4, 2:5, 0] == 0)
    assert np.all(masks[:1, :, 0] == 1)
    assert np.all(masks[4:, :, 0] == 1)
    assert np.all(masks[:, :2, 0] == 1)
    assert np.all(masks[:, 5:, 0] == 1)
    assert np.all(masks[1:4, 2:5, 1] == 2)
    assert np.all(masks[:1, :, 1] == 0)
    assert np.all(masks[4:, :, 1] == 0)
    assert np.all(masks[:, :2, 1] == 0)
    assert np.all(masks[:, 5:, 1] == 0)


def test_cutmix_instance_mask_empty_base() -> None:
    cutmix = CutMix(p=1.0, alpha=1.0)
    mask2 = np.full((6, 8, 1), 2, dtype=np.uint8)

    masks = cutmix.apply_to_instance_mask(
        [np.array([]), mask2],
        image_shapes=[(6, 8), (6, 8)],
        x1=2,
        y1=1,
        x2=5,
        y2=4,
    )

    assert masks.shape == (6, 8, 1)
    assert np.all(masks[1:4, 2:5, 0] == 2)
    assert np.all(masks[:1, :, 0] == 0)
    assert np.all(masks[4:, :, 0] == 0)


def test_cutmix_instance_mask_2d_base_empty_patch_source() -> None:
    cutmix = CutMix(p=1.0, alpha=1.0)
    mask1 = np.ones((6, 8), dtype=np.uint8)

    masks = cutmix.apply_to_instance_mask(
        [mask1, np.array([])],
        image_shapes=[(6, 8), (6, 8)],
        x1=2,
        y1=1,
        x2=5,
        y2=4,
    )

    assert masks.shape == (6, 8, 1)
    assert np.all(masks[1:4, 2:5, 0] == 0)
    assert np.all(masks[:1, :, 0] == 1)
    assert np.all(masks[4:, :, 0] == 1)


def test_cutmix_instance_mask_2d_patch_source() -> None:
    cutmix = CutMix(p=1.0, alpha=1.0, keep_aspect_ratio=False)
    mask1 = np.ones((6, 8, 1), dtype=np.uint8)
    mask2 = np.full((6, 8), 2, dtype=np.uint8)

    masks = cutmix.apply_to_instance_mask(
        [mask1, mask2],
        image_shapes=[(6, 8), (6, 8)],
        x1=2,
        y1=1,
        x2=5,
        y2=4,
    )

    assert masks.shape == (6, 8, 2)
    assert np.all(masks[1:4, 2:5, 1] == 2)


def test_cutmix_bboxes() -> None:
    cutmix = CutMix(p=1.0, alpha=1.0)
    bbox1 = np.array(
        [
            [0.0, 0.0, 0.3, 0.2, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )
    bbox2 = np.array([[0.2, 0.2, 0.8, 0.8, 1.0, 1.0]])

    bboxes = cutmix.apply_to_bboxes(
        [bbox1, bbox2],
        image_shapes=[(10, 10), (10, 10)],
        x1=4,
        y1=3,
        x2=9,
        y2=7,
    )

    expected = np.array(
        [
            [0.0, 0.0, 0.3, 0.2, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            [0.4, 0.3, 0.8, 0.7, 1.0, 1.0],
        ]
    )
    np.testing.assert_allclose(bboxes, expected)


def test_cutmix_bboxes_strict_visibility() -> None:
    cutmix = CutMix(p=1.0, alpha=1.0, bbox_min_visibility=1.0)
    bbox1 = np.array(
        [
            [0.0, 0.0, 0.3, 0.2, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )
    bbox2 = np.array([[0.2, 0.2, 0.8, 0.8, 1.0, 1.0]])

    bboxes = cutmix.apply_to_bboxes(
        [bbox1, bbox2],
        image_shapes=[(10, 10), (10, 10)],
        x1=4,
        y1=3,
        x2=9,
        y2=7,
    )

    # The second box is fully covered by the patch, so it is collapsed to
    # zero area instead of removed, keeping the row count aligned with the
    # instance labels. `BatchCompose` drops the two together afterwards.
    expected = np.array(
        [
            [0.0, 0.0, 0.3, 0.2, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            [0.4, 0.3, 0.8, 0.7, 1.0, 1.0],
        ]
    )
    np.testing.assert_allclose(bboxes, expected)


def test_cutmix_bboxes_min_visibility_keeps_all() -> None:
    cutmix = CutMix(p=1.0, alpha=1.0, bbox_min_visibility=0.0)
    bbox1 = np.array([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0]])

    bboxes = cutmix.apply_to_bboxes(
        [bbox1, np.array([])],
        image_shapes=[(10, 10), (10, 10)],
        x1=0,
        y1=0,
        x2=10,
        y2=10,
    )

    np.testing.assert_allclose(bboxes, bbox1)


def test_cutmix_bboxes_min_visibility_drops_mostly_covered() -> None:
    cutmix = CutMix(p=1.0, alpha=1.0, bbox_min_visibility=0.5)
    bbox1 = np.array([[0.0, 0.0, 0.4, 0.4, 0.0, 0.0]])

    bboxes = cutmix.apply_to_bboxes(
        [bbox1, np.array([])],
        image_shapes=[(10, 10), (10, 10)],
        x1=1,
        y1=1,
        x2=4,
        y2=4,
    )

    np.testing.assert_allclose(bboxes, [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])


def test_cutmix_bboxes_keep_strategy_leaves_the_box_whole() -> None:
    """The default keeps the box: the object still fills it."""
    cutmix = CutMix(p=1.0, alpha=1.0)
    bbox1 = np.array([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0]])

    bboxes = cutmix.apply_to_bboxes(
        [bbox1, np.array([])],
        image_shapes=[(10, 10), (10, 10)],
        x1=6,
        y1=0,
        x2=10,
        y2=10,
    )

    np.testing.assert_allclose(bboxes, bbox1)


def test_cutmix_bboxes_clip_strategy_shrinks_to_the_visible_side() -> None:
    cutmix = CutMix(p=1.0, alpha=1.0, occluded_bbox_strategy="clip")
    bbox1 = np.array([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0]])

    bboxes = cutmix.apply_to_bboxes(
        [bbox1, np.array([])],
        image_shapes=[(10, 10), (10, 10)],
        x1=6,
        y1=0,
        x2=10,
        y2=10,
    )

    np.testing.assert_allclose(bboxes, [[0.0, 0.0, 0.6, 1.0, 0.0, 0.0]])


def test_cutmix_bboxes_clip_picks_the_largest_remaining_side() -> None:
    """A patch inside a box cannot be clipped out of it.

    The box is cut at one of the patch's edges instead, keeping whichever
    of the four strips that leaves is biggest - here the one above the
    patch, at 40 pixels against 30 for each of the other three.
    """
    cutmix = CutMix(p=1.0, alpha=1.0, occluded_bbox_strategy="clip")
    bbox1 = np.array([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0]])

    bboxes = cutmix.apply_to_bboxes(
        [bbox1, np.array([])],
        image_shapes=[(10, 10), (10, 10)],
        x1=3,
        y1=4,
        x2=7,
        y2=7,
    )

    np.testing.assert_allclose(bboxes, [[0.0, 0.0, 1.0, 0.4, 0.0, 0.0]])


def test_cutmix_bboxes_clip_leaves_untouched_boxes_alone() -> None:
    cutmix = CutMix(p=1.0, alpha=1.0, occluded_bbox_strategy="clip")
    bbox1 = np.array([[0.0, 0.0, 0.4, 0.4, 0.0, 0.0]])

    bboxes = cutmix.apply_to_bboxes(
        [bbox1, np.array([])],
        image_shapes=[(10, 10), (10, 10)],
        x1=6,
        y1=6,
        x2=10,
        y2=10,
    )

    np.testing.assert_allclose(bboxes, bbox1)


def test_cutmix_bboxes_clip_collapses_a_fully_covered_box() -> None:
    """Cutting away a box the patch covers whole must not invert it.

    Every candidate strip is empty here, and a negative-width box would
    survive the area filters further down the pipeline.
    """
    cutmix = CutMix(
        p=1.0,
        alpha=1.0,
        bbox_min_visibility=0.0,
        occluded_bbox_strategy="clip",
    )
    bbox1 = np.array([[0.2, 0.2, 0.4, 0.4, 0.0, 0.0]])

    bboxes = cutmix.apply_to_bboxes(
        [bbox1, np.array([])],
        image_shapes=[(10, 10), (10, 10)],
        x1=0,
        y1=0,
        x2=10,
        y2=10,
    )

    left, top, right, bottom = bboxes[0, :4]
    assert right >= left
    assert bottom >= top
    assert (right - left) * (bottom - top) == 0


def test_cutmix_empty_bboxes() -> None:
    cutmix = CutMix(p=1.0, alpha=1.0)

    bboxes = cutmix.apply_to_bboxes(
        [np.array([]), np.array([])],
        image_shapes=[(10, 10), (10, 10)],
        x1=4,
        y1=3,
        x2=9,
        y2=7,
    )

    assert bboxes.shape == (0, 6)


def test_cutmix_keypoints() -> None:
    cutmix = CutMix(p=1.0, alpha=1.0)
    keypoints1 = np.array(
        [[2.0, 2.0, 0.0, 0.0, 2.0], [6.0, 6.0, 0.0, 0.0, 2.0]]
    )
    keypoints2 = keypoints1.copy()

    keypoints = cutmix.apply_to_keypoints(
        [keypoints1, keypoints2],
        image_shapes=[(8, 8), (8, 8)],
        x1=1,
        y1=1,
        x2=4,
        y2=4,
    )

    np.testing.assert_array_equal(keypoints[:, -1], [0.0, 2.0, 2.0, 0.0])


def test_cutmix_keypoints_zero_area_patch() -> None:
    cutmix = CutMix(p=1.0, alpha=1.0)
    keypoints1 = np.array([[2.0, 2.0, 0.0, 0.0, 2.0]])
    keypoints2 = np.array([[6.0, 6.0, 0.0, 0.0, 2.0]])

    keypoints = cutmix.apply_to_keypoints(
        [keypoints1, keypoints2],
        image_shapes=[(8, 8), (8, 8)],
        x1=2,
        y1=1,
        x2=2,
        y2=7,
    )

    np.testing.assert_array_equal(keypoints, keypoints1)


def test_cutmix_non_spatial_labels_zero_area_patch() -> None:
    cutmix = CutMix(p=1.0, alpha=1.0)
    array1 = np.array([[1.0, 2.0]])
    metadata1 = np.array([[3.0]])
    classification1 = np.array([1.0, 0.0])

    params = {"x1": 2, "y1": 1, "x2": 2, "y2": 7}

    np.testing.assert_array_equal(
        cutmix.apply_to_array([array1, np.array([[4.0, 5.0]])], **params),
        array1,
    )
    np.testing.assert_array_equal(
        cutmix.apply_to_metadata([metadata1, np.array([[6.0]])], **params),
        metadata1,
    )
    np.testing.assert_array_equal(
        cutmix.apply_to_classification(
            [classification1, np.array([0.0, 1.0])], **params
        ),
        classification1,
    )


def test_cutmix_empty_keypoints() -> None:
    cutmix = CutMix(p=1.0, alpha=1.0)

    keypoints = cutmix.apply_to_keypoints(
        [np.array([]), np.array([])],
        image_shapes=[(8, 8), (8, 8)],
        x1=1,
        y1=1,
        x2=4,
        y2=4,
    )

    assert keypoints.shape == (0, 5)


def test_resize_invalid_target_type() -> None:
    cutmix = CutMix(p=1.0, alpha=1.0)

    with pytest.raises(ValueError, match="Unsupported target type"):
        cutmix._resize(
            np.zeros((1, 1, 1), dtype=np.uint8),
            [(1, 1), (1, 1)],
            "invalid",  # type: ignore[arg-type]
        )


def test_bbox_patch_helpers_empty_results() -> None:
    bboxes = np.array([[0.0, 0.0, 0.2, 0.2, 0.0, 0.0]])

    # The box lies entirely outside the patch, so clipping collapses it.
    clipped = CutMix._clip_bboxes_to_patch(
        bboxes, height=10, width=10, x1=4, y1=4, x2=8, y2=8
    )
    np.testing.assert_allclose(clipped, [[0.4, 0.4, 0.4, 0.4, 0.0, 0.0]])

    # A patch with no area means the second image is dropped whole.
    zero_area = CutMix._clip_bboxes_to_patch(
        bboxes, height=10, width=10, x1=4, y1=4, x2=4, y2=8
    )
    assert zero_area.shape == (0, 6)

    unchanged = CutMix._occlude_bboxes(
        bboxes,
        height=10,
        width=10,
        x1=4,
        y1=4,
        x2=4,
        y2=8,
        min_visibility=0.5,
    )
    np.testing.assert_array_equal(unchanged, bboxes)


def test_keypoint_and_dimension_helpers_empty_or_dimensional() -> None:
    keypoints = np.array([])
    assert (
        CutMix._mark_keypoints_in_patch(
            keypoints, x1=1, y1=1, x2=4, y2=4, visible_inside=True
        ).size
        == 0
    )

    squeezed = CutMix._match_dimensions(
        np.ones((2, 3, 1), dtype=np.uint8),
        np.ones((2, 3), dtype=np.uint8),
    )
    assert squeezed.shape == (2, 3)

    expanded = CutMix._match_dimensions(
        np.ones((2, 3), dtype=np.uint8),
        np.ones((2, 3, 1), dtype=np.uint8),
    )
    assert expanded.shape == (2, 3, 1)


@pytest.mark.parametrize(
    "alpha", [0, -1.0, float("nan"), float("inf"), float("-inf")]
)
def test_invalid_alpha(alpha: float) -> None:
    """``np.random.beta`` only fails once the transform is applied.

    A non-finite ``alpha`` passes ``alpha <= 0``, so without an explicit
    check the error surfaces mid-training rather than at configuration
    time.
    """
    with pytest.raises(ValueError, match="greater than 0"):
        CutMix(alpha=alpha)


def test_invalid_occluded_bbox_strategy() -> None:
    with pytest.raises(ValueError, match="'keep' or 'clip'"):
        CutMix(occluded_bbox_strategy="crop")  # type: ignore[arg-type]


def test_invalid_bbox_min_visibility() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        CutMix(bbox_min_visibility=1.5)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        CutMix(bbox_min_visibility=-0.1)


def _instance_labels(first_id: int) -> Labels:
    """Two instances whose masks cover exactly their own box.

    Sharing the box outline means any box that keeps enough visible area to
    survive keeps a non-empty mask too, so an empty mask is a mispairing
    rather than an artifact of the patch position.
    """
    mask = np.zeros((2, 320, 320), dtype=np.uint8)
    mask[0, 48:112, 48:112] = first_id
    mask[1, 208:272, 208:272] = first_id + 1
    return {
        "task/boundingbox": np.array(
            [[0.0, 0.15, 0.15, 0.20, 0.20], [0.0, 0.65, 0.65, 0.20, 0.20]]
        ),
        "task/instance_segmentation": mask,
        "task/metadata/id": np.array([first_id, first_id + 1]),
    }


@pytest.mark.parametrize("strategy", ["keep", "clip"])
@pytest.mark.parametrize("seed", range(40))
def test_dropped_bboxes_take_their_instance_labels_with_them(
    seed: int, strategy: str
) -> None:
    """A box the patch covers must not leave its mask and id behind.

    CutMix used to delete the occluded rows from the bounding boxes alone,
    so the instance masks and metadata it concatenated alongside them
    stayed one entry longer and every later instance was paired with the
    wrong box.
    """
    targets = {
        "task/boundingbox": "boundingbox",
        "task/instance_segmentation": "instance_segmentation",
        "task/metadata/id": "metadata",
    }
    engine = AlbumentationsEngine(
        320,
        320,
        targets,
        dict.fromkeys(targets, 1),
        ["image"],
        [
            {
                "name": "CutMix",
                "params": {"p": 1.0, "occluded_bbox_strategy": strategy},
            }
        ],
        seed=seed,
    )
    image = np.zeros((320, 320, 3), dtype=np.uint8)

    _, output = engine.apply(
        [
            ({"image": image}, _instance_labels(1)),
            ({"image": image}, _instance_labels(11)),
        ]
    )

    boxes = output["task/boundingbox"]
    masks = output["task/instance_segmentation"]
    ids = output["task/metadata/id"]
    assert boxes.shape[0] == masks.shape[0] == ids.shape[0]
    # Each mask was painted with its own id, so this pins down which
    # instance every surviving row actually came from.
    for mask, instance_id in zip(masks, ids, strict=True):
        assert np.unique(mask[mask != 0]).tolist() == [instance_id]

    if strategy == "keep":
        # Under "clip" the box is deliberately smaller than the part of
        # the mask left showing, so its centroid can fall outside.
        assert_boxes_match_masks(boxes, masks)


def test_occluded_bboxes_do_not_mutate_the_input() -> None:
    """The batch arrays belong to the caller and may be read-only."""
    bboxes = np.array([[0.0, 0.0, 0.4, 0.4, 0.0, 0.0]])
    bboxes.setflags(write=False)

    CutMix._occlude_bboxes(
        bboxes,
        height=10,
        width=10,
        x1=0,
        y1=0,
        x2=10,
        y2=10,
        min_visibility=0.5,
    )

    np.testing.assert_allclose(bboxes, [[0.0, 0.0, 0.4, 0.4, 0.0, 0.0]])
