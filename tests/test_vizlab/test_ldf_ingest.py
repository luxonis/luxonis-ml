"""vizlab renders Luxonis Data Format objects natively."""

import numpy as np
import pytest

from luxonis_ml.ldf import (
    BBoxAnnotation,
    DatasetRecord,
    Detection,
    InstanceSegmentationAnnotation,
    KeypointAnnotation,
)
from luxonis_ml.vizlab import (
    BBox,
    Classification,
    Image,
    Keypoints,
    Mask,
    Palette,
    SemanticMask,
    VizConfig,
    visualize_record,
)
from luxonis_ml.vizlab.annotations.base import RenderContext
from luxonis_ml.vizlab.canvas import Canvas


def _ctx() -> RenderContext:
    return RenderContext(
        canvas=Canvas.from_rgba(np.zeros((10, 10, 4), np.uint8))
    )


def test_bbox_from_ldf():
    ann = BBoxAnnotation(x=0.1, y=0.2, w=0.3, h=0.4)
    box = BBox.from_ldf(ann, label="car")
    assert (box.x, box.y, box.w, box.h) == (0.1, 0.2, 0.3, 0.4)
    assert box.angle == 0.0
    assert box.label == "car"


def test_keypoints_from_ldf_preserves_visibility():
    ann = KeypointAnnotation(keypoints=[(0.1, 0.2, 2), (0.3, 0.4, 0)])
    kp = Keypoints.from_ldf(ann)
    # Reuses the LDF keypoint list directly.
    assert len(kp.keypoints) == 2
    assert kp.keypoints[0][2] == 2
    assert kp.keypoints[1][2] == 0


def test_mask_from_ldf():
    ann = InstanceSegmentationAnnotation(mask=np.eye(4, dtype=np.uint8))  # type: ignore
    mask = Mask.from_ldf(ann, label="thing")
    assert mask.label == "thing"
    assert mask.to_numpy().shape == (4, 4)


def test_keypoints_from_ldf_edges_and_names():
    ann = KeypointAnnotation(keypoints=[(0.1, 0.2, 2), (0.3, 0.4, 2)])
    kp = Keypoints.from_ldf(
        ann, edges=[(0, 1)], keypoint_names=["a", "b"], point_labels="names"
    )
    assert kp.edges == [(0, 1)]
    assert kp.keypoint_names == ["a", "b"]


def test_semantic_mask_from_ldf_builds_id_map():
    from luxonis_ml.ldf import SegmentationAnnotation

    road = SegmentationAnnotation(mask=np.array([[1, 1], [0, 0]], np.uint8))  # type: ignore
    sky = SegmentationAnnotation(mask=np.array([[0, 0], [1, 1]], np.uint8))  # type: ignore
    sm = SemanticMask.from_ldf([("road", road), ("sky", sky)])
    assert sm.labels is not None
    assert set(np.unique(sm.labels)) == {1, 2}
    assert sm.names == {1: "road", 2: "sky"}


def test_classification_from_ldf():
    chip = Classification.from_ldf(["cat", "dog"])
    assert list(chip.tags) == ["cat", "dog"]


def test_image_add_detection_builds_tree():
    det = Detection(
        class_name="car",
        boundingbox={"x": 0.1, "y": 0.1, "w": 0.4, "h": 0.4},  # type: ignore
        keypoints={"keypoints": [(0.2, 0.2, 2)]},  # type: ignore
        sub_detections={  # type: ignore
            "driver": {
                "class_name": "driver",
                "boundingbox": {"x": 0.15, "y": 0.15, "w": 0.1, "h": 0.1},
            }
        },
    )
    img = Image(np.zeros((50, 50, 3), np.uint8)).add(det)
    # One top-level annotation (the car box) ...
    assert len(img.annotations) == 1
    root = img.annotations[0]
    assert isinstance(root, BBox)
    # ... with the keypoints and the driver sub-detection as children.
    assert any(isinstance(c, Keypoints) for c in root.children)
    assert any(isinstance(c, BBox) for c in root.children)
    assert img.render().shape == (50, 50, 4)


def test_image_add_individual_annotation_models():
    img = Image(np.zeros((20, 20, 3), np.uint8))
    img.add(BBoxAnnotation(x=0.1, y=0.1, w=0.2, h=0.2))
    img.add(KeypointAnnotation(keypoints=[(0.5, 0.5, 2)]))
    assert isinstance(img.annotations[0], BBox)
    assert isinstance(img.annotations[1], Keypoints)


def test_image_add_rejects_unknown_type():
    with pytest.raises(TypeError):
        Image(np.zeros((10, 10, 3), np.uint8)).add(object())


def test_visualize_record_renders_with_panel():
    det = Detection(
        class_name="car",
        boundingbox={"x": 0.1, "y": 0.1, "w": 0.3, "h": 0.3},  # type: ignore
    )
    record = DatasetRecord.model_construct(
        files={},
        annotation=[det],
        task_name="det",
        sample_metadata={"frame": 3},
    )
    cfg = VizConfig(palette=Palette(["car"]))
    img = visualize_record(record, np.zeros((60, 60, 3), np.uint8), config=cfg)
    # sample_metadata attaches a side-panel, widening the output.
    assert img.render().shape[1] > 60


def test_color_determinism_across_records():
    """The same class gets the same color given a pre-seeded palette."""
    palette = Palette(["car", "person"])
    ann = BBoxAnnotation(x=0.1, y=0.1, w=0.2, h=0.2)
    box = BBox.from_ldf(ann, label="person", palette=palette)
    assert box.resolve_color(_ctx()) == Palette(["car", "person"]).color_for(
        "person"
    )


def test_metadata_annotations_from_boxless_detections():
    """Box-less metadata becomes an InfoCard; boxed metadata does not."""
    from luxonis_ml.vizlab import InfoCard
    from luxonis_ml.vizlab.convert import metadata_annotations

    boxless = Detection(
        class_name="pose",
        keypoints=KeypointAnnotation(keypoints=[(0.3, 0.3, 2)]),
        metadata={"action": "running"},
    )
    boxed = Detection(
        class_name="car",
        boundingbox=BBoxAnnotation(x=0.1, y=0.1, w=0.2, h=0.2),
        metadata={"id": 7},  # shown on hover, not in a card
    )
    cards = metadata_annotations([boxless, boxed])
    assert len(cards) == 1
    assert isinstance(cards[0], InfoCard)
    assert cards[0].rows == ["pose", "  action: running"]

    # No box-less metadata -> no cards.
    assert metadata_annotations([boxed]) == []


def test_metadata_annotations_cards_a_lone_boxed_object():
    """A single boxed object cards its metadata; two or more stay hover-only."""
    from luxonis_ml.vizlab import InfoCard
    from luxonis_ml.vizlab.convert import metadata_annotations

    car = Detection(
        class_name="car",
        boundingbox=BBoxAnnotation(x=0.1, y=0.1, w=0.2, h=0.2),
        metadata={"track_id": 42, "speed": 12.4},
    )
    person = Detection(
        class_name="person",
        boundingbox=BBoxAnnotation(x=0.6, y=0.6, w=0.2, h=0.3),
        metadata={"track_id": 7},
    )

    # Lone object -> a metadata card with its real values.
    cards = metadata_annotations([car], lone_object_card=True)
    assert len(cards) == 1
    assert isinstance(cards[0], InfoCard)
    assert cards[0].rows == ["car", "  track_id: 42", "  speed: 12.4"]

    # More than one object -> hover only, even with the flag set.
    assert metadata_annotations([car, person], lone_object_card=True) == []
    # Without the flag, a lone boxed object is still hover-only (default).
    assert metadata_annotations([car]) == []


def test_metadata_annotations_promotes_text_key():
    """The recognized-text key renders as a separate, prominent card."""
    from luxonis_ml.vizlab import InfoCard
    from luxonis_ml.vizlab.convert import metadata_annotations
    from luxonis_ml.vizlab.style import DEFAULT_STYLE

    det = Detection(
        class_name="ocr",
        keypoints=KeypointAnnotation(keypoints=[(0.5, 0.5, 2)]),
        metadata={"text": "HELLO", "conf": 0.9},
    )
    cards = metadata_annotations([det], text_key="text")
    text_cards = [
        c for c in cards if isinstance(c, InfoCard) and c.rows == ["HELLO"]
    ]
    meta_cards = [
        c for c in cards if isinstance(c, InfoCard) and c.title == "metadata"
    ]
    assert len(text_cards) == 1
    # Rendered larger than the default style.
    assert text_cards[0].style is not None
    assert text_cards[0].style.font_size > DEFAULT_STYLE.font_size
    assert meta_cards[0].rows == ["ocr", "  conf: 0.9"]


def test_metadata_annotations_recurses_sub_detections():
    from luxonis_ml.vizlab import InfoCard
    from luxonis_ml.vizlab.convert import metadata_annotations

    parent = Detection(
        class_name="car",
        boundingbox=BBoxAnnotation(x=0.0, y=0.0, w=0.5, h=0.5),
        sub_detections={
            "plate": Detection(
                boundingbox=None,  # type: ignore[arg-type]
                metadata={"note": "AB123"},
            )
        },
    )
    cards = metadata_annotations([parent], text_key="text")
    assert len(cards) == 1
    assert isinstance(cards[0], InfoCard)
    assert cards[0].rows == ["note: AB123"]


def test_boxed_text_metadata_goes_on_chip():
    """A boxed detection's text metadata becomes the box payload."""
    from luxonis_ml.vizlab.convert import VizConfig, detection_to_annotations

    det = Detection(
        class_name="word",
        boundingbox=BBoxAnnotation(x=0.1, y=0.1, w=0.3, h=0.2),
        metadata={"text": "STOP"},
    )
    annotations = detection_to_annotations(det, VizConfig())
    assert annotations[0].payload == "STOP"


def test_visualize_record_adds_metadata_card():
    """A record whose only metadata is box-less renders an in-image card."""
    from luxonis_ml.vizlab import InfoCard

    det = Detection(
        class_name="scene",
        keypoints=KeypointAnnotation(keypoints=[(0.5, 0.5, 2)]),
        metadata={"weather": "sunny"},
    )
    record = DatasetRecord.model_construct(
        files={}, annotation=[det], task_name="scene"
    )
    img = visualize_record(record, np.zeros((60, 60, 3), np.uint8))
    assert any(isinstance(a, InfoCard) for a in img.annotations)
