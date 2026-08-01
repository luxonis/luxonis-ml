"""vizlab renders Luxonis Data Format objects natively."""

from pathlib import Path

import numpy as np
import pytest

from luxonis_ml.ldf import (
    ArrayAnnotation,
    BBoxAnnotation,
    DatasetRecord,
    Detection,
    InstanceSegmentationAnnotation,
    KeypointAnnotation,
    SegmentationAnnotation,
)
from luxonis_ml.vizlab import (
    DARK_THEME,
    ArrayField,
    BBox,
    Classification,
    Image,
    Keypoints,
    Mask,
    Palette,
    RenderOptions,
    SemanticMask,
    visualize_record,
)
from luxonis_ml.vizlab.adapters.ldf import (
    _metadata_to_panel_data,
    to_render_annotations,
)
from luxonis_ml.vizlab.annotations.base import RenderContext
from luxonis_ml.vizlab.render.canvas import Canvas


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


def test_image_add_detection_attaches_instance_mask() -> None:
    detection = Detection(
        class_name="car",
        boundingbox=BBoxAnnotation(x=0.1, y=0.1, w=0.4, h=0.4),
        instance_segmentation=InstanceSegmentationAnnotation(
            mask=np.ones((12, 12), np.uint8)  # type: ignore[arg-type]
        ),
    )

    image = Image(np.zeros((20, 20, 3), np.uint8)).add(detection)

    root = image.annotations[0]
    assert isinstance(root, BBox)
    assert any(isinstance(child, Mask) for child in root.children)


def test_image_add_individual_annotation_models():
    img = Image(np.zeros((20, 20, 3), np.uint8))
    img.add(BBoxAnnotation(x=0.1, y=0.1, w=0.2, h=0.2))
    img.add(KeypointAnnotation(keypoints=[(0.5, 0.5, 2)]))
    assert isinstance(img.annotations[0], BBox)
    assert isinstance(img.annotations[1], Keypoints)


def test_image_add_individual_mask_annotation_models() -> None:
    img = Image(np.zeros((20, 20, 3), np.uint8))
    img.add(
        InstanceSegmentationAnnotation(
            mask=np.eye(4, dtype=np.uint8)  # type: ignore[arg-type]
        )
    )
    img.add(
        SegmentationAnnotation(
            mask=np.eye(4, dtype=np.uint8)  # type: ignore[arg-type]
        )
    )

    assert all(isinstance(annotation, Mask) for annotation in img.annotations)


def test_to_render_annotations_expands_dataset_record() -> None:
    record = DatasetRecord.model_construct(
        files={},
        annotation=[
            Detection(
                class_name="car",
                boundingbox=BBoxAnnotation(x=0.1, y=0.1, w=0.2, h=0.2),
                metadata={"id": 7},
            )
        ],
        task_name="objects",
    )

    annotations = to_render_annotations(record)

    assert any(isinstance(annotation, BBox) for annotation in annotations)


def test_image_add_rejects_unknown_type():
    with pytest.raises(TypeError):
        Image(np.zeros((10, 10, 3), np.uint8)).add(object())  # type: ignore[arg-type]


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
    options = RenderOptions(theme=DARK_THEME.with_palette(Palette(["car"])))
    img = visualize_record(
        record, np.zeros((60, 60, 3), np.uint8), options=options
    )
    # sample_metadata attaches a side-panel, widening the output.
    assert img.render().shape[1] > 60


def test_visualize_record_merges_explicit_panel_data() -> None:
    record = DatasetRecord.model_construct(
        files={},
        annotation=[],
        task_name="empty",
        sample_metadata={},
    )

    rendered = visualize_record(
        record,
        np.zeros((30, 40, 3), np.uint8),
        panel={"review": {"status": "ready"}},
    )

    assert rendered.width > 40


def test_metadata_panel_normalization_recurses_json_values() -> None:
    assert _metadata_to_panel_data(
        {"path": "a.jpg", "flags": [True, None], 3: 1.25}
    ) == {
        "path": "a.jpg",
        "flags": [True, None],
        "3": 1.25,
    }


def test_visualize_record_collects_nested_record_annotations(
    tmp_path: Path,
) -> None:
    """Nested masks, class tags, and arrays reach record-level output."""
    array_path = tmp_path / "embedding.npy"
    # Proportional to the 32x48 image below, so it may be drawn over it.
    np.save(array_path, np.zeros((4, 6), np.float32))
    root = Detection(
        class_name="vehicle",
        boundingbox=BBoxAnnotation(x=0.1, y=0.1, w=0.6, h=0.6),
        sub_detections={
            "surface": Detection(
                class_name="road",
                segmentation=SegmentationAnnotation(
                    mask=np.ones((8, 8), np.uint8)  # type: ignore[arg-type]
                ),
            ),
            "scene": Detection(class_name="outdoor"),
        },
    )
    record = _record("det", root)
    image = visualize_record(record, np.zeros((32, 32, 3), np.uint8))
    assert isinstance(image, Image)

    semantic = next(
        annotation
        for annotation in image.annotations
        if isinstance(annotation, SemanticMask)
    )
    classification = next(
        annotation
        for annotation in image.annotations
        if isinstance(annotation, Classification)
    )
    assert semantic.names == {1: "road"}
    assert list(classification.tags) == ["outdoor"]
    assert (
        sum(isinstance(annotation, BBox) for annotation in image.annotations)
        == 1
    )

    array_record = _record(
        "det",
        Detection(
            sub_detections={
                "features": Detection(
                    class_name=None,
                    array=ArrayAnnotation(path=array_path),
                )
            },
            class_name=None,
        ),
    )
    # An array on its own adds no side panel: reporting its shape as text was
    # a stand-in for not being able to draw it, and the field annotations
    # replaced that.
    plain = visualize_record(array_record, np.zeros((32, 32, 3), np.uint8))
    assert isinstance(plain, Image)
    assert plain.width == 32
    assert not any(
        isinstance(annotation, ArrayField)
        for annotation in plain.annotations  # off unless asked for
    )

    drawn = visualize_record(
        array_record,
        np.zeros((32, 48, 3), np.uint8),
        options=RenderOptions(array_view="overlay"),
    )
    assert isinstance(drawn, Image)
    assert any(
        isinstance(annotation, ArrayField) for annotation in drawn.annotations
    )


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
    from luxonis_ml.vizlab.adapters.ldf import metadata_annotations

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
    from luxonis_ml.vizlab.adapters.ldf import metadata_annotations

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


def test_metadata_annotations_treats_all_metadata_uniformly():
    """Every metadata key, "text" included, goes in the one metadata card."""
    from luxonis_ml.vizlab import InfoCard
    from luxonis_ml.vizlab.adapters.ldf import metadata_annotations

    det = Detection(
        class_name="ocr",
        keypoints=KeypointAnnotation(keypoints=[(0.5, 0.5, 2)]),
        metadata={"text": "HELLO", "conf": 0.9},
    )
    cards = metadata_annotations([det])
    assert len(cards) == 1
    assert isinstance(cards[0], InfoCard)
    assert cards[0].title == "metadata"
    assert cards[0].rows == ["ocr", "  text: HELLO", "  conf: 0.9"]


def test_metadata_annotations_recurses_sub_detections():
    from luxonis_ml.vizlab import InfoCard
    from luxonis_ml.vizlab.adapters.ldf import metadata_annotations

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
    cards = metadata_annotations([parent])
    assert len(cards) == 1
    assert isinstance(cards[0], InfoCard)
    assert cards[0].rows == ["note: AB123"]


def _record(task_name: str, *detections: Detection) -> DatasetRecord:
    return DatasetRecord.model_construct(
        files={}, annotation=list(detections), task_name=task_name
    )


def test_blend_drops_classification_chip_next_to_spatial() -> None:
    """Blending a classification task with a detection drops the corner chip."""
    from luxonis_ml.vizlab.adapters.ldf import blend_records_to_annotations

    classification = _record("classification", Detection(class_name="car"))
    detection = _record(
        "detection",
        Detection(
            class_name="car",
            boundingbox=BBoxAnnotation(x=0.1, y=0.1, w=0.3, h=0.3),
        ),
    )
    blended = blend_records_to_annotations([classification, detection])
    assert not any(isinstance(a, Classification) for a in blended)
    assert any(isinstance(a, BBox) for a in blended)


def test_blend_drops_mask_chip_for_a_class_a_box_already_labels() -> None:
    """A semantic mask's chip is dropped when a box shares its class."""
    from luxonis_ml.vizlab.adapters.ldf import blend_records_to_annotations

    car = np.zeros((20, 30), np.uint8)
    car[2:8, 2:12] = 1
    road = np.zeros((20, 30), np.uint8)
    road[12:18, 2:28] = 1
    car_seg = SegmentationAnnotation(mask=car)  # type: ignore[call-arg]
    road_seg = SegmentationAnnotation(mask=road)  # type: ignore[call-arg]
    detection = _record(
        "detection",
        Detection(
            class_name="car",
            boundingbox=BBoxAnnotation(x=0.1, y=0.1, w=0.3, h=0.3),
        ),
    )
    segmentation = _record(
        "segmentation",
        Detection(class_name="car", segmentation=car_seg),
        Detection(class_name="road", segmentation=road_seg),
    )
    palette = Palette(["car", "road"])
    options = RenderOptions(theme=DARK_THEME.with_palette(palette))
    blended = blend_records_to_annotations([detection, segmentation], options)

    masks = {m.label: m for m in blended if isinstance(m, Mask)}
    assert set(masks) == {"car", "road"}  # both masks keep their label...
    assert (
        masks["car"].label_chip is False
    )  # ...but car's redundant chip hides
    assert masks["road"].label_chip is True  # road (no box) keeps its chip
    # Hiding the chip must not change the fill/contour color: a suppressed mask
    # still resolves to its class color, matching the box (the regression).
    assert masks["car"].resolve_color(_ctx()) == palette.color_for("car")


def test_mask_label_chip_false_hides_chip_but_still_draws(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``label_chip=False`` skips the chip while fill and contour still render."""
    import luxonis_ml.vizlab.annotations.mask as mask_mod

    calls: list[str] = []
    monkeypatch.setattr(
        mask_mod, "place_label", lambda *a, **k: calls.append("chip")
    )

    disc = np.zeros((20, 20), np.uint8)
    disc[6:14, 6:14] = 1
    shown = Mask.from_ldf(SegmentationAnnotation(mask=disc), label="car")  # type: ignore[call-arg]
    hidden = Mask.from_ldf(SegmentationAnnotation(mask=disc), label="car")  # type: ignore[call-arg]
    hidden.label_chip = False

    Image(np.zeros((20, 20, 3), np.uint8)).add(shown).render()
    assert calls == ["chip"]  # labeled mask draws exactly one chip
    calls.clear()

    out = Image(np.zeros((20, 20, 3), np.uint8)).add(hidden).render()
    assert calls == []  # no chip drawn...
    assert out[10, 10, 3] > 0  # ...but the fill is still painted


def test_blend_keeps_mask_chip_when_no_box_labels_its_class() -> None:
    """Without a box of the same class, the mask keeps its own chip."""
    from luxonis_ml.vizlab.adapters.ldf import blend_records_to_annotations

    road = np.zeros((20, 30), np.uint8)
    road[12:18, 2:28] = 1
    road_seg = SegmentationAnnotation(mask=road)  # type: ignore[call-arg]
    segmentation = _record(
        "segmentation",
        Detection(class_name="road", segmentation=road_seg),
    )
    blended = blend_records_to_annotations([segmentation])
    assert all(
        m.label_chip for m in blended if isinstance(m, Mask)
    )  # chip kept
    assert [m.label for m in blended if isinstance(m, Mask)] == ["road"]


def test_blend_keeps_classification_when_it_is_the_only_content() -> None:
    """With nothing but class tags, the classification chips are kept."""
    from luxonis_ml.vizlab.adapters.ldf import blend_records_to_annotations

    blended = blend_records_to_annotations(
        [
            _record("car", Detection(class_name="car")),
            _record("motorbike", Detection(class_name="motorbike")),
        ]
    )
    assert len(blended) == 2
    assert all(isinstance(a, Classification) for a in blended)


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
    assert isinstance(img, Image)
    assert any(isinstance(a, InfoCard) for a in img.annotations)


def test_visualize_record_keeps_every_array_detection(
    tmp_path: Path,
) -> None:
    """Several array-bearing detections in one record all reach the scene."""
    paths = tmp_path / "first.npy", tmp_path / "second.npy"
    np.save(paths[0], np.zeros((4, 6), np.float32))
    np.save(paths[1], np.ones((4, 6), np.float32))
    record = _record(
        "det",
        *(
            Detection(class_name=None, array=ArrayAnnotation(path=path))
            for path in paths
        ),
    )
    drawn = visualize_record(
        record,
        np.zeros((32, 48, 3), np.uint8),
        options=RenderOptions(array_view="overlay"),
    )
    assert isinstance(drawn, Image)
    fields = [a for a in drawn.annotations if isinstance(a, ArrayField)]
    assert len(fields) == 2
