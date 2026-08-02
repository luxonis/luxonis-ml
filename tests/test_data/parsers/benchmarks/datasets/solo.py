"""Benchmark datasets for the SOLO parser.

The generated dataset follows the Unity SOLO layout: a ``train``,
``valid`` and ``test`` split, each holding the four definition JSON files
the parser validates and a number of ``sequence.<n>`` directories with
one frame per step.

Every annotation type the parser understands is present in each frame:

* ``SemanticSegmentationAnnotation``, one class instance per mask colour,
  each yielding its own ``segmentation`` record,
* ``BoundingBox2DAnnotation``, ``InstanceSegmentationAnnotation`` and
  ``KeypointAnnotation``, which share instance ids so the parser merges
  them into one record per instance.

Also exercised: class names and keypoint labels ordered by id rather than
by position, a second frame file repeating the annotations of a step the
parser already emitted, a capture with no annotations at all, a mask
colour per instance so that mask decoding does real work, and one
annotation whose ``@type`` omits the ``type.unity.com/unity.solo.``
prefix, which the parser matches by suffix.

Not covered: the flat layout where the dataset root is itself a single
split. It is mutually exclusive with the split-per-directory layout, and
the latter is what Unity emits and what ``detect`` targets.
"""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tests.test_data.parsers.benchmarks.harness import (
    BenchmarkCase,
    Scale,
    bbox,
    benchmark_dataset,
    class_names,
    keypoint_names,
    keypoints,
    write_images,
    write_masks,
)

SPLITS = ("train", "valid", "test")
SOLO_PREFIX = "type.unity.com/unity.solo."

STEPS_PER_SEQUENCE = 10
"""Frames written per sequence directory."""

SEMANTIC_CLASSES = 4
"""Classes visible in a semantic mask, capped by the scale's classes."""

INSTANCE_COLOR_OFFSET = 100
"""Keeps instance colours distinct from the semantic ones."""


def _color(index: int) -> tuple[int, int, int]:
    """Return a deterministic RGB colour, distinct for every index."""
    return (
        (53 * index + 17) % 256,
        (97 * index + 33) % 256,
        (29 * index + 61) % 256,
    )


def _bgr(color: tuple[int, int, int]) -> tuple[int, int, int]:
    """Reorder an RGB colour into the channel order OpenCV writes."""
    red, green, blue = color
    return (blue, green, red)


def _annotation_definitions(scale: Scale) -> dict[str, Any]:
    """Return the ``annotation_definitions.json`` payload.

    Class ids and keypoint indices run backwards so the parser has to sort
    the names it reports instead of taking them in file order.
    """
    names = class_names(scale.classes)
    labels = keypoint_names(scale.keypoints)
    return {
        "annotationDefinitions": [
            {
                "@type": f"{SOLO_PREFIX}SemanticSegmentationAnnotation",
                "id": "semantic segmentation",
                "spec": [
                    {"label_name": name, "pixel_value": [*_color(index), 255]}
                    for index, name in enumerate(names)
                ],
            },
            {
                "@type": f"{SOLO_PREFIX}BoundingBox2DAnnotation",
                "id": "bounding box",
                "spec": [
                    {"label_name": name, "label_id": scale.classes - index}
                    for index, name in enumerate(names)
                ],
            },
            {
                "@type": f"{SOLO_PREFIX}InstanceSegmentationAnnotation",
                "id": "instance segmentation",
                "spec": [],
            },
            {
                "@type": f"{SOLO_PREFIX}KeypointAnnotation",
                "id": "keypoints",
                "template": {
                    "templateId": "skeleton",
                    "keypoints": [
                        {"label": label, "index": scale.keypoints - 1 - index}
                        for index, label in enumerate(labels)
                    ],
                },
            },
        ]
    }


def _annotations(scale: Scale, semantic_classes: int) -> list[dict[str, Any]]:
    """Return the annotation blocks of one annotated capture.

    Mask filenames carry a ``__N__`` placeholder that
    :func:`_render_frame` replaces with the step number.
    """
    names = class_names(scale.classes)
    size = scale.image_size
    boxes: list[dict[str, Any]] = []
    instances: list[dict[str, Any]] = []
    skeletons: list[dict[str, Any]] = []
    for index in range(scale.annotations_per_image):
        instance_id = index + 1
        class_id = index % scale.classes
        x, y, width, height = bbox(index)
        boxes.append(
            {
                "labelId": class_id,
                "labelName": names[class_id],
                "instanceId": instance_id,
                "origin": [x * size, y * size],
                "dimension": [width * size, height * size],
            }
        )
        instances.append(
            {
                "instanceId": instance_id,
                "color": [*_color(INSTANCE_COLOR_OFFSET + index), 255],
            }
        )
        skeletons.append(
            {
                "instanceId": instance_id,
                "labelId": class_id,
                "pose": "unset",
                "keypoints": [
                    {
                        "index": order,
                        "location": [kx * size, ky * size],
                        "state": state,
                    }
                    for order, (kx, ky, state) in enumerate(
                        keypoints(index, scale.keypoints)
                    )
                ],
            }
        )
    return [
        {
            "@type": f"{SOLO_PREFIX}SemanticSegmentationAnnotation",
            "id": "semantic segmentation",
            "filename": "step__N__.camera.semantic.png",
            "instances": [
                {
                    "labelName": names[index],
                    "pixelValue": [*_color(index), 255],
                }
                for index in range(semantic_classes)
            ],
        },
        {
            "@type": f"{SOLO_PREFIX}BoundingBox2DAnnotation",
            "id": "bounding box",
            "values": boxes,
        },
        {
            # The parser matches annotation types by suffix, so the
            # unprefixed spelling has to be understood as well.
            "@type": "InstanceSegmentationAnnotation",
            "id": "instance segmentation",
            "filename": "step__N__.camera.instance.png",
            "instances": instances,
        },
        {
            "@type": f"{SOLO_PREFIX}KeypointAnnotation",
            "id": "keypoints",
            "values": skeletons,
        },
    ]


def _frame_templates(scale: Scale, semantic_classes: int) -> tuple[str, str]:
    """Return the frame JSON without and with an unannotated capture.

    Serializing the annotations once and substituting the step number per
    frame keeps generation cheap; the payload is by far the largest part
    of a SOLO dataset.
    """
    size = scale.image_size
    capture = {
        "@type": f"{SOLO_PREFIX}RGBCamera",
        "id": "camera",
        "filename": "step__N__.camera.jpg",
        "dimension": [size, size],
        "annotations": _annotations(scale, semantic_classes),
    }
    background = {
        "@type": f"{SOLO_PREFIX}RGBCamera",
        "id": "background",
        "filename": "step__N__.camera.background.jpg",
        "dimension": [size, size],
        "annotations": [],
    }
    return (
        json.dumps({"step": "__STEP__", "captures": [capture]}),
        json.dumps({"step": "__STEP__", "captures": [capture, background]}),
    )


def _render_frame(template: str, step: int) -> str:
    """Fill ``template`` in for ``step``.

    The quoted placeholder is replaced without its quotes so the step is
    serialized as the number Unity writes.
    """
    return template.replace('"__STEP__"', str(step)).replace(
        "__N__", str(step)
    )


def _write_sequence(
    sequence_path: Path,
    scale: Scale,
    semantic_colors: Sequence[tuple[int, int, int]],
    instance_colors: Sequence[tuple[int, int, int]],
    templates: tuple[str, str],
) -> None:
    """Write one sequence directory with its images, masks and frames."""
    steps = range(STEPS_PER_SEQUENCE)
    names = [f"step{step}.camera" for step in steps]
    write_images(
        sequence_path,
        [*names, "step0.camera.background"],
        scale.image_size,
    )
    write_masks(
        sequence_path,
        [f"{name}.semantic" for name in names],
        scale.image_size,
        semantic_colors,
    )
    write_masks(
        sequence_path,
        [f"{name}.instance" for name in names],
        scale.image_size,
        instance_colors,
    )

    plain, with_background = templates
    for step in steps:
        payload = _render_frame(with_background if step == 0 else plain, step)
        (sequence_path / f"step{step}.frame_data.json").write_text(payload)
        if step == 0:
            # Unity splits a step over several frame files; the parser
            # must emit the annotations of a step only once.
            (sequence_path / "step0.frame_data.copy.json").write_text(payload)


def _write_split(split_path: Path, scale: Scale, sequences: int) -> None:
    """Write one SOLO split holding ``sequences`` sequences."""
    split_path.mkdir(parents=True, exist_ok=True)
    semantic_classes = min(SEMANTIC_CLASSES, scale.classes)
    (split_path / "annotation_definitions.json").write_text(
        json.dumps(_annotation_definitions(scale))
    )
    (split_path / "metadata.json").write_text(
        json.dumps(
            {
                "unityVersion": "2022.3.0f1",
                "perceptionVersion": "1.0.0",
                "totalFrames": sequences * STEPS_PER_SEQUENCE,
                "totalSequences": sequences,
                "sensors": ["camera"],
            }
        )
    )
    (split_path / "metric_definitions.json").write_text(
        json.dumps({"metricDefinitions": []})
    )
    (split_path / "sensor_definitions.json").write_text(
        json.dumps(
            {
                "sensorDefinitions": [
                    {
                        "@type": f"{SOLO_PREFIX}RGBCamera",
                        "id": "camera",
                        "modality": "camera",
                    }
                ]
            }
        )
    )

    templates = _frame_templates(scale, semantic_classes)
    semantic_colors = [
        _bgr(_color(index)) for index in range(semantic_classes)
    ]
    instance_colors = [
        _bgr(_color(INSTANCE_COLOR_OFFSET + index))
        for index in range(scale.annotations_per_image)
    ]
    for sequence in range(sequences):
        _write_sequence(
            split_path / f"sequence.{sequence}",
            scale,
            semantic_colors,
            instance_colors,
            templates,
        )


@benchmark_dataset("solo")
def build_solo(root: Path, scale: Scale) -> BenchmarkCase:
    """Three SOLO splits carrying every supported annotation type."""
    # SOLO is the heaviest supported format per image - two PNG masks and
    # a frame JSON holding a full skeleton per instance — but the frame
    # payload is serialized once and reused, so the common image count
    # still generates in about a second and is kept unshrunk.
    sequences = max(1, scale.images // STEPS_PER_SEQUENCE)
    for split in SPLITS:
        _write_split(root / split, scale, sequences)

    frames = len(SPLITS) * sequences * STEPS_PER_SEQUENCE
    semantic_classes = min(SEMANTIC_CLASSES, scale.classes)
    return BenchmarkCase(
        dataset_type="solo",
        source=root,
        features=(
            "semantic segmentation",
            "instance segmentation",
            "detection",
            "keypoints",
            "instance merging",
            "unprefixed annotation types",
            "repeated frame files",
            "unannotated captures",
            "multi-split",
            "mask decode",
        ),
        # The unannotated capture of every sequence is a real file the
        # parser stats, but it never reaches a record.
        images=frames + len(SPLITS) * sequences,
        expected_records=frames
        * (semantic_classes + scale.annotations_per_image),
        expected_files=frames,
    )
