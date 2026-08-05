"""Rebuild LDF records from the arrays returned by `LuxonisLoader`."""

from collections import defaultdict
from pathlib import Path

import numpy as np

from luxonis_ml.data.utils.task_utils import get_task_name, get_task_type
from luxonis_ml.ldf import (
    ArrayAnnotation,
    BBoxAnnotation,
    Category,
    DatasetRecord,
    Detection,
    InstanceSegmentationAnnotation,
    KeypointAnnotation,
    SegmentationAnnotation,
)
from luxonis_ml.typing import Labels

_BACKGROUND = "background"


def _split_loader_key(key: str) -> tuple[str, str]:
    """Split a loader key while preserving a nested task's full path."""
    task_type = get_task_type(key)
    suffix = f"/{task_type}"
    if key.endswith(suffix):
        return key[: -len(suffix)], task_type
    return get_task_name(key), task_type


def labels_to_records(
    labels: Labels,
    *,
    classes: dict[str, dict[str, int]],
    images: dict[str, np.ndarray] | None = None,
    categorical_encodings: dict[str, dict[str, int]] | None = None,
    render_background: bool = False,
) -> dict[str, DatasetRecord]:
    """Convert `LuxonisLoader` label arrays into canonical LDF records.

    Args:
        labels: Arrays keyed by ``"task_name/task_type"``.
        classes: Class names and ids grouped by task name.
        images: Images to attach to each record. A placeholder path is used
            when omitted.
        categorical_encodings: Encodings used to restore categorical metadata.
        render_background: Keep the semantic-segmentation background class as a
            drawable mask instead of dropping it.

    Returns:
        One `DatasetRecord` per task name.

    """
    grouped: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    for key, array in labels.items():
        task_name, task_type = _split_loader_key(key)
        grouped[task_name][task_type] = array

    files: dict[str, np.ndarray | Path] = (
        dict(images) if images else {"image": Path("<loader>")}
    )

    records: dict[str, DatasetRecord] = {}
    for task_name, task_types in grouped.items():
        # Strip class names before use, so labels render cleanly and the
        # "background" check matches even when a name has stray whitespace.
        id_to_name = {
            index: name.strip()
            for name, index in classes.get(task_name, {}).items()
        }
        detections = _build_detections(
            task_name,
            task_types,
            id_to_name,
            categorical_encodings or {},
            render_background,
        )
        # Nested loader task names ("det/face") are not valid identifiers, so
        # the record is built unvalidated rather than rejected.
        records[task_name] = DatasetRecord.model_construct(
            files=files,
            annotation=detections,
            task_name=task_name,
            sample_metadata={},
        )
    return records


def _build_detections(
    task_name: str,
    task_types: dict[str, np.ndarray],
    id_to_name: dict[int, str],
    categorical_encodings: dict[str, dict[str, int]],
    render_background: bool = False,
) -> list[Detection]:
    detections: list[Detection] = []
    detections.extend(
        _instance_detections(
            task_name, task_types, id_to_name, categorical_encodings
        )
    )
    detections.extend(
        _semantic_detections(task_types, id_to_name, render_background)
    )
    # Bounding boxes and semantic masks carry their classes directly, so their
    # derived classification vector would be redundant. Keypoints and instance
    # masks do not encode class ids, however, so retain their vector as
    # record-level class tags when none of the reconstructed instances is named.
    if not any(detection.class_name is not None for detection in detections):
        detections.extend(_classification_detections(task_types, id_to_name))
    return detections


def _instance_detections(
    task_name: str,
    task_types: dict[str, np.ndarray],
    id_to_name: dict[int, str],
    categorical_encodings: dict[str, dict[str, int]],
) -> list[Detection]:
    boxes = task_types.get("boundingbox")
    keypoints = task_types.get("keypoints")
    instance_masks = task_types.get("instance_segmentation")
    arrays = task_types.get("array")

    # Metadata arrays are per-instance too, so a metadata-only task (e.g. OCR
    # with just ``metadata/text``) still yields one detection per entry.
    metadata_lengths = [
        len(np.asarray(arr))
        for task_type, arr in task_types.items()
        if task_type.startswith("metadata/") and np.asarray(arr).ndim == 1
    ]
    spatial_lengths = [
        len(arr)
        for arr in (boxes, keypoints, instance_masks, arrays)
        if arr is not None
    ]
    n_instances = max([*spatial_lengths, *metadata_lengths], default=0)
    metadata = _decode_metadata(
        task_name, task_types, n_instances, categorical_encodings
    )

    detections: list[Detection] = []
    for i in range(n_instances):
        class_id: int | None = None
        class_name: str | None = None
        boundingbox: BBoxAnnotation | None = None
        keypoint_annotation: KeypointAnnotation | None = None
        instance_segmentation: InstanceSegmentationAnnotation | None = None
        array: ArrayAnnotation | None = None
        if boxes is not None and i < len(boxes):
            row = boxes[i]
            class_id = int(row[0])
            name = id_to_name.get(class_id)
            # "background" is a bookkeeping class, not a real label.
            class_name = None if name == _BACKGROUND else name
            boundingbox = BBoxAnnotation(
                x=float(row[1]),
                y=float(row[2]),
                w=float(row[3]),
                h=float(row[4]),
            )
        if keypoints is not None and i < len(keypoints):
            triples = np.asarray(keypoints[i], dtype=float).reshape(-1, 3)
            keypoint_annotation = KeypointAnnotation.model_validate(
                {
                    "keypoints": [
                        (float(x), float(y), round(v)) for x, y, v in triples
                    ]
                }
            )
        if instance_masks is not None and i < len(instance_masks):
            instance_segmentation = (
                InstanceSegmentationAnnotation.model_validate(
                    {"mask": np.asarray(instance_masks[i]).astype(np.uint8)}
                )
            )
        if arrays is not None and i < len(arrays):
            data, slot_id = _unpad_class_slot(np.asarray(arrays[i]), class_id)
            array = ArrayAnnotation(array=data)
            if class_name is None and slot_id is not None:
                name = id_to_name.get(slot_id)
                class_name = None if name == _BACKGROUND else name
        detections.append(
            Detection(
                instance_id=i,
                class_name=class_name,
                boundingbox=boundingbox,
                keypoints=keypoint_annotation,
                instance_segmentation=instance_segmentation,
                array=array,
                metadata=metadata.get(i, {}),
            )
        )
    return detections


def _unpad_class_slot(
    row: np.ndarray, class_id: int | None
) -> tuple[np.ndarray, int | None]:
    """Return the stored array and its identifiable class slot."""
    if class_id is not None and class_id < len(row):
        return row[class_id], class_id
    for slot_id, slot in enumerate(row):
        if np.any(slot):
            return slot, slot_id
    # An all-zero array has no identifiable class slot.
    return row[0], None


def _semantic_detections(
    task_types: dict[str, np.ndarray],
    id_to_name: dict[int, str],
    render_background: bool = False,
) -> list[Detection]:
    semantic = task_types.get("segmentation")
    if semantic is None:
        return []
    detections: list[Detection] = []
    for class_id, channel in enumerate(np.asarray(semantic)):
        name = id_to_name.get(class_id)
        if (name == _BACKGROUND and not render_background) or not np.any(
            channel
        ):
            continue
        detections.append(
            Detection(
                class_name=name,
                segmentation=SegmentationAnnotation.model_validate(
                    {"mask": np.asarray(channel).astype(np.uint8)}
                ),
            )
        )
    return detections


def _classification_detections(
    task_types: dict[str, np.ndarray], id_to_name: dict[int, str]
) -> list[Detection]:
    classification = task_types.get("classification")
    if classification is None:
        return []
    detections: list[Detection] = []
    for class_id in np.flatnonzero(np.asarray(classification)):
        name = id_to_name.get(int(class_id))
        if name is not None and name != _BACKGROUND:
            detections.append(Detection(class_name=name))
    return detections


def _decode_metadata(
    task_name: str,
    task_types: dict[str, np.ndarray],
    n_instances: int,
    categorical_encodings: dict[str, dict[str, int]],
) -> dict[int, dict[str, int | float | str | Category]]:
    """Recover per-instance metadata dicts keyed by instance index.

    Only metadata arrays whose length matches the instance count are attached,
    and within those the padded rows of instances that do not carry the field
    are skipped. Categorical ids are decoded back to their string names when
    an encoding for the task is available.
    """
    per_instance: dict[int, dict[str, int | float | str | Category]] = (
        defaultdict(dict)
    )
    for task_type, array in task_types.items():
        if not task_type.startswith("metadata/"):
            continue
        key = task_type[len("metadata/") :]
        values = np.asarray(array)
        if values.ndim != 1 or len(values) != n_instances:
            continue
        full_task = f"{task_name}/{task_type}"
        decoder = None
        if full_task in categorical_encodings:
            decoder = {
                index: name
                for name, index in categorical_encodings[full_task].items()
            }
        for i, value in enumerate(values):
            item = value.item() if hasattr(value, "item") else value
            if item is None:
                continue
            if decoder is not None and isinstance(item, (int, float)):
                item = decoder.get(int(item), item)
            if isinstance(item, (str, int, float, Category)):
                per_instance[i][key] = item
            else:
                per_instance[i][key] = str(item)
    return dict(per_instance)
