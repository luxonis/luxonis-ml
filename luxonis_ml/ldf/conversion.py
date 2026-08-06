"""Conversions between canonical LDF records and loader arrays."""

from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path

import numpy as np
from PIL import Image

from luxonis_ml.typing import Labels, LoaderOutput, Params

from .annotation import (
    Annotation,
    ArrayAnnotation,
    BBoxAnnotation,
    Category,
    ClassificationAnnotation,
    DatasetRecord,
    Detection,
    InstanceSegmentationAnnotation,
    KeypointAnnotation,
    SegmentationAnnotation,
)

_ANNOTATION_FIELDS = (
    "boundingbox",
    "keypoints",
    "segmentation",
    "instance_segmentation",
    "array",
)
_BACKGROUND = "background"


def _task_type(key: str) -> str:
    parts = key.split("/")
    if len(parts) > 1 and parts[-2] == "metadata":
        return f"metadata/{parts[-1]}"
    return parts[-1]


def _split_loader_key(key: str) -> tuple[str, str]:
    task_type = _task_type(key)
    suffix = f"/{task_type}"
    if key.endswith(suffix):
        return key[: -len(suffix)], task_type
    return "", task_type


def _loader_key(task_name: str, task_type: str) -> str:
    return f"{task_name}/{task_type}"


def _flatten_annotations(
    annotations: Mapping[str, Sequence[Detection]],
) -> dict[str, list[Detection]]:
    grouped: dict[str, list[Detection]] = defaultdict(list)

    def add(task_name: str, detection: Detection) -> None:
        grouped[task_name].append(detection)
        for name, sub_detection in detection.sub_detections.items():
            add(f"{task_name}/{name}", sub_detection)

    for task_name, detections in annotations.items():
        grouped.setdefault(task_name, [])
        for detection in detections:
            add(task_name, detection)
    return dict(grouped)


def _infer_tasks(
    annotations: Mapping[str, Sequence[Detection]],
) -> dict[str, list[str]]:
    tasks: dict[str, set[str]] = defaultdict(set)
    for task_name, detections in annotations.items():
        tasks.setdefault(task_name, set())
        for detection in detections:
            tasks[task_name].update(detection.get_task_types())
    return {
        task_name: sorted(task_types)
        for task_name, task_types in tasks.items()
    }


def _infer_classes(
    annotations: Mapping[str, Sequence[Detection]],
) -> dict[str, dict[str, int]]:
    class_names: dict[str, set[str]] = defaultdict(set)
    for task_name, detections in annotations.items():
        class_names.setdefault(task_name, set())
        for detection in detections:
            if detection.class_name is not None:
                class_names[task_name].add(detection.class_name)
    return {
        task_name: {
            name: index
            for index, name in enumerate(
                sorted(
                    names,
                    key=lambda name: (
                        (0, "") if name == _BACKGROUND else (1, name.lower())
                    ),
                )
            )
        }
        for task_name, names in class_names.items()
    }


def _infer_categorical_encodings(
    annotations: Mapping[str, Sequence[Detection]],
) -> dict[str, dict[str, int]]:
    values: dict[str, set[str]] = defaultdict(set)
    for task_name, detections in annotations.items():
        for detection in detections:
            for name, value in detection.metadata.items():
                if isinstance(value, Category):
                    values[_loader_key(task_name, f"metadata/{name}")].add(
                        str(value)
                    )
    return {
        task: {value: index for index, value in enumerate(sorted(task_values))}
        for task, task_values in values.items()
    }


def _load_images(record: DatasetRecord) -> dict[str, np.ndarray]:
    images: dict[str, np.ndarray] = {}
    for source_name, source in record.files.items():
        if isinstance(source, np.ndarray):
            images[source_name] = source
            continue
        with Image.open(source) as image:
            images[source_name] = np.asarray(image.convert("RGB"))
    return images


def _class_id(
    task_name: str,
    detection: Detection,
    classes: Mapping[str, Mapping[str, int]],
) -> int:
    if detection.class_name is None:
        return 0
    try:
        return classes[task_name][detection.class_name]
    except KeyError:
        raise ValueError(
            f"Class '{detection.class_name}' is not defined for task "
            f"'{task_name}'."
        ) from None


def _spatial_instance_order(
    detections: Sequence[Detection],
) -> list[int] | None:
    for field in (
        "boundingbox",
        "instance_segmentation",
        "keypoints",
        "array",
    ):
        instance_ids = [
            detection.instance_id
            for detection in detections
            if getattr(detection, field) is not None
        ]
        if instance_ids:
            return sorted(instance_ids)
    return None


def _align_metadata(
    values: Sequence[str | int | float | Category],
    value_instance_ids: list[int],
    instance_order: list[int] | None,
) -> np.ndarray:
    if instance_order is None:
        order = sorted(range(len(values)), key=lambda i: value_instance_ids[i])
        return np.array([values[i] for i in order])

    free_rows: dict[int, deque[int]] = defaultdict(deque)
    for row, instance_id in enumerate(instance_order):
        free_rows[instance_id].append(row)

    aligned: list[str | int | float | Category | None] = [None] * len(
        instance_order
    )
    for value, instance_id in zip(values, value_instance_ids, strict=True):
        rows = free_rows.get(instance_id)
        if not rows:
            return np.array(values)
        aligned[rows.popleft()] = value

    if any(value is None for value in aligned):
        return np.array(aligned, dtype=object)
    return np.array(aligned)


def _empty_label(
    task_name: str,
    task_type: str,
    *,
    classes: Mapping[str, Mapping[str, int]],
    n_keypoints: Mapping[str, int],
    image_shape: tuple[int, int],
) -> np.ndarray:
    height, width = image_shape
    if task_type == "boundingbox":
        return np.zeros((0, 5))
    if task_type == "keypoints":
        if task_name not in n_keypoints:
            raise ValueError(
                f"Cannot create an empty keypoint label for task '{task_name}' "
                "without its keypoint count. Pass `n_keypoints`."
            )
        return np.zeros((0, n_keypoints[task_name] * 3))
    if task_type == "instance_segmentation":
        return np.zeros((0, height, width))
    if task_type in {"segmentation", "classification"}:
        task_classes = classes.get(task_name)
        if not task_classes:
            raise ValueError(
                f"Cannot create an empty {task_type} label for task "
                f"'{task_name}' without its class mapping. Pass `classes`."
            )
        if task_type == "segmentation":
            return np.zeros((len(task_classes), height, width))
        return np.zeros((len(task_classes),))
    if task_type.startswith("metadata/") or task_type == "array":
        return np.array([])
    raise ValueError(f"Unknown task type: {task_type}")


def record_to_loader_output(
    record: DatasetRecord,
    *,
    tasks: dict[str, list[str]] | None = None,
    classes: dict[str, dict[str, int]] | None = None,
    categorical_encodings: dict[str, dict[str, int]] | None = None,
    n_keypoints: dict[str, int] | None = None,
    keep_categorical_as_strings: bool = False,
    images: dict[str, np.ndarray] | None = None,
) -> LoaderOutput:
    """Convert a canonical record into loader images and label arrays."""
    annotations = _flatten_annotations(record.annotation)
    inferred_tasks = _infer_tasks(annotations)
    if tasks is None:
        empty_tasks = [
            task_name
            for task_name, task_types in inferred_tasks.items()
            if not task_types
        ]
        if empty_tasks:
            raise ValueError(
                "Cannot infer task types for empty tasks "
                f"{sorted(empty_tasks)}. Pass the dataset's `tasks` mapping."
            )
        effective_tasks = inferred_tasks
    else:
        effective_tasks = {
            task: list(task_types) for task, task_types in tasks.items()
        }
        unknown_tasks = set(annotations) - set(effective_tasks)
        if unknown_tasks:
            raise ValueError(
                f"Record contains tasks absent from `tasks`: {sorted(unknown_tasks)}"
            )
        for task_name, inferred_types in inferred_tasks.items():
            missing_types = set(inferred_types) - set(
                effective_tasks[task_name]
            )
            if missing_types:
                raise ValueError(
                    f"Task '{task_name}' contains undeclared task types: "
                    f"{sorted(missing_types)}"
                )
        for task_name in effective_tasks:
            annotations.setdefault(task_name, [])

    effective_classes = (
        _infer_classes(annotations)
        if classes is None
        else {task: dict(mapping) for task, mapping in classes.items()}
    )
    effective_encodings = (
        _infer_categorical_encodings(annotations)
        if categorical_encodings is None
        else {
            task: dict(mapping)
            for task, mapping in categorical_encodings.items()
        }
    )
    keypoint_counts = n_keypoints or {}

    if images is None:
        output_images = _load_images(record)
    else:
        if set(images) != set(record.files):
            raise ValueError(
                "The `images` override must contain exactly the record's "
                "source names."
            )
        output_images = dict(images)
    first_image = next(iter(output_images.values()))
    image_shape = (first_image.shape[0], first_image.shape[1])

    labels: Labels = {}
    for task_name, task_types in effective_tasks.items():
        detections = annotations[task_name]
        spatial_order = _spatial_instance_order(detections)
        for task_type in task_types:
            key = _loader_key(task_name, task_type)
            if task_type.startswith("metadata/"):
                metadata_name = task_type[len("metadata/") :]
                rows = [
                    (
                        detection.instance_id,
                        detection.metadata[metadata_name],
                    )
                    for detection in detections
                    if metadata_name in detection.metadata
                ]
                if not rows:
                    labels[key] = _empty_label(
                        task_name,
                        task_type,
                        classes=effective_classes,
                        n_keypoints=keypoint_counts,
                        image_shape=image_shape,
                    )
                    continue
                values = [value for _, value in rows]
                if (
                    not keep_categorical_as_strings
                    and key in effective_encodings
                ):
                    try:
                        values = [
                            effective_encodings[key][str(value)]
                            for value in values
                        ]
                    except KeyError as error:
                        raise ValueError(
                            f"Metadata value {error.args[0]!r} is not encoded "
                            f"for task '{key}'."
                        ) from None
                labels[key] = _align_metadata(
                    values,
                    [instance_id for instance_id, _ in rows],
                    spatial_order,
                )
                continue

            annotation_rows: list[tuple[int, Annotation, int]] = []
            for detection in detections:
                if task_type == "classification":
                    if detection.class_name is None:
                        continue
                    annotation: Annotation = ClassificationAnnotation()
                elif task_type in _ANNOTATION_FIELDS:
                    value = getattr(detection, task_type)
                    if value is None:
                        continue
                    annotation = value
                else:
                    raise ValueError(f"Unknown task type: {task_type}")
                annotation_rows.append(
                    (
                        detection.instance_id,
                        annotation,
                        _class_id(task_name, detection, effective_classes),
                    )
                )

            if not annotation_rows:
                labels[key] = _empty_label(
                    task_name,
                    task_type,
                    classes=effective_classes,
                    n_keypoints=keypoint_counts,
                    image_shape=image_shape,
                )
                continue
            annotation_rows.sort(key=lambda row: row[0])
            task_annotations = [row[1] for row in annotation_rows]
            class_ids = [row[2] for row in annotation_rows]
            labels[key] = task_annotations[0].combine_to_numpy(
                task_annotations,
                class_ids,
                len(effective_classes.get(task_name, {})),
            )

    return LoaderOutput(
        output_images,
        labels,
        deepcopy(record.sample_metadata),
        classes=effective_classes,
        categorical_encodings=effective_encodings,
    )


def labels_to_record(
    labels: Labels,
    *,
    classes: dict[str, dict[str, int]],
    images: dict[str, np.ndarray] | None = None,
    sample_metadata: Params | None = None,
    categorical_encodings: dict[str, dict[str, int]] | None = None,
    render_background: bool = False,
) -> DatasetRecord:
    """Convert loader label arrays into one canonical LDF record."""
    grouped: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    for key, array in labels.items():
        task_name, task_type = _split_loader_key(key)
        grouped[task_name][task_type] = array

    files: dict[str, np.ndarray | Path] = (
        dict(images) if images else {"image": Path("<loader>")}
    )
    annotations: dict[str, list[Detection]] = {}
    for task_name, task_types in grouped.items():
        id_to_name = {
            index: name.strip()
            for name, index in classes.get(task_name, {}).items()
        }
        annotations[task_name] = _build_detections(
            task_name,
            task_types,
            id_to_name,
            categorical_encodings or {},
            render_background,
        )
    return DatasetRecord.model_construct(
        files=files,
        annotation=annotations,
        sample_metadata=deepcopy(sample_metadata or {}),
    )


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
    if class_id is not None and class_id < len(row):
        return row[class_id], class_id
    for slot_id, slot in enumerate(row):
        if np.any(slot):
            return slot, slot_id
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
        full_task = _loader_key(task_name, task_type)
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


__all__ = ["labels_to_record", "record_to_loader_output"]
