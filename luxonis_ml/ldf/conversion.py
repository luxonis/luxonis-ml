"""Conversions between LDF records and the arrays a loader returns.

A record describes a sample the way it is stored: detections grouped by task,
each holding its own annotations. A loader returns the same sample the way a
model consumes it: one array per task and task type, with every instance a row.

The two directions are `record_to_loader_output` and `labels_to_record`, and
they are reached through `DatasetRecord.to_loader_output` and
`LoaderOutput.to_ldf`. Both need the `DatasetSchema` of the dataset, because a
record names its classes while an array numbers them.

Instances are paired by row index. Row :math:`i` of every array of one task
describes the same instance, which is the invariant the augmentation engine
already relies on when it filters keypoints and masks by their bounding box.
"""

from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from copy import deepcopy

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
from .schema import SCHEMA_METADATA_KEY, DatasetSchema
from .tasks import get_task_group, get_task_type

__all__ = ["labels_to_record", "record_to_loader_output"]

#: Task types that describe one instance each, in the order they are trusted
#: to define the instance order of a task.
_INSTANCE_TASK_TYPES = (
    "boundingbox",
    "instance_segmentation",
    "keypoints",
    "array",
)

_ANNOTATION_TYPES: dict[str, type[Annotation]] = {
    "boundingbox": BBoxAnnotation,
    "keypoints": KeypointAnnotation,
    "segmentation": SegmentationAnnotation,
    "instance_segmentation": InstanceSegmentationAnnotation,
    "array": ArrayAnnotation,
}

_METADATA_PREFIX = "metadata/"
#: Classification carries no data, so every row shares one annotation.
_CLASSIFICATION = ClassificationAnnotation()


def record_to_loader_output(
    record: DatasetRecord,
    schema: DatasetSchema,
    *,
    images: dict[str, np.ndarray] | None = None,
    keep_categorical_as_strings: bool = False,
    include_empty: bool = True,
) -> LoaderOutput:
    """Convert a record into the arrays a loader returns.

    Args:
        record: Record to convert.
        schema: Schema of the dataset the record belongs to.
        images: Images keyed by source name. Read from the record's files
            when omitted.
        keep_categorical_as_strings: Whether categorical metadata keeps its
            string values instead of the integers the schema encodes them as.
        include_empty: Whether every task of the schema gets a label, even
            when the record has no annotation for it.

    Returns:
        The images, one label per task and task type, and the record's
        metadata with the schema attached.

    """
    annotations = _flatten_sub_detections(record.annotation)
    images = _load_images(record) if images is None else dict(images)
    image_shape = next(iter(images.values())).shape[:2]

    task_types: dict[str, set[str]] = (
        {name: set(types) for name, types in schema.tasks.items()}
        if include_empty
        else {}
    )
    for task_name, detections in annotations.items():
        present = task_types.setdefault(task_name, set())
        for detection in detections:
            present |= detection.get_task_types()

    labels: Labels = {}
    for task_name, types in task_types.items():
        # Already in instance-ID order: `_flatten_sub_detections` sorts each
        # task before it collects the children, so that a child keeps the
        # row of its parent. Sorting again here would undo that.
        detections = annotations.get(task_name, [])
        instance_order = _instance_order(detections)
        for task_type in sorted(types):
            labels[f"{task_name}/{task_type}"] = _build_label(
                task_name,
                task_type,
                detections,
                instance_order,
                schema,
                image_shape=image_shape,
                keep_categorical_as_strings=keep_categorical_as_strings,
            )

    metadata = deepcopy(record.sample_metadata)
    metadata[SCHEMA_METADATA_KEY] = schema.as_metadata
    return LoaderOutput(images, labels, metadata)


def labels_to_record(
    labels: Labels,
    schema: DatasetSchema,
    *,
    images: dict[str, np.ndarray] | None = None,
    sample_metadata: Params | None = None,
    keep_background: bool = False,
) -> DatasetRecord:
    """Convert loader labels back into one record.

    Args:
        labels: Label arrays keyed by ``"task_name/task_type"``.
        schema: Schema of the dataset the labels came from.
        images: Images keyed by source name, held in memory by the record.
        sample_metadata: Record-level metadata to keep on the record.
        keep_background: Whether the background class of a semantic mask
            becomes a detection of its own. It is dropped by default, as the
            loader adds it to fill the pixels no class claimed.

    Returns:
        One record holding every detection the labels describe.

    """
    by_task: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    for key, array in labels.items():
        by_task[get_task_group(key)][get_task_type(key)] = array

    annotations = {
        task_name: _build_detections(
            task_name, task_types, schema, keep_background=keep_background
        )
        for task_name, task_types in by_task.items()
    }
    # Every part is already a validated model, and validating the record
    # would report our own `files` as the deprecated input name.
    return DatasetRecord.model_construct(
        files=dict(images or {}),
        annotation=_nest_sub_detections(annotations),
        sample_metadata=deepcopy(sample_metadata or {}),
    )


def _flatten_sub_detections(
    annotations: Mapping[str, Sequence[Detection]],
) -> dict[str, list[Detection]]:
    """Promote every sub-detection to a task of its own.

    A sub-detection is annotated separately from its parent, so a loader sees
    ``"driver/face"`` as an ordinary task.

    The parents are put in instance-ID order here, before their children are
    collected, so a child keeps the row of its parent. Sorting each task
    separately would not: a sub-detection carries the default instance ID, so
    its own sort leaves it where it was written while the parent moves.
    """
    flattened: dict[str, list[Detection]] = defaultdict(list)

    def add(task_name: str, detection: Detection) -> None:
        flattened[task_name].append(detection)
        for name, sub_detection in detection.sub_detections.items():
            add(f"{task_name}/{name}", sub_detection)

    for task_name, detections in annotations.items():
        flattened.setdefault(task_name, [])
        for detection in sorted(
            detections, key=lambda detection: detection.instance_id
        ):
            add(task_name, detection)
    return dict(flattened)


def _nest_sub_detections(
    annotations: Mapping[str, Sequence[Detection]],
) -> dict[str, list[Detection]]:
    """Put every sub-task back under the detection it belongs to.

    A task named ``"driver/face"`` is the ``"face"`` sub-detection of the
    ``"driver"`` task. Its detections pair with their parents by row index,
    which is the same order the instance IDs gave them.
    """
    nested = {
        name: list(detections) for name, detections in annotations.items()
    }
    # The deepest tasks nest first, so a chain such as "a/b/c" ends up under
    # "a/b" before "a/b" itself moves under "a".
    for task_name in sorted(
        nested, key=lambda name: name.count("/"), reverse=True
    ):
        if "/" not in task_name:
            # A task at the top level. Without this the default task, whose
            # name is empty, would claim every other task as its sub-task.
            continue
        parent_name, _, sub_name = task_name.rpartition("/")
        if not sub_name or parent_name not in nested:
            continue
        parents = nested[parent_name]
        children = nested.pop(task_name)
        for index, child in enumerate(children):
            if index < len(parents):
                parents[index].sub_detections[sub_name] = child
            else:
                # A sub-detection with no parent of its own keeps its task,
                # because dropping it would lose an annotation.
                nested.setdefault(task_name, []).append(child)
    return nested


def _load_images(record: DatasetRecord) -> dict[str, np.ndarray]:
    images: dict[str, np.ndarray] = {}
    for source_name, file in record.files.items():
        if isinstance(file, np.ndarray):
            images[source_name] = file
            continue
        with Image.open(file) as image:
            images[source_name] = np.asarray(image.convert("RGB"))
    return images


def _instance_order(detections: Sequence[Detection]) -> list[int] | None:
    """Return the instance IDs of the rows of a task, in row order."""
    for task_type in _INSTANCE_TASK_TYPES:
        instance_ids = [
            detection.instance_id
            for detection in detections
            if getattr(detection, task_type) is not None
        ]
        if instance_ids:
            return instance_ids
    return None


def _build_label(
    task_name: str,
    task_type: str,
    detections: Sequence[Detection],
    instance_order: list[int] | None,
    schema: DatasetSchema,
    *,
    image_shape: tuple[int, ...],
    keep_categorical_as_strings: bool,
) -> np.ndarray:
    if task_type.startswith(_METADATA_PREFIX):
        return _build_metadata_label(
            task_name,
            task_type,
            detections,
            instance_order,
            schema,
            keep_categorical_as_strings=keep_categorical_as_strings,
        )

    annotations: list[Annotation] = []
    class_ids: list[int] = []
    for detection in detections:
        if task_type == "classification":
            # The class is the whole annotation, so a detection without one
            # contributes nothing, and the rest share one annotation object.
            annotation = (
                _CLASSIFICATION if detection.class_name is not None else None
            )
        else:
            annotation = getattr(detection, task_type, None)
        if annotation is None:
            continue
        annotations.append(annotation)
        class_ids.append(
            0
            if detection.class_name is None
            else schema.class_id(task_name, detection.class_name)
        )

    if not annotations:
        return _empty_label(task_name, task_type, schema, image_shape)
    return annotations[0].combine_to_numpy(
        annotations, class_ids, schema.n_classes(task_name)
    )


def _build_metadata_label(
    task_name: str,
    task_type: str,
    detections: Sequence[Detection],
    instance_order: list[int] | None,
    schema: DatasetSchema,
    *,
    keep_categorical_as_strings: bool,
) -> np.ndarray:
    name = task_type[len(_METADATA_PREFIX) :]
    rows = [
        (detection.instance_id, detection.metadata[name])
        for detection in detections
        if name in detection.metadata
    ]
    if not rows:
        return np.zeros(0)

    values = [value for _, value in rows]
    encoding = schema.categorical_encodings.get(f"{task_name}/{task_type}")
    if encoding is not None and not keep_categorical_as_strings:
        values = [encoding[str(value)] for value in values]
    return _align_to_instances(
        values, [instance_id for instance_id, _ in rows], instance_order
    )


def _align_to_instances(
    values: Sequence[str | int | float | Category],
    instance_ids: Sequence[int],
    instance_order: Sequence[int] | None,
) -> np.ndarray:
    """Put per-instance values into the row order of their task.

    ``instance_order`` holds the instance ID of each row of the task, and is
    ``None`` when the task has no instance-shaped label to take an order
    from. The result has one value per row, and a row whose instance carries
    no value of this kind is filled with ``None``.
    """
    if instance_order is None:
        order = sorted(range(len(values)), key=lambda i: instance_ids[i])
        return np.array([values[i] for i in order])

    free_rows: dict[int, deque[int]] = defaultdict(deque)
    for row, instance_id in enumerate(instance_order):
        free_rows[instance_id].append(row)

    aligned: list[str | int | float | Category | None] = [None] * len(
        instance_order
    )
    for value, instance_id in zip(values, instance_ids, strict=True):
        rows = free_rows.get(instance_id)
        if not rows:
            # More values than rows, or an instance the task does not have.
            # The pairing is unknown, so the values stay as they came.
            return np.array(values)
        aligned[rows.popleft()] = value

    if any(value is None for value in aligned):
        return np.array(aligned, dtype=object)
    return np.array(aligned)


def _empty_label(
    task_name: str,
    task_type: str,
    schema: DatasetSchema,
    image_shape: tuple[int, ...],
) -> np.ndarray:
    height, width = image_shape[0], image_shape[1]
    if task_type == "boundingbox":
        return np.zeros((0, 5))
    if task_type == "keypoints":
        return np.zeros((0, schema.n_keypoints.get(task_name, 0) * 3))
    if task_type == "instance_segmentation":
        return np.zeros((0, height, width))
    if task_type == "segmentation":
        return np.zeros((schema.n_classes(task_name), height, width))
    if task_type == "classification":
        return np.zeros(schema.n_classes(task_name))
    return np.zeros(0)


def _build_detections(
    task_name: str,
    task_types: Mapping[str, np.ndarray],
    schema: DatasetSchema,
    *,
    keep_background: bool,
) -> list[Detection]:
    detections = _instance_detections(task_name, task_types, schema)
    detections.extend(
        _semantic_detections(
            task_name, task_types, schema, keep_background=keep_background
        )
    )
    if all(detection.class_name is None for detection in detections):
        detections.extend(
            _classification_detections(task_name, task_types, schema)
        )
    return detections


def _instance_detections(
    task_name: str,
    task_types: Mapping[str, np.ndarray],
    schema: DatasetSchema,
) -> list[Detection]:
    """Rebuild the detections that describe one instance each."""
    split: dict[str, list[tuple[Annotation, int | None]]] = {}
    for task_type in _INSTANCE_TASK_TYPES:
        array = task_types.get(task_type)
        if array is not None and len(array):
            split[task_type] = _ANNOTATION_TYPES[task_type].split_from_numpy(
                array
            )

    metadata = _split_metadata(task_name, task_types, schema)
    # The metadata is keyed by row index, and a row in the middle can carry
    # no value, so the count of the keys is not the number of rows.
    metadata_rows = [max(metadata) + 1] if metadata else []
    n_instances = max(
        [len(rows) for rows in split.values()] + metadata_rows, default=0
    )

    detections = []
    for index in range(n_instances):
        fields: dict[str, Annotation] = {}
        class_id: int | None = None
        for task_type, rows in split.items():
            if index >= len(rows):
                continue
            annotation, row_class_id = rows[index]
            fields[task_type] = annotation
            if class_id is None:
                class_id = row_class_id
        detections.append(
            Detection.model_validate(
                {
                    "instance_id": index,
                    "class_name": _class_name(task_name, class_id, schema),
                    "metadata": metadata.get(index, {}),
                    **fields,
                }
            )
        )
    return detections


def _semantic_detections(
    task_name: str,
    task_types: Mapping[str, np.ndarray],
    schema: DatasetSchema,
    *,
    keep_background: bool,
) -> list[Detection]:
    array = task_types.get("segmentation")
    if array is None:
        return []
    detections = []
    for annotation, class_id in SegmentationAnnotation.split_from_numpy(array):
        class_name = _class_name(task_name, class_id, schema)
        # Only the channel the loader added is dropped. A dataset that
        # declares a `background` class annotated it itself, and dropping
        # that one would delete the user's own mask.
        if (
            class_name == "background"
            and task_name in schema.synthesized_background
            and not keep_background
        ):
            continue
        detections.append(
            Detection(class_name=class_name, segmentation=annotation)
        )
    return detections


def _classification_detections(
    task_name: str,
    task_types: Mapping[str, np.ndarray],
    schema: DatasetSchema,
) -> list[Detection]:
    """Rebuild the classes no other annotation of the task accounts for.

    Every detection with a class contributes to the vector, so rebuilding it
    beside them would double each class it already describes.
    """
    array = task_types.get("classification")
    if array is None:
        return []
    return [
        Detection(class_name=_class_name(task_name, class_id, schema))
        for _, class_id in ClassificationAnnotation.split_from_numpy(array)
    ]


def _split_metadata(
    task_name: str,
    task_types: Mapping[str, np.ndarray],
    schema: DatasetSchema,
) -> dict[int, dict[str, int | float | str | Category]]:
    """Return the metadata of each instance, keyed by its row index."""
    per_instance: dict[int, dict[str, int | float | str | Category]] = (
        defaultdict(dict)
    )
    for task_type, array in task_types.items():
        if not task_type.startswith(_METADATA_PREFIX):
            continue
        name = task_type[len(_METADATA_PREFIX) :]
        encoding = schema.categorical_encodings.get(f"{task_name}/{task_type}")
        decoding = (
            {index: value for value, index in encoding.items()}
            if encoding is not None
            else None
        )
        # An object array holds the values themselves, a numeric one holds
        # numpy scalars, and a row with no value of this kind holds None.
        for index, value in enumerate(np.asarray(array)):
            item = value.item() if isinstance(value, np.generic) else value
            if item is None:
                continue
            if decoding is not None and isinstance(item, (int, float)):
                per_instance[index][name] = Category(
                    decoding.get(int(item), str(item))
                )
            elif isinstance(item, (str, int, float)):
                per_instance[index][name] = item
            else:
                per_instance[index][name] = str(item)
    return dict(per_instance)


def _class_name(
    task_name: str, class_id: int | None, schema: DatasetSchema
) -> str | None:
    if class_id is None:
        return None
    return schema.class_name(task_name, class_id)
