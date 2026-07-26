"""Reconstruct canonical LDF records from `LuxonisLoader` output.

`LuxonisLoader` returns per-task numpy arrays keyed by ``"task_name/task_type"``.
For visualization it is convenient to turn that runtime representation back into
the canonical Luxonis Data Format: this module rebuilds `Detection` objects
(pairing boxes, keypoints, and instance masks by row index) and groups them into
one `DatasetRecord` per task name.

The reconstructed records are meant for rendering (see
`luxonis_ml.vizlab.convert.visualize_record`): the base image is supplied
separately, so the records are built with a placeholder ``files`` entry via
``DatasetRecord.model_construct`` and never written back to disk.
"""

from collections import defaultdict
from pathlib import Path

import numpy as np

from luxonis_ml.data.utils.task_utils import get_task_name, get_task_type
from luxonis_ml.ldf import (
    BBoxAnnotation,
    DatasetRecord,
    Detection,
    InstanceSegmentationAnnotation,
    KeypointAnnotation,
    SegmentationAnnotation,
)
from luxonis_ml.typing import Labels, Params

_BACKGROUND = "background"


def loader_output_to_records(
    labels: Labels,
    *,
    classes: dict[str, dict[str, int]],
    categorical_encodings: dict[str, dict[str, int]] | None = None,
    render_background: bool = False,
) -> dict[str, DatasetRecord]:
    """Convert `LuxonisLoader` label arrays into canonical LDF records.

    Args:
        labels: The loader's ``LoaderOutput.labels`` — a mapping of
            ``"task_name/task_type"`` to numpy arrays (bbox ``(N, 5)``,
            keypoints ``(N, 3K)``, semantic segmentation ``(C, H, W)``, instance
            segmentation ``(N, H, W)``, classification ``(C,)``).
        classes: ``LuxonisDataset.get_classes()`` — a mapping of task name to a
            ``{class_name: id}`` dict, used to recover class names from ids.
        categorical_encodings: ``LuxonisDataset.get_categorical_encodings()``,
            used to decode categorical metadata ids back to their string values.
        render_background: Keep the semantic-segmentation background class as a
            drawable mask (named ``"background"``). When ``False`` (the default)
            the background channel is dropped, mirroring how detection and
            classification treat it as a bookkeeping non-label.

    Returns:
        One `DatasetRecord` per task name, each holding a list of `Detection`
        objects. The image is not attached; pass it separately to the renderer.

    Example:
        >>> import numpy as np
        >>> labels = {"det/boundingbox": np.array([[0, 0.1, 0.2, 0.3, 0.4]])}
        >>> classes = {"det": {"car": 0}}
        >>> records = loader_output_to_records(labels, classes=classes)
        >>> record = records["det"]
        >>> det = record._annotations()[0]
        >>> det.class_name, det.boundingbox.w
        ('car', 0.3)

    """
    grouped: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    for key, array in labels.items():
        grouped[get_task_name(key)][get_task_type(key)] = array

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
        records[task_name] = DatasetRecord.model_construct(
            files={"image": Path("<loader>")},
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
    # An LDF detection/segmentation task also emits a `classification` vector
    # derived from its instances' classes; only surface classification as
    # standalone detections for pure image-level classification tasks.
    if not detections:
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

    # Metadata arrays are per-instance too, so a metadata-only task (e.g. OCR
    # with just ``metadata/text``) still yields one detection per entry.
    metadata_lengths = [
        len(np.asarray(arr))
        for task_type, arr in task_types.items()
        if task_type.startswith("metadata/") and np.asarray(arr).ndim == 1
    ]
    spatial_lengths = [
        len(arr)
        for arr in (boxes, keypoints, instance_masks)
        if arr is not None
    ]
    n_instances = max([*spatial_lengths, *metadata_lengths], default=0)
    metadata = _decode_metadata(
        task_name, task_types, n_instances, categorical_encodings
    )

    detections: list[Detection] = []
    for i in range(n_instances):
        fields: dict = {"instance_id": i}
        if boxes is not None and i < len(boxes):
            row = boxes[i]
            class_id = int(row[0])
            name = id_to_name.get(class_id)
            # "background" is a bookkeeping class, not a real label.
            fields["class_name"] = None if name == _BACKGROUND else name
            fields["boundingbox"] = BBoxAnnotation(
                x=float(row[1]),
                y=float(row[2]),
                w=float(row[3]),
                h=float(row[4]),
            )
        if keypoints is not None and i < len(keypoints):
            triples = np.asarray(keypoints[i], dtype=float).reshape(-1, 3)
            fields["keypoints"] = KeypointAnnotation.model_validate(
                {
                    "keypoints": [
                        (float(x), float(y), round(v)) for x, y, v in triples
                    ]
                }
            )
        if instance_masks is not None and i < len(instance_masks):
            fields["instance_segmentation"] = (
                InstanceSegmentationAnnotation.model_validate(
                    {"mask": np.asarray(instance_masks[i]).astype(np.uint8)}
                )
            )
        if i in metadata:
            fields["metadata"] = metadata[i]
        detections.append(Detection(**fields))
    return detections


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
) -> dict[int, Params]:
    """Recover per-instance metadata dicts keyed by instance index.

    Only metadata arrays whose length matches the instance count are attached;
    categorical values are decoded back to their string names when an encoding
    for the task is available.
    """
    per_instance: dict[int, Params] = defaultdict(dict)
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
            if decoder is not None:
                item = decoder.get(int(item), item)
            per_instance[i][key] = item
    return dict(per_instance)
