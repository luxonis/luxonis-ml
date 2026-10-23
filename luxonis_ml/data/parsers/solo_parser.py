import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger

from luxonis_ml.data import DatasetIterator
from luxonis_ml.utils.path import resolve_manifest_path

from .base_parser import BaseParser, ParserOutput


class SOLOParser(BaseParser):
    """Parse a directory with SOLO annotations into LDF.

    Expected format::

        dataset_dir/
        ├── train/
        │   ├── metadata.json
        │   ├── sensor_definitions.json
        │   ├── annotation_definitions.json
        │   ├── metric_definitions.json
        │   └── sequence.<SequenceNUM>/
        │       ├── step<StepNUM>.camera.jpg
        │       ├── step<StepNUM>.frame_data.json
        │       └── (OPTIONAL: step<StepNUM>.camera.semantic segmentation.jpg)
        ├── valid/
        └── test/

    This is the default format returned by Unity simulation engine.
    """

    @staticmethod
    def validate_split(split_path: Path) -> dict[str, Any] | None:
        """Validate whether a split directory has the expected SOLO
        format.

        Args:
            split_path: Path to a split directory.

        Returns:
            Keyword arguments for ``from_split``, or ``None`` if the split
            is not in the expected format.

        """
        if not split_path.exists():
            return None
        # check if all json files are present
        for json_fname in [
            "annotation_definitions.json",
            "metadata.json",
            "metric_definitions.json",
            "sensor_definitions.json",
        ]:
            json_path = next(split_path.glob(json_fname), None)
            if not json_path:
                return None
        with open(split_path / "metadata.json") as json_file:
            metadata_dict = json.load(json_file)
        # check if all sequences are present
        total_sequences_expected = metadata_dict["totalSequences"]
        total_sequences = len(
            [d for d in split_path.glob("sequence*") if d.is_dir()]
        )
        if total_sequences != total_sequences_expected:
            logger.warning(
                f"Expected {total_sequences_expected} based on metadata.json, "
                f"but found {total_sequences} sequences."
            )
        return {"split_path": split_path}

    def from_dir(
        self, dataset_dir: Path
    ) -> tuple[list[Path], list[Path], list[Path]]:
        """Parse all SOLO data in a source dataset directory.

        Args:
            dataset_dir: Source dataset directory.

        Returns:
            Added images for the train, validation, and test splits.

        """
        added_train_imgs = self._parse_split(split_path=dataset_dir / "train")
        added_valid_imgs = self._parse_split(split_path=dataset_dir / "valid")
        added_test_imgs = self._parse_split(split_path=dataset_dir / "test")

        return added_train_imgs, added_valid_imgs, added_test_imgs

    def from_split(self, split_path: Path) -> ParserOutput:
        """Parse one SOLO split into LDF records.

        Args:
            split_path: Directory with SOLO sequences and annotations.

        Returns:
            Parser output containing annotation records, skeleton metadata,
            and added images.

        Raises:
            FileNotFoundError: If the split directory, annotation
                definitions file, referenced image, or referenced mask does
                not exist.
            ValueError: If no bounding-box class names can be identified
                from ``annotation_definitions.json``.

        """
        if not split_path.exists():
            raise FileNotFoundError(f"{split_path} path non-existent.")

        annotation_definitions_path = (
            split_path / "annotation_definitions.json"
        )
        if annotation_definitions_path.exists():
            with open(annotation_definitions_path) as json_file:
                annotation_definitions_dict = json.load(json_file)
        else:
            raise FileNotFoundError(
                f"{annotation_definitions_path} path non-existent."
            )

        bbox_class_names = self._get_solo_bbox_class_names(
            annotation_definitions_dict
        )

        if not bbox_class_names:
            raise ValueError("No class_names identified. ")

        keypoint_labels = self._get_solo_keypoint_names(
            annotation_definitions_dict
        )

        skeletons = {
            class_name: {"labels": keypoint_labels}
            for class_name in bbox_class_names
        }

        def generator() -> DatasetIterator:
            for sequence_path in split_path.glob("sequence*"):
                # Separate JSON files can have the same annotations.
                processed_per_step: dict[str, set[str]] = {}
                for frame_path in sequence_path.glob("*.frame_data*.json"):
                    frame = json.loads(frame_path.read_text())
                    processed = processed_per_step.setdefault(
                        frame["step"], set()
                    )
                    for capture in frame.get("captures", []):
                        yield from _parse_capture(
                            sequence_path, capture, processed
                        )

        return generator(), skeletons, []

    def _get_solo_annotation_types(
        self, annotation_definitions_dict: dict[str, Any]
    ) -> list[str]:
        """List all annotation types present in the dataset.

        Args:
            annotation_definitions_dict: Parsed ``annotation_definitions.json``.

        Returns:
            Annotation type names.

        """
        annotation_types = []
        for definition in annotation_definitions_dict["annotationDefinitions"]:
            annotation_types.append(
                definition["@type"].replace("type.unity.com/unity.solo.", "")
            )
        return annotation_types

    def _get_solo_bbox_class_names(
        self, annotation_definitions_dict: dict[str, Any]
    ) -> list[str]:
        """List class names for BoundingBox2DAnnotation type.

        Args:
            annotation_definitions_dict: Parsed ``annotation_definitions.json``.

        Returns:
            Bounding box class names.

        """
        class_names = []
        for definition in annotation_definitions_dict["annotationDefinitions"]:
            annotation_type = definition["@type"].replace(
                "type.unity.com/unity.solo.", ""
            )
            if annotation_type == "BoundingBox2DAnnotation":
                names = [spec["label_name"] for spec in definition["spec"]]
                ids = [spec["label_id"] for spec in definition["spec"]]
                class_names = [
                    c for _, c in sorted(zip(ids, names, strict=True))
                ]
        return class_names

    def _get_solo_keypoint_names(
        self, annotation_definitions_dict: dict[str, Any]
    ) -> list[str]:
        """List keypoint labels for all classes.

        Args:
            annotation_definitions_dict: Parsed ``annotation_definitions.json``.

        Returns:
            Keypoint labels.

        """
        keypoint_labels = []
        for definition in annotation_definitions_dict["annotationDefinitions"]:
            annotation_type = definition["@type"].replace(
                "type.unity.com/unity.solo.", ""
            )
            if annotation_type == "KeypointAnnotation":
                keypoints = definition["template"]["keypoints"]
                labels = [keypoint["label"] for keypoint in keypoints]
                ids = [keypoint["index"] for keypoint in keypoints]
                keypoint_labels = [
                    c for _, c in sorted(zip(ids, labels, strict=True))
                ]
        return keypoint_labels


_ANNOTATION_KINDS = (
    "SemanticSegmentationAnnotation",
    "BoundingBox2DAnnotation",
    "InstanceSegmentationAnnotation",
    "KeypointAnnotation",
)


def _parse_capture(
    sequence_path: Path, capture: dict[str, Any], processed: set[str]
) -> DatasetIterator:
    """Yield the records of one camera capture.

    Each annotation kind is read once per step. ``processed`` holds the
    kinds that an earlier capture of the same step already read.

    Yields:
        Records with the capture image as ``file``. The records of the
        semantic masks come first, one per class, then one record per
        instance with its box, plus its keypoints and instance mask if
        the capture has them.

    Raises:
        FileNotFoundError: If the capture image, or a mask that an
            annotation references, does not exist.

    """
    img_fname = capture["filename"]
    img_w, img_h = capture["dimension"]
    annotations = capture["annotations"]
    img_path = resolve_manifest_path(sequence_path, img_fname)
    if not img_path.exists():
        raise FileNotFoundError(f"{img_path} not existent.")

    bounding_boxes: dict[int, dict[str, Any]] = {}
    instance_keypoints: dict[int, list[tuple[float, float, int]]] = {}
    instance_masks: dict[int, np.ndarray] = {}
    for anno in annotations:
        kind = next(
            (
                k
                for k in _ANNOTATION_KINDS
                if k not in processed and anno["@type"].endswith(k)
            ),
            None,
        )
        if kind is None:
            continue
        processed.add(kind)
        if kind == "SemanticSegmentationAnnotation":
            yield from _semantic_records(sequence_path, anno, img_path)
        elif kind == "BoundingBox2DAnnotation":
            bounding_boxes.update(_bounding_boxes(anno, img_w, img_h))
        elif kind == "InstanceSegmentationAnnotation":
            instance_masks.update(_instance_masks(sequence_path, anno))
        else:
            instance_keypoints.update(_instance_keypoints(anno, img_w, img_h))

    yield from _instance_records(
        img_path, bounding_boxes, instance_keypoints, instance_masks
    )


def _read_mask(sequence_path: Path, anno: dict[str, Any]) -> np.ndarray:
    """Read a color-coded mask image as one integer per pixel."""
    mask_path = resolve_manifest_path(sequence_path, anno["filename"])
    if not mask_path.exists():
        raise FileNotFoundError(f"{mask_path} not existent.")
    mask = cv2.imread(str(mask_path))
    if mask is None:
        raise ValueError(f"Failed to read mask image from {mask_path}.")
    return (
        (mask[..., 0].astype(np.uint32) << 16)
        | (mask[..., 1].astype(np.uint32) << 8)
        | mask[..., 2].astype(np.uint32)
    )


def _color_mask(mask: np.ndarray, color: list[int]) -> np.ndarray:
    """Select the pixels of one RGBA color from an integer mask."""
    r, g, b, _ = color
    return (mask == ((b << 16) | (g << 8) | r)).astype(np.uint8)


def _semantic_records(
    sequence_path: Path, anno: dict[str, Any], img_path: Path
) -> DatasetIterator:
    """Yield one segmentation record per class of a semantic mask."""
    mask = _read_mask(sequence_path, anno)
    for instance in anno.get("instances", []):
        yield {
            "file": img_path,
            "annotation": {
                "class": instance["labelName"],
                "segmentation": {
                    "mask": _color_mask(mask, instance["pixelValue"]),
                },
            },
        }


def _bounding_boxes(
    anno: dict[str, Any], img_w: int, img_h: int
) -> dict[int, dict[str, Any]]:
    """Map each instance ID to its class and normalized bounding box."""
    boxes = {}
    for value in anno.get("values", []):
        class_name = value["labelName"]
        origin = value["origin"]
        dimension = value["dimension"]
        xmin, ymin = origin
        bbox_w, bbox_h = dimension
        instance_id = value["instanceId"]
        boxes[instance_id] = {
            "class": class_name,
            "boundingbox": {
                "x": xmin / img_w,
                "y": ymin / img_h,
                "w": bbox_w / img_w,
                "h": bbox_h / img_h,
            },
        }
    return boxes


def _instance_masks(
    sequence_path: Path, anno: dict[str, Any]
) -> dict[int, np.ndarray]:
    """Map each instance ID to its binary mask."""
    mask = _read_mask(sequence_path, anno)
    masks = {}
    for instance in anno.get("instances", []):
        masks[instance["instanceId"]] = _color_mask(mask, instance["color"])
    return masks


def _instance_keypoints(
    anno: dict[str, Any], img_w: int, img_h: int
) -> dict[int, list[tuple[float, float, int]]]:
    """Map each instance ID to its normalized keypoints."""
    result = {}
    for value in anno.get("values", []):
        keypoints = []
        for keypoint in value["keypoints"]:
            x, y = keypoint["location"]
            visibility = keypoint["state"]
            keypoints.append((x / img_w, y / img_h, visibility))
        result[value["instanceId"]] = keypoints
    return result


def _instance_records(
    img_path: Path,
    bounding_boxes: dict[int, dict[str, Any]],
    instance_keypoints: dict[int, list[tuple[float, float, int]]],
    instance_masks: dict[int, np.ndarray],
) -> DatasetIterator:
    """Yield one record per instance ID that every present kind shares.

    Bounding boxes, keypoints and instance masks of one instance belong
    together, so an instance that one of the present kinds lacks is
    dropped.

    Yields:
        Records with ``img_path`` as ``file``. Each annotation holds the
        class, the instance ID and the bounding box, plus the keypoints
        and the instance mask if those kinds are present. Nothing is
        yielded if no kind is present.

    """
    present = [
        d for d in (bounding_boxes, instance_keypoints, instance_masks) if d
    ]
    shared = set.intersection(*(set(d) for d in present)) if present else set()
    for instance_id in shared:
        annotation = {
            "class": bounding_boxes[instance_id]["class"],
            "instance_id": instance_id,
            "boundingbox": bounding_boxes[instance_id]["boundingbox"],
        }
        if instance_keypoints:
            annotation["keypoints"] = {
                "keypoints": instance_keypoints[instance_id]
            }
        if instance_masks:
            annotation["instance_segmentation"] = {
                "mask": instance_masks[instance_id]
            }
        yield {"file": img_path, "annotation": annotation}
