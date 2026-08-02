import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger

from luxonis_ml.data import DatasetIterator
from luxonis_ml.utils.path import resolve_manifest_path

from .parser_plugin import SplitParserPlugin


class SOLOParser(SplitParserPlugin):
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

    dataset_types = ("solo",)

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
        with open(split_path / "metadata.json", encoding="utf-8") as json_file:
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

    def _split_records(self, split_path: Path) -> DatasetIterator:
        """Stream one SOLO split as LDF records.

        The definitions are read and validated before the walk starts, so a
        malformed split fails without a record being pulled.

        `_split_files` is left unimplemented: which captures yield records
        is only known once every frame JSON has been read, which is the
        parse itself.

        Args:
            split_path: Directory with SOLO sequences and annotations.

        Returns:
            Annotation records of the split, streamed in one walk.

        Raises:
            FileNotFoundError: If the split directory or the annotation
                definitions file does not exist. Streaming raises it for a
                referenced image or mask that does not exist.
            ValueError: If no bounding-box class names can be identified
                from ``annotation_definitions.json``. Streaming raises it
                for a mask that cannot be decoded.

        """
        if not split_path.exists():
            raise FileNotFoundError(f"{split_path} path non-existent.")

        annotation_definitions_path = (
            split_path / "annotation_definitions.json"
        )
        if annotation_definitions_path.exists():
            with open(
                annotation_definitions_path, encoding="utf-8"
            ) as json_file:
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

        self._skeletons.update(
            {
                class_name: {"labels": keypoint_labels}
                for class_name in bbox_class_names
            }
        )

        def generator() -> DatasetIterator:
            """Walk the split once, yielding records."""
            for sequence_path in split_path.glob("sequence*"):
                processed_annotations_per_step: dict[
                    str, set
                ] = {}  # Separate JSON files can have the same annotations.
                for frame_path in sequence_path.glob("*.frame_data*.json"):
                    frame = json.loads(frame_path.read_text())

                    current_step = frame["step"]
                    processed = processed_annotations_per_step.setdefault(
                        current_step, set()
                    )

                    for capture in frame.get("captures", []):
                        img_fname = capture["filename"]
                        img_w, img_h = capture["dimension"]
                        annotations = capture["annotations"]
                        img_path = resolve_manifest_path(
                            sequence_path, img_fname
                        )
                        if not img_path.exists():
                            raise FileNotFoundError(
                                f"{img_path} not existent."
                            )
                        instance_segmentations: dict[Any, Any] = {}
                        instance_keypoints: dict[Any, Any] = {}
                        bounding_boxes: dict[Any, Any] = {}
                        for anno in annotations:
                            anno_type = anno["@type"]
                            if (
                                "SemanticSegmentationAnnotation"
                                not in processed
                                and anno_type.endswith(
                                    "SemanticSegmentationAnnotation"
                                )
                            ):
                                processed.add("SemanticSegmentationAnnotation")

                                mask_fname = anno["filename"]
                                mask_path = resolve_manifest_path(
                                    sequence_path, mask_fname
                                )
                                if not mask_path.exists():
                                    raise FileNotFoundError(
                                        f"{mask_path} not existent."
                                    )
                                mask_int = self._read_mask_int(mask_path)

                                for instance in anno.get("instances", []):
                                    class_name = instance["labelName"]
                                    r, g, b, _ = instance["pixelValue"]
                                    target_int = (b << 16) | (g << 8) | r
                                    curr_mask = self._instance_mask(
                                        mask_int, target_int
                                    )
                                    yield {
                                        "file": img_path,
                                        "annotation": {
                                            "class": class_name,
                                            "segmentation": {
                                                "mask": curr_mask,
                                            },
                                        },
                                    }

                            elif (
                                "BoundingBox2DAnnotation" not in processed
                                and anno_type.endswith(
                                    "BoundingBox2DAnnotation"
                                )
                            ):
                                processed.add("BoundingBox2DAnnotation")
                                bbox_annotations = anno.get("values", [])

                                for bbox_annotation in bbox_annotations:
                                    instance_id = bbox_annotation["instanceId"]
                                    class_name = bbox_annotation["labelName"]
                                    origin = bbox_annotation["origin"]
                                    dimension = bbox_annotation["dimension"]
                                    xmin, ymin = origin
                                    bbox_w, bbox_h = dimension

                                    bounding_boxes[instance_id] = (
                                        class_name,
                                        {
                                            "x": xmin / img_w,
                                            "y": ymin / img_h,
                                            "w": bbox_w / img_w,
                                            "h": bbox_h / img_h,
                                        },
                                    )

                            elif (
                                "InstanceSegmentationAnnotation"
                                not in processed
                                and anno_type.endswith(
                                    "InstanceSegmentationAnnotation"
                                )
                            ):
                                processed.add("InstanceSegmentationAnnotation")

                                mask_fname = anno["filename"]
                                mask_path = resolve_manifest_path(
                                    sequence_path, mask_fname
                                )
                                if not mask_path.exists():
                                    raise FileNotFoundError(
                                        f"{mask_path} not existent."
                                    )
                                mask_int = self._read_mask_int(mask_path)

                                for instance in anno.get("instances", []):
                                    r, g, b, _ = instance["color"]
                                    target_int = (b << 16) | (g << 8) | r
                                    curr_mask = self._instance_mask(
                                        mask_int, target_int
                                    )
                                    instance_id = instance["instanceId"]

                                    instance_segmentations[instance_id] = {
                                        "mask": curr_mask,
                                    }

                            elif (
                                "KeypointAnnotation" not in processed
                                and anno_type.endswith("KeypointAnnotation")
                            ):
                                processed.add("KeypointAnnotation")
                                keypoint_annotations = anno.get("values", [])

                                for (
                                    keypoints_annotation
                                ) in keypoint_annotations:
                                    keypoints = []
                                    for keypoint in keypoints_annotation[
                                        "keypoints"
                                    ]:
                                        x, y = keypoint["location"]
                                        visibility = keypoint["state"]
                                        keypoints.append(
                                            (x / img_w, y / img_h, visibility)
                                        )

                                    instance_id = keypoints_annotation[
                                        "instanceId"
                                    ]

                                    instance_keypoints[instance_id] = {
                                        "keypoints": keypoints,
                                    }
                        # Hard dependencies between bbox, keypoints and instance_segmentations
                        non_empty_annotations = []
                        if bounding_boxes:
                            non_empty_annotations.append(bounding_boxes)
                        if instance_keypoints:
                            non_empty_annotations.append(instance_keypoints)
                        if instance_segmentations:
                            non_empty_annotations.append(
                                instance_segmentations
                            )

                        # The merged record is anchored on the bounding box,
                        # which carries the class name, so a capture with
                        # keypoints or segmentations but no boxes yields
                        # nothing rather than failing the lookup below.
                        if bounding_boxes:
                            common_instance_ids = set.intersection(
                                *[
                                    set(ann.keys())
                                    for ann in non_empty_annotations
                                ]
                            )
                        else:
                            common_instance_ids = set()

                        for instance_id in common_instance_ids:
                            class_name, boundingbox = bounding_boxes[
                                instance_id
                            ]
                            annotation_entry = {
                                "class": class_name,
                                "instance_id": instance_id,
                                "boundingbox": boundingbox,
                            }
                            if instance_keypoints:
                                annotation_entry["keypoints"] = (
                                    instance_keypoints[instance_id]
                                )
                            if instance_segmentations:
                                annotation_entry["instance_segmentation"] = (
                                    instance_segmentations[instance_id]
                                )

                            yield {
                                "file": img_path,
                                "annotation": annotation_entry,
                            }

        return generator()

    @staticmethod
    def _read_mask_int(mask_path: Path) -> np.ndarray:
        """Read a mask image and pack its BGR channels into one integer."""
        mask = cv2.imread(str(mask_path))
        if mask is None:
            raise ValueError(f"Failed to read mask image from {mask_path}.")
        return (
            (mask[..., 0].astype(np.uint32) << 16)
            | (mask[..., 1].astype(np.uint32) << 8)
            | mask[..., 2].astype(np.uint32)
        )

    @staticmethod
    def _instance_mask(mask_int: np.ndarray, target_int: int) -> np.ndarray:
        """Extract the binary mask of one instance colour."""
        return (mask_int == target_int).astype(np.uint8)

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
