import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from luxonis_ml.data import DatasetIterator

from .base_parser import BaseParser, ParserOutput


class SOLOParser(BaseParser):
    """Parses directory with SOLO annotations to LDF.

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
    def validate_split(split_path: Path) -> Optional[Dict[str, Any]]:
        """Validates if a split subdirectory is in an expected format.

        @type split_path: Path
        @param split_path: Path to split directory.
        @rtype: Optional[Dict[str, Any]]
        @return: Dictionary with kwargs to pass to L{from_split} method
            or C{None} if the split is not in the expected format.
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

    @staticmethod
    def validate(dataset_dir: Path) -> bool:
        """Validates if the dataset is in an expected format.

        @type dataset_dir: Path
        @param dataset_dir: Path to source dataset directory.
        @rtype: bool
        @return: True if the dataset is in the expected format.
        """
        for split in ["train", "valid", "test"]:
            split_path = dataset_dir / split
            if SOLOParser.validate_split(split_path) is None:
                return False
        return True

    def from_dir(
        self, dataset_dir: Path
    ) -> Tuple[List[Path], List[Path], List[Path]]:
        """Parses all present data to L{LuxonisDataset} format.

        @type dataset_dir: str
        @param dataset_dir: Path to source dataset directory.
        @rtype: Tuple[List[Path], List[Path], List[Path]]
        @return: Tuple with added images for train, valid and test
            splits.
        """

        added_train_imgs = self._parse_split(split_path=dataset_dir / "train")
        added_valid_imgs = self._parse_split(split_path=dataset_dir / "valid")
        added_test_imgs = self._parse_split(split_path=dataset_dir / "test")

        return added_train_imgs, added_valid_imgs, added_test_imgs

    def from_split(self, split_path: Path) -> ParserOutput:
        """Parses data in a split subdirectory from SOLO format to
        L{LuxonisDataset} format.

        @type split_path: Path
        @param split_path: Path to directory with sequences of images
            and annotations.
        @rtype: L{ParserOutput}
        @return: C{LuxonisDataset} generator, list of class names,
            skeleton dictionary for keypoints and list of added images.
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
                processed_annotations_per_step: Dict[
                    str, set
                ] = {}  # Seperate json files can have the same annotations in them
                for frame_path in sequence_path.glob("*.frame_data*.json"):
                    frame = json.loads(frame_path.read_text())

                    curent_step = frame["step"]
                    if curent_step not in processed_annotations_per_step:
                        processed_annotations_per_step[curent_step] = set()

                    for capture in frame.get("captures", []):
                        img_fname = capture["filename"]
                        img_w, img_h = capture["dimension"]
                        annotations = capture["annotations"]
                        img_path = sequence_path / img_fname
                        if not img_path.exists():
                            raise FileNotFoundError(
                                f"{img_path} not existent."
                            )
                        instance_segmentations = {}
                        instance_keypoints = {}
                        bounding_boxes = {}
                        for anno in annotations:
                            if (
                                "SemanticSegmentationAnnotation"
                                not in processed_annotations_per_step[
                                    curent_step
                                ]
                                and anno["@type"].endswith(
                                    "SemanticSegmentationAnnotation"
                                )
                            ):
                                processed_annotations_per_step[
                                    curent_step
                                ].add("SemanticSegmentationAnnotation")

                                mask_fname = anno["filename"]
                                mask_path = sequence_path / mask_fname
                                mask = cv2.imread(mask_path)

                                mask_int = (
                                    (mask[..., 0].astype(np.uint32) << 16)
                                    | (mask[..., 1].astype(np.uint32) << 8)
                                    | mask[..., 2].astype(np.uint32)
                                )

                                for instance in anno.get("instances", []):
                                    class_name = instance["labelName"]
                                    r, g, b, _ = instance["pixelValue"]
                                    target_int = (b << 16) | (g << 8) | r
                                    curr_mask = (
                                        mask_int == target_int
                                    ).astype(np.uint8)
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
                                "BoundingBox2DAnnotation"
                                not in processed_annotations_per_step[
                                    curent_step
                                ]
                                and anno["@type"].endswith(
                                    "BoundingBox2DAnnotation"
                                )
                            ):
                                processed_annotations_per_step[
                                    curent_step
                                ].add("BoundingBox2DAnnotation")
                                bbox_annotations = anno.get("values", [])

                                for bbox_annotation in bbox_annotations:
                                    class_name = bbox_annotation["labelName"]
                                    origin = bbox_annotation["origin"]
                                    dimension = bbox_annotation["dimension"]
                                    xmin, ymin = origin
                                    bbox_w, bbox_h = dimension

                                    instance_id = bbox_annotation["instanceId"]
                                    bounding_boxes[instance_id] = {
                                        "file": img_path,
                                        "annotation": {
                                            "class": class_name,
                                            "instance_id": instance_id,
                                            "boundingbox": {
                                                "x": xmin / img_w,
                                                "y": ymin / img_h,
                                                "w": bbox_w / img_w,
                                                "h": bbox_h / img_h,
                                            },
                                        },
                                    }

                            elif (
                                "InstanceSegmentationAnnotation"
                                not in processed_annotations_per_step[
                                    curent_step
                                ]
                                and anno["@type"].endswith(
                                    "InstanceSegmentationAnnotation"
                                )
                            ):
                                processed_annotations_per_step[
                                    curent_step
                                ].add("InstanceSegmentationAnnotation")

                                mask_fname = anno["filename"]
                                mask_path = sequence_path / mask_fname
                                mask = cv2.imread(mask_path)

                                mask_int = (
                                    (mask[..., 0].astype(np.uint32) << 16)
                                    | (mask[..., 1].astype(np.uint32) << 8)
                                    | mask[..., 2].astype(np.uint32)
                                )

                                for instance in anno.get("instances", []):
                                    r, g, b, _ = instance["color"]
                                    target_int = (b << 16) | (g << 8) | r
                                    curr_mask = (
                                        mask_int == target_int
                                    ).astype(np.uint8)
                                    instance_id = instance["instanceId"]

                                    instance_segmentations[instance_id] = {
                                        "file": img_path,
                                        "annotation": {
                                            "instance_id": instance_id,
                                            "instance_segmentation": {
                                                "mask": curr_mask,
                                            },
                                        },
                                    }

                            elif (
                                "KeypointAnnotation"
                                not in processed_annotations_per_step[
                                    curent_step
                                ]
                                and anno["@type"].endswith(
                                    "KeypointAnnotation"
                                )
                            ):
                                processed_annotations_per_step[
                                    curent_step
                                ].add("KeypointAnnotation")
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
                                        "file": img_path,
                                        "annotation": {
                                            "instance_id": instance_id,
                                            "keypoints": {
                                                "keypoints": keypoints,
                                            },
                                        },
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

                        if non_empty_annotations:
                            common_instance_ids = set.intersection(
                                *[
                                    set(ann.keys())
                                    for ann in non_empty_annotations
                                ]
                            )
                        else:
                            common_instance_ids = set()

                        for instance_id in common_instance_ids:
                            annotation_entry = {
                                "class": bounding_boxes[instance_id][
                                    "annotation"
                                ]["class"],
                                "instance_id": instance_id,
                                "boundingbox": bounding_boxes[instance_id][
                                    "annotation"
                                ]["boundingbox"],
                            }
                            if instance_keypoints:
                                annotation_entry["keypoints"] = (
                                    instance_keypoints[instance_id][
                                        "annotation"
                                    ]["keypoints"]
                                )
                            if instance_segmentations:
                                annotation_entry["instance_segmentation"] = (
                                    instance_segmentations[instance_id][
                                        "annotation"
                                    ]["instance_segmentation"]
                                )

                            yield {
                                "file": img_path,
                                "annotation": annotation_entry,
                            }

        return generator(), skeletons, []

    def _get_solo_annotation_types(
        self, annotation_definitions_dict: Dict[str, Any]
    ) -> List[str]:
        """List all annotation types present in the dataset.

        @type annotation_definitions_dict: dict
        @param annotation_definitions_dict: annotation_definitions.json read as dict.
        @rtype: List[str]
        @return: List of annotation types (e.g. ["BoundingBox2DAnnotation", "SemanticSegmentationAnnotation", ...]).
        """

        annotation_types = []
        for definition in annotation_definitions_dict["annotationDefinitions"]:
            annotation_types.append(
                definition["@type"].replace("type.unity.com/unity.solo.", "")
            )
        return annotation_types

    def _get_solo_bbox_class_names(
        self, annotation_definitions_dict: Dict[str, Any]
    ) -> List[str]:
        """List class names for BoundingBox2DAnnotation type.

        @type annotation_definitions_dict: dict
        @param annotation_definitions_dict: annotation_definitions.json read as dict.
        @rtype: List[str]
        @return: List of class names (e.g. ["car", "motorbike", ...]).
        """

        class_names = []
        for definition in annotation_definitions_dict["annotationDefinitions"]:
            annotation_type = definition["@type"].replace(
                "type.unity.com/unity.solo.", ""
            )
            if annotation_type == "BoundingBox2DAnnotation":
                names = [spec["label_name"] for spec in definition["spec"]]
                ids = [spec["label_id"] for spec in definition["spec"]]
                class_names = [c for _, c in sorted(zip(ids, names))]
        return class_names

    def _get_solo_keypoint_names(
        self, annotation_definitions_dict: Dict[str, Any]
    ) -> List[str]:
        """List keypoint labels for all classes.

        @type annotation_definitions_dict: dict
        @param annotation_definitions_dict: annotation_definitions.json read as dict.
        @rtype: List[str]
        @return: List of keypoint labels (e.g. ["wheel_front_motorbike", "wheel_back_motorbike", ...]).
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
                keypoint_labels = [c for _, c in sorted(zip(ids, labels))]
        return keypoint_labels
