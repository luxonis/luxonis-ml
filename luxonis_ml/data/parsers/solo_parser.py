import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pycocotools.mask as mask_util

from luxonis_ml.data import DatasetGenerator

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
        @return: Dictionary with kwargs to pass to L{from_split} method or C{None} if
            the split is not in the expected format.
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
        with open(os.path.join(split_path, "metadata.json")) as json_file:
            metadata_dict = json.load(json_file)
        # check if all sequences are present
        total_sequences_expected = metadata_dict["totalSequences"]
        total_sequences = len(
            [
                d
                for d in glob.glob(os.path.join(split_path, "sequence*"))
                if os.path.isdir(d)
            ]
        )
        if not total_sequences == total_sequences_expected:
            return None
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
        self,
        dataset_dir: Path,
    ) -> Tuple[List[str], List[str], List[str]]:
        """Parses all present data to L{LuxonisDataset} format.

        @type dataset_dir: str
        @param dataset_dir: Path to source dataset directory.
        @rtype: Tuple[List[str], List[str], List[str]]
        @return: Tuple with added images for train, valid and test splits.
        """

        added_train_imgs = self._parse_split(split_path=dataset_dir / "train")
        added_valid_imgs = self._parse_split(split_path=dataset_dir / "valid")
        added_test_imgs = self._parse_split(split_path=dataset_dir / "test")

        return added_train_imgs, added_valid_imgs, added_test_imgs

    def from_split(
        self,
        split_path: Path,
    ) -> ParserOutput:
        """Parses data in a split subdirectory from SOLO format to L{LuxonisDataset}
        format.

        @type split_path: Path
        @param split_path: Path to directory with sequences of images and annotations.
        @rtype: L{ParserOutput}
        @return: C{LuxonisDataset} generator, list of class names, skeleton dictionary
            for keypoints and list of added images.
        """

        if not os.path.exists(split_path):
            raise Exception(f"{split_path} path non-existent.")

        annotation_definitions_path = os.path.join(
            split_path, "annotation_definitions.json"
        )
        if os.path.exists(annotation_definitions_path):
            with open(annotation_definitions_path) as json_file:
                annotation_definitions_dict = json.load(json_file)
        else:
            raise Exception(f"{annotation_definitions_path} path non-existent.")

        annotation_types = self._get_solo_annotation_types(annotation_definitions_dict)

        class_names = self._get_solo_bbox_class_names(annotation_definitions_dict)
        # TODO: We make an assumption here that bbox class_names are also valid for all other annotation types in the dataset. Is this OK?
        # TODO: Can we imagine a case where classes between annotation types are different? Which class names to return in this case?
        if class_names == []:
            raise Exception("No class_names identified. ")

        keypoint_labels = self._get_solo_keypoint_names(annotation_definitions_dict)

        skeletons = {
            class_name: {"labels": keypoint_labels} for class_name in class_names
        }
        # TODO: setting skeletons by assigning all keypoint names to each class_name. Is this OK?
        # if NOT, set them manually with LuxonisDataset.set_skeletons() as SOLO format does not
        # encode which keypoint_name belongs to which class

        def generator() -> DatasetGenerator:
            for dir_path, dir_names, _ in os.walk(split_path):
                for dir_name in dir_names:
                    if not dir_name.startswith("sequence"):
                        continue

                    sequence_path = os.path.join(dir_path, dir_name)

                    for frame_fname in glob.glob(
                        os.path.join(sequence_path, "*.frame_data.json")
                    ):  # single sequence can have multiple steps
                        frame_path = os.path.join(sequence_path, frame_fname)
                        if not os.path.exists(frame_path):
                            raise FileNotFoundError(f"{frame_path} not existent.")
                        with open(frame_path) as f:
                            frame = json.load(f)

                        for capture in frame["captures"]:
                            img_fname = capture["filename"]
                            img_w, img_h = capture["dimension"]
                            annotations = capture["annotations"]
                            img_path = os.path.join(sequence_path, img_fname)
                            if not os.path.exists(img_path):
                                raise FileNotFoundError(f"{img_path} not existent.")

                            if "BoundingBox2DAnnotation" in annotation_types:
                                for anno in annotations:
                                    if anno["@type"].endswith(
                                        "BoundingBox2DAnnotation"
                                    ):
                                        bbox_annotations = anno["values"]

                                for bbox_annotation in bbox_annotations:
                                    class_name = bbox_annotation["labelName"]

                                    origin = bbox_annotation["origin"]
                                    dimension = bbox_annotation["dimension"]
                                    xmin, ymin = origin
                                    bbox_w, bbox_h = dimension

                                    yield {
                                        "file": img_path,
                                        "class": class_name,
                                        "type": "box",
                                        "value": (
                                            xmin / img_w,
                                            ymin / img_h,
                                            bbox_w / img_w,
                                            bbox_h / img_h,
                                        ),
                                    }

                            if "SemanticSegmentationAnnotation" in annotation_types:
                                for anno in annotations:
                                    if anno["@type"].endswith(
                                        "SemanticSegmentationAnnotation"
                                    ):
                                        sseg_annotations = anno

                                mask_fname = sseg_annotations["filename"]
                                mask_path = os.path.join(sequence_path, mask_fname)
                                mask = cv2.imread(mask_path)

                                for instance in sseg_annotations["instances"]:
                                    class_name = instance["labelName"]
                                    r, g, b, _ = instance["pixelValue"]
                                    curr_mask = np.zeros_like(mask)
                                    curr_mask[np.all(mask == [b, g, r], axis=2)] = 1
                                    curr_mask = np.max(curr_mask, axis=2)  # 3D->2D
                                    curr_mask = np.asfortranarray(curr_mask)
                                    curr_rle = mask_util.encode(curr_mask)
                                    value = (
                                        curr_rle["size"][0],
                                        curr_rle["size"][1],
                                        curr_rle["counts"],
                                    )

                                    yield {
                                        "file": img_path,
                                        "class": class_name,
                                        "type": "segmentation",
                                        "value": value,
                                    }

                            if "KeypointAnnotation" in annotation_types:
                                for anno in annotations:
                                    if anno["@type"].endswith("KeypointAnnotation"):
                                        keypoint_annotations = anno["values"]

                                for keypoints_annotation in keypoint_annotations:
                                    label_id = keypoints_annotation["labelId"]
                                    keypoints = []
                                    for keypoint in keypoints_annotation["keypoints"]:
                                        x, y = keypoint["location"]
                                        visibility = keypoint["state"]
                                        keypoints.append(
                                            (x / img_w, y / img_h, visibility)
                                        )

                                    class_name = class_names[label_id]

                                    yield {
                                        "file": img_path,
                                        "class": class_name,
                                        "type": "keypoints",
                                        "value": keypoints,
                                    }

        added_images = self._get_added_images(generator)

        return (
            generator,
            class_names,
            skeletons,
            added_images,
        )

    def _get_solo_annotation_types(self, annotation_definitions_dict: dict) -> list:
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

    def _get_solo_bbox_class_names(self, annotation_definitions_dict):
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

    def _get_solo_keypoint_names(self, annotation_definitions_dict):
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
