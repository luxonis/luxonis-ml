import csv
import json
import logging
import os
import os.path as osp
import xml.etree.ElementTree as ET
from typing import Callable, Dict, List, Literal, Optional, Tuple

import cv2
import numpy as np
import pycocotools.mask as mask_util
import yaml

from luxonis_ml.data import DatasetGenerator, DatasetGeneratorFunction, LuxonisDataset
from luxonis_ml.enums import DatasetType

ParserOutput = Tuple[DatasetGeneratorFunction, List[str], Dict[str, Dict], List[str]]
"""Type alias for parser output.

Contains a function to create the annotation generator, list of classes names, skeleton
dictionary for keypoints and list of added images.
"""


def parsing_wrapper(func: Callable) -> Callable:
    """Wrapper for parsing functions that adds data to LDF.

    @type func: Callable
    @param func: Parsing function
    @rtype: Callable
    @return: Wrapper function
    """

    def wrapper(*args, **kwargs):
        dataset = args[0].dataset
        generator, class_names, skeletons, added_images = func(*args, **kwargs)
        dataset.set_classes(class_names)
        dataset.set_skeletons(skeletons)
        dataset.add(generator)

        return added_images

    return wrapper


class LuxonisParser:
    def __init__(self, **ldf_kwargs):
        """A parser class used for parsing common dataset formats to LDF.

        @type ldf_kwargs: Dict[str, Any]
        @param ldf_kwargs: Init parameters for L{LuxonisDataset}.
        """

        self.logger = logging.getLogger(__name__)
        self.dataset_exists = LuxonisDataset.exists(
            dataset_name=ldf_kwargs["dataset_name"]
        )
        self.dataset = LuxonisDataset(**ldf_kwargs)

    def parse_dir(
        self,
        dataset_type: DatasetType,
        dataset_dir: str,
        **parser_kwargs,
    ) -> LuxonisDataset:
        """Parses all present data in LuxonisDataset format. Check under selected parser
        function for expected directory structure.

        @type dataset_type: DatasetType
        @param dataset_type: Source dataset type
        @type dataset_dir: str
        @param dataset_dir: Path to source dataset directory.
        @type parser_kwargs: Dict[str, Any]
        @param parser_kwargs: Additional kwargs for specific parser function.
        @rtype: LuxonisDataset
        @return: Output LDF with all images and annotations parsed.
        """
        if dataset_type == DatasetType.LDF:
            return self.dataset

        if self.dataset_exists:
            self.logger.warning(
                "There already exists an LDF dataset with this name. "
                "Skipping parsing and using that one instead."
            )
            return self.dataset

        if dataset_type == DatasetType.COCO:
            self.from_coco_dir(dataset_dir, **parser_kwargs)
        elif dataset_type == DatasetType.VOC:
            self.from_voc_dir(dataset_dir)
        elif dataset_type == DatasetType.DARKNET:
            self.from_darknet_dir(dataset_dir)
        elif dataset_type == DatasetType.YOLOV6:
            self.from_yolov6_dir(dataset_dir)
        elif dataset_type == DatasetType.YOLOV4:
            self.from_yolov4_dir(dataset_dir)
        elif dataset_type == DatasetType.CREATEML:
            self.from_create_ml_dir(dataset_dir)
        elif dataset_type == DatasetType.TFCSV:
            self.from_tensorflow_csv_dir(dataset_dir)
        elif dataset_type == DatasetType.CLSDIR:
            self.from_class_dir_dir(dataset_dir)
        elif dataset_type == DatasetType.SEGMASK:
            self.from_seg_mask_dir(dataset_dir)

        return self.dataset

    def parse_raw_dir(
        self,
        dataset_type: DatasetType,
        split: Optional[Literal["train", "val", "test"]] = None,
        random_split: bool = False,
        split_ratios: Optional[List[float]] = None,
        **parser_kwargs,
    ) -> LuxonisDataset:
        """Parses data in specific directory, should be used if adding/changing only
        specific split. Check under selected parser function for expected directory
        structure.

        @type dataset_type: DatasetType
        @param dataset_type: Source dataset type
        @type split: Optional[Literal["train", "val", "test"]]
        @param split: Split under which data will be added.
        @type random_split: bool
        @param random_split: If random splits should be made.
        @type split_ratios: Optional[List[float]]
        @param split_ratios: Ratios for random splits.
        @type parser_kwargs: Dict[str, Any]
        @param parser_kwargs: Additional kwargs for specific parser function.
        @rtype: LuxonisDataset
        @return: Output LDF with all images and annotations parsed.
        """
        if dataset_type == DatasetType.LDF:
            pass
        elif dataset_type == DatasetType.COCO:
            added_images = self.from_coco_format(**parser_kwargs)
        elif dataset_type == DatasetType.VOC:
            added_images = self.from_voc_format(**parser_kwargs)
        elif dataset_type == DatasetType.DARKNET:
            added_images = self.from_darknet_format(**parser_kwargs)
        elif dataset_type == DatasetType.YOLOV6:
            added_images = self.from_yolov6_format(**parser_kwargs)
        elif dataset_type == DatasetType.YOLOV4:
            added_images = self.from_yolov4_format(**parser_kwargs)
        elif dataset_type == DatasetType.CREATEML:
            added_images = self.from_create_ml_format(**parser_kwargs)
        elif dataset_type == DatasetType.TFCSV:
            added_images = self.from_tensorflow_csv_format(**parser_kwargs)
        elif dataset_type == DatasetType.CLSDIR:
            added_images = self.from_class_dir_format(**parser_kwargs)
        elif dataset_type == DatasetType.SEGMASK:
            added_images = self.from_seg_mask_format(**parser_kwargs)

        if split:
            self.dataset.make_splits(definitions={split: added_images})
        elif random_split:
            split_ratios = split_ratios or [0.8, 0.1, 0.1]
            self.dataset.make_splits(split_ratios)

        return self.dataset

    def from_coco_dir(
        self,
        dataset_dir: str,
        use_keypoint_ann: bool = False,
        keypoint_ann_paths: Optional[Dict[str, str]] = None,
        split_val_to_test: bool = True,
    ) -> None:
        """Parses directory with COCO annotations to LDF.
        Expected format: "train", "validation" and "test" directories. Each one has "data" dir
        with images and "labels.json" file with annotations. This is default format returned
        when using fiftyone package.

        @type dataset_dir: str
        @param dataset_dir: Path to dataset directory

        @type use_keypoint_ann: bool
        @param use_keypoint_ann: If keypoint annotations should be used. Defaults to False.

        @type keypoint_ann_paths: Optional[Dict[str, str]]
        @param keypoint_ann_paths: Path to keypoint annotations for each split.
            Defaults to None.

        @type split_val_to_test: bool
        @param split_val_to_test: If part of validation data should be used as test data.
            Defaults to True.
        """
        if use_keypoint_ann and not keypoint_ann_paths:
            keypoint_ann_paths = {
                "train": "raw/person_keypoints_train2017.json",
                "val": "raw/person_keypoints_val2017.json",
                "test": "raw/person_keypoints_test2017.json",  # NOTE: this file is not present by default
            }

        train_ann_path = (
            osp.join(dataset_dir, keypoint_ann_paths["train"])
            if use_keypoint_ann
            else osp.join(dataset_dir, "train", "labels.json")
        )
        added_train_imgs = self.from_coco_format(
            image_dir=osp.join(dataset_dir, "train", "data"),
            annotation_path=train_ann_path,
        )

        val_ann_path = (
            osp.join(dataset_dir, keypoint_ann_paths["val"])
            if use_keypoint_ann
            else osp.join(dataset_dir, "validation", "labels.json")
        )
        _added_val_imgs = self.from_coco_format(
            image_dir=osp.join(dataset_dir, "validation", "data"),
            annotation_path=val_ann_path,
        )

        if not split_val_to_test:
            # NOTE: test split annotations are not included by default
            test_ann_path = (
                osp.join(dataset_dir, keypoint_ann_paths["test"])
                if use_keypoint_ann
                else osp.join(dataset_dir, "test", "labels.json")
            )
            added_test_imgs = self.from_coco_format(
                image_dir=osp.join(dataset_dir, "test", "data"),
                annotation_path=test_ann_path,
            )

        if split_val_to_test:
            split_point = round(len(_added_val_imgs) * 0.5)
            added_val_imgs = _added_val_imgs[:split_point]
            added_test_imgs = _added_val_imgs[split_point:]
        else:
            added_val_imgs = _added_val_imgs

        self.dataset.make_splits(
            definitions={
                "train": added_train_imgs,
                "val": added_val_imgs,
                "test": added_test_imgs,
            }
        )

    @parsing_wrapper
    def from_coco_format(self, image_dir: str, annotation_path: str) -> ParserOutput:
        """Parses annotations from COCO format to LDF. Annotations include
        classification, segmentation, object detection and keypoints if present.

        @type image_dir: str
        @param image_dir: Path to directory with images
        @type annotation_path: str
        @param annotation_path: Path to annotation json file
        """
        with open(annotation_path) as f:
            annotation_data = json.load(f)

        coco_images = annotation_data["images"]
        coco_annotations = annotation_data["annotations"]
        coco_categories = annotation_data["categories"]
        categories = {cat["id"]: cat["name"] for cat in coco_categories}

        class_names = list(categories.values())
        skeletons = {}
        for cat in coco_categories:
            if "keypoints" in cat.keys() and "skeleton" in cat.keys():
                skeletons[categories[cat["id"]]] = {
                    "labels": cat["keypoints"],
                    "edges": (np.array(cat["skeleton"]) - 1).tolist(),
                }

        def generator() -> DatasetGenerator:
            for img in coco_images:
                img_id = img["id"]

                path = osp.join(osp.abspath(image_dir), img["file_name"])
                if not osp.exists(path):
                    continue

                img_anns = [
                    ann for ann in coco_annotations if ann["image_id"] == img_id
                ]

                height = img["height"]
                width = img["width"]

                for ann in img_anns:
                    class_name = categories[ann["category_id"]]
                    yield {
                        "file": path,
                        "class": class_name,
                        "type": "classification",
                        "value": True,
                    }

                    seg = ann["segmentation"]
                    if isinstance(seg, list):
                        poly = []
                        for s in seg:
                            poly_arr = np.array(s).reshape(-1, 2)
                            poly += [
                                tuple([poly_arr[i, 0] / width, poly_arr[i, 1] / height])
                                for i in range(len(poly_arr))
                            ]
                        yield {
                            "file": path,
                            "class": class_name,
                            "type": "polyline",
                            "value": poly,
                        }

                    x, y, w, h = ann["bbox"]
                    yield {
                        "file": path,
                        "class": class_name,
                        "type": "box",
                        "value": tuple([x / width, y / height, w / width, h / height]),
                    }

                    if "keypoints" in ann.keys():
                        kpts = np.array(ann["keypoints"]).reshape(-1, 3)
                        keypoints = []
                        for kp in kpts:
                            keypoints.append(
                                tuple(
                                    [
                                        float(kp[0] / width),
                                        float(kp[1] / height),
                                        int(kp[2]),
                                    ]
                                )
                            )
                        yield {
                            "file": path,
                            "class": class_name,
                            "type": "keypoints",
                            "value": keypoints,
                        }

        added_images = self._get_added_images(generator)

        return generator, class_names, skeletons, added_images

    def from_voc_dir(self, dataset_dir: str) -> None:
        """Parses directory with VOC annotations to LDF.

        Expected format: "train", "valid" and "test" directories. Each one has images
        and .xml annotations. This is default format returned when using Roboflow.

        @type dataset_dir: str
        @param dataset_dir: Path to dataset directory
        """
        added_train_imgs = self.from_voc_format(
            image_dir=osp.join(dataset_dir, "train"),
            annotation_dir=osp.join(dataset_dir, "train"),
        )
        added_val_imgs = self.from_voc_format(
            image_dir=osp.join(dataset_dir, "valid"),
            annotation_dir=osp.join(dataset_dir, "valid"),
        )
        added_test_imgs = self.from_voc_format(
            image_dir=osp.join(dataset_dir, "test"),
            annotation_dir=osp.join(dataset_dir, "test"),
        )

        self.dataset.make_splits(
            definitions={
                "train": added_train_imgs,
                "val": added_val_imgs,
                "test": added_test_imgs,
            }
        )

    @parsing_wrapper
    def from_voc_format(
        self,
        image_dir: str,
        annotation_dir: str,
    ) -> ParserOutput:
        """Parses annotations from VOC format to LDF. Annotations include classification
        and object detection.

        @type image_dir: str
        @param image_dir: Path to directory with images
        @type annotation_dir: str
        @param annotation_dir: Path to directory with .xml annotations
        @rtype: Tuple[Generator, List[str], Dict[str, Dict], List[str]]
        @return: Annotation generator, list of classes names, skeleton dictionary for
            keypoints and list of added images.
        """
        anno_files = [i for i in os.listdir(annotation_dir) if i.endswith(".xml")]

        class_names = set()
        images_annotations = []
        for anno_file in anno_files:
            anno_xml = osp.join(annotation_dir, anno_file)
            annotation_data = ET.parse(anno_xml)
            root = annotation_data.getroot()

            filename_item = root.find("filename")
            path = osp.join(osp.abspath(image_dir), filename_item.text)
            if not osp.exists(path):
                continue

            curr_annotations = {"path": path, "classes": [], "bboxes": []}
            size_item = root.find("size")
            height = float(size_item.find("height").text)
            width = float(size_item.find("width").text)

            for object_item in root.findall("object"):
                class_name = object_item.find("name").text
                curr_annotations["classes"].append(class_name)
                class_names.add(class_name)

                bbox_info = object_item.find("bndbox")
                if bbox_info:
                    bbox_xywh = np.array(
                        [
                            float(bbox_info.find("xmin").text),
                            float(bbox_info.find("ymin").text),
                            float(bbox_info.find("xmax").text)
                            - float(bbox_info.find("xmin").text),
                            float(bbox_info.find("ymax").text)
                            - float(bbox_info.find("ymin").text),
                        ]
                    )
                    bbox_xywh[::2] /= width
                    bbox_xywh[1::2] /= height
                    bbox_xywh = bbox_xywh.tolist()
                    curr_annotations["bboxes"].append((class_name, bbox_xywh))
            images_annotations.append(curr_annotations)

        def generator() -> DatasetGenerator:
            for curr_annotations in images_annotations:
                path = curr_annotations["path"]
                for class_name in curr_annotations["classes"]:
                    yield {
                        "file": path,
                        "class": class_name,
                        "type": "classification",
                        "value": True,
                    }
                for bbox_class, bbox in curr_annotations["bboxes"]:
                    yield {
                        "file": path,
                        "class": bbox_class,
                        "type": "box",
                        "value": tuple(bbox),
                    }

        added_images = self._get_added_images(generator)

        return generator, list(class_names), {}, added_images

    def from_darknet_dir(self, dataset_dir: str) -> None:
        """Parses directory with DarkNet annotations to LDF.

        Expected format: "train", "valid" and "test" directories. Each one has images,
        .txt annotations and "darknet.labels" file with all present class names.
        This is default format returned when using Roboflow.

        @type dataset_dir: str
        @param dataset_dir: Path to dataset directory
        """
        added_train_imgs = self.from_darknet_format(
            image_dir=osp.join(dataset_dir, "train"),
            classes_path=osp.join(dataset_dir, "train", "_darknet.labels"),
        )
        added_val_imgs = self.from_darknet_format(
            image_dir=osp.join(dataset_dir, "valid"),
            classes_path=osp.join(dataset_dir, "valid", "_darknet.labels"),
        )
        added_test_imgs = self.from_darknet_format(
            image_dir=osp.join(dataset_dir, "test"),
            classes_path=osp.join(dataset_dir, "test", "_darknet.labels"),
        )

        self.dataset.make_splits(
            definitions={
                "train": added_train_imgs,
                "val": added_val_imgs,
                "test": added_test_imgs,
            }
        )

    @parsing_wrapper
    def from_darknet_format(self, image_dir: str, classes_path: str) -> ParserOutput:
        """Parses annotations from Darknet format to LDF. Annotations include
        classification and object detection.

        @type image_dir: str
        @param image_dir: Path to directory with images
        @type classes_path: str
        @param classes_path: Path to file with class names
        @rtype: Tuple[Generator, List[str], Dict[str, Dict], List[str]]
        @return: Annotation generator, list of classes names, skeleton dictionary for
            keypoints and list of added images.
        """
        with open(classes_path) as f:
            class_names = {i: line.rstrip() for i, line in enumerate(f.readlines())}

        def generator() -> DatasetGenerator:
            images = [img for img in os.listdir(image_dir) if img.endswith(".jpg")]
            for img_path in images:
                path = osp.join(osp.abspath(image_dir), img_path)
                ann_path = osp.join(image_dir, img_path.replace(".jpg", ".txt"))
                with open(ann_path) as f:
                    annotation_data = f.readlines()

                for ann_line in annotation_data:
                    class_id, x_center, y_center, width, height = [
                        float(i) for i in ann_line.split(" ")
                    ]
                    class_name = class_names[class_id]
                    yield {
                        "file": path,
                        "class": class_name,
                        "type": "classification",
                        "value": True,
                    }

                    bbox_xywh = [
                        x_center - width / 2,
                        y_center - height / 2,
                        width,
                        height,
                    ]
                    yield {
                        "file": path,
                        "class": class_name,
                        "type": "box",
                        "value": tuple(bbox_xywh),
                    }

        added_images = self._get_added_images(generator)

        return generator, list(class_names.values()), {}, added_images

    def from_yolov6_dir(self, dataset_dir: str) -> None:
        """Parses annotations from YoloV6 annotations to LDF.

        Expected format: "images" and "labels" directories on top level and then "train",
        "valid" and "test" directories in each of them. Images are in directories under
        "images" and annotations in directories under "labels" as .txt files. On top level
        there is also "data.yaml" with names of all present classes. This is default
        format returned when using Roboflow.

        @type dataset_dir: str
        @param dataset_dir: Path to dataset directory
        """
        classes_path = osp.join(dataset_dir, "data.yaml")
        added_train_imgs = self.from_yolov6_format(
            image_dir=osp.join(dataset_dir, "images", "train"),
            annotation_dir=osp.join(dataset_dir, "labels", "train"),
            classes_path=classes_path,
        )
        added_val_imgs = self.from_yolov6_format(
            image_dir=osp.join(dataset_dir, "images", "valid"),
            annotation_dir=osp.join(dataset_dir, "labels", "valid"),
            classes_path=classes_path,
        )
        added_test_imgs = self.from_yolov6_format(
            image_dir=osp.join(dataset_dir, "images", "test"),
            annotation_dir=osp.join(dataset_dir, "labels", "test"),
            classes_path=classes_path,
        )

        self.dataset.make_splits(
            definitions={
                "train": added_train_imgs,
                "val": added_val_imgs,
                "test": added_test_imgs,
            }
        )

    @parsing_wrapper
    def from_yolov6_format(
        self, image_dir: str, annotation_dir: str, classes_path: str
    ) -> ParserOutput:
        """Parses annotations from YoloV6 format to LDF. Annotations include
        classification and object detection.

        @type image_dir: str
        @param image_dir: Path to directory with images
        @type annotation_dir: str
        @param annotation_dir: Path to directory with annotations
        @type classes_path: str
        @param classes_path: Path to yaml file with classes names
        @rtype: Tuple[Generator, List[str], Dict[str, Dict], List[str]]
        @return: Annotation generator, list of classes names, skeleton dictionary for
            keypoints and list of added images.
        """
        with open(classes_path) as f:
            classes_data = yaml.safe_load(f)
        class_names = {
            i: class_name for i, class_name in enumerate(classes_data["names"])
        }

        def generator() -> DatasetGenerator:
            for ann_file in os.listdir(annotation_dir):
                ann_path = osp.join(osp.abspath(annotation_dir), ann_file)
                path = osp.join(
                    osp.abspath(image_dir), ann_file.replace(".txt", ".jpg")
                )
                if not osp.exists(path):
                    continue

                with open(ann_path) as f:
                    annotation_data = f.readlines()

                for ann_line in annotation_data:
                    class_id, x_center, y_center, width, height = [
                        float(i) for i in ann_line.split(" ")
                    ]
                    class_name = class_names[class_id]
                    yield {
                        "file": path,
                        "class": class_name,
                        "type": "classification",
                        "value": True,
                    }

                    bbox_xywh = [
                        x_center - width / 2,
                        y_center - height / 2,
                        width,
                        height,
                    ]
                    yield {
                        "file": path,
                        "class": class_name,
                        "type": "box",
                        "value": tuple(bbox_xywh),
                    }

        added_images = self._get_added_images(generator())

        return generator, list(class_names.values()), {}, added_images

    def from_yolov4_dir(self, dataset_dir: str) -> None:
        """Parses directory with YoloV4 annotations to LDF.

        Expected format: "train", "valid" and "test" directories. Each one has images,
        "_annotations.txt" file with annotations and "_classes.txt" file with all present
        class names. This is default format returned when using Roboflow.

        @type dataset_dir: str
        @param dataset_dir: Path to dataset directory.
        """
        added_train_imgs = self.from_yolov4_format(
            image_dir=osp.join(dataset_dir, "train"),
            annotation_path=osp.join(dataset_dir, "train", "_annotations.txt"),
            classes_path=osp.join(dataset_dir, "train", "_classes.txt"),
        )
        added_val_imgs = self.from_yolov4_format(
            image_dir=osp.join(dataset_dir, "valid"),
            annotation_path=osp.join(dataset_dir, "valid", "_annotations.txt"),
            classes_path=osp.join(dataset_dir, "valid", "_classes.txt"),
        )
        added_test_imgs = self.from_yolov4_format(
            image_dir=osp.join(dataset_dir, "test"),
            annotation_path=osp.join(dataset_dir, "test", "_annotations.txt"),
            classes_path=osp.join(dataset_dir, "test", "_classes.txt"),
        )

        self.dataset.make_splits(
            definitions={
                "train": added_train_imgs,
                "val": added_val_imgs,
                "test": added_test_imgs,
            }
        )

    @parsing_wrapper
    def from_yolov4_format(
        self, image_dir: str, annotation_path: str, classes_path: str
    ) -> ParserOutput:
        """Parses annotations from YoloV4 format to LDF. Annotations include
        classification and object detection.

        @type image_dir: str
        @param image_dir: Path to directory with images
        @type annotation_path: str
        @param annotation_path: Path to annotation file
        @type classes_path: str
        @param classes_path: Path to file with class names
        @rtype: Tuple[Generator, List[str], Dict[str, Dict], List[str]]
        @return: Annotation generator, list of classes names, skeleton dictionary for
            keypoints and list of added images.
        """
        with open(classes_path) as f:
            class_names = {i: line.rstrip() for i, line in enumerate(f.readlines())}

        def generator() -> DatasetGenerator:
            with open(annotation_path) as f:
                annotation_data = [line.rstrip() for line in f.readlines()]

            for ann_line in annotation_data:
                data = ann_line.split(" ")
                img_path = data[0]

                path = osp.join(osp.abspath(image_dir), img_path)
                if not osp.exists(path):
                    continue
                else:
                    img = cv2.imread(path)
                    shape = img.shape
                    height, width = shape[0], shape[1]

                for ann_data in data[1:]:
                    curr_ann_data = ann_data.split(",")
                    class_name = class_names[int(curr_ann_data[4])]
                    yield {
                        "file": path,
                        "class": class_name,
                        "type": "classification",
                        "value": True,
                    }

                    bbox_xyxy = [float(i) for i in curr_ann_data[:4]]
                    bbox_xywh = [
                        bbox_xyxy[0] / width,
                        bbox_xyxy[1] / height,
                        (bbox_xyxy[2] - bbox_xyxy[0]) / width,
                        (bbox_xyxy[3] - bbox_xyxy[1]) / height,
                    ]
                    yield {
                        "file": path,
                        "class": class_name,
                        "type": "box",
                        "value": tuple(bbox_xywh),
                    }

        added_images = self._get_added_images(generator)

        return generator, list(class_names.values()), {}, added_images

    def from_create_ml_dir(self, dataset_dir: str) -> None:
        """Parses directory with CreateML annotations to LDF.

        Expected format: "train", "valid" and "test" directories. Each one has images
        and "_annotations.createml.json" file with annotations. This is default format
        returned when using Roboflow.

        @type dataset_dir: str
        @param dataset_dir: Path to dataset directory
        """

        added_train_imgs = self.from_create_ml_format(
            image_dir=osp.join(dataset_dir, "train"),
            annotation_path=osp.join(
                dataset_dir, "train", "_annotations.createml.json"
            ),
        )
        added_val_imgs = self.from_create_ml_format(
            image_dir=osp.join(dataset_dir, "valid"),
            annotation_path=osp.join(
                dataset_dir, "valid", "_annotations.createml.json"
            ),
        )
        added_test_imgs = self.from_create_ml_format(
            image_dir=osp.join(dataset_dir, "test"),
            annotation_path=osp.join(dataset_dir, "test", "_annotations.createml.json"),
        )

        self.dataset.make_splits(
            definitions={
                "train": added_train_imgs,
                "val": added_val_imgs,
                "test": added_test_imgs,
            }
        )

    @parsing_wrapper
    def from_create_ml_format(
        self, image_dir: str, annotation_path: str
    ) -> ParserOutput:
        """Parses annotations from CreateML format to LDF. Annotations include
        classification and object detection.

        @type image_dir: str
        @param image_dir: Path to directory with images
        @type annotation_path: str
        @param annotation_path: Path to annotation json file
        @rtype: Tuple[Generator, List[str], Dict[str, Dict], List[str]]
        @return: Annotation generator, list of classes names, skeleton dictionary for
            keypoints and list of added images.
        """
        with open(annotation_path) as f:
            annotations_data = json.load(f)

        class_names = set()
        images_annotations = []
        for annotations in annotations_data:
            path = osp.join(osp.abspath(image_dir), annotations["image"])
            if not osp.exists(path):
                continue
            else:
                img = cv2.imread(path)
                shape = img.shape
                height, width = shape[0], shape[1]

            curr_annotations = {"path": path, "classes": [], "bboxes": []}
            for curr_ann in annotations["annotations"]:
                class_name = curr_ann["label"]
                curr_annotations["classes"].append(class_name)
                class_names.add(class_name)

                bbox_ann = curr_ann["coordinates"]
                bbox_xywh = [
                    (bbox_ann["x"] - bbox_ann["width"] / 2) / width,
                    (bbox_ann["y"] - bbox_ann["height"] / 2) / height,
                    bbox_ann["width"] / width,
                    bbox_ann["height"] / height,
                ]
                curr_annotations["bboxes"].append((class_name, bbox_xywh))
            images_annotations.append(curr_annotations)

        def generator() -> DatasetGenerator:
            for curr_annotations in images_annotations:
                path = curr_annotations["path"]
                for class_name in curr_annotations["classes"]:
                    yield {
                        "file": path,
                        "class": class_name,
                        "type": "classification",
                        "value": True,
                    }
                for bbox_class, bbox in curr_annotations["bboxes"]:
                    yield {
                        "file": path,
                        "class": bbox_class,
                        "type": "box",
                        "value": tuple(bbox),
                    }

        added_images = self._get_added_images(generator)

        return generator, list(class_names), {}, added_images

    def from_tensorflow_csv_dir(self, dataset_dir: str) -> None:
        """Parses directory with TensorflowCSV annotations to LDF.

        Expected format: "train", "valid" and "test" directories. Each one has images
        and "_annotations.csv" file with annotations. This is default format
        returned when using Roboflow.

        @type dataset_dir: str
        @param dataset_dir: Path to dataset directory
        """
        added_train_imgs = self.from_tensorflow_csv_format(
            image_dir=osp.join(dataset_dir, "train"),
            annotation_path=osp.join(dataset_dir, "train", "_annotations.csv"),
        )
        added_val_imgs = self.from_tensorflow_csv_format(
            image_dir=osp.join(dataset_dir, "valid"),
            annotation_path=osp.join(dataset_dir, "valid", "_annotations.csv"),
        )
        added_test_imgs = self.from_tensorflow_csv_format(
            image_dir=osp.join(dataset_dir, "test"),
            annotation_path=osp.join(dataset_dir, "test", "_annotations.csv"),
        )

        self.dataset.make_splits(
            definitions={
                "train": added_train_imgs,
                "val": added_val_imgs,
                "test": added_test_imgs,
            }
        )

    @parsing_wrapper
    def from_tensorflow_csv_format(
        self, image_dir: str, annotation_path: str
    ) -> ParserOutput:
        """Parses annotations from TensorflowCSV format to LDF. Annotations include
        classification and object detection.

        @type image_dir: str
        @param image_dir: Path to directory with images
        @type annotation_path: str
        @param annotation_path: Path to annotation CSV file
        @rtype: Tuple[Generator, List[str], Dict[str, Dict], List[str]]
        @return: Annotation generator, list of classes names, skeleton dictionary for
        """
        with open(annotation_path) as f:
            reader = csv.reader(f, delimiter=",")

            class_names = set()
            images_annotations = {}
            for i, row in enumerate(reader):
                if i == 0:
                    idx_fname = row.index("filename")
                    idx_class = row.index("class")
                    idx_xmin = row.index("xmin")
                    idx_ymin = row.index("ymin")
                    idx_xmax = row.index("xmax")
                    idx_ymax = row.index("ymax")
                    idx_height = row.index("height")
                    idx_width = row.index("width")
                else:
                    path = osp.join(osp.abspath(image_dir), row[idx_fname])
                    if not osp.exists(path):
                        continue
                    if path not in images_annotations:
                        images_annotations[path] = {"classes": [], "bboxes": []}

                    class_name = row[idx_class]
                    images_annotations[path]["classes"].append(class_name)
                    class_names.add(class_name)

                    height = float(row[idx_height])
                    width = float(row[idx_width])
                    xmin = float(row[idx_xmin])
                    ymin = float(row[idx_ymin])
                    xmax = float(row[idx_xmax])
                    ymax = float(row[idx_ymax])
                    bbox_xywh = np.array([xmin, ymin, xmax - xmin, ymax - ymin])
                    bbox_xywh[::2] /= width
                    bbox_xywh[1::2] /= height
                    bbox_xywh = bbox_xywh.tolist()
                    images_annotations[path]["bboxes"].append((class_name, bbox_xywh))

        def generator() -> DatasetGenerator:
            for path in images_annotations:
                curr_annotations = images_annotations[path]
                for class_name in curr_annotations["classes"]:
                    yield {
                        "file": path,
                        "class": class_name,
                        "type": "classification",
                        "value": True,
                    }
                for bbox_class, bbox in curr_annotations["bboxes"]:
                    yield {
                        "file": path,
                        "class": bbox_class,
                        "type": "box",
                        "value": tuple(bbox),
                    }

        added_images = self._get_added_images(generator)

        return generator, list(class_names), {}, added_images

    def from_class_dir_dir(self, dataset_dir: str) -> None:
        """Parses directory with ClassificationDirectory annotations to LDF.
        Expected format: "train", "valid" and "test" directories. Each one has
        subdirectories with class name and images with this class inside. This is
        default format when using Roboflow.

        @type dataset_dir: str
        @param dataset_dir: Path to dataset directory
        """
        added_train_imgs = self.from_class_dir_format(
            class_dir=osp.join(dataset_dir, "train"),
        )
        added_val_imgs = self.from_class_dir_format(
            class_dir=osp.join(dataset_dir, "valid"),
        )
        added_test_imgs = self.from_class_dir_format(
            class_dir=osp.join(dataset_dir, "test"),
        )

        self.dataset.make_splits(
            definitions={
                "train": added_train_imgs,
                "val": added_val_imgs,
                "test": added_test_imgs,
            }
        )

    @parsing_wrapper
    def from_class_dir_format(self, class_dir: str) -> ParserOutput:
        """Parses annotations from classification directory format to LDF. Annotations
        include classification.

        @type class_dir: str
        @param class_dir: Path to top level directory
        @rtype: Tuple[Generator, List[str], Dict[str, Dict], List[str]]
        @return: Annotation generator, list of classes names, skeleton dictionary for
            keypoints and list of added images.
        """
        class_names = [
            d for d in os.listdir(class_dir) if osp.isdir(osp.join(class_dir, d))
        ]

        def generator() -> DatasetGenerator:
            for class_name in class_names:
                images = os.listdir(osp.join(class_dir, class_name))
                for img_path in images:
                    path = osp.join(osp.abspath(class_dir), class_name, img_path)
                    yield {
                        "file": path,
                        "class": class_name,
                        "type": "classification",
                        "value": True,
                    }

        added_images = self._get_added_images(generator)

        return generator, class_names, {}, added_images

    def from_seg_mask_dir(self, dataset_dir: str) -> None:
        """Parses directory with SegmentationMask annotations to LDF.

        Expected format: "train", "valid" and "test" directories. Each one has
        images (.jpg), their masks (.png) and "_classes.csv" with mappings between
        pixel value and class name. This is default format returned when using
        Roboflow.

        @type dataset_dir: str
        @param dataset_dir: Path to dataset directory
        """
        added_train_imgs = self.from_seg_mask_format(
            image_dir=osp.join(dataset_dir, "train"),
            seg_dir=osp.join(dataset_dir, "train"),
            classes_path=osp.join(dataset_dir, "train", "_classes.csv"),
        )
        added_val_imgs = self.from_seg_mask_format(
            image_dir=osp.join(dataset_dir, "valid"),
            seg_dir=osp.join(dataset_dir, "valid"),
            classes_path=osp.join(dataset_dir, "valid", "_classes.csv"),
        )
        added_test_imgs = self.from_seg_mask_format(
            image_dir=osp.join(dataset_dir, "test"),
            seg_dir=osp.join(dataset_dir, "test"),
            classes_path=osp.join(dataset_dir, "test", "_classes.csv"),
        )

        self.dataset.make_splits(
            definitions={
                "train": added_train_imgs,
                "val": added_val_imgs,
                "test": added_test_imgs,
            }
        )

    @parsing_wrapper
    def from_seg_mask_format(
        self, image_dir: str, seg_dir: str, classes_path: str
    ) -> ParserOutput:
        """Parses annotations with SegmentationMask format to LDF.

        Annotations include classification and segmentation.

        @type image_dir: str
        @param image_dir: Path to directory with images
        @type seg_dir: str
        @param seg_dir: Path to directory with segmentation mask
        @type classes_path: str
        @param classes_path: Path to CSV file with class names
        @rtype: Tuple[Generator, List[str], Dict[str, Dict], List[str]]
        @return: Annotation generator, list of classes names, skeleton dictionary for
            keypoints and list of added images
        """
        with open(classes_path) as f:
            reader = csv.reader(f, delimiter=",")

            class_names = {}
            for i, row in enumerate(reader):
                if i == 0:
                    idx_pixel_val = row.index("Pixel Value")

                    # NOTE: space prefix included
                    idx_class = row.index(" Class")
                else:
                    class_names[int(row[idx_pixel_val])] = row[idx_class]

        def generator() -> DatasetGenerator:
            images = [i for i in os.listdir(image_dir) if i.endswith(".jpg")]
            for image_path in images:
                mask_path = image_path.replace(".jpg", "_mask.png")
                mask_path = osp.abspath(osp.join(seg_dir, mask_path))
                path = osp.abspath(osp.join(image_dir, image_path))
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                ids = np.unique(mask)
                for id in ids:
                    class_name = class_names[id]
                    yield {
                        "file": path,
                        "class": class_name,
                        "type": "classification",
                        "value": True,
                    }

                    curr_seg_mask = np.zeros_like(mask)
                    curr_seg_mask[mask == id] = 1
                    curr_seg_mask = np.asfortranarray(
                        curr_seg_mask
                    )  # pycocotools requirement
                    curr_rle = mask_util.encode(curr_seg_mask)
                    value = (
                        curr_rle["size"][0],
                        curr_rle["size"][1],
                        curr_rle["counts"],
                    )
                    yield {
                        "file": path,
                        "class": class_name,
                        "type": "segmentation",
                        "value": value,
                    }

        added_images = self._get_added_images(generator)
        return generator, list(class_names.values()), {}, added_images

    def _get_added_images(self, generator: DatasetGeneratorFunction) -> List[str]:
        """Returns list of unique images added by the generator function.

        @type generator: L{DatasetGeneratorFunction}
        @param generator: Generator function
        @rtype: List[str]
        @return: List of added images by generator function
        """
        added_images = set()
        for item in generator():
            added_images.add(item["file"])
        return list(added_images)
