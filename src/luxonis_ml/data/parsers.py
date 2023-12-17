import os
import json
import csv
import yaml
import cv2
import logging
import numpy as np
import xml.etree.ElementTree as ET
from typing import Generator, List, Dict, Tuple, Any, Callable, Optional, Literal

from luxonis_ml.data import LuxonisDataset
from luxonis_ml.enums import DatasetType


class LuxonisParser:
    """Class used for parsing other common dataset formats to LDF

    Attributes:
        dataset (LuxonisDataset): LuxonisDataset where data is stored
    """

    def __init__(self, **ldf_kwargs: Dict[str, Any]):
        """Initializes LuxonisDataset

        Args:
            ldf_kwargs (Dict[str, Any]): Init parameters for LuxonisDataset
        """
        self.logger = logging.getLogger(__name__)

        # if LuxonisDataset.exists(dataset_name=ldf_kwargs["dataset_name"]):
        #     self.dataset_exists = True
        # else:
        #     self.dataset_exists = False
        self.dataset_exists = False

        self.dataset = LuxonisDataset(**ldf_kwargs)
        # NOTE: remove this, for testing only
        self.dataset.delete_dataset()
        self.dataset = LuxonisDataset(**ldf_kwargs)

    def parsing_wrapper(func: Callable) -> Callable:
        """Wrapper for parsing functions that adds data to LDF

        Args:
            func (Callable): Parsing function

        Returns:
            Callable: Wrapper function
        """

        def wrapper(*args, **kwargs):
            dataset = args[0].dataset
            generator, class_names, skeletons, added_images = func(*args, **kwargs)
            dataset.set_classes(class_names)
            dataset.set_skeletons(skeletons)
            dataset.add(generator)

            return added_images

        return wrapper

    def parse(
        self,
        dataset_type: DatasetType,
        dataset_dir: str,
        **parser_kwargs: Dict[str, Any],
    ) -> LuxonisDataset:
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
        else:
            raise ValueError(f"Parsing from `{dataset_type}` not supported.")

        return self.dataset

    def parse_single_dir(
        self,
        dataset_type: DatasetType,
        split: Optional[Literal["train", "val", "test"]] = None,
        random_split: bool = False,
        split_ratios: Optional[List[float]] = None,
        **parser_kwargs: Dict[str, Any],
    ) -> LuxonisDataset:
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
        else:
            raise KeyError(f"Parsing from `{dataset_type}` not supported.")

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
    ):
        if use_keypoint_ann and not keypoint_ann_paths:
            keypoint_ann_paths = {
                "train": "raw/person_keypoints_train2017.json",
                "val": "raw/person_keypoints_val2017.json",
                "test": "raw/person_keypoints_test2017.json",  # NOTE: this is not present by default
            }

        train_ann_path = (
            os.path.join(dataset_dir, keypoint_ann_paths["train"])
            if use_keypoint_ann
            else os.path.join(dataset_dir, "train", "labels.json")
        )
        added_train_imgs = self.from_coco_format(
            image_dir=os.path.join(dataset_dir, "train", "data"),
            annotation_path=train_ann_path,
        )

        val_ann_path = (
            os.path.join(dataset_dir, keypoint_ann_paths["val"])
            if use_keypoint_ann
            else os.path.join(dataset_dir, "validation", "labels.json")
        )
        _added_val_imgs = self.from_coco_format(
            image_dir=os.path.join(dataset_dir, "validation", "data"),
            annotation_path=val_ann_path,
        )

        if not split_val_to_test:
            # NOTE: test split annotations are not included by default
            test_ann_path = (
                os.path.join(dataset_dir, keypoint_ann_paths["test"])
                if use_keypoint_ann
                else os.path.join(dataset_dir, "test", "labels.json")
            )
            added_test_imgs = self.from_coco_format(
                image_dir=os.path.join(dataset_dir, "test", "data"),
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
    def from_coco_format(
        self, image_dir: str, annotation_path: str
    ) -> Tuple[Generator, List[str], Dict[str, Dict], List[str]]:
        """Parses annotations from COCO format to LDF. Annotations include classification,
        segmentation, object detection and keypoints if present.

        Args:
            image_dir (str): Path to directory with images
            annotation_path (str): Path to annotation json file

        Returns:
            Tuple[Generator, List[str], Dict[str, Dict], List[str]]: Annotation generator,
            list of classes names, skeleton dictionary for keypoints and list of added images

        Yields:
            Iterator[Tuple[Generator, List[str], Dict[str, Dict]]]: Annotation data
        """
        with open(annotation_path) as file:
            annotation_data = json.load(file)

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

        def generator() -> Dict[str, Any]:
            for img in coco_images:
                img_id = img["id"]
                img_anns = [
                    ann for ann in coco_annotations if ann["image_id"] == img_id
                ]
                path = os.path.join(os.path.abspath(image_dir), img["file_name"])
                if not os.path.exists(path):
                    continue

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
                                [
                                    float(kp[0] / width),
                                    float(kp[1] / height),
                                    int(kp[2]),
                                ]
                            )
                        yield {
                            "file": path,
                            "class": class_name,
                            "type": "keypoints",
                            "value": keypoints,
                        }

        added_images = self._get_added_images(generator)

        return generator, class_names, skeletons, added_images

    def from_voc_dir(self, dataset_dir: str):
        train_ann_path = os.path.join(dataset_dir, "ImageSets", "Main", "train.txt")
        with open(train_ann_path) as f:
            train_split = f.readlines()

        with open(os.path.join(dataset_dir, "ImageSets", "Main", "val.txt")) as f:
            val_test_split = f.readlines()
            split_point = round(len(val_test_split) * 0.5)
            val_split = val_test_split[:split_point]
            test_split = val_test_split[split_point:]

        # create test_new.txt and val_new.txt files with new splits
        val_ann_path = os.path.join(dataset_dir, "ImageSets", "Main", "val_new.txt")
        with open(val_ann_path) as f:
            f.writelines(val_split)
        test_ann_path = os.path.join(dataset_dir, "ImageSets", "Main", "test_new.txt")
        with open(test_ann_path) as f:
            f.writelines(test_split)

        image_dir = os.path.join(dataset_dir, "JPEGImages")
        ann_dir = os.path.join(dataset_dir, "Annotations")

        added_train_imgs = self.from_voc_dir(
            image_dir=image_dir, annotation_path=train_ann_path, annotation_dir=ann_dir
        )
        added_val_imgs = self.from_voc_dir(
            image_dir=image_dir, annotation_path=val_ann_path, annotation_dir=ann_dir
        )
        added_test_imgs = self.from_voc_dir(
            image_dir=image_dir, annotation_path=test_ann_path, annotation_dir=ann_dir
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
        annotation_path: str,
        annotation_dir: str,
    ) -> Tuple[Generator, List[str], Dict[str, Dict], List[str]]:
        """Parses annotations from VOC format to LDF. Annotations include classification
        and object detection.

        Args:
            image_dir (str): Path to directory with images
            annotation_path (str): Path to txt file with image names
            annotation_dir (str): Path to directory with annotations

        Returns:
            Tuple[Generator, List[str], Dict[str, Dict], List[str]]: Annotation generator,
            list of classes names, skeleton dictionary for keypoints and list of added images

        Yields:
            Iterator[Tuple[Generator, List[str], Dict[str, Dict]]]: Annotation data
        """
        with open(annotation_path) as f:
            filenames = [l.rstrip() for l in f.readlines()]

        class_names = set()
        images_annotations = []
        for anno_file in os.listdir(annotation_dir):
            anno_xml = os.path.join(annotation_dir, anno_file)
            annotation_data = ET.parse(anno_xml)
            root = annotation_data.getroot()

            filename_item = root.find("filename")
            if filename_item not in filenames:
                continue

            path = os.path.join(os.path.abspath(image_dir), filename_item.text)
            if not os.path.exists(path):
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
                    bbox_xyxy = [float(i.text) for i in bbox_info]
                    bbox_xywh = np.array(
                        [
                            bbox_xyxy[0],
                            bbox_xyxy[1],
                            bbox_xyxy[2] - bbox_xyxy[0],
                            bbox_xyxy[3] - bbox_xyxy[1],
                        ]
                    )
                    bbox_xywh[::2] /= width
                    bbox_xywh[1::2] /= height
                    bbox_xywh = bbox_xywh.tolist()
                    curr_annotations["bboxes"].append((class_name, bbox_xywh))

            images_annotations.append(curr_annotations)

        def generator() -> Dict[str, Any]:
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

    def from_darknet_dir(self, dataset_dir: str):
        added_train_imgs = self.from_darknet_format(
            image_dir=os.path.join(dataset_dir, "train"),
            classes_path=os.path.join(dataset_dir, "_darknet.labels"),
        )
        added_val_imgs = self.from_darknet_format(
            image_dir=os.path.join(dataset_dir, "valid"),
            classes_path=os.path.join(dataset_dir, "_darknet.labels"),
        )
        added_test_imgs = self.from_darknet_format(
            image_dir=os.path.join(dataset_dir, "test"),
            classes_path=os.path.join(dataset_dir, "_darknet.labels"),
        )

        self.dataset.make_splits(
            definitions={
                "train": added_train_imgs,
                "val": added_val_imgs,
                "test": added_test_imgs,
            }
        )

    @parsing_wrapper
    def from_darknet_format(
        self, image_dir: str, classes_path: str
    ) -> Tuple[Generator, List[str], Dict[str, Dict], List[str]]:
        """Parses annotations from Darknet format to LDF. Annotations include
        classification and object detection.

        Args:
            image_dir (str): Path to directory with images
            classes_path (str): Path to file with class names

        Returns:
            Tuple[Generator, List[str], Dict[str, Dict], List[str]]: Annotation generator,
            list of classes names, skeleton dictionary for keypoints and list of added images

        Yields:
            Iterator[Tuple[Generator, List[str], Dict[str, Dict]]]: Annotation data
        """
        with open(classes_path) as f:
            class_names = {i: line.rstrip() for i, line in enumerate(f.readlines())}

        def generator() -> Dict[str, Any]:
            images = [img for img in os.listdir(image_dir) if img.endswith(".jpg")]
            for img_path in images:
                path = os.path.join(os.path.abspath(image_dir), img_path)
                ann_path = os.path.join(image_dir, img_path.replace(".jpg", ".txt"))
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

    def from_yolov6_dir(self, dataset_dir: str):
        classes_path = os.path.join(dataset_dir, "data.yaml")
        added_train_imgs = self.from_yolov6_format(
            image_dir=os.path.join(dataset_dir, "images", "train"),
            annotation_dir=os.path.join(dataset_dir, "labels", "train"),
            classes_path=classes_path,
        )
        added_val_imgs = self.from_yolov6_format(
            image_dir=os.path.join(dataset_dir, "images", "valid"),
            annotation_dir=os.path.join(dataset_dir, "labels", "valid"),
            classes_path=classes_path,
        )
        added_test_imgs = self.from_yolov6_format(
            image_dir=os.path.join(dataset_dir, "images", "test"),
            annotation_dir=os.path.join(dataset_dir, "labels", "test"),
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
    ) -> Tuple[Generator, List[str], Dict[str, Dict], List[str]]:
        """Parses annotations from YoloV6 format to LDF. Annotations include
        classification and object detection.

        Args:
            image_dir (str): Path to directory with images
            annotation_dir (str): Path to directory with annotations
            classes_path (str): Path to yaml file with classes names

        Returns:
            Tuple[Generator, List[str], Dict[str, Dict], List[str]]: Annotation generator,
            list of classes names, skeleton dictionary for keypoints and list of added images

        Yields:
            Iterator[Tuple[Generator, List[str], Dict[str, Dict]]]: Annotation data
        """
        with open(classes_path) as f:
            classes_data = yaml.safe_load(f)
        class_names = {
            i: class_name for i, class_name in enumerate(classes_data["names"])
        }

        def generator() -> Dict[str, Any]:
            for ann_file in os.listdir(annotation_dir):
                ann_path = os.path.join(os.path.abspath(annotation_dir), ann_file)
                path = os.path.join(
                    os.path.abspath(image_dir), ann_file.replace(".txt", ".jpg")
                )
                if not os.path.exists(path):
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

        added_images = self._get_added_images(generator)

        return generator, list(class_names.values()), {}, added_images

    def from_yolov4_dir(self, dataset_dir: str):
        added_train_imgs = self.from_yolov4_format(
            image_dir=os.path.join(dataset_dir, "train"),
            annotation_path=os.path.join(dataset_dir, "train", "_annotations.txt"),
            classes_path=os.path.join(dataset_dir, "train", "_classes.txt"),
        )
        added_val_imgs = self.from_yolov4_format(
            image_dir=os.path.join(dataset_dir, "valid"),
            annotation_path=os.path.join(dataset_dir, "valid", "_annotations.txt"),
            classes_path=os.path.join(dataset_dir, "valid", "_classes.txt"),
        )
        added_test_imgs = self.from_yolov4_format(
            image_dir=os.path.join(dataset_dir, "test"),
            annotation_path=os.path.join(dataset_dir, "test", "_annotations.txt"),
            classes_path=os.path.join(dataset_dir, "test", "_classes.txt"),
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
    ) -> Tuple[Generator, List[str], Dict[str, Dict], List[str]]:
        """Parses annotations from YoloV4 format to LDF. Annotations include
        classification and object detection.

        Args:
            image_dir (str): Path to directory with images
            annotation_path (str): Path to annotation file
            classes_path (str): Path to file with class names

        Returns:
            Tuple[Generator, List[str], Dict[str, Dict], List[str]]: Annotation generator,
            list of classes names, skeleton dictionary for keypoints and list of added images

        Yields:
            Iterator[Tuple[Generator, List[str], Dict[str, Dict]]]: Annotation data
        """
        with open(classes_path) as f:
            class_names = {i: line.rstrip() for i, line in enumerate(f.readlines())}

        def generator() -> Dict[str, Any]:
            with open(annotation_path) as f:
                annotation_data = [line.rstrip() for line in f.readlines()]

            for ann_line in annotation_data:
                data = ann_line.split(" ")
                img_path = data[0]

                path = os.path.join(os.path.abspath(image_dir), img_path)
                if not os.path.exists(path):
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

    def from_create_ml_dir(self, dataset_dir: str):
        added_train_imgs = self.from_create_ml_format(
            image_dir=os.path.join(dataset_dir, "train"),
            annotation_path=os.path.join(
                dataset_dir, "train", "_annotations.createml.json"
            ),
        )
        added_val_imgs = self.from_create_ml_format(
            image_dir=os.path.join(dataset_dir, "valid"),
            annotation_path=os.path.join(
                dataset_dir, "valid", "_annotations.createml.json"
            ),
        )
        added_test_imgs = self.from_create_ml_format(
            image_dir=os.path.join(dataset_dir, "test"),
            annotation_path=os.path.join(
                dataset_dir, "test", "_annotations.createml.json"
            ),
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
    ) -> Tuple[Generator, List[str], Dict[str, Dict], List[str]]:
        """Parses annotations from CreateML format to LDF. Annotations include classification
        and object detection.

        Args:
            image_dir (str): Path to directory with images
            annotation_path (str): Path to annotation json file

        Returns:
            Tuple[Generator, List[str], Dict[str, Dict], List[str]]: Annotation generator,
            list of classes names, skeleton dictionary for keypoints and list of added images

        Yields:
            Iterator[Tuple[Generator, List[str], Dict[str, Dict]]]: Annotation data
        """
        with open(annotation_path) as f:
            annotations_data = json.load(f)

        class_names = set()
        images_annotations = []
        for annotations in annotations_data:
            path = os.path.join(os.path.abspath(image_dir), annotations["image"])
            if not os.path.exists(path):
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

        def generator() -> Dict[str, Any]:
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

    def from_tensorflow_csv_dir(self, dataset_dir: str):
        added_train_imgs = self.from_tensorflow_csv_format(
            image_dir=os.path.join(dataset_dir, "train"),
            annotation_path=os.path.join(dataset_dir, "train", "_annotations.csv"),
        )
        added_val_imgs = self.from_tensorflow_csv_format(
            image_dir=os.path.join(dataset_dir, "valid"),
            annotation_path=os.path.join(dataset_dir, "valid", "_annotations.csv"),
        )
        added_test_imgs = self.from_tensorflow_csv_format(
            image_dir=os.path.join(dataset_dir, "test"),
            annotation_path=os.path.join(dataset_dir, "test", "_annotations.csv"),
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
    ) -> Tuple[Generator, List[str], Dict[str, Dict], List[str]]:
        """Parses annotations from TensorflowCSV format to LDF. Annotations include classification
        and object detection.

        Args:
            image_dir (str): Path to directory with images
            annotation_path (str): Path to annotation CSV file

        Returns:
            Tuple[Generator, List[str], Dict[str, Dict], List[str]]: Annotation generator,
            list of classes names, skeleton dictionary for keypoints and list of added images

        Yields:
            Iterator[Tuple[Generator, List[str], Dict[str, Dict]]]: Annotation data
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
                    path = os.path.join(os.path.abspath(image_dir), row[idx_fname])
                    if not os.path.exists(path):
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

        def generator() -> Dict[str, Any]:
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

    def from_class_dir_dir(self, dataset_dir: str):
        added_train_imgs = self.from_class_dir_format(
            class_dir=os.path.join(dataset_dir, "train"),
        )
        added_val_imgs = self.from_class_dir_format(
            class_dir=os.path.join(dataset_dir, "valid"),
        )
        added_test_imgs = self.from_class_dir_format(
            class_dir=os.path.join(dataset_dir, "test"),
        )

        self.dataset.make_splits(
            definitions={
                "train": added_train_imgs,
                "val": added_val_imgs,
                "test": added_test_imgs,
            }
        )

    @parsing_wrapper
    def from_class_dir_format(
        self, class_dir: str
    ) -> Tuple[Generator, List[str], Dict[str, Dict], List[str]]:
        """Parses annotations from classification directory format to LDF. Annotations
        include classification.

        Args:
            class_dir (str): Path to top level directory

        Returns:
            Tuple[Generator, List[str], Dict[str, Dict], List[str]]: Annotation generator,
            list of classes names, skeleton dictionary for keypoints and list of added images

        Yields:
            Iterator[Tuple[Generator, List[str], Dict[str, Dict]]]: Annotation data
        """
        class_names = [
            d
            for d in os.listdir(class_dir)
            if os.path.isdir(os.path.join(class_dir, d))
        ]

        def generator() -> Dict[str, Any]:
            for class_name in class_names:
                images = os.listdir(os.path.join(class_dir, class_name))
                for img_path in images:
                    path = os.path.join(
                        os.path.abspath(class_dir), class_name, img_path
                    )
                    yield {
                        "file": path,
                        "class": class_name,
                        "type": "classification",
                        "value": True,
                    }

        added_images = self._get_added_images(generator)

        return generator, class_names, {}, added_images

    # # TODO: Test once segmentation PR merged
    # @parsing_wrapper
    # def _from_seg_mask_format(
    #     self, image_dir: str, seg_dir: str, annotation_path: str
    # ) -> Tuple[Generator, List[str], Dict[str, Dict]]:
    #     """Parses annotations from segmentation masks to LDF. Annotations include
    #     classification and segmentation.

    #     Args:
    #         image_dir (str): Path to directory with images
    #         seg_dir (str): Path to directory with segmentation masks
    #         annotation_path (str): Path to annotation CSV file

    #     Returns:
    #         Tuple[Generator, List[str], Dict[str, Dict]]: Annotation generator,
    #         list of classes names and skeleton dictionary for keypoints

    #     Yields:
    #         Iterator[Tuple[Generator, List[str], Dict[str, Dict]]]: Annotation data
    #     """
    #     with open(annotation_path) as f:
    #         reader = csv.reader(f, delimiter=",")

    #         class_names = {}
    #         for i, row in enumerate(reader):
    #             if i == 0:
    #                 idx_pixel_val = row.index("Pixel Value")
    #                 idx_class = row.index(" Class")  # space prefix included
    #             else:
    #                 class_names[int(row[idx_pixel_val])] = row[idx_class]

    #     def generator() -> Dict[str, Any]:
    #         images = [i for i in os.listdir(image_dir) if i.endswith(".jpg")]
    #         for image_path in images:
    #             mask_path = image_path.removesuffix(".jpg") + "_mask.png"
    #             mask_path = os.path.abspath(os.path.join(seg_dir, mask_path))
    #             path = os.path.abspath(os.path.join(image_dir, image_path))
    #             mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    #             ids = np.unique(mask)
    #             for id in ids:
    #                 class_name = class_names[id]
    #                 yield {
    #                     "file": path,
    #                     "class": class_name,
    #                     "type": "classification",
    #                     "value": True,
    #                 }

    #                 binary_mask = (mask == id).astype(np.uint8)
    #                 yield {
    #                     "file": path,
    #                     "class": class_name,
    #                     "type": "segmentation",
    #                     "value": binary_mask,
    #                 }

    #     return generator, list(class_names.values()), {}

    def _get_added_images(self, generator: Generator) -> List[str]:
        added_images = set()
        for item in generator():
            added_images.add(item["file"])
        return list(added_images)


if __name__ == "__main__":
    # parser = LuxonisParser(dataset_name="coco_test")
    # dataset = parser.parse(
    #     dataset_type=DatasetType.COCO,
    #     dataset_dir="/home/klemen/fiftyone/coco-2017-all/",
    # )
    parser = LuxonisParser(dataset_name="coco_test_person")
    dataset = parser.parse(
        dataset_type=DatasetType.COCO,
        dataset_dir="/home/klemen/fiftyone/coco-2017-person/",
        use_keypoint_ann=True,
    )
