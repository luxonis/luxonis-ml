import os
import json
import csv
import yaml
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from typing import Generator, List, Dict, Tuple, Any, Callable

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
            generator, class_names, skeletons = func(*args, **kwargs)

            dataset.set_classes(class_names)
            dataset.set_skeletons(skeletons)
            dataset.add(generator)

        return wrapper

    def parse(
        self, dataset_type: DatasetType, **parser_kwargs: Dict[str, Any]
    ) -> LuxonisDataset:
        """Parses annotations from specified dataset type to LDF

        Args:
            dataset_type (DatasetType): Input dataset type
            parser_kwargs (Dict[str, Any]): Parameters specific to chosen
            dataset type parser

        Raises:
            KeyError: Raised if input dataset type not supported

        Returns:
            LuxonisDataset: LuxonisDataset with parsed annotations
        """
        if dataset_type == DatasetType.LDF:
            pass
        elif dataset_type == DatasetType.COCO:
            self._from_coco_format(**parser_kwargs)
        elif dataset_type == DatasetType.VOC:
            self._from_voc_format(**parser_kwargs)
        elif dataset_type == DatasetType.DARKNET:
            self._from_darknet_format(**parser_kwargs)
        elif dataset_type == DatasetType.YOLOV6:
            self._from_yolov6_format(**parser_kwargs)
        elif dataset_type == DatasetType.YOLOV4:
            self._from_yolov4_format(**parser_kwargs)
        elif dataset_type == DatasetType.CREATEML:
            self._from_create_ml_format(**parser_kwargs)
        elif dataset_type == DatasetType.TFCSV:
            self._from_tensorflow_csv_format(**parser_kwargs)
        elif dataset_type == DatasetType.CLSDIR:
            self._from_class_dir_format(**parser_kwargs)
        # elif dataset_type == DatasetType.SEGMASK:
        #     self._from_seg_mask_format(**parser_kwargs)
        else:
            raise KeyError(f"Parsing from `{dataset_type}` not supported.")
        return self.dataset

    @parsing_wrapper
    def _from_coco_format(
        self, image_dir: str, annotation_path: str
    ) -> Tuple[Generator, List[str], Dict[str, Dict]]:
        """Parses annotations from COCO format to LDF. Annotations include classification,
        segmentation, object detection and keypoints if present.

        Args:
            image_dir (str): Path to directory with images
            annotation_path (str): Path to annotation json file

        Returns:
            Tuple[Generator, List[str], Dict[str, Dict]]: Annotation generator,
            list of classes names and skeleton dictionary for keypoints

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
                                [poly_arr[i, 0] / width, poly_arr[i, 1] / height]
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
                        "value": [x / width, y / height, w / width, h / height],
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

        return generator, class_names, skeletons

    @parsing_wrapper
    def _from_voc_format(
        self,
        image_dir: str,
        annotation_dir: str,
    ) -> Tuple[Generator, List[str], Dict[str, Dict]]:
        """Parses annotations from VOC format to LDF. Annotations include classification
        and object detection.

        Args:
            image_dir (str): Path to directory with images
            annotation_dir (str): Path to directory with annotations

        Returns:
            Tuple[Generator, List[str], Dict[str, Dict]]: Annotation generator,
            list of classes names and skeleton dictionary for keypoints

        Yields:
            Iterator[Tuple[Generator, List[str], Dict[str, Dict]]]: Annotation data
        """
        # NOTE: also includes segmentation masks but not for all images
        class_names = set()
        images_annotations = []
        for anno_file in os.listdir(annotation_dir):
            anno_xml = os.path.join(annotation_dir, anno_file)
            annotation_data = ET.parse(anno_xml)
            root = annotation_data.getroot()

            filename_item = root.find("filename")
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
                        "value": bbox,
                    }

        return generator, list(class_names), {}

    @parsing_wrapper
    def _from_darknet_format(
        self, image_dir: str, classes_path: str
    ) -> Tuple[Generator, List[str], Dict[str, Dict]]:
        """Parses annotations from Darknet format to LDF. Annotations include
        classification and object detection.

        Args:
            image_dir (str): Path to directory with images
            classes_path (str): Path to file with class names

        Returns:
            Tuple[Generator, List[str], Dict[str, Dict]]: Annotation generator,
            list of classes names and skeleton dictionary for keypoints

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
                        "value": bbox_xywh,
                    }

        return generator, list(class_names.values()), {}

    @parsing_wrapper
    def _from_yolov6_format(
        self, image_dir: str, annotation_dir: str, classes_path: str
    ) -> Tuple[Generator, List[str], Dict[str, Dict]]:
        """Parses annotations from YoloV6 format to LDF. Annotations include
        classification and object detection.

        Args:
            image_dir (str): Path to directory with images
            annotation_dir (str): Path to directory with annotations
            classes_path (str): Path to yaml file with classes names

        Returns:
            Tuple[Generator, List[str], Dict[str, Dict]]: Annotation generator,
            list of classes names and skeleton dictionary for keypoints

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
                        "value": bbox_xywh,
                    }

        return generator, list(class_names.values()), {}

    @parsing_wrapper
    def _from_yolov4_format(
        self, image_dir: str, annotation_path: str, classes_path: str
    ) -> Tuple[Generator, List[str], Dict[str, Dict]]:
        """Parses annotations from YoloV4 format to LDF. Annotations include
        classification and object detection.

        Args:
            image_dir (str): Path to directory with images
            annotation_path (str): Path to annotation file
            classes_path (str): Path to file with class names

        Returns:
            Tuple[Generator, List[str], Dict[str, Dict]]: Annotation generator,
            list of classes names and skeleton dictionary for keypoints

        Yields:
            Iterator[Tuple[Generator, List[str], Dict[str, Dict]]]: Annotation data
        """
        with open(classes_path) as f:
            class_names = {i: line.rstrip() for i, line in enumerate(f.readlines())}

        def generate() -> Dict[str, Any]:
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
                        "value": bbox_xywh,
                    }

        return generate, list(class_names.values()), {}

    @parsing_wrapper
    def _from_create_ml_format(
        self, image_dir: str, annotation_path: str
    ) -> Tuple[Generator, List[str], Dict[str, Dict]]:
        """Parses annotations from CreateML format to LDF. Annotations include classification
        and object detection.

        Args:
            image_dir (str): Path to directory with images
            annotation_path (str): Path to annotation json file

        Returns:
            Tuple[Generator, List[str], Dict[str, Dict]]: Annotation generator,
            list of classes names and skeleton dictionary for keypoints

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
                        "value": bbox,
                    }

        return generator, list(class_names), {}

    @parsing_wrapper
    def _from_tensorflow_csv_format(
        self, image_dir: str, annotation_path: str
    ) -> Tuple[Generator, List[str], Dict[str, Dict]]:
        """Parses annotations from TensorflowCSV format to LDF. Annotations include classification
        and object detection.

        Args:
            image_dir (str): Path to directory with images
            annotation_path (str): Path to annotation CSV file

        Returns:
            Tuple[Generator, List[str], Dict[str, Dict]]: Annotation generator,
            list of classes names and skeleton dictionary for keypoints

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
                        "value": bbox,
                    }

        return generator, list(class_names), {}

    @parsing_wrapper
    def _from_class_dir_format(
        self, class_dir: str
    ) -> Tuple[Generator, List[str], Dict[str, Dict]]:
        """Parses annotations from classification directory format to LDF. Annotations
        include classification.

        Args:
            class_dir (str): Path to top level directory

        Returns:
            Tuple[Generator, List[str], Dict[str, Dict]]: Annotation generator,
            list of classes names and skeleton dictionary for keypoints

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

        return generator, class_names, {}

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
