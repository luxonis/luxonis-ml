import json, yaml
import os
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path
import warnings
import random
import shutil

# from luxonis_ml.data.dataset_type import DatasetType as dt
from luxonis_ml.data.dataset import BucketStorage
from enum import Enum
import threading
from threading import Thread
import xml.etree.ElementTree as ET
from luxonis_ml.data import *
import csv


class DatasetType(Enum):
    LDF = "LDF"
    S3DIR = "S3DIR"
    COCO = "COCO"
    CDT = "ClassificationDirectoryTree"
    CTA = "ClassificationWithTextAnnotations"
    FOD = "FiftyOneDetection"
    CML = "CreateML"
    VOC = "VOC"
    YOLO4 = "YOLO4"
    YOLO5 = "YOLO5"
    TFODD = "TFObjectDetectionDataset"
    TFODC = "TFObjectDetectionCSV"
    YAML = "YAML"
    NUMPY = "numpy"
    UNKNOWN = "unknown"


"""
Functions used with LuxonisDataset to convert other data formats to LDF
"""


class LuxonisParser:
    def __init__(self):
        self.percentage = 0
        self.error_message = None
        self.parsing_in_progress = False
        self.source_name = "image"

    # This wrapper enables us to decorate parser functions to do some thing before and after every parser call
    def parsing_wrapper(func):
        def inner(*args, **kwargs):
            args[0].parsing_in_progress = True

            bucket_storage = BucketStorage.S3 if args[1][1] == DatasetType.S3DIR else BucketStorage.LOCAL
            with LuxonisDataset(
                args[1][0], bucket_storage=bucket_storage
            ) as dataset:
                print("setting component")
                custom_components = [
                    LDFComponent(
                        name="image",
                        htype=HType.IMAGE,  # data component type
                        itype=IType.BGR,  # image type
                    )
                ]
                # Just set source to some default name, as of right now, I think that parser users will care about its name
                print("setting source")
                dataset.create_source(
                    name=args[0].source_name, custom_components=custom_components
                )
                # once we create dataset, we just replace "dataset_info" with an actual "dataset" object and everything should work
                args = list(args)
                args[1] = dataset

                print("started parsing..")
                func(*tuple(args), **kwargs)

            args[0].parsing_in_progress = False
            args[0].percentage = 100.0

        return inner

    @parsing_wrapper
    def from_s3_directory_format(
        self,
        dataset,
        image_dir,
        splits=None,
        dataset_size=None,
        override_main_component=None,
    ):
        ## define main component
        if override_main_component is not None:
            component_name = override_main_component
        else:
            component_name = dataset.source.main_component

        ## initialize additions
        contents = dataset.client.list_objects(Bucket=dataset.bucket, Prefix=image_dir)[
            "Contents"
        ]

        if splits is None:
            idxs = np.arange(len(contents))
            np.random.shuffle(idxs)
            i1 = round(0.8 * len(contents))
            i2 = round(0.9 * len(contents))
            splits = {contents[idxs[n]]["Key"]: "train" for n in range(0, i1)}
            val = {contents[idxs[n]]["Key"]: "val" for n in range(i1, i2)}
            test = {contents[idxs[n]]["Key"]: "test" for n in range(i2, len(contents))}
            splits.update(val)
            splits.update(test)

        additions = []
        for metadata in contents:
            filepath = metadata["Key"]
            additions.append(
                {component_name: {"filepath": filepath, "split": splits[filepath]}}
            )

        dataset.add(additions, note="S3 directory parser", from_bucket=True)

    @parsing_wrapper
    def from_coco_format(
        self,
        dataset,
        image_dir,
        annotation_path,
        split,
        dataset_size=None,
        override_main_component=None,
    ):
        """
        Constructs a LDF dataset from a COCO type dataset.
        Arguments:
            dataset: [LuxonisDataset] LDF dataset instance
                     The parsing_wrapper decorator will replace this with an actual dataset object, 
                        so through this argument we pass  dataset_info = (dataset_name:str, dataset_type:DatasetType)
            image_dir: [string] path to root directory containing images
            annotation_path: [string] path to json annotations file
            split: [string] 'train', 'val', or 'test'
            dataset_size: [int] number of data instances to include in our dataset (if None include all)
            override_main_component: [LDFComponent] provide another LDFComponent if not using the main component from the LDFSource
        Returns:
            None
        """

        ## define main component
        if override_main_component is not None:
            component_name = override_main_component
        else:
            component_name = dataset.source.main_component

        ## load data
        with open(annotation_path) as file:
            coco = json.load(file)
        coco_images = coco["images"]
        coco_annotations = coco["annotations"]
        coco_categories = coco["categories"]
        categories = {cat["id"]: cat["name"] for cat in coco_categories}
        
        keypoint_definitions = {
            cat["id"]: {"keypoints": cat["keypoints"], "skeleton": cat["skeleton"]}
            for cat in coco_categories
            if "keypoints" in cat.keys() and "skeleton" in cat.keys()
        }

        additions = []
        ## dataset construction loop
        iter = tqdm(coco_images)
        for image in iter:
            self.percentage = round((iter.n / iter.total) * 100, 2)
            
            image_name = image["file_name"]
            image_id = image["id"]
            image_width = image["width"]
            image_height = image["height"]

            image_path = os.path.join(image_dir, image_name)
            if os.path.exists(image_path):
                image = cv2.imread(image_path)
            else:
                warnings.warn(f"skipping {image_path} as it does no exist!")
                continue
            
            annotations = [
                ann for ann in coco_annotations if ann["image_id"] == image_id
            ]
            for annotation in annotations:
                addition_instance = {
                    component_name: {
                        "filepath": image_path, 
                        "split": split
                    }
                }

                cat_id = annotation["category_id"]
                class_name = categories[cat_id]
                addition_instance[component_name]["class"] = class_name                    

                if "segmentation" in annotation.keys():
                    segmentation = annotation["segmentation"]

                    # If segmentation is in polygon format
                    if isinstance(segmentation, list) and len(segmentation) > 0:
                        mask = np.zeros((image_height, image_width), dtype=np.uint8)
                        for polygon in segmentation:
                            polygon = np.array(polygon).reshape((-1, 2))
                            cv2.fillPoly(mask, [polygon.astype(int)], 1)
                        addition_instance[component_name]["segmentation"] = mask
                    
                    # If segmentation is in mask format (RLE)
                    # Note: This requires additional decoding. Placeholder for now.
                    elif isinstance(segmentation, dict) and "counts" in segmentation:
                        # Decode the RLE to get the numpy array mask
                        # mask = decode_RLE(segmentation)
                        # addition_instance[component_name]["segmentation"] = mask
                        pass

                if "bbox" in annotation.keys():
                    # Convert COCO's [xmin, ymin, width, height] to normalized format
                    xmin, ymin, width, height = annotation["bbox"]
                    
                    # Normalize the coordinates
                    norm_xmin = xmin / image_width
                    norm_ymin = ymin / image_height
                    norm_width = width / image_width
                    norm_height = height / image_height
                    
                    # Clip the bounding box dimensions to ensure they don't exceed the image boundaries
                    norm_width = min(norm_width, 1 - norm_xmin)
                    norm_height = min(norm_height, 1 - norm_ymin)
                    
                    normalized_bbox = [
                        class_name,
                        norm_xmin,
                        norm_ymin,
                        norm_width,
                        norm_height
                    ]
                    addition_instance[component_name]["boxes"] = [normalized_bbox]

                if "keypoints" in annotation.keys():
                    keypoints = annotation["keypoints"]
                    formatted_keypoints = [[class_name, keypoints]]
                    addition_instance[component_name]["keypoints"] = formatted_keypoints

                additions.append(addition_instance)
            
            ## limit dataset size
            if dataset_size is not None and iter.n + 1 >= dataset_size:
                break
        
        # set the dataset's classes
        dataset.set_classes(list(categories.values()))

        # Using the dataset's add method
        dataset.add(additions)

    @parsing_wrapper
    def from_voc_format(
        self,
        dataset,
        image_dir,
        xml_annotation_files_paths,
        split,
        dataset_size=None,
        override_main_component=None,
    ):
        """
        Constructs a LDF dataset from a VOC type dataset.
        Arguments:
            dataset: [LuxonisDataset] LDF dataset instance
                     The parsing_wrapper decorator will replace this with an actual dataset object, 
                        so through this argument we pass  dataset_info = (dataset_name:str, dataset_type:DatasetType)
            source_name: [string] name of the LDFSource to add to
            image_dir: [string] path to root directory containing images
            xml_annotation_files_paths: [list of strings] path to xml annotation files
            split: [string] 'train', 'val', or 'test'
            dataset_size: [int] number of data instances to include in our dataset (if None include all)
            override_main_component: [LDFComponent] provide another LDFComponent if not using the main component from the LDFSource
        Returns:
            None
        Note:
            only bounding boxes supported for now
        """

        ## define main component
        if override_main_component is not None:
            component_name = override_main_component
        else:
            component_name = dataset.source.main_component

        ## dataset construction loop
        iter = tqdm(xml_annotation_files_paths)
        for xml_annotation_file_path in iter:
            self.percentage = round((iter.n / iter.total) * 100, 2)

            new_ann = {component_name: {"annotations": []}}

            instance_tree = ET.parse(xml_annotation_file_path)
            instance_root = instance_tree.getroot()

            for child in instance_root:
                if child.tag == "filename":
                    image_path = os.path.join(image_dir, child.text)
                    if os.path.exists(image_path):
                        image = cv2.imread(image_path)
                    else:
                        warnings.warn(f"skipping {image_path} as it does no exist!")
                        continue

                if child.tag == "object":
                    new_ann_instance = {}

                    for info in child:
                        if info.tag == "name":
                            new_ann_instance["class_name"] = info.text
                            class_id = dataset._add_class(new_ann_instance)
                            new_ann_instance["class"] = class_id
                        if info.tag == "bndbox":
                            for point in info:
                                if point.tag == "xmin":
                                    bbox_xmin = int(point.text)
                                if point.tag == "ymin":
                                    bbox_ymin = int(point.text)
                                if point.tag == "xmax":
                                    bbox_xmax = int(point.text)
                                if point.tag == "ymax":
                                    bbox_ymax = int(point.text)
                            coco_bbox = [
                                bbox_xmin,
                                bbox_ymin,
                                bbox_xmax - bbox_xmin,
                                bbox_ymax - bbox_ymin,
                            ]  # x_min, y_min, width, height
                            new_ann_instance["bbox"] = coco_bbox

                    new_ann[component_name]["annotations"].append(new_ann_instance)

            ## add to dataset
            dataset.add_data(
                self.source_name, {component_name: image, "json": new_ann}, split=split
            )

            ## limit dataset size
            if dataset_size is not None and iter.n + 1 >= dataset_size:
                break

    @parsing_wrapper
    def from_yolo4_format(
        self,
        dataset,
        image_dir,
        txt_annotations_file_path,
        classes_txt_file_path,
        split,
        dataset_size=None,
        override_main_component=None,
    ):
        """
        Constructs a LDF dataset from a YOLO4 type dataset.
        Arguments:
            dataset: [LuxonisDataset] LDF dataset instance
                     The parsing_wrapper decorator will replace this with an actual dataset object, 
                        so through this argument we pass  dataset_info = (dataset_name:str, dataset_type:DatasetType)
            source_name: [string] name of the LDFSource to add to
            image_dir: [string] path to the directory where images are stored
            split: [string] 'train', 'val', or 'test'
            dataset_size: [int] number of data instances to include in our dataset (if None include all)
            override_main_component: [LDFComponent] provide another LDFComponent if not using the main component from the LDFSource
        Returns:
            None
        Note:
            only bounding boxes supported for now
        """

        ## define main component
        if override_main_component is not None:
            component_name = override_main_component
        else:
            component_name = dataset.source.main_component

        ## get class names
        if not os.path.exists(classes_txt_file_path):
            raise RuntimeError("classes file path non-existent.")
        with open(classes_txt_file_path, "r", encoding="utf-8-sig") as text:
            classes = text.readlines()
            classes = [class_name.replace("\n", "") for class_name in classes]

        ## dataset construction loop
        with open(txt_annotations_file_path, "r", encoding="utf-8-sig") as text:
            iter = tqdm(text.readlines())
            for line in iter:
                self.percentage = round((iter.n / iter.total) * 100, 2)

                new_ann = {component_name: {"annotations": []}}

                image_name = line.split(" ")[0]
                image_path = os.path.join(image_dir, image_name)
                if os.path.exists(image_path):
                    image = cv2.imread(image_path)
                else:
                    warnings.warn(f"skipping {image_path} as it does no exist!")
                    continue

                for annotation in line.split(" ")[1:]:
                    new_ann_instance = {}

                    bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax, class_idx = [
                        int(x) for x in annotation.split(",")
                    ]

                    new_ann_instance["class_name"] = classes[class_idx]
                    class_id = dataset._add_class(new_ann_instance)
                    new_ann_instance["class"] = class_id

                    coco_bbox_format = [
                        bbox_xmin,
                        bbox_ymin,
                        bbox_xmax - bbox_xmin,
                        bbox_ymax - bbox_ymin,
                    ]  # x_min, y_min, width, height
                    new_ann_instance["bbox"] = coco_bbox_format
                    new_ann[component_name]["annotations"].append(new_ann_instance)

                ## add to dataset
                dataset.add_data(
                    self.source_name,
                    {component_name: image, "json": new_ann},
                    split=split,
                )

                ## limit dataset size
                if dataset_size is not None and iter.n + 1 >= dataset_size:
                    break

    @parsing_wrapper
    def from_yolo5_format(
        self,
        dataset,
        image_dir,
        # txt_annotation_files_paths, # list of paths to the text annotation files where each line encodes a bounding box
        split,
        dataset_size=None,
        override_main_component=None,
    ):
        """
        Constructs a LDF dataset from a YOLO5 type dataset.
        Arguments:
            dataset: [LuxonisDataset] LDF dataset instance
                     The parsing_wrapper decorator will replace this with an actual dataset object, 
                        so through this argument we pass  dataset_info = (dataset_name:str, dataset_type:DatasetType)
            source_name: [string] name of the LDFSource to add to
            image_dir: [string] path to the directory where images are stored
            split: [string] 'train', 'val', or 'test'
            dataset_size: [int] number of data instances to include in our dataset (if None include all)
            override_main_component: [LDFComponent] provide another LDFComponent if not using the main component from the LDFSource
        Returns:
            None
        Note:
            only bounding boxes supported for now
        """

        ## define main component
        if override_main_component is not None:
            component_name = override_main_component
        else:
            component_name = dataset.source.main_component

        ## get class names
        root_path = os.path.split(os.path.split(image_dir)[0])[0]  # go two levels up
        yaml_files = [
            fname for fname in os.listdir(root_path) if fname.endswith(".yaml")
        ]
        if len(yaml_files) > 1:
            raise RuntimeError("Multiple YAML files - possible ambiguity")
        else:
            yaml_path = yaml_files[0]
            yolo = yaml.safe_load(Path(os.path.join(root_path, yaml_path)).read_text())
            classes = yolo["names"]
            if yolo["nc"] != len(classes):
                raise Exception(f"nc in YOLO YAML file does not match names!")

        ## dataset construction loop
        iter = tqdm(os.listdir(image_dir))
        for image_name in iter:
            self.percentage = round((iter.n / iter.total) * 100, 2)

            image_path = os.path.join(image_dir, image_name)
            if os.path.exists(image_path):
                image = cv2.imread(image_path)
            else:
                warnings.warn(f"skipping {image_path} as it does no exist!")
                continue

            ext = image_path.split(".")[-1]
            label_path = image_path.replace("/images/", "/labels/").replace(
                f".{ext}", ".txt"
            )
            if not os.path.exists(label_path):
                warnings.warn(
                    f"Skipping image {image_path} - label {label_path} not found!"
                )
                continue

            # add YOLO data
            new_ann = {component_name: {"annotations": []}}

            h, w = image.shape[0], image.shape[1]

            with open(label_path) as file:
                lines = file.readlines()
                rows = len(lines)
                data = np.array(
                    [
                        float(num)
                        for line in lines
                        for num in line.replace("\n", "").split(" ")
                    ]
                )
                data = data.reshape(rows, -1).tolist()

            for row in data:
                new_ann_instance = {}
                yolo_class_id = int(row[0])
                class_name = classes[yolo_class_id]
                new_ann_instance["class_name"] = class_name
                class_id = dataset._add_class(new_ann_instance)
                new_ann_instance["class"] = class_id

                # bounding box

                bbox_xcenter = row[1] * w
                bbox_ycenter = row[2] * h

                # bbox_xmin = annotation["coordinates"]["x"]
                # bbox_ymin = annotation["coordinates"]["y"]
                bbox_width = row[3] * w
                bbox_height = row[4] * h

                bbox_xmin = bbox_xcenter - bbox_width / 2
                bbox_ymin = bbox_ycenter - bbox_height / 2

                coco_bbox_format = [bbox_xmin, bbox_ymin, bbox_width, bbox_height]
                new_ann_instance["bbox"] = coco_bbox_format
                new_ann[component_name]["annotations"].append(new_ann_instance)

                #############################
                # new_ann_instance['bbox'] = [, , , ]
                # new_ann[component_name]['annotations'].append(new_ann_instance)

            ## add to dataset
            dataset.add_data(
                self.source_name, {component_name: image, "json": new_ann}, split=split
            )

            ## limit dataset size
            if dataset_size is not None and iter.n + 1 >= dataset_size:
                break

    @parsing_wrapper
    def from_tfodc_format(
        self,
        dataset,
        image_dir,
        csv_file_path,
        split,
        dataset_size=None,
        override_main_component=None,
    ):
        """
        Constructs a LDF dataset from a TFObjectDetectionCSV type dataset.
        Arguments:
            dataset: [LuxonisDataset] LDF dataset instance
                     The parsing_wrapper decorator will replace this with an actual dataset object, 
                        so through this argument we pass  dataset_info = (dataset_name:str, dataset_type:DatasetType)
            source_name: [string] name of the LDFSource to add to
            image_folder_path: [string] path to the directory where images are stored
            csv_file_path: [string] path to csv file where annotations are stored
            split: [string] 'train', 'val', or 'test'
            dataset_size: [int] number of data instances to include in our dataset (if None include all)
            override_main_component: [LDFComponent] provide another LDFComponent if not using the main component from the LDFSource
        Returns:
            None
        Note:
            only bounding boxes supported for now
        """

        ## define main component
        if override_main_component is not None:
            component_name = override_main_component
        else:
            component_name = dataset.source.main_component

        ## extract annotations for each image
        annotations = {}
        reader = csv.reader(open(csv_file_path), delimiter=",")
        for n, row in enumerate(reader):
            if n == 0:
                idx_fname = row.index("filename")
                idx_class = row.index("class")
                idx_xmin = row.index("xmin")
                idx_ymin = row.index("ymin")
                idx_xmax = row.index("xmax")
                idx_ymax = row.index("ymax")

            else:
                image_name = row[idx_fname]
                class_name = row[idx_class]
                xmin = int(row[idx_xmin])
                ymin = int(row[idx_ymin])
                xmax = int(row[idx_xmax])
                ymax = int(row[idx_ymax])

                if image_name not in annotations:
                    annotations[image_name] = []
                annotations[image_name].append((class_name, xmin, ymin, xmax, ymax))

        ## dataset construction loop
        iter = tqdm(annotations.keys())
        for image_name in iter:
            self.percentage = round((iter.n / iter.total) * 100, 2)

            image_path = os.path.join(image_dir, image_name)
            if os.path.exists(image_path):
                image = cv2.imread(image_path)
            else:
                warnings.warn(f"skipping {image_path} as it does no exist!")
                continue

            new_ann = {component_name: {"annotations": []}}

            for annotation in annotations[image_name]:
                class_name, bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = annotation

                new_ann_instance = {}

                new_ann_instance["class_name"] = class_name
                class_id = dataset._add_class(new_ann_instance)
                new_ann_instance["class"] = class_id

                coco_bbox = [
                    bbox_xmin,
                    bbox_ymin,
                    bbox_xmax - bbox_xmin,
                    bbox_ymax - bbox_ymin,
                ]  # x_min, y_min, width, height
                new_ann_instance["bbox"] = coco_bbox
                new_ann[component_name]["annotations"].append(new_ann_instance)

            ## add data to the provided LDF dataset instance
            dataset.add_data(
                self.source_name, {component_name: image, "json": new_ann}, split=split
            )

            ## limit dataset size
            if dataset_size is not None and iter.n + 1 >= dataset_size:
                break

    @parsing_wrapper
    def from_cml_format(
        self,
        dataset,
        image_dir,
        annotation_path,
        split,
        dataset_size=None,
        override_main_component=None,
    ):
        """
        Constructs a LDF dataset from a CreateML type dataset.
        Arguments:
            dataset: [LuxonisDataset] LDF dataset instance
                     The parsing_wrapper decorator will replace this with an actual dataset object, 
                        so through this argument we pass  dataset_info = (dataset_name:str, dataset_type:DatasetType)
            source_name: [string] name of the LDFSource to add to
            image_dir: [string] path to the directory where images are stored
            annotation_path: [string] path to json file where annotations are stored
            split: [string] 'train', 'val', or 'test'
            dataset_size: [int] number of data instances to include in our dataset (if None include all)
            override_main_component: [LDFComponent] provide another LDFComponent if not using the main component from the LDFSource
        Returns:
            None
        Note:
            only bounding boxes supported for now
        """

        ## define main component
        if override_main_component is not None:
            component_name = override_main_component
        else:
            component_name = dataset.source.main_component

        ## read annotations file
        with open(annotation_path) as file:
            cml_annotations = json.load(file)

        ## dataset construction loop
        iter = tqdm(cml_annotations)
        for annotations_instance in iter:
            self.percentage = round((iter.n / iter.total) * 100, 2)

            image_name = annotations_instance["image"]
            image_path = os.path.join(image_dir, image_name)
            if os.path.exists(image_path):
                image = cv2.imread(image_path)
            else:
                warnings.warn(f"skipping {image_path} as it does no exist!")
                continue

            new_ann = {component_name: {"annotations": []}}

            annotations = annotations_instance["annotations"]
            for annotation in annotations:
                class_name = annotation["label"]
                bbox_xcenter = annotation["coordinates"]["x"]
                bbox_ycenter = annotation["coordinates"]["y"]

                bbox_width = annotation["coordinates"]["width"]
                bbox_height = annotation["coordinates"]["height"]

                bbox_xmin = bbox_xcenter - bbox_width / 2
                bbox_ymin = bbox_ycenter - bbox_height / 2

                new_ann_instance = {}

                new_ann_instance["class_name"] = class_name
                class_id = dataset._add_class(new_ann_instance)
                new_ann_instance["class"] = class_id

                coco_bbox_format = [bbox_xmin, bbox_ymin, bbox_width, bbox_height]
                new_ann_instance["bbox"] = coco_bbox_format
                new_ann[component_name]["annotations"].append(new_ann_instance)

            ## add data to the provided LDF dataset instance
            dataset.add_data(
                self.source_name, {component_name: image, "json": new_ann}, split=split
            )

            ## limit dataset size
            if dataset_size is not None and iter.n + 1 >= dataset_size:
                break

    @parsing_wrapper
    def from_numpy_format(
        self,
        dataset,
        images,
        labels,
        split,
        dataset_size=None,
        override_main_component=None,
    ):
        """
        Constructs a LDF dataset from data provided in numpy arrays.
        Arguments:
            dataset: [LuxonisDataset] LDF dataset instance
                     The parsing_wrapper decorator will replace this with an actual dataset object, 
                        so through this argument we pass  dataset_info = (dataset_name:str, dataset_type:DatasetType)
            source_name: [string] name of the LDFSource to add to
            images: [numpy.array] numpy.array of RGB images of shape (N, image_height, image_width) or (N, image_height, image_width, color)
            labels: [numpy.array] classification labels in numpy.array of shape (N,)
            split: [string] 'train', 'val', or 'test'
            dataset_size: [int] number of data instances to include in our dataset (if None include all)
            override_main_component: [LDFComponent] provide another LDFComponent if not using the main component from the LDFSource
        Returns:
            None
        """

        ## define main component
        if override_main_component is not None:
            component_name = override_main_component
        else:
            component_name = dataset.source.main_component

        ## dataset size limit
        if dataset_size is not None:
            images = images[:dataset_size]
            labels = labels[:dataset_size]

        ## dataset construction loop
        iter = tqdm(zip(images, labels))
        for image, label in iter:
            self.percentage = round((iter.n / iter.total) * 100, 2)

            new_ann = {component_name: {"annotations": []}}
            new_ann_instance = {}
            new_ann_instance["class_name"] = str(label)
            new_ann[component_name]["annotations"].append(new_ann_instance)

            dataset.add_data(
                self.source_name, {component_name: image, "json": new_ann}, split=split
            )

    def train_test_split_image_classification_directory_tree(
        self,
        directory_tree_path,
        destination_path1,
        destination_path2,
        split_proportion,
        folders_to_ignore=[],
    ):
        """
        Split directory tree into two parts. This is useful as some classification directory tree format datasets
        (e.g. Caltech101) do not separately provide data for training, validation and testing.
        Arguments:
            directory_tree_path: [string] path to the directory tree folder
            destination_path1: [string] path to the destination directory 1 - must be an existing and empty folder
            destination_path2: [string] path to the destination directory 2 - must be an existing and empty folder
            split_proportion: [float] proportion of dataset going into output directory 1
            folders_to_ignore: [list] list of folder names which should not be included in the split
        Returns:
            None
        """

        for folder_name in os.listdir(directory_tree_path):
            if folder_name in folders_to_ignore:
                continue

            image_names = os.listdir(os.path.join(directory_tree_path, folder_name))
            random.shuffle(image_names)
            split_idx = int(len(image_names) * split_proportion)
            image_names1 = image_names[:split_idx]
            image_names2 = image_names[split_idx:]

            os.mkdir(os.path.join(destination_path1, folder_name))
            os.mkdir(os.path.join(destination_path2, folder_name))

            for image_name in image_names1:
                shutil.copyfile(
                    src=os.path.join(directory_tree_path, folder_name, image_name),
                    dst=os.path.join(destination_path1, folder_name, image_name),
                )
            for image_name in image_names2:
                shutil.copyfile(
                    src=os.path.join(directory_tree_path, folder_name, image_name),
                    dst=os.path.join(destination_path2, folder_name, image_name),
                )

    @parsing_wrapper
    def from_image_classification_directory_tree_format(
        self,
        dataset,
        class_folders_paths,
        split,
        dataset_size=None,
        override_main_component=None,
    ):
        """
        Constructs a LDF dataset from a directory tree whose subfolders define image classes.
        Arguments:
            dataset: [LuxonisDataset] LDF dataset instance
                     The parsing_wrapper decorator will replace this with an actual dataset object, 
                        so through this argument we pass  dataset_info = (dataset_name:str, dataset_type:DatasetType)
            source_name: [string] name of the LDFSource to add to
            class_folders_paths: [list of strings] paths to folders containing images of specific class
            split: [string] 'train', 'val', or 'test'
            dataset_size: [int] number of data instances to include in the LDF dataset (if None include all)
            override_main_component: [LDFComponent] provide another LDFComponent if not using the main component from the LDFSource
        Returns:
            None
        """

        ## define main component
        if override_main_component is not None:
            component_name = override_main_component
        else:
            component_name = dataset.source.main_component

        if len(class_folders_paths) == 0:
            raise RuntimeError("Directory tree is empty")

        ## dataset construction loop
        iter1 = tqdm(class_folders_paths)
        for class_folder_path in iter1:
            class_folder_name = os.path.split(class_folder_path)[-1]
            iter2 = tqdm(os.listdir(class_folder_path))
            for image_name in iter2:
                self.percentage = round((iter1.n / iter1.total) * 100, 2) + round(
                    ((iter2.n / iter2.total) / iter1.total) * 100, 2
                )

                image_path = os.path.join(class_folder_path, image_name)
                if os.path.exists(image_path):
                    image = cv2.imread(image_path)
                else:
                    warnings.warn(f"skipping {image_path} as it does no exist!")
                    continue

                ## structure annotations
                new_ann = {component_name: {"annotations": []}}
                new_ann_instance = {}
                new_ann_instance["class_name"] = str(class_folder_name)
                new_ann[component_name]["annotations"].append(new_ann_instance)

                ## add data to the provided LDF dataset instance
                dataset.add_data(
                    self.source_name,
                    {component_name: image, "json": new_ann},
                    split=split,
                )

                ## limit dataset size
                if dataset_size is not None and iter.n + 1 >= dataset_size:
                    break

    @parsing_wrapper
    def from_image_classification_with_text_annotations_format(
        self,
        dataset,
        image_dir,
        info_file_path,
        split,
        delimiter=" ",
        dataset_size=None,
        override_main_component=None,
    ):
        """
        Constructs a LDF dataset based on image paths and labels from text annotations.
        Arguments:
            dataset: [LuxonisDataset] LDF dataset instance
                     The parsing_wrapper decorator will replace this with an actual dataset object, 
                        so through this argument we pass  dataset_info = (dataset_name:str, dataset_type:DatasetType)
            source_name: [string] name of the LDFSource to add to
            image_dir: [string] path to the directory where images are stored
            info_file_path: [string] path to the text annotations file where each line encodes a name and the associated class of an image
            split: [string] 'train', 'val', or 'test'
            delimiter: [string] how image names and classes are separated in the info file (e.g. " ", "," or ";")
            dataset_size: [int] number of data instances to include in our dataset (if None include all)
            override_main_component: [LDFComponent] provide another LDFComponent if not using the main component from the LDFSource
        Returns:
            None
        """

        ## define main component
        if override_main_component is not None:
            component_name = override_main_component
        else:
            component_name = dataset.source.main_component

        if not os.path.exists(info_file_path):
            raise RuntimeError("Info file path non-existent.")

        ## dataset construction loop
        with open(info_file_path) as f:
            lines = f.readlines()
            iter = tqdm(lines)
            for line in iter:
                self.percentage = round((iter.n / iter.total) * 100, 2)
                try:
                    image_path, label = line.split(delimiter)
                except:
                    raise RuntimeError(
                        "Unable to split the info file based on the provided delimiter."
                    )

                label = label.strip()

                image_path = os.path.join(image_dir, image_path)
                if os.path.exists(image_path):
                    image = cv2.imread(image_path)
                else:
                    warnings.warn(f"skipping {image_path} as it does no exist!")
                    continue

                new_ann = {component_name: {"annotations": []}}
                new_ann_instance = {}
                new_ann_instance["class_name"] = str(label)
                new_ann[component_name]["annotations"].append(new_ann_instance)

                ## add data to the provided LDF dataset instance
                dataset.add_data(
                    self.source_name,
                    {component_name: image, "json": new_ann},
                    split=split,
                )

                ## limit dataset size
                if dataset_size is not None and iter.n + 1 >= dataset_size:
                    break

    # Defining this below all the functions
    DATASET_TYPE_TO_FUNCTION = {
        DatasetType.S3DIR: from_s3_directory_format,
        DatasetType.COCO: from_coco_format,
        DatasetType.YOLO4: from_yolo4_format,
        DatasetType.YOLO5: from_yolo5_format,
        DatasetType.TFODC: from_tfodc_format,
        DatasetType.CML: from_cml_format,
        DatasetType.CDT: from_image_classification_directory_tree_format,
        DatasetType.CTA: from_image_classification_with_text_annotations_format,
        DatasetType.NUMPY: from_numpy_format,
        DatasetType.VOC: from_voc_format,
    }

    def get_percentage(self):
        return self.percentage

    def get_error_message(self):
        return self.error_message

    def get_parsing_in_progress(self):
        return self.parsing_in_progress

    def parse_to_dataset(
        self, dataset_type, dataset_name, *args, new_thread=False, **kwargs
    ):
        # Order of args passed in parsing functions has to be (self, dataset_info) + other args

        dataset_info = (dataset_name, dataset_type)
        if not new_thread:
            LuxonisParser.DATASET_TYPE_TO_FUNCTION[dataset_type](
                self, dataset_info, *args, **kwargs
            )
        else:

            def thread_exception_hook(args):
                self.error_message = str(args.exc_value)
                self.parsing_in_progress = False

            threading.excepthook = thread_exception_hook

            self.thread = threading.Thread(
                target=LuxonisParser.DATASET_TYPE_TO_FUNCTION[dataset_type],
                args=(self, dataset_info) + args,
                kwargs=kwargs,
                daemon=True,
            )
            self.thread.start()
