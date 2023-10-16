import json
import os
import yaml
from yaml.scanner import ScannerError
from pathlib import Path

from luxonis_ml.data.parsers import DatasetType


# TODO: place for common parser args
def recognize(dataset_path: str) -> DatasetType:
    """
    dataset_path (str): Path to the root folder of the dataset.

    NOTE: Dataset type checking is done by some significant property of the dataset (has to contain json file, yaml file,..).
    """
    
    dataset_path = dataset_path if dataset_path[-1] != "/" else dataset_path[:-1]

    if not os.path.isdir(dataset_path):
        raise Exception("Invalid path name - not a directory.")

    def get_files(directory, extension):
        return [
            f"{directory}/{file}"
            for file in os.listdir(directory)
            if file.endswith(extension)
        ]

    json_files, yaml_files, csv_files, npy_files, txt_files, xml_files = [], [], [], [], [], []

    # Check the main directory
    json_files.extend(get_files(dataset_path, ".json"))
    yaml_files.extend(get_files(dataset_path, ".yaml"))
    csv_files.extend(get_files(dataset_path, ".csv"))
    npy_files.extend(get_files(dataset_path, ".npy"))
    txt_files.extend(get_files(dataset_path, ".txt"))
    xml_files.extend(get_files(dataset_path, ".xml"))

    # Check subdirectories one level deep
    for dir_file in os.listdir(dataset_path):
        sub_dir = f"{dataset_path}/{dir_file}"
        if os.path.isdir(sub_dir):
            json_files.extend(get_files(sub_dir, ".json"))
            yaml_files.extend(get_files(sub_dir, ".yaml"))
            csv_files.extend(get_files(sub_dir, ".csv"))
            npy_files.extend(get_files(sub_dir, ".npy"))
            txt_files.extend(get_files(sub_dir, ".txt"))
            xml_files.extend(get_files(sub_dir, ".xml"))

    dirs = [
        f"{dataset_path}/{dir_file}"
        for dir_file in os.listdir(dataset_path)
        if os.path.isdir(f"{dataset_path}/{dir_file}")
    ]

    # COCO Dataset
    if json_files and dirs:
        with open(json_files[0], 'r') as f:
            data = json.load(f)
        if 'images' in data and 'annotations' in data and 'categories' in data:
            return DatasetType.COCO

    # CreateML Dataset
    if json_files:
        with open(json_files[0], 'r') as f:
            data = json.load(f)
        if 'images' in data and 'annotations' in data:
            if any('label' in ann and 'coordinates' in ann for ann in data['annotations']):
                return DatasetType.CML
            
    # YOLOv4 Dataset
    single_annotation_file = any(['.txt' in f and '/labels/' not in f for f in txt_files])
    classes_file = any(['class' in f or 'classes' in f for f in txt_files])
    if single_annotation_file and classes_file and dirs:# and cfg_files:
        return DatasetType.YOLO4
    
    # YOLOv5 Dataset
    def check_yolov5_structure(root_path):
        for dirpath, dirnames, filenames in os.walk(root_path):
            if "images" in dirnames or "labels" in dirnames:
                image_dir_present = os.path.isdir(os.path.join(dirpath, "images"))
                label_dir_present = os.path.isdir(os.path.join(dirpath, "labels"))
                if image_dir_present and label_dir_present:
                    return True
        return False
    
    if yaml_files and check_yolov5_structure(dataset_path):
        return DatasetType.YOLO5
    
    # VOC Dataset
    voc_dirs = ['Annotations', 'JPEGImages', 'ImageSets']
    if all([os.path.isdir(f"{dataset_path}/{dir_name}") for dir_name in voc_dirs]):
        return DatasetType.VOC
    if xml_files and dirs:
        return DatasetType.VOC
    
    # TFObjectDetectionCSV Dataset
    if csv_files and dirs:
        train_labels = any(['train_labels.csv' in f for f in csv_files])
        test_labels = any(['test_labels.csv' in f for f in csv_files])
        if train_labels or test_labels:
            return DatasetType.TFODC
    if csv_files and dirs:
        return DatasetType.TFODC
    
    # Numpy Dataset
    if npy_files:
        return DatasetType.NUMPY

    # ClassificationDirectoryTree Dataset
    if dirs and not json_files and not yaml_files and not csv_files and not txt_files:
        return DatasetType.CDT

    # ClassificationWithTextAnnotations Dataset
    if txt_files and dirs:
        return DatasetType.CTA
    
    return DatasetType.UNKNOWN
