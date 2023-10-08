import json
import os
import yaml
from yaml.scanner import ScannerError
from pathlib import Path


# TODO: place for common parser args
def recognize(dataset_path: str) -> str:
    """
    dataset_path (str): Path to the root folder of the dataset.

    NOTE: Dataset type checking is done by some significant property of the dataset (has to contain json file, yaml file,..).
    """
    
    dataset_path = dataset_path if dataset_path[-1] != "/" else dataset_path[:-1]

    if not os.path.isdir(dataset_path):
        raise Exception("Invalid path name - not a directory.")

    file_dir_list = [
        f"{dataset_path}/{dir_file}" for dir_file in os.listdir(dataset_path)
    ]
    json_files = list(
        filter(lambda file_dir: file_dir.split(".")[-1] == "json", file_dir_list)
    )
    yaml_files = list(
        filter(lambda file_dir: file_dir.split(".")[-1] == "yaml", file_dir_list)
    )
    csv_files = list(
        filter(lambda file_dir: file_dir.split(".")[-1] == "csv", file_dir_list)
    )
    npy_files = list(
        filter(lambda file_dir: file_dir.split(".")[-1] == "npy", file_dir_list)
    )
    txt_files = list(
        filter(lambda file_dir: file_dir.split(".")[-1] == "txt", file_dir_list)
    )
    xml_files = list(
        filter(lambda file_dir: file_dir.split(".")[-1] == "xml", file_dir_list)
    )
    dirs = list(filter(lambda file_dir: os.path.isdir(file_dir), file_dir_list))

    # COCO Dataset
    if json_files and dirs:
        with open(json_files[0], 'r') as f:
            data = json.load(f)
        if 'images' in data and 'annotations' in data and 'categories' in data:
            return "COCO"

    # CreateML Dataset
    if json_files:
        with open(json_files[0], 'r') as f:
            data = json.load(f)
        if 'images' in data and 'annotations' in data:
            if any('label' in ann and 'coordinates' in ann for ann in data['annotations']):
                return "CreateML"
            
    # YOLOv4 Dataset
    single_annotation_file = any(['.txt' in f and '/labels/' not in f for f in txt_files])
    classes_file = any(['class' in f or 'classes' in f for f in txt_files])
    if single_annotation_file and classes_file and dirs:# and cfg_files:
        return "YOLO4"
    
    # YOLOv5 Dataset
    image_dir_present = os.path.isdir(os.path.join(dataset_path, "images"))
    label_dir_present = os.path.isdir(os.path.join(dataset_path, "labels"))
    if yaml_files and label_dir_present and image_dir_present:
        return "YOLO5"
    
    # VOC Dataset
    voc_dirs = ['Annotations', 'JPEGImages', 'ImageSets']
    if all([os.path.isdir(f"{dataset_path}/{dir_name}") for dir_name in voc_dirs]):
        return "VOC"
    if xml_files and dirs:
        return "VOC"
    
    # TFObjectDetectionCSV Dataset
    if csv_files and dirs:
        train_labels = any(['train_labels.csv' in f for f in csv_files])
        test_labels = any(['test_labels.csv' in f for f in csv_files])
        if train_labels or test_labels:
            return "TFObjectDetectionCSV"
    if csv_files and dirs:
        return "TFObjectDetectionCSV"
    
    # Numpy Dataset
    if npy_files:
        return "numpy"

    # ClassificationDirectoryTree Dataset
    if dirs and not json_files and not yaml_files and not csv_files and not txt_files:
        return "ClassificationDirectoryTree"

    # ClassificationWithTextAnnotations Dataset
    if txt_files and dirs:
        return "ClassificationWithTextAnnotations"

    
    return "unknown"
