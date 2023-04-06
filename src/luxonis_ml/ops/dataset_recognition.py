import json
import os
import yaml
from yaml.scanner import ScannerError
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd

## image formats supported by cv2
IMAGE_EXTENSIONS = [
    "bmp", 
    "dib",
    "jpeg",
    "jpg",
    "jpe",
    "jp2",
    "png",
    "WebP",
    "webp",
    "pbm",
    "pgm",
    "ppm",
    "pxm",
    "pnm",
    "sr",
    "ras",
    "tiff",
    "tif",
    "exr",
    "hdr",
    "pic"
    ]

def list_files(root="", extensions=[]):
    file_paths = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            file_paths.append(os.path.join(path, name))
    return [file_path for file_path in file_paths if file_path.split(".")[-1] in extensions]

def list_all_image_files(root):
    return list_files(root, IMAGE_EXTENSIONS)

def list_all_json_files(root):
    return list_files(root, ["json"])

def list_all_xml_files(root):
    return list_files(root, ["xml"])

def list_all_txt_files(root):
    return list_files(root, ["txt"])

def list_all_tfrecord_files(root):
    return list_files(root, ["tfrecord"]) 

def list_all_yaml_files(root):
    return list_files(root, ["yaml"]) 

def list_all_csv_files(root):
    return list_files(root, ["csv"]) 

def find_path(file_name, file_paths):
    for file_path in file_paths:
        if file_path.endswith(file_name):
            return file_path.replace(file_name,"")
    return None

# TODO: place for common parser args
def recognize(dataset_path: str) -> str:
    """
    dataset_path (str): Path to the root folder of the dataset.

    NOTE: Dataset type checking is done by some significant property of the dataset (has to contain json file, yaml file,..).
    """

    dataset_path = dataset_path if dataset_path[-1] != '/' else dataset_path[:-1]

    if not os.path.isdir(dataset_path):
        raise Exception("Invalid path name - not a directory.")

    ## get dataset characteristics
    image_files = list_all_image_files(dataset_path)
    image_names = [os.path.split(image_file)[-1] for image_file in image_files]
    json_files = list_all_json_files(dataset_path)
    xml_files = list_all_xml_files(dataset_path)
    txt_files = list_all_txt_files(dataset_path)
    tfrecord_files = list_all_tfrecord_files(dataset_path)
    yaml_files = list_all_yaml_files(dataset_path)
    csv_files = list_all_csv_files(dataset_path)

    n_of_images = len(image_files)
    if n_of_images == 0:
        if tfrecord_files:
            pass # images packed in the TFRECORD file
        else:
            raise Exception("No images found.")

    ## Recognition based on JSON - COCO, FiftyOneImageDetection, and CreateML data formats
    if json_files:
        
        if len(json_files) > 1:
            raise Exception("Multiple JSON files present - possible ambiguity.")

        try:
            with open(json_files[0]) as file:
                json_file = json.load(file)
        except ValueError:
            raise Exception(f"{json_file} is not a valid json file.")

        annotations_path = os.path.split(json_files[0])[0]
        image_paths = []

        if isinstance(json_file, dict):
            if all(key in json_file for key in ['info', 'licenses', 'images', 'annotations', 'categories']):
                for image in json_file["images"]:
                    image_name = image["file_name"]
                    image_path = find_path(image_name, image_files)
                    if image_path == None:
                        return "unmatching json annotations"
                    image_paths.append(image_path)
                
                if len(set(image_paths)) > 1:
                    return "multiple image paths - possible ambiguity."
                images_path = image_paths[0]

                return "COCO", {"images_path": images_path, "annotations_path": annotations_path}
            
            if all(key in json_file for key in ['classes', 'labels']):
                if False: # TODO
                    return "unmatching json annotations"
                return "FiftyOneDetection"
            
        if isinstance(json_file, list):
            if all(key in json_file[0] for key in ["image", "annotations"]):
                for data_instance in json_file:
                    image_name = data_instance["image"]
                    image_path = find_path(image_name, image_files)
                    if image_path == None:
                        return "unmatching json annotations"
                    image_paths.append(image_path)
                
                if len(set(image_paths)) > 1:
                    return "multiple image paths - possible ambiguity."
                images_path = image_paths[0]

                return "CreateML", {"images_path": images_path, "annotations_path": annotations_path}

    ## Recognize based on XML - PascalVOC data format
    if xml_files:
        for xml_file in xml_files:
            instance_tree = ET.parse(xml_file)
            instance_root = instance_tree.getroot()
            for child in instance_root:
                if child.tag == "filename":
                    if child.text not in image_names:
                        return "unmatching xml annotations"
        return "VOC"

    ## Recognize based on TXT - YOLO4, YOLO5, and KITTY data formats
    for txt_file in txt_files:
        with open(txt_file, 'r', encoding='utf-8-sig') as text:
            possible_formats = []
            for line in text.readlines():
                line_split = line.split(" ")
                
                if len(line_split) == 2:
                    if line_split[0] in image_names:
                        if line_split[1].split(",")[0] == line_split[1]:
                            possible_formats.append("ClassificationWithTextAnnotations")
                        else:
                            possible_formats.append("YOLO4")
                
                elif len(line_split) > 2:
                    if line_split[0] in image_names:
                        possible_formats.append("YOLO4")
                    else:
                        possible_formats.append("YOLO5 or KITTY")

            if len(set(possible_formats)) == 1:
                return possible_formats[0]

    ## Recognize based on TFRECORD - TFObjectDetectionDataset
    if tfrecord_files:
        if len(json_files) > 1:
            raise Exception("Multiple TFRECORD files present - possible ambiguity.")
        else:
            return "TFObjectDetectionDataset"

    ## Recognize based on CSV - TFObjectDetectionCSV
    if csv_files:
        for csv_file in csv_files:
            for instance_name in pd.read_csv(csv_file)["filename"]:
                if instance_name not in image_names:
                    return "unmatching csv annotations"
        return "TFObjectDetectionCSV"

    ## No characteristic files
    if not json_files and not xml_files and not txt_files:
        if not tfrecord_files and not yaml_files and not csv_files:
            return "ClassificationDirectoryTree"

    ## Recognize based on YAML files
    # TODO - legacy code - not sure what this detects
    if yaml_files:
        if len(yaml_files) > 1:
            raise Exception("Possible YOLO dataset but multiple yaml files present - possible ambiguity.")
        yaml_file = yaml_files[0]
        try:
            yaml.safe_load(Path(yaml_file).read_text())
        except ScannerError:
            raise Exception(f"{yaml_file} is not a valid yaml file.")
        return "YAML"

    return "Unknown dataset"
    # TODO: - rest of datasets + appropriate parsers call