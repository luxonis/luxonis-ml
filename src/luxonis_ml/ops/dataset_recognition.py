import json
import os
import yaml
from yaml.scanner import ScannerError
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd

def list_files(root="", extensions=[]):
    file_paths = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            file_paths.append(os.path.join(path, name))
    return [file_path for file_path in file_paths if file_path.split(".")[-1] in extensions]

def list_all_image_files(root):
    cv2_supported_image_formats = [
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
    return list_files(root, cv2_supported_image_formats)

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

def get_unique_paths(file_paths):
    return list(set([os.path.split(file_path)[0] for file_path in file_paths]))

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
    json_files = list_all_json_files(dataset_path)
    xml_files = list_all_xml_files(dataset_path)
    txt_files = list_all_txt_files(dataset_path)
    tfrecord_files = list_all_tfrecord_files(dataset_path)
    yaml_files = list_all_yaml_files(dataset_path)
    csv_files = list_all_csv_files(dataset_path)

    image_names = [os.path.split(image_file)[-1] for image_file in image_files]

    image_extensions = set([image_file.split(".")[-1] for image_file in image_files])
    
    n_of_images = len(image_files)
    if n_of_images == 0:
        if tfrecord_files:
            pass # images packed in the TFRECORD file
        else:
            raise Exception("No images found.")

    image_dirs = get_unique_paths(image_files)  
    if len(image_dirs) > 1:
        if not any ([json_files, xml_files, txt_files, tfrecord_files, yaml_files, csv_files]):
            return "ClassificationDirectoryTree", {"image_dirs": image_dirs}
        else:
            return "multiple directories with image files - possible ambiguity."
    
    xml_dirs = get_unique_paths(xml_files)
    if len(xml_dirs) > 1:
        return "multiple directories with xml files - possible ambiguity."

    ## Recognition based on JSON - COCO, FiftyOneImageDetection, and CreateML data formats
    if json_files:
        
        if len(json_files) > 1:
            raise Exception("Multiple JSON files present - possible ambiguity.")

        try:
            with open(json_files[0]) as file:
                json_file = json.load(file)
        except ValueError:
            raise Exception(f"{json_file} is not a valid json file.")

        if isinstance(json_file, dict):
            if all(key in json_file for key in ['info', 'licenses', 'images', 'annotations', 'categories']):
                for image in json_file["images"]:
                    image_name = image["file_name"]
                    if image_name not in image_names:
                        return "unmatching json annotations"
                return "COCO", {"image_dir": image_dirs[0], "json_file_path": json_files[0]}
            
            if all(key in json_file for key in ['classes', 'labels']):
                if False: # TODO
                    return "unmatching json annotations"
                return "FiftyOneDetection"
            
        if isinstance(json_file, list):
            if all(key in json_file[0] for key in ["image", "annotations"]):
                for data_instance in json_file:
                    image_name = data_instance["image"]
                    if image_name not in image_names:
                        return "unmatching json annotations"
                return "CreateML", {"image_dir": image_dirs[0], "json_file_path": json_files[0]}

    ## Recognize based on XML - PascalVOC data format
    if xml_files:
        for xml_file in xml_files:
            instance_tree = ET.parse(xml_file)
            instance_root = instance_tree.getroot()
            for child in instance_root:
                if child.tag == "filename":
                    image_name = child.text
                    if image_name not in image_names:
                        return "unmatching xml annotations"

        return "VOC", {"image_dir": image_dirs[0], "xml_files_paths": xml_files}

    ## Recognize based on TXT - YOLO4, YOLO5, and KITTY data formats
    if txt_files:
        possible_dataset_types = []
        relevant_txt_files = [] # exclude README files etc.
        for txt_file in txt_files:
            relevant_txt_file = False

            for ext in image_extensions:
                if os.path.split(txt_file)[1].replace(".txt", f".{ext}") in image_names:
                    possible_dataset_types.append("YOLO5 or KITTY")
                    relevant_txt_file = True

            with open(txt_file, 'r', encoding='utf-8-sig') as text:
                for line in text.readlines():
                    line_split = line.split(" ")
                    if line_split[0] in image_names:
                        if len(line_split) == 2:
                                if line_split[1].split(",")[0] == line_split[1]:
                                    possible_dataset_types.append("ClassificationWithTextAnnotations")
                                else:
                                    possible_dataset_types.append("YOLO4")
                        elif len(line_split) > 2:
                                possible_dataset_types.append("YOLO4")
                        relevant_txt_file = True

            if relevant_txt_file:
                relevant_txt_files.append(txt_file)
            
        if len(set(possible_dataset_types)) == 1:
            return possible_dataset_types[0], {"image_dir": image_dirs[0], "txt_annotation_files_paths": relevant_txt_files}

    ## Recognize based on TFRECORD - TFObjectDetectionDataset
    if tfrecord_files:
        if len(json_files) > 1:
            raise Exception("Multiple TFRECORD files present - possible ambiguity.")
        else:
            return "TFObjectDetectionDataset", {"tfrecord_file_path": tfrecord_files[0]}

    ## Recognize based on CSV - TFObjectDetectionCSV
    if csv_files:

        if len(csv_files) > 1:
            raise Exception("Multiple CSV files present - possible ambiguity.")

        for instance_name in pd.read_csv(csv_files[0])["filename"]:
            if instance_name not in image_names:
                return "unmatching csv annotations"
        
        return "TFObjectDetectionCSV", {"image_dir": image_dirs[0], "csv_file_path": csv_files[0]}

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