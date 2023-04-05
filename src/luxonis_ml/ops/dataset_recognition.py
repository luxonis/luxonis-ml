import json
import os
import yaml
from yaml.scanner import ScannerError
from pathlib import Path

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

# TODO: place for common parser args
def recognize(dataset_path: str) -> str:
    """
    dataset_path (str): Path to the root folder of the dataset.

    NOTE: Dataset type checking is done by some significant property of the dataset (has to contain json file, yaml file,..).
    """

    dataset_path = dataset_path if dataset_path[-1] != '/' else dataset_path[:-1]

    if not os.path.isdir(dataset_path):
        raise Exception("Invalid path name - not a directory.")

    ## get characteristics
    image_files = list_all_image_files(dataset_path)
    image_names = [os.path.split(image_file)[-1] for image_file in image_files]
    json_files = list_all_json_files(dataset_path)
    xml_files = list_all_xml_files(dataset_path)
    txt_files = list_all_txt_files(dataset_path)
    tfrecord_files = list_all_tfrecord_files(dataset_path)

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

        if isinstance(json_file, dict):
            if all(key in json_file for key in ['info', 'licenses', 'images', 'annotations', 'categories']):
                for image in json_file["images"]:
                    if image["file_name"] not in image_names:
                        return "unmatching json annotations"
                return "COCO"
            
            if all(key in json_file for key in ['classes', 'labels']):
                if False: # TODO
                    return "unmatching json annotations"
                return "FiftyOneDetection"
            
        if isinstance(json_file, list):
            if all(key in json_file[0] for key in ["image", "annotations"]):
                for data_instance in json_file:
                    if data_instance["image"] not in image_names:
                        return "unmatching json annotations"
                return "CreateML"

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
            for line in text.readlines():
                #breakpoint()
                if len(line.split(" ")) == 2:
                    if line.split(" ")[0] in image_names:
                        return "ClassificationWithTextAnnotations"
                elif len(line.split(" ")) >= 5:
                    if line.split(" ")[0] in image_names:
                        return "YOLO4"
                    return "YOLO5 or KITTY"

    ## Recognize based on TFRECORD - TFObjectDetectionDataset
    if tfrecord_files:
        if len(json_files) > 1:
            raise Exception("Multiple TFRECORD files present - possible ambiguity.")
        else:
            return "TFObjectDetectionDataset"

    ## No characteristic files
    if not json_files and not xml_files and not txt_files and not tfrecord_files:
        return "ClassificationDirectoryTree"

    if yaml_files and dirs:
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
    ...
