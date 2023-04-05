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

    if json_files and dirs:
        if len(json_files) > 1:
            raise Exception("Possible COCO dataset but multiple json files present - possible ambiguity.")
        json_file = json_files[0]

        try:
            with open(json_file) as file:
                json.load(file)
        except ValueError:
            raise Exception(f"{json_file} is not a valid json file.")

        # NOTE: if we want to validate the file further, we can use `if all(key in coco for key in ['images', 'annotations', 'categories']):``
        return "COCO"

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
