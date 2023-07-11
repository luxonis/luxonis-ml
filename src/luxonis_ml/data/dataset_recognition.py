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
    dirs = list(filter(lambda file_dir: os.path.isdir(file_dir), file_dir_list))

    if json_files and dirs:
        if len(json_files) > 1:
            raise Exception(
                "Possible COCO dataset but multiple json files present - possible ambiguity."
            )
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
            raise Exception(
                "Possible YOLO dataset but multiple yaml files present - possible ambiguity."
            )
        yaml_file = yaml_files[0]
        try:
            yaml.safe_load(Path(yaml_file).read_text())
        except ScannerError:
            raise Exception(f"{yaml_file} is not a valid yaml file.")

        return "YAML"

    return "Unknown dataset"
    # TODO: - rest of datasets + appropriate parsers call
    ...
