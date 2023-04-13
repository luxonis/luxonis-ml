#!/usr/bin/env python3

import json, yaml
import os
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path
import warnings
import random
import shutil

from luxonis_ml.ops.dataset_recognition import recognize
from luxonis_ml.ops.parser import Parser
from luxonis_ml.ops.dataset_type import DatasetType



def recognize_and_load_ldf(
        dataset_path,
        dataset,
        source_name,
        split,
        new_thread=True,
        dataset_size=None,
        override_main_component=None
    ):
    """
    Based on the provided path, automatically detects a dataset type and constructs a LDF dataset.
    Arguments:
        dataset_path (str): Path to the root folder of the dataset.
        dataset: [LuxonisDataset] LDF dataset instance
        source_name: [string] name of the LDFSource to add to
        split: [string] 'train', 'val', or 'test'
        dataset_size: [int] number of data instances to include in our dataset (if None include all)
        override_main_component: [LDFComponent] provide another LDFComponent if not using the main component from the LDFSource
    Returns:
        None
    """
    
    dataset_type, dataset_info = recognize(dataset_path)

    parser = Parser()

    if dataset_type.value == "LDF":
        print("Already LDF")
    
    elif dataset_type.value == "ClassificationDirectoryTree":

        class_folders_paths = dataset_info["image_dirs"] #image_dirs

        parser.parse_to_ldf(
            DatasetType.CDT, 
            new_thread=new_thread,
            dataset=dataset, 
            source_name=source_name, 
            class_folders_paths=class_folders_paths,
            split=split,
            dataset_size=dataset_size,
            override_main_component=override_main_component
        )

        while(parser.get_percentage() < 100):
            print(parser.get_percentage())
    
    elif dataset_type.value == "ClassificationWithTextAnnotations":

        image_folder_path = dataset_info["image_dir"]
        info_file_path = dataset_info["txt_annotation_files_paths"][0]
        delimiter=" " #TODO: automatically detect required delimiter

        parser.parse_to_ldf(
            DatasetType.CTA, 
            new_thread=new_thread,
            dataset=dataset, 
            source_name=source_name, 
            image_folder_path=image_folder_path,
            info_file_path=info_file_path,
            split=split,
            delimiter=delimiter, #TODO: automatically detect required delimiter
            dataset_size=dataset_size,
            override_main_component=override_main_component
        )

        while(parser.get_percentage() < 100):
            print(parser.get_percentage())

    elif dataset_type.value == "COCO":

        image_dir = dataset_info["image_dir"]
        annotation_path = dataset_info["json_file_path"]

        parser.parse_to_ldf(
            DatasetType.COCO, 
            new_thread=new_thread,
            dataset=dataset, 
            source_name=source_name, 
            image_dir=image_dir, 
            annotation_path=annotation_path, 
            split=split,
            override_main_component=override_main_component
        )

        while(parser.get_percentage() < 100):
            print(parser.get_percentage())
    
    elif dataset_type.value == "YOLO5":
        
        image_folder_path = dataset_info["image_dir"]

        parser.parse_to_ldf(
            DatasetType.YOLO5, 
            new_thread=new_thread,
            dataset=dataset, 
            source_name=source_name, 
            image_folder_path=image_folder_path,
            split=split,
            dataset_size=dataset_size,
            override_main_component=override_main_component
        )

        while(parser.get_percentage() < 100):
            print(parser.get_percentage())

    elif dataset_type.value == "unknown":
        print("Cannot recognize dataset type")

    else:
        print("Cannot load the provided dataset")