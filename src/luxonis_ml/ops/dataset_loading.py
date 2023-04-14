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

from luxonis_ml.ops import *

from luxonis_ml.ops.dataset_recognition import recognize
from luxonis_ml.ops.parser import Parser
from luxonis_ml.ops.dataset_type import DatasetType



def recognize_and_load_ldf(
        dataset_path,
        output_path=".",
        split="train",
        new_thread=False,
        dataset_size=None,
        override_main_component=None
    ):
    """
    Based on the provided path, automatically detects a dataset type and constructs a LDF dataset.
    Arguments:
        dataset_path [string]: Path to the root folder of the dataset.
        output_path [string]: Path to the output folder
        split: [string] 'train', 'val', or 'test'
        dataset_size: [int] number of data instances to include in our dataset (if None include all)
        override_main_component: [LDFComponent] provide another LDFComponent if not using the main component from the LDFSource
    Returns:
        If new_thread=True: returns [Parser] object if conversion started successfully, otherwise [None]
        If new_thread=False: returns [Bool] True if conversion was succesful, otherwise [Bool] False 
    """
    
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    DATASET_DIR = f"{str(output_path)}/{str(dataset_path.name)}_ldf"

    dataset_type, dataset_info = recognize(dataset_path)
    source_name = dataset_type.value

    if dataset_type.value == "LDF":
        print("Already a LDF")
        return None
    
    parser = Parser()

    ## initialize a local LDF repository
    with LuxonisDataset(DATASET_DIR) as dataset:
        custom_components = [
            LDFComponent(name="image", 
                        htype=HType.IMAGE, # data component type
                        itype=IType.BGR # image type
            )
        ]
        dataset.create_source(
            name=source_name, custom_components=custom_components
        )
    
        if dataset_type.value == "ClassificationDirectoryTree":

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

        elif dataset_type.value == "unknown":
            print("Cannot recognize dataset type")
            if(new_thread):
                return None
            else:
                return False

        else:
            print("Cannot load the provided dataset")
            if(new_thread):
                return None
            else:
                return False
            
    ## make training, validation, and testing data available as a WebDataset
    with LuxonisDataset(local_path=DATASET_DIR) as dataset:
        query = f"SELECT basename FROM df WHERE split='{split}';"
        dataset.to_webdataset(split, query)

    # If we are running in another thread, we return parser (for progress inspection), other True for succesful parsing
    if(new_thread):
        return parser
    else:
        return True
        
    