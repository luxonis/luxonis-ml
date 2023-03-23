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

"""
Functions used with LuxonisDataset to convert other data formats to LDF
"""

def from_coco(dataset, source_name, image_dir, annotation_path, split='train', override_main_component=None):
    """
    dataset: LuxonisDataset instance
    source_name: name of the LDFSource to add to
    image_dir: path to root directory containing images with COCO basenames
    annotation_path: path to COCO annotation file
    split: train, val, or test split
    override_main_component: provide another LDFComponent if not using the main component from the LDFSource
    """

    with open(annotation_path) as file:
        coco = json.load(file)

    images = coco['images']
    annotations = coco['annotations']
    coco_categories = coco['categories']
    categories = {cat['id']:cat['name'] for cat in coco_categories}
    keypoint_definitions = {cat['id']:{'keypoints':cat['keypoints'],'skeleton':cat['skeleton']} \
                            for cat in coco_categories if 'keypoints' in cat.keys() and 'skeleton' in cat.keys()}

    if override_main_component is not None:
        component_name = override_main_component
    else:
        component_name = dataset.sources[source_name].main_component

    for image in tqdm(images):
        new_ann = {
            component_name: {
                'annotations': []
            }
        }
        fn = image['file_name']
        image_id = image['id']
        image_path = f"{image_dir}/{fn}"
        if os.path.exists(image_path):
            img = cv2.imread(image_path)
            anns = [ann for ann in annotations if ann['image_id'] == image_id]
            for ann in anns:
                cat_id = ann['category_id']
                class_name = categories[cat_id]

                new_ann_instance = {}
                new_ann_instance['class_name'] = class_name
                class_id = dataset._add_class(new_ann_instance)
                new_ann_instance['class'] = class_id

                if cat_id in keypoint_definitions.keys():
                    dataset._add_keypoint_definition(class_id, keypoint_definitions[cat_id])

                if 'segmentation' in ann.keys():
                    segmentation = ann['segmentation']
                    if isinstance(segmentation, list) and len(segmentation) > 0: # polygon format
                        new_ann_instance['segmentation'] = {'type':'polygon', 'task':'instance', 'data':segmentation}
                    elif isinstance(segmentation, dict) and len(segmentation['counts']) > 0:
                        new_ann_instance['segmentation'] = {'type':'mask', 'task':'instance', 'data':segmentation}
                if 'bbox' in ann.keys():
                    new_ann_instance['bbox'] = ann['bbox']
                if 'keypoints' in ann.keys():
                    new_ann_instance['keypoints'] = ann['keypoints']

                new_ann[component_name]['annotations'].append(new_ann_instance)

            dataset.add_data(source_name, {
                component_name: img,
                'json': new_ann
            }, split=split)

        else:
            warnings.warn(f"skipping {fn} as it does no exist!")

def from_yolo(dataset, source_name, yaml_path, split='all', override_main_component=None):
    """
    dataset: LuxonisDataset instance
    source_name: name of the LDFSource to add to
    yaml_path: path to YAML file for YOLO format (be careful about relative paths in this file)
    split: train, val, or test split
    override_main_component: provide another LDFComponent if not using the main component from the LDFSource

    Note: only bounding boxes supported for now
    """

    yolo = yaml.safe_load(Path(yaml_path).read_text())
    classes = yolo['names']
    if yolo['nc'] != len(classes):
        raise Exception(f"nc in YOLO YAML file does not match names!")

    if split == 'all': splits = ['train', 'val', 'test']
    else: splits = [split]

    if override_main_component is not None:
        component_name = override_main_component
    else:
        component_name = dataset.sources[source_name].main_component

    for split in splits:
        path = yolo[split]
        if not os.path.exists(path):
            if 'path' in yolo.keys():
                path = f"{yolo['path']}/{path}"

        if not os.path.exists(path):
            raise Exception(f"Cannot find {split} file {path} from YOLO YAML file!")

        dir_path = path.replace(path.split('/')[-1], '')
        with open(path) as file:
            lines = file.readlines()
            image_paths = [line.replace('\n','') for line in lines]
            image_paths = [f"{dir_path}/{path}" for path in image_paths]

        for image_path in tqdm(image_paths):
            if not os.path.exists(image_path):
                warnings.warn(f"Skipping image {image_path} - not found!")
                continue
            ext = image_path.split('.')[-1]
            label_path = image_path.replace('/images/', '/labels/').replace(f".{ext}", '.txt')
            if not os.path.exists(label_path):
                warnings.warn(f"Skipping image {image_path} - label {label_path} not found!")
                continue

            # add YOLO data
            new_ann = {
                component_name: {
                    'annotations': []
                }
            }

            img = cv2.imread(image_path)
            h, w = img.shape[0], img.shape[1]

            with open(label_path) as file:
                lines = file.readlines()
                rows = len(lines)
                data = np.array([float(num) for line in lines for num in line.replace('\n','').split(' ')])
                data = data.reshape(rows, -1).tolist()

            for row in data:
                new_ann_instance = {}
                yolo_class_id = int(row[0])
                class_name = classes[yolo_class_id]
                new_ann_instance['class_name'] = class_name
                class_id = dataset._add_class(new_ann_instance)
                new_ann_instance['class'] = class_id

                # bounding box
                new_ann_instance['bbox'] = [row[1]*w, row[2]*h, row[3]*w, row[4]*h]

                new_ann[component_name]['annotations'].append(new_ann_instance)

            dataset.add_data(source_name, {
                component_name: img,
                'json': new_ann
            }, split=split)

def from_numpy_format(
        dataset, 
        source_name, 
        images, 
        labels, 
        split, 
        dataset_size=None, 
        override_main_component=None
    ):
    """
    Constructs a LDF dataset from data provided in numpy arrays.
    Arguments:
        dataset: [LuxonisDataset] LDF dataset instance
        source_name: [string] name of the LDFSource to add to
        images: [numpy.array] numpy.array of images of shape (N, image_height, image_width) or (N, image_height, image_width, color)
        labels: [numpy.array] classification labels in numpy.array of shape (N,)
        split: [string] 'train', 'val', or 'test'
        dataset_size: [int] number of data instances to include in our dataset (if None include all)
        override_main_component: [LDFComponent] provide another LDFComponent if not using the main component from the LDFSource
    Returns:
        None
    """

    ## define source component name
    if override_main_component is not None:
        component_name = override_main_component
    else:
        component_name = dataset.sources[source_name].main_component
    
    ## dataset size limit
    if dataset_size is not None:
        images = images[:dataset_size]
        labels = labels[:dataset_size]

    for image, label in zip(images, labels):

        ## structure annotations
        new_ann = {component_name: {"annotations": []}}
        new_ann_instance = {}
        new_ann_instance["class_name"] = str(label)
        new_ann[component_name]["annotations"].append(new_ann_instance)

        ## add data to the provided LDF dataset instance
        dataset.add_data(
            source_name, {component_name: image, "json": new_ann}, split=split
        )

def train_test_split_image_classification_directory_tree(
        directory_tree_path, 
        destination_path1, 
        destination_path2,
        split_proportion,
        folders_to_ignore=[]
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

        image_names = os.listdir(f"{directory_tree_path}/{folder_name}")
        random.shuffle(image_names)
        split_idx = int(len(image_names)*split_proportion)
        image_names1 = image_names[:split_idx]
        image_names2 = image_names[split_idx:]

        os.mkdir(f"{destination_path1}/{folder_name}")
        os.mkdir(f"{destination_path2}/{folder_name}")
        
        for image_name in image_names1:
            shutil.copyfile(src=f"{directory_tree_path}/{folder_name}/{image_name}", 
                            dst=f"{destination_path1}/{folder_name}/{image_name}")
        for image_name in image_names2:
            shutil.copyfile(src=f"{directory_tree_path}/{folder_name}/{image_name}", 
                            dst=f"{destination_path2}/{folder_name}/{image_name}")

def from_image_classification_directory_tree_format(
        dataset, 
        source_name, 
        directory_root, 
        split,
        dataset_size=None,
        override_main_component=None
    ):
    """
    Constructs a LDF dataset from a directory tree whose subfolders define image classes.
    Arguments:
        dataset: [LuxonisDataset] LDF dataset instance
        source_name: [string] name of the LDFSource to add to
        directory_root: [string] path to the root of directory tree
        split: [string] 'train', 'val', or 'test'
        dataset_size: [int] number of data instances to include in the LDF dataset (if None include all)
        override_main_component: [LDFComponent] provide another LDFComponent if not using the main component from the LDFSource
    Returns:
        None
    """

    ## define source component name
    if override_main_component is not None:
        component_name = override_main_component
    else:
        component_name = dataset.sources[source_name].main_component
    
    if len(os.listdir(directory_root)) == 0:
        raise RuntimeError('Directory tree is empty')

    count = 0
    for class_folder_name in os.listdir(directory_root):
        class_folder_path = os.path.join(directory_root, class_folder_name)

        for image_name in os.listdir(class_folder_path):
            ## check dataset size limit
            count += 1
            if dataset_size is not None and count > dataset_size:
                break

            image_path = os.path.join(directory_root, class_folder_name, image_name)
            if os.path.exists(image_path):
                
                ## read image
                image = cv2.imread(image_path)

                ## structure annotations
                new_ann = {component_name: {"annotations": []}}
                new_ann_instance = {}
                new_ann_instance["class_name"] = str(class_folder_name)
                new_ann[component_name]["annotations"].append(new_ann_instance)

                ## add data to the provided LDF dataset instance
                dataset.add_data(
                    source_name, {component_name: image, "json": new_ann}, split=split
                )

            else:
                raise RuntimeError('A non-valid image path was encountered')
            
def from_image_classification_with_text_annotations_format(
        dataset, 
        source_name, 
        image_folder_path,
        info_file_path,
        delimiter,
        split,
        dataset_size=None,
        override_main_component=None
    ):

    """
    Constructs a LDF dataset based on image paths and labels from text annotations.
    Arguments:
        dataset: [LuxonisDataset] LDF dataset instance
        source_name: [string] name of the LDFSource to add to
        image_folder_path: [string] path to the directory where images are stored
        info_file_path: [string] path to the text annotations file where each line encodes a name and the associated class of an image
        delimiter: [string] how image names and classes are separated in the info file (e.g. " ", "," or ";")
        split: [string] 'train', 'val', or 'test'
        dataset_size: [int] number of data instances to include in our dataset (if None include all)
        override_main_component: [LDFComponent] provide another LDFComponent if not using the main component from the LDFSource
    Returns:
        None
    """

    ## define source component name
    if override_main_component is not None:
        component_name = override_main_component
    else:
        component_name = dataset.sources[source_name].main_component
    
    if not os.path.exists(info_file_path):
        raise RuntimeError('Info file path non-existent.')

    count = 0
    with open(info_file_path) as f:
        
        for line in f:

            try:
                image_path, label = line.split(delimiter)
            except:
                raise RuntimeError('Unable to split the info file based on the provided delimiter.')

            ## read image
            image = cv2.imread(os.path.join(image_folder_path, image_path))

            ## structure annotations
            new_ann = {component_name: {"annotations": []}}
            new_ann_instance = {}
            new_ann_instance["class_name"] = str(label)
            new_ann[component_name]["annotations"].append(new_ann_instance)

            ## add data to the provided LDF dataset instance
            dataset.add_data(
                source_name, {component_name: image, "json": new_ann}, split=split
            )

            ## dataset size limit
            count += 1
            if dataset_size is not None and count >= dataset_size:
                break