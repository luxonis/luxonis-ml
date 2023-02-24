#!/usr/bin/env python3

import json, yaml
import os
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path

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
