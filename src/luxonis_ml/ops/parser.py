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
from luxonis_ml.ops.dataset_type import DatasetType as dt
import threading
from threading import Thread

"""
Functions used with LuxonisDataset to convert other data formats to LDF
"""
class Parser:

    def __init__(self):
        self.percentage = 0
        self.error_message = None
        self.parsing_in_progress = False

    def parsing_wrapper(func):
        def inner(*args, **kwargs):
            args[0].parsing_in_progress = True
            func(*args, **kwargs)
            args[0].parsing_in_progress = False
            args[0].percentage = 100
        return inner
    
    @parsing_wrapper
    def from_coco_format(
            self,
            dataset, 
            source_name, 
            image_dir, 
            annotation_path, 
            split, 
            override_main_component=None
        ):
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

        iter = tqdm(images)
        for image in iter:
            self.percentage = round((iter.n/iter.total)*100, 2)
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

    @parsing_wrapper
    def from_yolo5_format(
            self,
            dataset, 
            source_name, 
            image_folder_path,
            #txt_annotation_files_paths, # list of paths to the text annotation files where each line encodes a bounding box
            split, 
            dataset_size=None, 
            override_main_component=None
        ):
        """
        Constructs a LDF dataset from a YOLO5 type dataset.
        Arguments:
            dataset: [LuxonisDataset] LDF dataset instance
            source_name: [string] name of the LDFSource to add to
            image_folder_path: [string] path to the directory where images are stored
            split: [string] 'train', 'val', or 'test'
            dataset_size: [int] number of data instances to include in our dataset (if None include all)
            override_main_component: [LDFComponent] provide another LDFComponent if not using the main component from the LDFSource
        Returns:
            None
        Note: 
            only bounding boxes supported for now
        """

        ## get classes
        root_path = os.path.split(os.path.split(image_folder_path)[0])[0] #go two levels up
        yaml_files = [fname for fname in os.listdir(root_path) if fname.endswith('.yaml')]
        if len(yaml_files) > 1:
            raise RuntimeError('Multiple YAML files - possible ambiguity')
        else:
            yaml_path = yaml_files[0]
            yolo = yaml.safe_load(Path(os.path.join(root_path,yaml_path)).read_text())
            classes = yolo['names']
            if yolo['nc'] != len(classes):
                raise Exception(f"nc in YOLO YAML file does not match names!")

        ## define source component name
        if override_main_component is not None:
            component_name = override_main_component
        else:
            component_name = dataset.sources[source_name].main_component
        
        if not os.path.exists(image_folder_path):
            raise RuntimeError('image folder path non-existent.')

        count = 0
        for image_name in tqdm(os.listdir(image_folder_path)):
            image_path = os.path.join(image_folder_path, image_name)

            if not os.path.exists(image_path):
                warnings.warn(f"Skipping image {image_path} - not found!")
                continue
            ext = image_path.split('.')[-1]
            label_path = image_path.replace('/images/', '/labels/').replace(f".{ext}", '.txt')
            if not os.path.exists(label_path):
                warnings.warn(f"Skipping image {image_path} - label {label_path} not found!")
                continue

            ## check dataset size limit
            count += 1
            if dataset_size is not None and count > dataset_size:
                break

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

    @parsing_wrapper
    def from_numpy_format(
            self,
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
            images: [numpy.array] numpy.array of RGB images of shape (N, image_height, image_width) or (N, image_height, image_width, color)
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

            ## change image to BGR
            image = image[:,:,::-1]

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
            self,
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

    @parsing_wrapper
    def from_image_classification_directory_tree_format(
            self,
            dataset, 
            source_name, 
            class_folders_paths, 
            split,
            dataset_size=None,
            override_main_component=None
        ):
        """
        Constructs a LDF dataset from a directory tree whose subfolders define image classes.
        Arguments:
            dataset: [LuxonisDataset] LDF dataset instance
            source_name: [string] name of the LDFSource to add to
            class_folders_paths: [list of strings] paths to folders containing images of specific class
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
        
        if len(class_folders_paths) == 0:
            raise RuntimeError('Directory tree is empty')

        count = 0
        iter1 = tqdm(class_folders_paths)
        for class_folder_path in iter1:
            class_folder_name = os.path.split(class_folder_path)[-1]
            iter2 = tqdm(os.listdir(class_folder_path))
            for image_name in iter2:
                self.percentage = round((iter1.n/iter1.total)*100, 2) + round(((iter2.n/iter2.total)/iter1.total)*100, 2)
                ## check dataset size limit
                count += 1
                if dataset_size is not None and count > dataset_size:
                    break

                image_path = os.path.join(class_folder_path, image_name)
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

    @parsing_wrapper    
    def from_image_classification_with_text_annotations_format(
            self,
            dataset, 
            source_name, 
            image_folder_path,
            info_file_path,
            split,
            delimiter=" ",
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
            split: [string] 'train', 'val', or 'test'
            delimiter: [string] how image names and classes are separated in the info file (e.g. " ", "," or ";")
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
            lines = f.readlines()
            iter = tqdm(lines)
            for line in iter:
                self.percentage = round((iter.n/iter.total)*100, 2)
                try:
                    image_path, label = line.split(delimiter)
                except:
                    raise RuntimeError('Unable to split the info file based on the provided delimiter.')

                if label.endswith('\n'):
                    label.replace('\n','')

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
    
    # Defining this below all the functions
    DATASET_TYPE_TO_FUNCTION = {
        dt.COCO: from_coco_format,
        dt.YOLO5: from_yolo5_format,
        dt.CDT: from_image_classification_directory_tree_format,
        dt.CTA: from_image_classification_with_text_annotations_format,
        dt.NUMPY: from_numpy_format,
    }

    def get_percentage(self):
        return self.percentage
    
    def get_error_message(self):
        return self.error_message
    
    def get_parsing_in_progress(self):
        return self.parsing_in_progress
    
    def parse_to_ldf(self, dataset_type, *args, new_thread = False, **kwargs):
        if not new_thread:
            Parser.DATASET_TYPE_TO_FUNCTION[dataset_type](self, *args, **kwargs)
        else:
            def thread_exception_hook(args):
                self.error_message = str(args.exc_value)
                self.parsing_in_progress = False
            threading.excepthook = thread_exception_hook

            self.thread = threading.Thread(
                target=Parser.DATASET_TYPE_TO_FUNCTION[dataset_type],
                args=(self, ) + args,
                kwargs=kwargs,
                daemon=True
            )
            self.thread.start()
        