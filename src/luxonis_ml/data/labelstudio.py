from .dataset import LuxonisDataset
import luxonis_ml.data.utils.labelstudio as lsUtils
from label_studio_sdk import Client, Project

from typing import List
import numpy as np
import os
import xml.etree.ElementTree as ET


class LabelStudioConnector:
    def __init__(self, dataset: LuxonisDataset) -> None:
        self.dataset = dataset
        self.client = Client(
            self.dataset._get_credentials("LABELSTUDIO_URL"),
            self.dataset._get_credentials("LABELSTUDIO_KEY"),
        )
        self.client.check_connection()

    def get_projects(self):
        return self.client.list_projects()

    def push(
        self,
        project_id: int,
        sample_ids: List,
    ) -> None:
        # TODO: add option to push annotations for cases such as pre-annotation

        tasks = []
        project = Project.get_from_id(self.client, project_id)

        for sample_id in sample_ids:
            sample = self.dataset.fo_dataset[sample_id]
            filepath = sample["filepath"]

            if self.dataset.bucket_type == "aws":
                tasks.append({"image": f"s3://{self.dataset.bucket}{filepath}"})
            else:
                raise NotImplementedError()

        project.import_tasks(tasks)

    def pull(self, project_id: int) -> None:
        additions = []
        project = Project.get_from_id(self.client, project_id)

        for task in project.export_tasks():
            filepath = task["data"]["image"]
            component_name = filepath.split("/")[
                -2
            ]  # TODO: maybe update this, as it's a bit hacky
            addition = {component_name: {"filepath": filepath}}

            cls = []
            boxes = []
            segmentation = None
            seg_name_to_class = {
                v: k
                for k, v in self.dataset.fo_dataset.mask_targets["segmentation"].items()
            }
            keypoints = []

            for annotation in task["annotations"]:
                annotation = annotation["result"]
                for ann in annotation:
                    if ann["type"] == "choices":
                        cls += ann["value"]["choices"]
                    if ann["type"] == "rectanglelabels":
                        # ih, iw = ann['original_height'], ann['original_width']
                        x, y = ann["value"]["x"] / 100, ann["value"]["y"] / 100
                        w, h = ann["value"]["width"] / 100, ann["value"]["height"] / 100
                        box_cls = ann["value"]["rectanglelabels"][0]
                        boxes.append([box_cls, x, y, w, h])
                    if ann["type"] == "brushlabels":
                        mask = lsUtils.rle_to_mask(
                            ann["value"]["rle"],
                            ann["original_height"],
                            ann["original_width"],
                        )
                        mask_cls = ann["value"]["brushlabels"][0]
                        if segmentation is None:
                            segmentation = np.zeros(
                                (ann["original_height"], ann["original_width"])
                            )
                        segmentation[mask > 128] = seg_name_to_class[mask_cls]
                    if ann["type"] == "keypointlabels":
                        pass

            if len(cls):
                addition[component_name]["class"] = cls
            if len(boxes):
                addition[component_name]["boxes"] = boxes
            if segmentation is not None:
                addition[component_name]["segmentation"] = segmentation
            if len(keypoints):
                addition[component_name]["keypoints"] = keypoints

            additions.append(addition)

        self.dataset.add(additions, media_exists=True)

    def create_project(self, project_name: str):
        label_config_str = lsUtils.get_xml_config()

        project = Project(
            self.dataset._get_credentials("LABELSTUDIO_URL"),
            self.dataset._get_credentials("LABELSTUDIO_KEY"),
        )
        project.start_project(title=project_name, label_config=label_config_str)

        return project.id
