from .dataset import LuxonisDataset
from label_studio_sdk import Client, Project

# from label_studio_converter import brush
from typing import List, Union
from pycocotools import mask as maskUtils
import numpy as np
import xml.etree.ElementTree as ET


class InputStream:
    def __init__(self, data):
        self.data = data
        self.i = 0

    def read(self, size):
        out = self.data[self.i : self.i + size]
        self.i += size
        return int(out, 2)


def access_bit(data, num):
    """from bytes array to bits by num position"""
    base = int(num // 8)
    shift = 7 - int(num % 8)
    return (data[base] & (1 << shift)) >> shift


def bytes2bit(data):
    """get bit string from bytes data"""
    return "".join([str(access_bit(data, i)) for i in range(len(data) * 8)])


def rle_to_mask(rle: List[int], height: int, width: int) -> np.array:
    """
    Converts rle to image mask
    Args:
        rle: your long rle
        height: original_height
        width: original_width

    Returns: np.array
    """

    rle_input = InputStream(bytes2bit(rle))

    num = rle_input.read(32)
    word_size = rle_input.read(5) + 1
    rle_sizes = [rle_input.read(4) + 1 for _ in range(4)]
    # print('RLE params:', num, 'values,', word_size, 'word_size,', rle_sizes, 'rle_sizes')

    i = 0
    out = np.zeros(num, dtype=np.uint8)
    while i < num:
        x = rle_input.read(1)
        j = i + 1 + rle_input.read(rle_sizes[rle_input.read(2)])
        if x:
            val = rle_input.read(word_size)
            out[i:j] = val
            i = j
        else:
            while i < j:
                val = rle_input.read(word_size)
                out[i] = val
                i += 1

    image = np.reshape(out, [height, width, 4])[:, :, 3]
    return image


def generate_random_color():
    return [np.random.randint(0, 255) for _ in range(3)]


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
                        mask = rle_to_mask(
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
        root = ET.Element("View")

        image = ET.SubElement(root, "Image")
        image.set("name", "image")
        image.set("value", "$image")
        image.set("zoom", "true")

        classes = self.dataset.fo_dataset.classes

        if "class" in classes and len(classes["class"]):
            choices = ET.SubElement(root, "Choices")
            choices.set("name", "choice")
            choices.set("toName", "image")
            for label in classes["class"]:
                choice = ET.SubElement(choices, "Choice")
                choice.set("value", label)

        if "boxes" in classes and len(classes["boxes"]):
            inst = ET.SubElement(root, "RectangleLabels")
            inst.set("name", "label")
            inst.set("toName", "image")
            for label in classes["boxes"]:
                r, g, b = generate_random_color()
                lbl = ET.SubElement(inst, "Label")
                lbl.set("value", label)
                lbl.set("background", f"rgba({r},{b},{g},0.7)")

        if "segmentation" in classes and len(classes["segmentation"]):
            inst = ET.SubElement(root, "BrushLabels")
            inst.set("name", "tag")
            inst.set("toName", "image")
            for label in classes["segmentation"]:
                r, g, b = generate_random_color()
                lbl = ET.SubElement(inst, "Label")
                lbl.set("value", label)
                lbl.set("background", f"rgba({r},{b},{g},0.7)")

        if "keypoints" in classes and len(classes["keypoints"]):
            inst = ET.SubElement(root, "KeyPointLabels")
            inst.set("name", "kp-1")
            inst.set("toName", "image")
            for label in classes["keypoints"]:
                r, g, b = generate_random_color()
                lbl = ET.SubElement(inst, "Label")
                lbl.set("value", label)
                lbl.set("background", f"rgba({r},{b},{g},0.7)")

        tree = ET.ElementTree(root)
        tree.write("tmp.xml", encoding="utf-8", xml_declaration=False)

        with open("tmp.xml") as file:
            label_config_str = file.read()

        project = Project(
            self.dataset._get_credentials("LABELSTUDIO_URL"),
            self.dataset._get_credentials("LABELSTUDIO_KEY"),
        )
        project.start_project(title=project_name, label_config=label_config_str)

        return project.id
