import numpy as np
import os
import xml.etree.ElementTree as ET
from typing import List


def generate_random_color():
    """Generates random RGB color"""

    return [np.random.randint(0, 255) for _ in range(3)]


def get_xml_config(dataset) -> str:
    """Returns the labelstudio XML for a LuxonisDataset"""

    root = ET.Element("View")

    image = ET.SubElement(root, "Image")
    image.set("name", "image")
    image.set("value", "$image")
    image.set("zoom", "true")

    classes = dataset.fo_dataset.classes

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

    os.remove("tmp.xml")

    return label_config_str
