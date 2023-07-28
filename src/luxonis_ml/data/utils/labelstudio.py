import numpy as np
import os
import xml.etree.ElementTree as ET
from typing import List


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


def get_xml_config(dataset) -> str:
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
