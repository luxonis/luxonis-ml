# LuxonisML Data

## Introduction

LuxonisML Data is a library for creating and interacting with datasets in the LuxonisDataFormat (LDF).

The lifecycle of an LDF dataset is as follows:

1. Creating new dataset
1. Adding data
1. Defining splits
1. (Optional) Adding new data or redefining splits
1. Loading the dataset with `LuxonisLoader` for use in a training pipeline
1. (Optional) Deleting the dataset if it is no longer needed

Each of these steps will be explained in more detail in the following examples.

## Table of Contents

- [LuxonisML Data](#luxonisml-data)
  - [Introduction](#introduction)
  - [Prerequisites](#prerequisites)
  - [LuxonisDataset](#luxonisdataset)
    - [Adding Data](#adding-data)
    - [Defining Splits](#defining-splits)
    - [CLI Reference](#cli-reference)
  - [LuxonisLoader](#luxonisloader)
  - [Dataset Loading](#dataset-loading)
  - [LuxonisParser](#luxonisparser)
    - [Dataset Creation](#dataset-creation)
    - [CLI Reference](#cli-reference)

## Prerequisites

We will be using our toy dataset `parking_lot` in all examples. The dataset consists of images of cars and motorcycles in a parking lot. Each image has a corresponding annotation in the form of a bounding box, keypoints and several segmentation masks.

**Dataset Annotations:**

| Task                        | Annotation Type   | Classes                                                                                                |
| --------------------------- | ----------------- | ------------------------------------------------------------------------------------------------------ |
| Object Detection            | Bounding Box      | car, motorcycle                                                                                        |
| Keypoint Detection          | Keypoints         | car, motorcycle                                                                                        |
| Color Segmentation          | Segmentation Mask | background, red, gree, blue                                                                            |
| Type Segmentation           | Segmentation Mask | backgeound, car, motorcycle                                                                            |
| Brand Segmentation          | Segmentation Mask | background, alfa-romeo, buick, ducati, harley, ferrari, infiniti, jeep, land-rover, roll-royce, yamaha |
| Binary Vehicle Segmentation | Segmentation Mask | vehicle                                                                                                |

To start, download the zipped dataset from TODO and extract it to a directory of your choice.

## LuxonisDataset

The first step is to create a new dataset and add some data to it.

The main part of the dataset is the `LuxonisDataset` class.
It serves as an abstraction over the dataset and provides methods
for adding data and creating splits.
You can create as many datasets as you want, each with a unique name.

Datasets can be stored locally or in one of the supported cloud storage providers.

First we import `LuxonisDataset` and create a dataset with the name `"parking_lot"`.

```python
from luxonisml.data import LuxonisDataset

dataset_name = "parking_lot"

dataset = LuxonisDataset(dataset_name)
```

> \[!NOTE\]
> By default, the dataset will be created locally. For details on different storage methods, see TODO.

> \[!NOTE\]
> If there already is a dataset with the same name, it will be loaded instead of creating a new one.
> If you want to always create a new dataset, you can pass `delete_existing=True` to the `LuxonisDataset` constructor.

### Adding Data

The next step is to add data to the dataset.
The data are provided as dictionaries with the following structure:

```python
{
    "file": str,  # path to the image file
    "annotation": Optional[dict]  # annotation of the file
}
```

The content of the `"annotation"` field depends on the task type.

To add the data to the dataset, we first define a generator function that yields the data.

```python
import json
from pathlib import Path

# path to the dataset, replace it with the actual path on your system
dataset_root = Path("data/parking_lot")

def generator():
    for annotation_dir in dataset_root.iterdir():
        with open(annotation_dir / "annotations.json") as f:
            data = json.load(f)

        # get the width and height of the image
        W = data["dimensions"]["width"]
        H = data["dimensions"]["height"]

        image_path = annotation_dir / data["filename"]

        for instance_id, bbox in data["BoundingBoxAnnotation"].items():

            # get unnormalized bounding box coordinates
            x, y = bbox["origin"]
            w, h = bbox["dimension"]

            # get the class name of the bounding box
            class_ = bbox["labelName"]
            yield {
                "file": image_path,
                "annotation": {
                    "type": "boundingbox",
                    "class": class_,

                    # normalized bounding box
                    "x": x / W,
                    "y": y / H,
                    "w": w / W,
                    "h": h / H,
                },
            }
```

The generator is then passed to the `add` method of the dataset.

```python
dataset.add(generator())
```

> \[!NOTE\]
> The `add` method accepts any iterable, not only generators.

### Defining Splits

The last step is to define the splits of the dataset.
Usually, the dataset is split into training, validation, and test sets.

To define splits, we use the `make_splits` method of the dataset.

```python

dataset.make_splits({
  "train": 0.7,
  "val": 0.2,
  "test": 0.1,
})
```

This will split the dataset across `"train"`, `"val"`, and `"test"` sets with the specified ratios.

For more refined control over the splits, you can pass a dictionary with the split names as keys and lists of file names as values:

```python

dataset.make_splits({
  "train": ["file1.jpg", "file2.jpg", ...],
  "val": ["file3.jpg", "file4.jpg", ...],
  "test": ["file5.jpg", "file6.jpg", ...],
})
```

Calling `make_splits` with no arguments will default to an 80/10/10 split.

In order for splits to be created, there must be some new data in the dataset. If no new data were added, calling `make_splits` will raise an error.
If you wish to delete old splits and create new ones using all the data, pass `redefine_splits=True` to the method call.

> \[!NOTE\]
> There are no restrictions on the split names,
> however for most cases one should stick to `"train"`, `"val"`, and `"test"`.

### CLI Reference

The `luxonis_ml` CLI provides a set of commands for managing datasets.
These commands are accessible via the `luxonis_ml data` command.

The available commands are:

- `luxonis_ml data parse <data_directory>` - parses data in the specified directory and creates a new dataset
- `luxonis_ml data ls` - lists all datasets
- `luxonis_ml data info <dataset_name>` - prints information about the dataset
- `luxonis_ml data inspect <dataset_name>` - renders the data in the dataset on screen using `cv2`
- `luxonis_ml data delete <dataset_name>` - deletes the dataset

For more information, run `luxonis_ml data --help` or pass the `--help` flag to any of the above commands.

## LuxonisLoader

`LuxonisLoader` provides an easy way to load datasets in the Luxonis Data Format. It is designed to work with the `LuxonisDataset` class and provides an abstraction over the dataset, allowing you to iterate over the stored data.

This guide covers the loading of datasets using the `LuxonisLoader` class.

The `LuxonisLoader` class can also take care of data augmentation, for more info see [Augmentation](#augmentation).

## Dataset Loading

To load a dataset with `LuxonisLoader`, we need an instance of `LuxonisDataset`, and we need to specify what view of the dataset we want to load.

The view can be either a single split (created by `LuxonisDataset.make_splits`) or a list of splits (for more complicated datasets).

```python
from luxonisml.data import LuxonisLoader, LuxonisDataset

dataset = LuxonisDataset("parking_lot")
loader = LuxonisLoader(dataset, view="train")
```

The `LuxonisLoader` can be iterated over to get pairs of images and annotations.

```python
for img, labels in loader:
    ...
```

## LuxonisParser

`LuxonisParser` offers a simple API for creating datasets from several common dataset formats. All of these formats are supported by `roboflow`.

The supported formats are:

- COCO - We support COCO JSON format in two variants:
  - [RoboFlow](https://roboflow.com/formats/coco-json)
  - [FiftyOne](https://docs.voxel51.com/user_guide/export_datasets.html#cocodetectiondataset-export)
- [Pascal VOC XML](https://roboflow.com/formats/pascal-voc-xml)
- [YOLO Darknet TXT](https://roboflow.com/formats/yolo-darknet-txt)
- [YOLOv4 PyTorch TXT](https://roboflow.com/formats/yolov4-pytorch-txt)
- [MT YOLOv6](https://roboflow.com/formats/mt-yolov6)
- [CreateML JSON](https://roboflow.com/formats/createml-json)
- [TensorFlow Object Detection CSV](https://roboflow.com/formats/tensorflow-object-detection-csv)
- Classification Directory - A directory with subdirectories for each class

```plaintext
dataset_dir/
├── train/
│   ├── class1/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   ├── class2/
│   └── ...
├── valid/
└── test/
```

- Segmentation Mask Directory - A directory with images and corresponding masks.

```plaintext
dataset_dir/
├── train/
│   ├── img1.jpg
│   ├── img1_mask.png
│   ├── ...
│   └── _classes.csv
├── valid/
└── test/
```

The masks are stored as grayscale PNG images where each pixel value corresponds to a class.
The mapping from pixel values to class is defined in the `_classes.csv` file.

```csv
Pixel Value, Class
0, background
1, class1
2, class2
3, class3
```

### Dataset Creation

The first step is to initialize the `LuxonisParser` object with the path to the dataset. Optionally, you can specify the name of the dataset and the type of the dataset.
If no name is specified, the dataset will be created with the name of the directory containing the dataset.
If no type is specified, the parser will try to infer the type from the dataset directory structure.

The dataset directory can either be a local directory or a directory in one of the supported cloud storage providers. The parser will automatically download the dataset if the provided path is a cloud storage URL.

The directory can also be a zip file containing the dataset.

```python
from luxonisml.data import LuxonisParser
from luxonis_ml.enums import DatasetType

dataset_dir = "path/to/dataset"

parser = LuxonisParser(
  dataset_dir=dataset_dir,
  dataset_name="my_dataset",
  dataset_type=DatasetType.COCO
)
```

After initializing the parser, you can parse the dataset to create a `LuxonisDataset` instance. The `LuxonisDataset` instance will contain the data from the dataset with splits for training, validation, and testing based on the dataset directory structure.

```python
dataset = parser.parse()
```

### CLI Reference

The parser can be invoked using the `luxonis_ml data parse` command.

```bash
luxonis_ml data parse path/to/dataset --name my_dataset --type coco
```

For more detailed information, run `luxonis_ml data parse --help`.
