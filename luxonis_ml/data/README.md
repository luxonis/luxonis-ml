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
  - [Table of Contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
  - [LuxonisDataset](#luxonisdataset)
    - [Adding Data](#adding-data)
    - [Defining Splits](#defining-splits)
    - [LuxonisML CLI](#luxonisml-cli)
  - [LuxonisLoader](#luxonisloader)
    - [Dataset Loading](#dataset-loading)
  - [LuxonisParser](#luxonisparser)
    - [Dataset Creation](#dataset-creation)
    - [LuxonisParser CLI](#luxonisparser-cli)
  - [Annotation Format](#annotation-format)
    - [Classification](#classification)
    - [Bounding Box](#bounding-box)
    - [Keypoints](#keypoints)
    - [Segmentation Mask](#segmentation-mask)
      - [Polyline](#polyline)
      - [Binary Mask](#binary-mask)
      - [Run-Length Encoding](#run-length-encoding)
    - [Array](#array)
  - [Augmentation](#augmentation)
    - [Augmentation Pipeline](#augmentation-pipeline)
    - [Example](#example)
    - [Usage with LuxonisLoader](#usage-with-luxonisloader)

## Prerequisites

We will be using our toy dataset `parking_lot` in all examples. The dataset consists of images of cars and motorcycles in a parking lot. Each image has a corresponding annotation in the form of a bounding box, keypoints and several segmentation masks.

**Dataset Annotations:**

| Task                        | Annotation Type   | Classes                                                                                                |
| --------------------------- | ----------------- | ------------------------------------------------------------------------------------------------------ |
| Object Detection            | Bounding Box      | car, motorcycle                                                                                        |
| Keypoint Detection          | Keypoints         | car, motorcycle                                                                                        |
| Color Segmentation          | Segmentation Mask | background, red, gree, blue                                                                            |
| Type Segmentation           | Segmentation Mask | background, car, motorcycle                                                                            |
| Brand Segmentation          | Segmentation Mask | background, alfa-romeo, buick, ducati, harley, ferrari, infiniti, jeep, land-rover, roll-royce, yamaha |
| Binary Vehicle Segmentation | Segmentation Mask | vehicle                                                                                                |

To start, download the zipped dataset from [here](https://drive.google.com/uc?export=download&id=1OAuLlL_4wRSzZ33BuxM6Uw2QYYgv_19N) and extract it to a directory of your choice.

## LuxonisDataset

The first step is to create a new dataset and add some data to it.

The main part of the dataset is the `LuxonisDataset` class.
It serves as an abstraction over the dataset and provides methods
for adding data and creating splits.
You can create as many datasets as you want, each with a unique name.

Datasets can be stored locally or in one of the supported cloud storage providers.

### Dataset Creation

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

After creating a dataset, the next step is to populate it with images and their annotations. This process involves preparing data entries and submitting them to the dataset through the `add` method.

#### Data Format

Each data entry should be a dictionary with the following structure:

```python
{
    "file": str,  # path to the image file
    "annotation": Optional[dict]  # annotation of the file
}
```

The content of the `"annotation"` field depends on the task type and follows the Annotation Format described later in this document.

#### Adding Data with a Generator Function

The recommended approach for adding data is to create a generator function that yields data entries one by one.

Here's an example that loads object detection annotations:

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
            class_name = bbox["labelName"]
            yield {
                "file": image_path,
                "annotation": {
                    "class": class_name,

                    "boundingbox": {
                      # normalized bounding box
                      "x": x / W,
                      "y": y / H,
                      "w": w / W,
                      "h": h / H,
                    }
                },
            }
```

The generator is then passed to the `add` method of the dataset.

#### Adding the Data to the Dataset

Once you've defined your data source, pass it to the dataset's add method:

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

### LuxonisML CLI

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

### Dataset Loading

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

- **COCO** - We support COCO JSON format in two variants:
  - [RoboFlow](https://roboflow.com/formats/coco-json)
  - [FiftyOne](https://docs.voxel51.com/user_guide/export_datasets.html#cocodetectiondataset-export)
- [**Pascal VOC XML**](https://roboflow.com/formats/pascal-voc-xml)
- [**YOLO Darknet TXT**](https://roboflow.com/formats/yolo-darknet-txt)
- [**YOLOv4 PyTorch TXT**](https://roboflow.com/formats/yolov4-pytorch-txt)
- [**MT YOLOv6**](https://roboflow.com/formats/mt-yolov6)
- [**CreateML JSON**](https://roboflow.com/formats/createml-json)
- [**TensorFlow Object Detection CSV**](https://roboflow.com/formats/tensorflow-object-detection-csv)
- **Classification Directory** - A directory with subdirectories for each class

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

- **Segmentation Mask Directory** - A directory with images and corresponding masks.

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

### LuxonisParser CLI

The parser can be invoked using the `luxonis_ml data parse` command.

```bash
luxonis_ml data parse path/to/dataset --name my_dataset --type coco
```

For more detailed information, run `luxonis_ml data parse --help`.

## Annotation Format

The Luxonis Data Format supports several task types, each with its own annotation structure.
Each annotation describes a single instance of the corresponding task type in the image.

### Classification

A single class label for the entire image.

```python
{
    # name of the class the image belongs to
    "class": str,
}
```

### Bounding Box

A single bounding box in the `"xywh"` format.
The coordinates and dimensions are relative to the image size.

```python
{
    # name of the class the bounding box belongs to
    "class": str,

    # unique identifier of the instance in the image
    "instance_id": Optional[int],

    # bounding box coordinates normalized to the image size
    "boundingbox": {
        "x": float, # x coordinate of the top-left corner
        "y": float, # y coordinate of the top-left corner
        "w": float, # width of the bounding box
        "h": float, # height of the bounding box
    },
}
```

### Keypoints

A list of keypoint coordinates in the form of `(x, y, visibility)`.
THe coordinates are relative to the image size.

The visibility can be:

- 0 for keypoints outside the image
- 1 for keypoints in the image but occluded
- 2 for fully visible keypoints

```python
{
    # name of the class the keypoints belong to
    "class": str,

    # unique identifier of the instance in the image
    "instance_id": Optional[int],

    # keypoints coordinates normalized to the image size
    "keypoints": {
        # List of keypoint coordinates as tuples of (x, y, visibility)
        # x, y are floating point coordinates relative to image size
        # visibility: 0=not visible, 1=occluded, 2=visible
        "keypoints": list[tuple[float, float, Literal[0, 1, 2]]],
    },
}
```

### Segmentation Mask

There are 3 options for how to specify the segmentation mask:

#### Polyline

The mask is described as a polyline. It is assumed that the last
point connects to the first one to form a polygon.
The coordinates are relative to the image size.

```python
{
    # name of the class this mask belongs to
    "class": str,

    "segmentation": {
        # height of the mask
        "height": int,
        
        # width of the mask
        "width": int,
        
        # list of (x, y) coordinates forming the polyline
        # coordinates are normalized with the image size
        # the polyline will be closed to form a polygon,
        #   i.e. the first and last point are the same
        "points": list[tuple[float, float]],
        },
}
```

#### Binary Mask

The mask is a binary 2D numpy array.

```python
{
    # name of the class this mask belongs to
    "class": str,

    # binary mask as a 2D numpy array
    # 0 for background, 1 for the object
    "segmentation": {
        "mask": np.ndarray,
        },
    }
```

#### Run-Length Encoding

The mask is described using the [Run-Length Encoding](https://en.wikipedia.org/wiki/Run-length_encoding) compression.

Run-length encoding compresses data by reducing the physical size
of a repeating string of characters.
This process involves converting the input data into a compressed format
by identifying and counting consecutive occurrences of each character.

The RLE is composed of the height and width of the mask image and the counts of the pixels belonging to the positive class.

```python
{
    # name of the class this mask belongs to
    "class": str,

    "segmentation":
    {
        # height of the mask
        "height": int,
        
        # width of the mask
        "width": int,

        # counts of the pixels belonging to the positive class
        "counts": list[int] | bytes,    
    },

}
```

> \[!NOTE\]
> The RLE format is not intended for regular use and is provided mainly to support datasets that may already be in this format.

> \[!NOTE\]
> Masks provided as numpy arrays are converted to RLE format internally.

### Array

An array of arbitrary data. This can be used for any custom data that doesn't fit into the other types.

```python
{
    # name of the class this array belongs to
    "class": str,

    "array":{
        # path to a `.npy` file containing the array data
        "path": str,
    },	
}
```

### Instance Segmentation

Instance segmentation combines bounding boxes and segmentation masks for each instance in the image.

```python
{
    # name of the class the bounding box belongs to
    "class": str,

    # unique identifier of the instance in the image
    "instance_id": Optional[int],

    # bounding box coordinates normalized to the image size
    "boundingbox": {
        "x": float, # x coordinate of the top-left corner
        "y": float, # y coordinate of the top-left corner
        "w": float, # width of the bounding box
        "h": float, # height of the bounding box
    },

    # Instance segmentation mask - Can be in any of the segmentation formats described above. We'll use the polyline format here.
    "instance_segmentation": {
        # height of the mask
        "height": int,
        
        # width of the mask
        "width": int,
        
        # list of (x, y) coordinates forming the polyline
        # coordinates are normalized with the image size
        # the polyline will be closed to form a polygon,
        #   i.e. the first and last point are the same
        "points": list[tuple[float, float]],
        },
}
```

### Metadata

Metadata is a flexible catch-all field for storing additional information about annotations that doesn't fit into the standard annotation types. This can be used for a variety of custom needs such as OCR text content, embedding IDs, or other application-specific data.

```python
{
    "metadata": {
        # Flexible key-value pairs for custom data
        key1: value1,
        key2: value2,
        # ...
    },
}
```

#### OCR Example

For Optical Character Recognition (OCR) tasks, metadata can store text content and properties:

```python
{
    "metadata": {
        "text": str,  # text content
        "color": Category("red"),  # text color
    },
}
```

Where the `Category` class should be imported from `luxonis_ml.data`.

For more information on using OCR annotations with training models, see the [OCR Heads documentation](https://github.com/luxonis/luxonis-train/tree/main/luxonis_train/nodes#ocr-heads).

#### Embeddings Example

For embedding-based tasks like re-identification or similarity search:

```python
{
    "metadata": {
        "id": int,  # unique identifier
        "color": Category("red"),  # text color
    },
}
```

For more information on using OCR annotations with training models, see the [Embedding Heads documentation.](https://github.com/luxonis/luxonis-train/tree/main/luxonis_train/nodes#embedding-heads).

## Augmentation

`LuxonisLoader` provides a way to use augmentations with your datasets. Augmentations are transformations applied to the data to increase the diversity of the dataset and improve the performance of the model.

### Augmentation Pipeline

The augmentations are specified as a list of dictionaries, where each dictionary represents a single augmentation. The structure of the dictionary is as follows:

```python
{
    "name": str,  # name of the augmentation
    "params": dict  # parameters of the augmentation
}
```

By default, we support most augmentations from the `albumentations` library. You can find the full list of augmentations and their parameters in the [Albumentations](https://albumentations.ai/docs/api_reference/augmentations/)documentation.

On top of that, we provide a handful of custom batch augmentations:

- `Mosaic4` - Mosaic augmentation with 4 images. Combines crops of 4 images into a single image in a mosaic pattern.
- `MixUp` - MixUp augmentation. Overlays two images with a random weight.

### Example

The following example demonstrates a simple augmentation pipeline:

```python

[
  {
    'name': 'HueSaturationValue',
    'params': {
      'p': 0.5,
      'hue_shift_limit': 3,
      'sat_shift_limit': 70,
      'val_shift_limit': 40,
    }
  },
  {
    'name': 'Rotate',
    'params': {
      'p': 0.6,
      'limit': 30,
      'border_mode': 0,
      'value': [0, 0, 0]
    }
  },
  {
    'name': 'Perspective',
    'params': {
      'p': 0.5,
      'scale': [0.04, 0.08],
      'keep_size': True,
      'pad_mode': 0,
      'pad_val': 0,
      'mask_pad_val': 0,
      'fit_output': False,
      'interpolation': 1,
    }
  },
  {
    'name': 'Affine',
    'params': {
      'p': 0.4,
      'scale': None,
      'translate_percent': None,
      'translate_px': None,
      'rotate': None,
      'shear': 10,
      'interpolation': 1,
      'mask_interpolation': 0,
      'cval': 0,
      'cval_mask': 0,
      'mode': 0,
      'fit_output': False,
      'keep_ratio': False,
      'rotate_method': 'largest_box',
    }
  },
]

```

> \[!NOTE\]
> The augmentations are **not** applied in order. Instead, an optimal order is determined based on the type of the augmentations to minimize the computational cost.

### Usage with LuxonisLoader

For this example, we will assume the augmentation example above is stored in a YAML file named `augmentations.yaml`.

```python

from luxonis_ml.data import LuxonisDataset, LuxonisLoader, AlbumentationsEngine
import json

dataset = LuxonisDataset("parking_lot")
loader = LuxonisLoader(
    dataset,
    view="train", 
    augmentation_config="augmentations.yaml", 
    augmentation_engine="albumentations",
    height=256,
    width=320,
    keep_aspect_ratio=True,
    color_space="RGB",
)
for img, labels in loader:
    ...
```
