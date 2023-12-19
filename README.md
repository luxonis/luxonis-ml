# LuxonisML

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![CI](https://github.com/luxonis/luxonis-ml/actions/workflows/ci.yaml/badge.svg)
![Coverage](media/coverage_badge.svg)

A collection of helper function and utilities.

## Luxonis Dataset Format

The Luxonis Dataset Format (LDF) is intended to standardize the way the ML team stores data. We can use LDF with the `luxonis_ml.ops` package. LDF should provide the foundation for tasks such as training, evaluation, querying, and visualization. The library is largely powered by the open source software from [Voxel51](https://voxel51.com/).

The work on this project is in an MVP state, so it may be missing some critical features or have some issues - please report any feedback!

### Installation and Setup

Currently, the `luxonis_ml` package is in a private repo. To install,

```bash
git clone https://github.com/luxonis/luxonis-ml && cd luxonis-ml
pip install -e .
```

Finally, we will want to install `s3fs-fuse` ([repo](https://github.com/s3fs-fuse/s3fs-fuse)) to mount S3 buckets to our local filesystem. Check the link for installation on other operating systems, but for Debian this is

```bash
sudo apt install s3fs
```

#### luxonis_ml config

After installing the `luxonis_ml` package with `pip`, you should be able to run

```bash
luxonis_ml config
```

It will ask you to enter the following variables:

```text
AWS Bucket: xxxxxxxxxxxxxxxxx
AWS Access Key: xxxxxxxxxxxxxxxxx
AWS Secret Access Key: xxxxxxxxxxxxxxxxx
AWS Endpoint URL: xxxxxxxxxxxxxxxxx
MONGO_URI: xxxxxxxxxxxxxxxxx
```

### LDF Structure

When initializing a `LuxonisDataset` in Python, we want to specify a team name and a dataset name. Additionally, we should use `with` statements to take advantage of the Python context manager. Let's initialize an LDF in Python.

```python
from luxonis_ml.ops import *
team_name = 'luxonis'
dataset_name = 'bdd'

with LuxonisDataset(team_name, dataset_name) as dataset:
    print(dataset.name)
```

#### Source

The source abstracts the dataset source to specify the type of training media and optionally groups of training media.

```python
with LuxonisDataset(team_name, dataset_name) as dataset:

    # this uses a default configuration which can be used for
    # any OAK-D-like hardware
    # In this example, we literally have OAK-D
    dataset.create_source(oak_d_default=True)
```

Here's a source for COCO

```python
with LuxonisDataset(team_name, dataset_name) as dataset:

    # custom source for our COCO 2017
    custom_components = [
        # the first component is main component, which is important for hashing
        LDFComponent(name="image", htype=HType.IMAGE, itype=IType.BGR)
    ]
    dataset.create_source(name="coco", custom_components=custom_components)
```

#### Components

In the code above, you can see that since COCO is not from a default configuration of Luxonis hardware, we want to create custom components for it to define the structure of the source. In the COCO example, it's simple. We don't need to define any components for annotations, we just want to define them for any images we might have. Obviously, we don't have multi-cam setups in the COCO dataset, so we just define one component called `image` that represents our input BGR image.

The default components for OAK-D looks like this:

```python
components = [
                LDFComponent('left', htype=HType.IMAGE, itype=IType.MONO),
                LDFComponent('right', htype=HType.IMAGE, itype=IType.MONO),
                LDFComponent('color', htype=HType.IMAGE, itype=IType.BGR),
                LDFComponent('disparity', htype=HType.IMAGE, itype=IType.DISPARITY),
                LDFComponent('depth', htype=HType.IMAGE, itype=IType.DEPTH)
            ]
```

You could see that this configuration could also be used for any OAK-D-like product. When adding data, you do not need to provide data for every single component.

Some more definitions:

- `HType`: The type of the component. Right now, there's really only `IMAGE`. But eventually, we will have video and point cloud.
- `IType`: For an image component, the type of the image. `BGR` is for any 3-channel image, `MONO` is for single-channel images, and `DEPTH` and `DISPARITY` can be uint16 single-channel images for subpixel disparity. The only difference between `DISPARITY` and `DEPTH` is a semantic difference.

#### Adding Data

Currently, data must be added with some manually written logic using the `dataset.add()` method. This expects as input a list of dictionaries, `additions`, in a specific format.

```text
additions [list]:
    addition [dict]:
        # here, keys and types of values are specified
        filepath [str]: local path to an image to be added
        class [str]: for Image Classification, the name of the class for the entire image
        boxes [list]: for Object Detection, a list of bounding boxes
            box [list]: a list of length 5 encoding the following information - [class, x, y, width, height], where x, y, width, and height are all normalized to the image dimensions
        segmentation [np.ndarray]: for Semantic Segmention, a HxW array where the pixels correspond to integer classes. The class names for these classes are defined by `dataset.set_mask_targets()`
        keypoints [list]: for Keypoint Detection, a list of keypoint instances
            points [list]: tuples of length 2 consisting of the class and a list of (x,y) points
```

Additionally, currently we need to define our dataset classes manually.

```python
with LuxonisDataset(team_name, dataset_name) as dataset:
    dataset.set_classes(
        ['apple', 'orange', 'tomato'] # this will automaticall set classes for all tasks
    )
    # if using segmentation
    dataset.set_mask_targets(
        {1: 'orange', 2: 'apple'} # this can be a subset of all classes and 0 is assumed to be background
    )
```

Afterwards, you should be able to see all of the classes and the classes by CV task (object detection, segmentation, etc.) with the following

```python
with LuxonisDataset(team_name, dataset_name) as dataset:

    classes, classes_by_task = dataset.get_classes()
    print(classes)
    print(classes_by_task)
```

### Dataset Versioning

Every time the `dataset.add` method is called, dataset versioning is handled in the background and stored as fiftyone dataset views. The library will detect if there are any "add" or "update" cases.

### Data Loaders

Finally, now we can take our `LuxonisDataset` and wrap it in a `LuxonisLoader`. The `LuxonisLoader` is the default pytorch dataset used in the training library. Generally, it loads a single image (by default, the main component of a source) along with a dictionary of annotations.

```python
import torch
with LuxonisDataset(team_name, dataset_name) as dataset:

    loader = LuxonisLoader(
        dataset,
        split='train'
    )

    pytorch_loader = torch.utils.data.DataLoader(
        loader,
        batch_size=2,
        collate_fn=loader.collate_fn
    )

    for imgs, dict in pytorch_loader:
        print(type(imgs), type(dict))
        break
```

The first element in the loaded tuple is the image(s) and the second is a dictionary where the keys may provide annotations in different formats.

- `imgs` is (batch size x number of channels x height x width)
- `dict['class']` is (batch size x number of classes)
- `dict['bbox']` is (number of boxes in batch x 6 \[image ID, class, x, y, width, height\])
- `dict['segmentation']` is (batch size x number of classes x height x width)
- `dict['keypoints']` is (batch size x (number of points\*2 + 1))
