# LuxonisML - Data

Check out [this notebook](../../examples/COCO_people_subset.ipynb) for a complete example of utilizing `LuxonisDataset` and `LuxonisLoader`.

## Table of Contents

- [LuxonisDataset](#luxonisdataset)
  - [Adding Data](#adding-data)
- [LuxonisLoader](#luxonisloader)

## LuxonisDataset

The `LuxonisDataset` is intended to standardize the way data is stored and managed in the Luxonis MLOps ecosystem. `LuxonisDataset` should provide the foundation for tasks such as training, evaluation, querying, and visualization.

To initialize a `LuxonisDataset` in Python, we want to specigy a dataset name.

```python
from luxonis_ml.data import LuxonisDataset
dataset_name = 'my_dataset'

dataset = LuxonisDataset(dataset_name)
print(dataset.name)
```

### Adding Data

`LuxonisDataset` will expect a generator that yields data in the following format:

```text
- file [str] : path to file on local disk or object storage
- class [str]: string specifying the class name or label name
- type [str] : the type of label or annotation
- value [Union[str, list, int, float, bool]]: the actual annotation value
    For here are the expected structures for `value`.
    The function will check to ensure `value` matches this for each annotation type

    value (classification) [bool] : Marks whether the class is present or not
        (e.g. True/False)
    value (box) [List[float]] : the normalized (0-1) x, y, w, and h of a bounding box
        (e.g. [0.5, 0.4, 0.1, 0.2])
    value (polyline) [List[List[float]]] : an ordered list of [x, y] polyline points
        (e.g. [[0.2, 0.3], [0.4, 0.5], ...])
    value (segmentation) [Tuple[int, int, List[int]]]: an RLE representation of (height, width, counts) based on the COCO convention
    value (keypoints) [List[List[float]]] : an ordered list of [x, y, visibility] keypoints for a keypoint skeleton instance
        (e.g. [[0.2, 0.3, 2], [0.4, 0.5, 2], ...])
    value (array) [str]: path to a numpy .npy file
```

Additionally, currently we need to define our dataset classes manually.

```python
dataset = LuxonisDataset(team_name, dataset_name)
dataset.set_classes(
    ['apple', 'orange', 'tomato'] # this will automatically set classes for all tasks
)
# if using segmentation
dataset.set_classes(
    ['orange', 'apple'],
    # this can be a subset of all classes and 0 is assumed to be background
    task="segmentation",
)
```

Afterwards, you should be able to see all of the classes and the classes by CV task (object detection, segmentation, etc.) with the following

```python
dataset = LuxonisDataset(team_name, dataset_name)

classes, classes_by_task = dataset.get_classes()
print(classes)
print(classes_by_task)
```

## LuxonisLoader

Finally, now we can take our `LuxonisDataset` and wrap it in a `LuxonisLoader`.
The `LuxonisLoader` is the default pytorch dataset used in the training library.
Generally, it loads a single image (by default, the main component of a source) along with a dictionary of annotations.

```python
from luxonis_ml.data import LuxonisDataset, LuxonisLoader
dataset = LuxonisDataset(team_name, dataset_name)

loader = LuxonisLoader(
    dataset,
    split='train'
)

for imgs, labels in loader:
    print(type(imgs), type(labels))
    break
```

The first element in the loaded tuple is the image(s) and the second is a dictionary where the keys may provide annotations in different formats.

- `imgs` is (batch size x number of channels x height x width)
- `dict['class']` is (batch size x number of classes)
- `dict['bbox']` is (number of boxes in batch x 6 \[image ID, class, x, y, width, height\])
- `dict['segmentation']` is (batch size x number of classes x height x width)
- `dict['keypoints']` is (batch size x (number of points\*2 + 1))
