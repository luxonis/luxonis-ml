# LuxonisML Loader

The `LuxonisLoader` class provides efficient access to dataset samples with configurable preprocessing options.

## Table of Contents

- [LuxonisML Loader](#luxonisml-loader)
  - [Parameters](#parameters)
  - [Output Structure](#output-structure)

## Parameters

### LuxonisLoader Constructor Parameters

| Parameter                     | Type                                      | Default            | Description                                                                                                                                                                          |
| ----------------------------- | ----------------------------------------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `dataset`                     | `LuxonisDataset`                          | Required           | The dataset to load data from                                                                                                                                                        |
| `view`                        | `Union[str, List[str]]`                   | `"train"`          | Dataset split to use ("train", "val", "test")                                                                                                                                        |
| `augmentation_engine`         | `str`                                     | `"albumentations"` | [Augmentation engine](../augmentations/README.md) to use.                                                                                                                            |
| `augmentation_config`         | `Optional[Union[List[Params], PathType]]` | `None`             | Configuration for the augmentations                                                                                                                                                  |
| `height`                      | `Optional[int]`                           | `None`             | Height of the output images                                                                                                                                                          |
| `width`                       | `Optional[int]`                           | `None`             | Width of the output images                                                                                                                                                           |
| `keep_aspect_ratio`           | `bool`                                    | `True`             | Whether to keep image aspect ratio                                                                                                                                                   |
| `exclude_empty_annotations`   | `bool`                                    | `False`            | Whether to exclude empty annotations                                                                                                                                                 |
| `color_space`                 | `Literal["RGB", "BGR"]`                   | `"RGB"`            | Color space of output images                                                                                                                                                         |
| `seed`                        | `Optional[int]`                           | `None`             | The random seed to use for the augmentations.                                                                                                                                        |
| `keep_categorical_as_strings` | `bool`                                    | `False`            | Whether to keep categorical metadata as strings                                                                                                                                      |
| `update_mode`                 | `UpdateMode`                              | `UpdateMode.ALL`   | Applicable to remote datasets. The loader internally calls the [`pull_from_cloud`](../datasets/README.md#pulling-from-remote-storage) method to download the dataset from the cloud. |
| `filter_task_names`           | `Optional[List[str]]`                     | `None`             | If provided, only include annotations for these specified tasks, ignoring any others in the data.                                                                                    |
| `class_order_per_task`        | `Optional[Dict[str, List[str]]]`          | `None`             | If provided, the classes for the specified tasks will be reordered permanently in the dataset.                                                                                       |

## Output Structure

`LuxonisLoader` implements `__getitem__`, so each `dataset[i]` returns a **tuple** of `(inputs, labels)`.

```python
(
    inputs: np.ndarray | Dict[str, np.ndarray],  # one or more image-like np arrays, where keys are source names
    labels: Labels                               # task-specific labels
)
```

### Accessing Labels

Labels can be accessed using specific keys in the `labels` dictionary. These keys follow the format:

```python
f"{task_name}/{task_type}"
```

If the dataset was created without specifying a `task_name`, the default keys will be:

```
/boundingbox, /classification, /segmentation, /instance_segmentation, /keypoints, /metadata/<key>
```

In the case of metadata, the `/metadata/<key>` format uses a field name that was provided when creating the dataset in place of `<key>` (e.g., `/metadata/id`, `/metadata/camera_angle`, etc.).

If a dataset was created using multiple `task_name`s—which is especially helpful for more structured or complex datasets—you might define one task for **segmentation** and another for **keypoint detection**. In that case, you would access the labels using keys like:

```
segmentation_task/segmentation, pose_task/keypoints
```

This naming convention makes it easier to manage, access, and process annotations in multi-task datasets.

### Labels Structure

| Task Type                 | Example Shape                    | Description                                                                                                              |
| ------------------------- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **classification**        | `(N_classes,)`                   | One-hot encoded classification vector.                                                                                   |
| **segmentation**          | `(C, H, W)`                      | One-hot encoded per-pixel class mask in CHW format, where `C` is the number of classes.                                  |
| **boundingbox**           | `(N_instances, 5)`               | Each row represents an object as `[cls, x_min, y_min, w, h]`.                                                            |
| **instance_segmentation** | `(N_instances, H, W)`            | Binary mask for each instance.                                                                                           |
| **keypoints**             | `(N_instances, n_keypoints * 3)` | Keypoints in `x, y, v` format (visibility flag), repeated per keypoint.                                                  |
| **metadata**              | —                                | Follows the same structure as yielded. Values are accessible via keys such as `"metadata/key1"`, `"metadata/key2"`, etc. |
