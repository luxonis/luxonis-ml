# LuxonisML Loader

The `LuxonisLoader` class provides efficient access to dataset samples with configurable preprocessing options.

## Table of Contents

- [LuxonisML Loader](#luxonisml-loader)
  - [Parameters](#parameters)

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
| `exclude_empty_annotations`   | `bool`                                    | `False`            | Whether to exclude empty annotations             Optional                                                                                                                            |
| `color_space`                 | `Literal["RGB", "BGR"]`                   | `"RGB"`            | Color space of output images                                                                                                                                                         |
| `keep_categorical_as_strings` | `bool`                                    | `False`            | Whether to keep categorical metadata as strings                                                                                                                                      |
| `update_mode`                 | `UpdateMode`                              | `UpdateMode.ALL`   | Applicable to remote datasets. The loader internally calls the [`pull_from_cloud`](../datasets/README.md#pulling-from-remote-storage) method to download the dataset from the cloud. |
| `filter_task_names`           | `Optional[List[str]]`                     | `None`             | If provided, only include annotations for these specified tasks, ignoring any others in the data.                                                                                    |
