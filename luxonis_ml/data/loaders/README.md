# LuxonisML Loader

The `LuxonisLoader` class provides efficient access to dataset samples with configurable preprocessing options.

## Table of Contents

- [LuxonisML Loader](#luxonisml-loader)
  - [Parameters](#parameters)

## Parameters

### LuxonisLoader Constructor Parameters

| Parameter                     | Type                                      | Default             | Description                                                    |
| ----------------------------- | ----------------------------------------- | ------------------- | -------------------------------------------------------------- |
| `dataset`                     | `LuxonisDataset`                          | Required            | The dataset to load data from                                  |
| `view`                        | `Union[str, List[str]]`                   | `"train"`           | Dataset split to use ("train", "val", "test")                  |
| `augmentation_engine`         | `str`                                     | `"albumentations"`  | Augmentation engine to use                                     |
| `augmentation_config`         | `Optional[Union[List[Params], PathType]]` | `None`              | Configuration for the augmentations                            |
| `height`                      | `Optional[int]`                           | `None`              | Height of the output images                                    |
| `width`                       | `Optional[int]`                           | `None`              | Width of the output images                                     |
| `keep_aspect_ratio`           | `bool`                                    | `True`              | Whether to keep image aspect ratio                             |
| `exclude_empty_annotations`   | `bool`                                    | `False`             | Whether to exclude empty annotations                           |
| `color_space`                 | `Literal["RGB", "BGR"]`                   | `"RGB"`             | Color space of output images                                   |
| `seed`                        | `Optional[int]`                           | `None`              | The random seed to use for the augmentations                   |
| `keep_categorical_as_strings` | `bool`                                    | `False`             | Whether to keep categorical metadata as strings                |
| `update_mode`                 | `UpdateMode`                              | `UpdateMode.ALWAYS` | Whether to always download dataset from cloud or only if empty |
