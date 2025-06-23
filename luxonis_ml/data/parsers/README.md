# LuxonisML Parsers

The `LuxonisParser` class provides functionality for converting various dataset formats to the Luxonis Dataset Format (LDF).

## Table of Contents

- [LuxonisML Parsers](#luxonisml-parsers)
  - [Parameters](#parameters)
  - [Parse Method Parameters](#parse-method-parameters)

## Parameters

### LuxonisParser Constructor Parameters

| Parameter        | Type                                   | Default  | Description                                                                                                                     |
| ---------------- | -------------------------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `dataset_dir`    | `str`                                  | Required | Path or URL to dataset directory (local path, `gcs://`, `s3://` or `roboflow://`)                                               |
| `dataset_name`   | `Optional[str]`                        | `None`   | Name for the dataset (if None, derived from directory name)                                                                     |
| `save_dir`       | `Optional[Union[Path, str]]`           | `None`   | Where to save downloaded datasets if remote URL is provided (if None, uses current directory)                                   |
| `dataset_plugin` | `Optional[str]`                        | `None`   | Dataset plugin to use (if None, uses `LuxonisDataset`)                                                                          |
| `dataset_type`   | `Optional[DatasetType]`                | `None`   | Force specific dataset format type instead of auto-detection                                                                    |
| `task_name`      | `Optional[Union[str, Dict[str, str]]]` | `None`   | Task name(s) for the dataset. Used to link the classes to the desired tasks, with class names as keys and task names as values. |

### Parse Method Parameters

| Parameter      | Type                         | Default | Description                                                                                                                                 |
| -------------- | ---------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `split`        | `Optional[str]`              | `None`  | Split name if parsing a single split                                                                                                        |
| `random_split` | `bool`                       | `True`  | Whether to create random splits                                                                                                             |
| `split_ratios` | `Optional[Dict[str, float]]` | `None`  | Ratios for train/validation/test splits. If set to `None`, the default behavior of the `LuxonisDataset`'s `make_splits` method will be used |
