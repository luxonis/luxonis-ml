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

| Parameter      | Type                              | Default | Description                                                                                              |
| -------------- | --------------------------------- | ------- | -------------------------------------------------------------------------------------------------------- |
| `split`        | `Optional[str]`                   | `None`  | Split name if parsing a single split                                                                     |
| `random_split` | `bool`                            | `True`  | Whether to create random splits                                                                          |
| `split_ratios` | `Optional[Dict[str, float\|int]]` | `None`  | Split specification (see below). If `None`, the default behavior of `LuxonisDataset.make_splits` is used |

### Split Ratio Modes

The `--split-ratio` CLI argument (or `split_ratios` parameter) supports two modes:

#### 1. Percentage Mode (Redistributes Samples)

Use decimal values that sum to 1.0 (e.g., `0.8,0.1,0.1`).

```bash
luxonis_ml data parse /path/to/dataset --split-ratio 0.8,0.1,0.1
```

**Important:** This mode will **redistribute and shuffle** all samples across splits. The original split boundaries from the source dataset will **not** be preserved.

#### 2. Raw Counts Mode (Preserves Original Splits)

Use integer values (e.g., `1000,100,50`).

```bash
luxonis_ml data parse /path/to/dataset --split-ratio 1000,100,50
```

This mode:

- Draws samples **only from the respective original split** (no cross-split borrowing)
- If the requested count exceeds available samples in that split, all available samples are used and a warning is shown
- Useful when you want to create a smaller subset while preserving the original train/val/test distribution

**Example:** If the original dataset has 10,000 train images and 2,000 val images, using `--split-ratio 500,100,0` will:

- Take 500 random samples from the original train split
- Take 100 random samples from the original val split
- Take 0 samples from the test split
