# LuxonisML Parsers

The `LuxonisParser` class provides functionality for converting various dataset formats to the Luxonis Dataset Format (LDF).

## Table of Contents

- [LuxonisML Parsers](#luxonisml-parsers)
  - [Supported Formats](#supported-formats)
  - [Parameters](#parameters)
  - [Parse Method Parameters](#parse-method-parameters)

## Supported Formats

### FiftyOneClassification

The FiftyOne Classification format is used by the [FiftyOne](https://voxel51.com/fiftyone/) library for image classification datasets.

**Format characteristics:**

- **Flat structure only** - This format does not support pre-defined train/valid/test splits
- When parsing, all data is treated as a single dataset and LDF's random splitting creates train/val/test splits internally
- When exporting back to this format, the flat structure is preserved (LDF splits are not reflected in the output)
- **Round-trip consistent** - Parse → Export produces the same flat structure

**Expected directory structure:**

```
dataset_dir/
├── data/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── labels.json
└── info.json (optional)
```

**labels.json structure:**

```json
{
    "classes": ["class1", "class2", ...],
    "labels": {
        "img1": 0,
        "img2": 1,
        ...
    }
}
```

Where each key in `labels` is the image filename (without extension) and the value is the index into the `classes` list.

**Usage:**

```python
from luxonis_ml.data.parsers import LuxonisParser
from luxonis_ml.enums import DatasetType

# Auto-detection
parser = LuxonisParser("path/to/fiftyone_dataset")
dataset = parser.parse()

# Explicit type
parser = LuxonisParser(
    "path/to/fiftyone_dataset",
    dataset_type=DatasetType.FIFTYONECLASSIFICATION
)
dataset = parser.parse()

# Export back to FiftyOneClassification format
dataset.export("output_path", DatasetType.FIFTYONECLASSIFICATION)
```

**Note on splits:** The FiftyOne Classification format is inherently flat. When you parse a dataset, LDF creates internal train/val/test splits (default 80/10/10) for training purposes. However, when you export back to FiftyOneClassification format, all images are exported to a single flat structure, preserving format consistency.

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
