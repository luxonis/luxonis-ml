# LuxonisML Parsers

The `LuxonisParser` class provides functionality for converting various dataset formats to the Luxonis Dataset Format (LDF).

## Table of Contents

- [LuxonisML Parsers](#luxonisml-parsers)
  - [Parameters](#parameters)
  - [Parse Method Parameters](#parse-method-parameters)
  - [Common dataset formats for evaluation](#common-datasets-for-evaluation)
    - [COCO-2017](#coco-2017-dataset-parsing)
      - [Supported Directory Formats](#supported-directory-formats)
      - [Parsed Annotation Types](#parsed-annotation-types)
      - [Keypoint Annotations (`use_keypoint_ann`)](#keypoint-annotations-use_keypoint_ann)
      - [Additional COCO-Specific Parameters](#additional-coco-specific-parameters)
      - [Examples](#examples)
    - [Imagenet-sample](#imagenet-sample-dataset-parsing)
      - [Supported Directory Formats](#supported-directory-formats-1)
      - [Labels Format](#labels-format)
      - [Parsed Annotation Types](#parsed-annotation-types-1)
      - [Automatic Annotation Cleaning](#automatic-annotation-cleaning)
      - [Examples](#examples-1)

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

## Common datasets for evaluation

### COCO-2017 Dataset Parsing

The COCO parser supports two directory layout variants: **FiftyOne** and **Roboflow**. It automatically detects which format is present.

#### Supported Directory Formats

##### FiftyOne Format

This is the default layout when downloading COCO via the [FiftyOne](https://docs.voxel51.com/) package.

With `fiftyone` installed locally as a pip package, individual COCO-2017 splits from the official dataset can be installed through the command:

`fiftyone zoo datasets load coco-2017 --split train/validation/test --kwargs max_samples=<max_sample_num>`

```
dataset_dir/
├── raw/                              # (optional) extra annotation files
│   ├── person_keypoints_train2017.json
│   └── person_keypoints_val2017.json
├── train/
│   ├── data/
│   │   ├── 000000000009.jpg
│   │   └── ...
│   └── labels.json
├── validation/
│   ├── data/
│   └── labels.json
└── test/
    ├── data/
    └── labels.json
```

##### Roboflow Format

```
dataset_dir/
├── train/
│   ├── img1.jpg
│   └── _annotations.coco.json
├── valid/
└── test/
```

#### Parsed Annotation Types

The COCO parser extracts the following annotation types from each instance:

| Annotation            | Source Field   | Notes                                                             |
| --------------------- | -------------- | ----------------------------------------------------------------- |
| Bounding box          | `bbox`         | Normalized to `[0, 1]` relative to image dimensions               |
| Segmentation          | `segmentation` | Polygon or RLE mask, stored as RLE                                |
| Instance segmentation | `segmentation` | Same mask, also stored under the instance segmentation task       |
| Keypoints             | `keypoints`    | Triplets `(x, y, visibility)`, normalized and clipped to `[0, 1]` |
| Classification        | `category_id`  | Category name mapped from the `categories` list                   |

Skeleton information (keypoint labels and edges) is automatically extracted from COCO categories that contain `keypoints` and `skeleton` fields.

#### Keypoint Annotations (`use_keypoint_ann`)

By default, keypoints are read from the same annotation file as bounding boxes and segmentations (e.g., `labels.json`). The standard COCO `labels.json` exported by FiftyOne uses the **instances** annotations, which contain bounding boxes and segmentations for all 80 COCO categories but **do not include keypoints**.

To parse person keypoints, set `use_keypoint_ann=True`. This tells the parser to read from the dedicated **person keypoints** annotation files (`person_keypoints_*.json`) instead of the default instance annotation files.

##### How It Works

When `use_keypoint_ann=True` (FiftyOne format only), the parser looks for keypoint-specific annotation files at these default paths:

| Split | Default Path                            |
| ----- | --------------------------------------- |
| train | `raw/person_keypoints_train2017.json`   |
| val   | `raw/person_keypoints_val2017.json`     |
| test  | `raw/person_keypoints_test2017.json` \* |

\* The test keypoints file is **not included** in the official COCO release. If missing, the validation split is automatically split 50/50 into val and test (see `split_val_to_test` below).

You can override these paths with the `keypoint_ann_paths` parameter:

```python
dataset = parser.parse(
    use_keypoint_ann=True,
    keypoint_ann_paths={
        "train": "path/to/custom_train_keypoints.json",
        "val": "path/to/custom_val_keypoints.json",
        "test": "path/to/custom_test_keypoints.json",
    },
)
```

> **Note:** `use_keypoint_ann` is only supported for the FiftyOne format. If a Roboflow-format dataset is detected, the parameter is ignored.

#### Additional COCO-Specific Parameters

| Parameter            | Type                       | Default | Description                                                                                                                                                                        |
| -------------------- | -------------------------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `use_keypoint_ann`   | `bool`                     | `False` | Use dedicated person keypoint annotation files instead of the default instance annotations. FiftyOne format only.                                                                  |
| `keypoint_ann_paths` | `Optional[Dict[str, str]]` | `None`  | Custom paths (relative to dataset dir) for keypoint annotation files, keyed by split name (`train`, `val`, `test`). Only used when `use_keypoint_ann=True`.                        |
| `split_val_to_test`  | `bool`                     | `True`  | When the test split has no annotations (common for COCO-2017), automatically split the validation set 50/50 into val and test. Set to `False` to keep the original val set intact. |

These parameters are passed as keyword arguments to `parser.parse()`.

#### Example: Parsing COCO-2017 with Keypoints

> **Note:** The `use_keypoint_ann` flag is not exposed via the CLI. Use the Python API for keypoint-specific parsing.

```python
from luxonis_ml.data import LuxonisParser

parser = LuxonisParser(
    "coco-2017",               # path to dataset directory
    dataset_name="coco-2017",
)

# Parse with keypoint annotations
dataset = parser.parse(
    use_keypoint_ann=True,
    split_ratios={"train": 0.5, "val": 0.4, "test": 0.1},
)
```

#### Example: CLI Parsing

```bash
# Basic COCO parsing (instances only, no keypoints)
luxonis_ml data parse ./coco-2017

# With a subset of samples
luxonis_ml data parse ./coco-2017 --split-ratio 1000,200,100
```

#### Corrupted Image Handling

The COCO-2017 train set contains a small number of images known to cause issues. The parser automatically filters these out when parsing the train split. No user action is required.

### Imagenet-sample Dataset Parsing

The ImageNet-sample parser handles the [FiftyOneImageClassificationDataset](https://docs.voxel51.com/user_guide/export_datasets.html#fiftyone-image-classification-dataset) format. It supports both a flat (single-directory) layout and a split-based layout.

#### Supported Directory Formats

##### Flat Format (ImageNet-sample default)

This is the layout produced by `fiftyone zoo datasets load imagenet-sample`. Since the dataset contains as a single split, random splits are applied at parse time.

With `fiftyone` installed locally as a pip package, the imagenet-sample dataset can be downloaded through the command:

`fiftyone zoo datasets load imagenet-sample`

```
dataset_dir/
├── data/
│   ├── 000007.jpg
│   └── ...
├── info.json
└── labels.json
```

##### Split Format

If the dataset is organized into split subdirectories, each split is parsed independently.

```
dataset_dir/
├── train/
│   ├── data/
│   │   ├── img1.jpg
│   │   └── ...
│   └── labels.json
├── validation/
│   ├── data/
│   └── labels.json
└── test/
    ├── data/
    └── labels.json
```

#### Labels Format

The `labels.json` file maps image stems to class indices:

```json
{
    "classes": ["tench", "goldfish", "great white shark", ...],
    "labels": {
        "000007": 0,
        "000083": 1,
        ...
    }
}
```

#### Parsed Annotation Types

| Annotation     | Source Field | Notes                                       |
| -------------- | ------------ | ------------------------------------------- |
| Classification | `labels`     | Class name resolved from the `classes` list |

#### Automatic Annotation Cleaning

The ImageNet FiftyOne export contains known issues that the parser automatically fixes when using the flat format:

| Issue               | Fix Applied                                                                                               |
| ------------------- | --------------------------------------------------------------------------------------------------------- |
| Duplicate "crane"   | First occurrence (bird, index 141) renamed to "crane bird" to distinguish from crane (machine, index 524) |
| Duplicate "maillot" | Second occurrence (index 646) renamed to "maillot swim suit"                                              |
| Misindexed label    | Image `006742` corrected from index 517 to 134                                                            |
| Misindexed label    | Image `031933` corrected from index 639 to 638                                                            |

A cleaned copy (`labels_fixed.json`) is saved alongside the original file.

#### Dataset Details

- **Samples:** 1,000 images
- **Classes:** 1,000 ImageNet categories
- **Task:** Image classification

#### Example: Parsing ImageNet-sample

```python
from luxonis_ml.data import LuxonisParser

parser = LuxonisParser(
    "imagenet-sample",               # path to dataset directory
    dataset_name="imagenet-sample",
)

# Parse with random splits
dataset = parser.parse(
    split_ratios={"train": 0.8, "val": 0.1, "test": 0.1},
)
```

#### Example: CLI Parsing

```bash
# Basic ImageNet-sample parsing
luxonis_ml data parse ./imagenet-sample

# With custom split ratios
luxonis_ml data parse ./imagenet-sample --split-ratio 0.8,0.1,0.1
```
