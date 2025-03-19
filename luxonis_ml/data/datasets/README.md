# LuxonisML Dataset

The `LuxonisDataset` class provides functionality for creating, managing, and interacting with datasets.

## Table of Contents

- [LuxonisML Dataset](#luxonisml-dataset)
  - [Parameters](#parameters)
  - [Core Methods](#core-methods)
    - [Adding Data](#adding-data)
    - [Creating Splits](#creating-splits)
    - [Merging Datasets](#merging-datasets)
    - [Cloning the Dataset](#cloning-the-dataset)

## Parameters

### LuxonisDataset Constructor Parameters

| Parameter         | Type            | Default               | Description                                           |
| ----------------- | --------------- | --------------------- | ----------------------------------------------------- |
| `dataset_name`    | `str`           | Required              | The unique name for the dataset                       |
| `team_id`         | `Optional[str]` | `None`                | Optional team identifier for the cloud                |
| `bucket_type`     | `BucketType`    | `BucketType.INTERNAL` | Whether to use external cloud buckets                 |
| `bucket_storage`  | `BucketStorage` | `BucketStorage.LOCAL` | Underlying storage (local, GCS, S3, Azure)            |
| `delete_existing` | `bool`          | `False`               | Whether to delete existing dataset with the same name |
| `delete_remote`   | `bool`          | `False`               | Whether to delete remote data when deleting dataset   |

## Core Methods

### Adding Data

The `add()` method is used to add data to a dataset.

#### Parameters

| Parameter    | Type              | Default     | Description                                           |
| ------------ | ----------------- | ----------- | ----------------------------------------------------- |
| `generator`  | `DatasetIterator` | Required    | Generator yielding dataset records                    |
| `batch_size` | `int`             | `1_000_000` | Number of annotation records to process in each batch |

### Creating Splits

The `make_splits()` method divides the dataset into separate splits (train/val/test) for machine learning workflows.

#### Parameters

| Parameter            | Type                                                                                          | Default | Description                                   |
| -------------------- | --------------------------------------------------------------------------------------------- | ------- | --------------------------------------------- |
| `splits`             | `Mapping[str, float]` or<br>`Tuple[float, float, float]` or<br>`Mapping[str, List[PathType]]` | `None`  | Proportions or explicit file paths for splits |
| `replace_old_splits` | `bool`                                                                                        | `False` | Whether to replace existing splits            |

### Merging Datasets

The `merge_with()` method combines data from another dataset into the current one.

#### Parameters

| Parameter          | Type             | Default  | Description                                               |
| ------------------ | ---------------- | -------- | --------------------------------------------------------- |
| `other`            | `LuxonisDataset` | Required | Dataset to merge with                                     |
| `inplace`          | `bool`           | `True`   | Whether to modify the current dataset or create a new one |
| `new_dataset_name` | `str`            | `None`   | Name for the new dataset if `inplace=False`               |

### Cloning the Dataset

The `clone()` method creates a complete copy of a dataset with a new name. It copies all data, metadata, and splits from the original dataset.

#### Parameters

| Parameter          | Type   | Default  | Description                                                                                |
| ------------------ | ------ | -------- | ------------------------------------------------------------------------------------------ |
| `new_dataset_name` | `str`  | Required | Name for the cloned dataset                                                                |
| `push_to_cloud`    | `bool` | `True`   | Whether to push the cloned dataset to cloud storage. Only if the current dataset is remote |
