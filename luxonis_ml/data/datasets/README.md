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
    - [Pulling from remote storage](#pulling-from-remote-storage)
    - [Pushing existing local dataset to cloud](#pushing-existing-local-dataset-to-cloud)
- [In-Depth Explanation of luxonis-ml Dataset Storage](#in-depth-explanation-of-luxonis-ml-dataset-storage)
  - [File Structure](#file-structure)
    - [Local dataset storage](#local-dataset-storage)
    - [Remote dataset storage](#remote-dataset-storage)
  - [Parquet file content](#parquet-file-content)
    - [Creating a Dataset Locally](#creating-a-dataset-locally)
    - [Creating a Dataset Remotely](#creating-a-dataset-remotely)

## Parameters

### LuxonisDataset Constructor Parameters

| Parameter        | Type            | Default               | Description                                                                                                                                                             |
| ---------------- | --------------- | --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dataset_name`   | `str`           | Required              | The unique name for the dataset                                                                                                                                         |
| `team_id`        | `Optional[str]` | `None`                | Optional team identifier for the cloud                                                                                                                                  |
| `bucket_type`    | `BucketType`    | `BucketType.INTERNAL` | Whether to use external cloud buckets                                                                                                                                   |
| `bucket_storage` | `BucketStorage` | `BucketStorage.LOCAL` | Underlying storage (local, GCS, S3, Azure)                                                                                                                              |
| `delete_local`   | `bool`          | `False`               | Whether to delete dataset from local storage.                                                                                                                           |
| `delete_remote`  | `bool`          | `False`               | Whether to delete the dataset from remote storage. This only applies to remote datasets. When recreating a cloud dataset, it is recommended to also use `delete_local`. |

## Core Methods

### Adding Data

The `add()` method is used to add data to a dataset.

#### Parameters

| Parameter    | Type              | Default     | Description                                                                                                                                                                                                                                                                   |
| ------------ | ----------------- | ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `generator`  | `DatasetIterator` | Required    | Generator yielding dataset records                                                                                                                                                                                                                                            |
| `batch_size` | `int`             | `1_000_000` | The number of annotation records to process in each batch before writing data to Parquet with `batch_size` rows. For remote datasets, this also defines the number of rows in the Parquet file before pushing the media (images) and annotations for that batch to the cloud. |

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

| Parameter          | Type   | Default  | Description                                                                                             |
| ------------------ | ------ | -------- | ------------------------------------------------------------------------------------------------------- |
| `new_dataset_name` | `str`  | Required | Name for the cloned dataset                                                                             |
| `push_to_cloud`    | `bool` | `True`   | Whether to push the cloned remote dataset back to the cloud storage, or just keep the local copy of it. |

### Pulling from remote storage

The `pull_from_cloud()` method is used to download the remote dataset locally.
The annotations and metadata subfolders of the dataset are always updated, overwriting any existing local definitions. For media files, if `update_mode=UpdateMode.ALWAYS`, all files are overwritten in the local media folder. If `update_mode=IF_EMPTY`, only missing media files are downloaded. A file is considered missing if it does not exist locally under the path specified in the "file" column of the Parquet file and does not exist as the `media/uuid.suffix`, where the UUID is derived from the "uuid" column of the Parquet file.

| Parameter     | Type         | Default              | Description                                                                                           |
| ------------- | ------------ | -------------------- | ----------------------------------------------------------------------------------------------------- |
| `update_mode` | `UpdateMode` | `UpdateMode.MISSING` | Whether to always download the dataset's media folder from the cloud, or only download missing files. |

### Pushing existing local dataset to cloud

The `push_to_cloud()` method is used to upload a local dataset to the specified bucket storage. Depending on the update mode, it can either always overwrite existing media files or only upload missing files to the media folder of the dataset. Annotations and metadata are always uploaded.

| Parameter        | Type            | Default              | Description                                                                                              |
| ---------------- | --------------- | -------------------- | -------------------------------------------------------------------------------------------------------- |
| `update_mode`    | `UpdateMode`    | `UpdateMode.MISSING` | Whether to always push (overwrite) the dataset’s media folder to the cloud or only upload missing files. |
| `bucket_storage` | `BucketStorage` | Required             | The cloud storage destination to which local media files should be uploaded (e.g., GCS, S3, Azure).      |

## In-Depth Explanation of luxonis-ml Dataset Storage

### File Structure

`LUXONISML_BASE_PATH` defaults to `Path.home() / "luxonis_ml"`
and `LUXONISML_TEAM_ID` defaults to `"offline"`.

**Local dataset storage**:

The dataset structure is as follows:

```plaintext
LUXONISML_BASE_PATH / data / LUXONISML_TEAM_ID / datasets / dataset_name
│
├── annotations
│   ├── 0000000000.parquet
│   ├── 0000000001.parquet
│   ├── 0000000002.parquet
│   ...
│   └── xxxxxxxxxx.parquet
│
├── media (Empty for local datasets unless images are pulled from the cloud when missing locally)
│
└── metadata
    ├── metadata.json
    └── splits.json
```

**Remote dataset storage**:

```plaintext
LUXONISML_BASE_PATH / data / LUXONISML_TEAM_ID / datasets / dataset_name
│
├── annotations
│   ├── 0000000000.parquet
│   ├── 0000000001.parquet
│   ├── 0000000002.parquet
│   ...
│   └── xxxxxxxxxx.parquet
│
├── media
│   ├── fe163bd9-6381-5533-9d41-c1735edf96d5.jpg
│   ├── ge163bd9-6311-5553-9d41-d1735edf96d5.jpg
│   ...
│   └── cae124jd3-6381-5123-9d41-d1735edf96d5.jpg
└── metadata
    ├── metadata.json
    └── splits.json
```

### Parquet file content

The parquet file content will look like:

```plaintext
                       file source_name task_name class_name  instance_id              task_type                                         annotation                                  uuid
0  absolute/path/to/COCO...       image    "coco"     person            0            boundingbox  {"x":0.438734375,"y":0.1052470588235294,"w":0....  20a54075-838a-5a83-bb70-39cb97501a3b
1  Absolute/path/to/COCO...       image    "coco"     person            0              keypoints  {"keypoints":[[0.5734375,0.19058823529411764,2...  20a54075-838a-5a83-bb70-39cb97501a3b
2  Absolute/path/to/COCO...       image    "coco"     person            0           segmentation  {"height":425,"width":640,"counts":"`id3>k<4K3...  20a54075-838a-5a83-bb70-39cb97501a3b
3  Absolute/path/to/COCO...       image    "coco"     person            0  instance_segmentation  {"height":425,"width":640,"counts":"`id3>k<4K3...  20a54075-838a-5a83-bb70-39cb97501a3b
4  Absolute/path/to/COCO...       image    "coco"     person            0         classification                                                 {}  20a54075-838a-5a83-bb70-39cb97501a3b
5  Absolute/path/to/COCO...       image    "coco"     person            0            boundingbox  {"x":0.701359375,"y":0.3757460317460317,"w":0....  83c3579c-d27c-5701-8436-31a72090343f
6  Absolute/path/to/COCO...       image    "coco"     person            0              keypoints  {"keypoints":[[0.74375,0.4158730158730159,2],[...  83c3579c-d27c-5701-8436-31a72090343f
7  Absolute/path/to/COCO...       image    "coco"     person            0           segmentation  {"height":315,"width":640,"counts":"[YZ46d91O1...  83c3579c-d27c-5701-8436-31a72090343f
8  Absolute/path/to/COCO...       image    "coco"     person            0  instance_segmentation  {"height":315,"width":640,"counts":"[YZ46d91O1...  83c3579c-d27c-5701-8436-31a72090343f
9  Absolute/path/to/COCO...       image    "coco"     person            0         classification                                                 {}  83c3579c-d27c-5701-8436-31a72090343f
```

Parquet content is determined by the `generator()` function provided by the user. For tasks such as `instance_segmentation` or `keypoints`, you must either supply an `instance_id` for annotations that are yielded separately but represent the same object (instance) or yield all annotations for each instance together (e.g., yield keypoints and bounding boxes together, or yield bounding boxes and instance segmentation masks together).

### Creating a Dataset Locally

```python
from luxonis_ml.data import LuxonisDataset, BucketStorage

dataset_name = "parking_lot"
bucket_storage = BucketStorage.LOCAL
dataset = LuxonisDataset(dataset_name, bucket_storage=bucket_storage, delete_local=True)

# Add data to the dataset
dataset.add(generator(), batch_size=100_000_000)

# Create splits with 80% train, 10% val and 10% test
dataset.make_splits((0.8, 0.1, 0.1))
```

This creates a local dataset with the structure shown above and deletes any files from a previous dataset with the same name.

If the dataset already exists but new data needs to be appended, use:

```python
dataset = LuxonisDataset(dataset_name, bucket_storage=bucket_storage, delete_local=False)
dataset.add(generator(), batch_size=100_000_000)
```

This will append new data to the dataset and overwrite annotations for images that already exist, even if they have different names but share the same informational content (the same UUID).

### Creating a Dataset Remotely

```python
from luxonis_ml.data import LuxonisDataset, BucketStorage

dataset_name = "parking_lot"
bucket_storage = BucketStorage.GCS # Or any other cloud storage
dataset = LuxonisDataset(dataset_name, bucket_storage=bucket_storage, delete_local=True)

# Add data to the dataset
dataset.add(generator(), batch_size=100_000_000)

# Create splits with 80% train, 10% val and 10% test
dataset.make_splits((0.8, 0.1, 0.1))
```

[A remote dataset functions similarly to a local dataset](#in-depth-explanation-of-luxonis-ml-dataset-storage). When a remote dataset is created, the same folder structure appears locally, and the equivalent structure appears in the cloud. The media folder is empty locally but is filled with images on the remote storage, where filenames become UUIDs with the appropriate suffix.

> \[!NOTE\]
> **IMPORTANT:** Be careful when creating a remote dataset with the same name as an already existing local dataset, because corruption of datasets may occur if not handled properly.
>
> Use `delete_local=True` and `delete_remote=True` to create a new dataset (deleting both local and remote storage) before calling `dataset.add()`, or use `dataset.push_to_cloud()` to push an existing local dataset to the cloud. To append data to an existing dataset using `dataset.> add()`, keep `delete_local=False` and `delete_remote=False`. In that case, ensure both local and remote datasets are healthy. If the local dataset might be corrupted but the remote version is healthy, use `delete_local=True` and `delete_remote=False` so that the local dataset is > deleted, while the remote stays intact.
