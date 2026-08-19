# LuxonisML

![Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white)
![Windows](https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![PyBadge](https://img.shields.io/pypi/pyversions/luxonis-ml?logo=data:image/svg+xml%3Bbase64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAxMDAgMTAwIj4KICA8ZGVmcz4KICAgIDxsaW5lYXJHcmFkaWVudCBpZD0icHlZZWxsb3ciIGdyYWRpZW50VHJhbnNmb3JtPSJyb3RhdGUoNDUpIj4KICAgICAgPHN0b3Agc3RvcC1jb2xvcj0iI2ZlNSIgb2Zmc2V0PSIwLjYiLz4KICAgICAgPHN0b3Agc3RvcC1jb2xvcj0iI2RhMSIgb2Zmc2V0PSIxIi8+CiAgICA8L2xpbmVhckdyYWRpZW50PgogICAgPGxpbmVhckdyYWRpZW50IGlkPSJweUJsdWUiIGdyYWRpZW50VHJhbnNmb3JtPSJyb3RhdGUoNDUpIj4KICAgICAgPHN0b3Agc3RvcC1jb2xvcj0iIzY5ZiIgb2Zmc2V0PSIwLjQiLz4KICAgICAgPHN0b3Agc3RvcC1jb2xvcj0iIzQ2OCIgb2Zmc2V0PSIxIi8+CiAgICA8L2xpbmVhckdyYWRpZW50PgogIDwvZGVmcz4KCiAgPHBhdGggZD0iTTI3LDE2YzAtNyw5LTEzLDI0LTEzYzE1LDAsMjMsNiwyMywxM2wwLDIyYzAsNy01LDEyLTExLDEybC0yNCwwYy04LDAtMTQsNi0xNCwxNWwwLDEwbC05LDBjLTgsMC0xMy05LTEzLTI0YzAtMTQsNS0yMywxMy0yM2wzNSwwbDAtM2wtMjQsMGwwLTlsMCwweiBNODgsNTB2MSIgZmlsbD0idXJsKCNweUJsdWUpIi8+CiAgPHBhdGggZD0iTTc0LDg3YzAsNy04LDEzLTIzLDEzYy0xNSwwLTI0LTYtMjQtMTNsMC0yMmMwLTcsNi0xMiwxMi0xMmwyNCwwYzgsMCwxNC03LDE0LTE1bDAtMTBsOSwwYzcsMCwxMyw5LDEzLDIzYzAsMTUtNiwyNC0xMywyNGwtMzUsMGwwLDNsMjMsMGwwLDlsMCwweiBNMTQwLDUwdjEiIGZpbGw9InVybCgjcHlZZWxsb3cpIi8+CgogIDxjaXJjbGUgcj0iNCIgY3g9IjY0IiBjeT0iODgiIGZpbGw9IiNGRkYiLz4KICA8Y2lyY2xlIHI9IjQiIGN4PSIzNyIgY3k9IjE1IiBmaWxsPSIjRkZGIi8+Cjwvc3ZnPgo=)
[![PyPI](https://img.shields.io/pypi/v/luxonis-ml?label=pypi%20package)](https://pypi.org/project/luxonis-ml/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/luxonis-ml)](https://pypi.org/project/luxonis-ml/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![CI](https://github.com/luxonis/luxonis-ml/actions/workflows/ci.yaml/badge.svg)
[![codecov](https://codecov.io/gh/luxonis/luxonis-ml/graph/badge.svg?token=01E7QTYXWU)](https://codecov.io/gh/luxonis/luxonis-ml)

<a name="overview"></a>

## 🌟 Overview

`LuxonisML` is the core library of the Luxonis MLOps stack. It defines the
`Luxonis Data Format` (LDF), and it provides the dataset, loader, parser,
tracking, and utility layers that the other Luxonis tools build on.
[`LuxonisTrain`](https://github.com/luxonis/luxonis-train),
[`ModelConverter`](https://github.com/luxonis/modelconverter), and
[`LuxonisEval`](https://github.com/luxonis/luxonis-eval) all depend on it.

<a name="key-features"></a>

### ✨ Key Features

- **One Dataset Format**: Build a computer vision dataset once, then train on it with any of the Luxonis tools.
- **Bring Your Own Data**: Convert `COCO`, `YOLO`, `VOC`, `RoboFlow`, and other common formats with a single call.
- **Storage Agnostic**: Keep the images locally, in `Google Cloud Storage`, or in `S3`, without a change to your training code.
- **Rich Annotations**: Bounding boxes, keypoints, semantic and instance segmentation, classification, arrays, and free-form metadata.
- **One Tracking API**: Log metrics to `TensorBoard`, `Weights & Biases`, or `MLflow` through one interface.

> [!WARNING]
> **The project is in a beta state and might be unstable or contain bugs - please report any feedback.**

<a name="quick-start"></a>

## 🚀 Quick Start

1. **Install `LuxonisML`**

   ```bash
   pip install luxonis-ml[data]
   ```

   This will create the `luxonis_ml` executable in your `PATH`.

1. **Convert a dataset into LDF**

   We will use a sample COCO dataset from `RoboFlow` in this example.

   ```bash
   luxonis_ml data parse "roboflow://team-roboflow/coco-128/2/coco" --name coco_test
   ```

   > [!IMPORTANT]
   > A `roboflow://` source needs the `ROBOFLOW_API_KEY` environment variable.
   > Get your key from the
   > [Roboflow settings](https://app.roboflow.com/settings/api). The
   > [Roboflow documentation](https://docs.roboflow.com/reference/authentication/authentication/find-your-roboflow-api-key)
   > gives the steps.

1. **Inspect the result**

   ```bash
   luxonis_ml data info coco_test
   luxonis_ml data inspect coco_test
   ```

1. **Load it in your training code**

   ```python
   from luxonis_ml.data import LuxonisDataset, LuxonisLoader

   loader = LuxonisLoader(LuxonisDataset("coco_test"), view="train")

   for sample in loader:
       images = sample.images
       labels = sample.labels
   ```

> [!NOTE]
> For hands-on examples of how to prepare data with `LuxonisML` and train AI models using `LuxonisTrain`, check out [this guide](https://github.com/luxonis/ai-tutorials/tree/main/training).

## 📜 Table Of Contents

- [🌟 Overview](#overview)
  - [✨ Key Features](#key-features)
- [🚀 Quick Start](#quick-start)
- [🧩 Modules](#modules)
- [🛠️ Installation](#installation)
- [📝 Usage](#usage)
  - [💾 Creating a Dataset](#creating-a-dataset)
  - [🔄 Converting an Existing Dataset](#converting-a-dataset)
  - [📤 Loading the Data](#loading-the-data)
  - [📈 Tracking an Experiment](#tracking-an-experiment)
- [💻 CLI](#cli)
- [🔑 Credentials](#credentials)
- [📚 Documentation](#documentation)
- [🤝 Contributing](#contributing)

<a name="modules"></a>

## 🧩 Modules

Each module links to its own API reference:

| Module                                                                                                                                            | Extra        | Purpose                                                                    |
| ------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ | -------------------------------------------------------------------------- |
| [`luxonis_ml.data`](https://docs.luxonis.com/software-v3/ai-inference/model-source/training/luxonis-ml/luxonis-ml-api-reference/data)             | `data`       | Dataset creation, conversion, loading, augmentation, and export.           |
| [`luxonis_ml.ldf`](https://docs.luxonis.com/software-v3/ai-inference/model-source/training/luxonis-ml/luxonis-ml-api-reference/ldf)               | `ldf`        | The annotation schemas of the Luxonis Data Format. A subset of `data`.     |
| [`luxonis_ml.tracker`](https://docs.luxonis.com/software-v3/ai-inference/model-source/training/luxonis-ml/luxonis-ml-api-reference/tracker)       | `tracker`    | One experiment tracking API for TensorBoard, Weights & Biases, and MLflow. |
| [`luxonis_ml.telemetry`](https://docs.luxonis.com/software-v3/ai-inference/model-source/training/luxonis-ml/luxonis-ml-api-reference/telemetry)   | `telemetry`  | A lightweight telemetry client with pluggable backends.                    |
| [`luxonis_ml.nn_archive`](https://docs.luxonis.com/software-v3/ai-inference/model-source/training/luxonis-ml/luxonis-ml-api-reference/nn_archive) | `nn_archive` | NN Archive creation and inspection.                                        |
| [`luxonis_ml.utils`](https://docs.luxonis.com/software-v3/ai-inference/model-source/training/luxonis-ml/luxonis-ml-api-reference/utils)           | `utils`      | Config, environment, filesystem, logging, graph, and registry helpers.     |

<a name="installation"></a>

## 🛠️ Installation

`LuxonisML` requires **Python 3.10** or higher. We recommend using a virtual
environment to manage dependencies.

**Install via `pip`**:

```bash
pip install luxonis-ml[data]
```

Each module has its own extra, so you install only what you use:

| Extra        | Installs the dependencies of                              |
| ------------ | --------------------------------------------------------- |
| `ldf`        | `luxonis_ml.ldf`                                          |
| `data`       | `luxonis_ml.data`, and `luxonis_ml.ldf` with it           |
| `tracker`    | `luxonis_ml.tracker`, except `mlflow` and `opencv-python` |
| `telemetry`  | `luxonis_ml.telemetry`, with the PostHog backend          |
| `nn_archive` | `luxonis_ml.nn_archive`                                   |
| `utils`      | `luxonis_ml.utils`                                        |
| `all`        | All of the above, and all of the extras below             |

The `data`, `ldf`, `tracker`, and `utils` modules fail on import when you do
not install their extra. The message names the extra. `luxonis_ml.telemetry`
is an exception: it imports without `posthog` and falls back to a no-op
backend.

### ☁️ Additional Dependencies

These extras add support for specific cloud services and integrations:

| Extra      | Adds support for                     |
| ---------- | ------------------------------------ |
| `gcs`      | Google Cloud Storage                 |
| `s3`       | AWS S3                               |
| `roboflow` | Dataset downloads from Roboflow      |
| `mlflow`   | MLflow tracking and artifact storage |

> [!NOTE]
> `LuxonisML` installs these four dependencies for you on first use. If you open a `gs://`, `s3://`, `mlflow://`, or `roboflow://` path and the package is absent, `LuxonisML` installs it and continues. Install the extra yourself when you want a reproducible environment or an offline machine.

**Examples**:

```bash
# the data module, with Google Cloud Storage and Roboflow support
pip install luxonis-ml[data,gcs,roboflow]

# everything
pip install luxonis-ml[all]
```

For a development environment, read [CONTRIBUTING.md](CONTRIBUTING.md). The
development tooling lives in `uv` dependency groups, not in a published extra.

<a name="usage"></a>

## 📝 Usage

<a name="creating-a-dataset"></a>

### 💾 Creating a Dataset

A dataset is a named collection of records. Each record points to one image and
carries one annotation. The coordinates are relative to the image size.

```python
from luxonis_ml.data import LuxonisDataset

dataset = LuxonisDataset("parking_lot")


def records():
    yield {
        "file": "images/frame_001.jpg",
        "task_name": "detection",
        "annotation": {
            "class": "car",
            "boundingbox": {
                "x": 0.1,
                "y": 0.2,
                "w": 0.3,
                "h": 0.4,
            },
        },
    }


dataset.add(records())
dataset.make_splits({"train": 0.8, "val": 0.1, "test": 0.1})
```

LDF also supports keypoints, semantic and instance segmentation,
classification, arrays, and free-form metadata. See
[`luxonis_ml.data.datasets`](https://docs.luxonis.com/software-v3/ai-inference/model-source/training/luxonis-ml/luxonis-ml-api-reference/data/datasets)
for the dataset contract, and the [`luxonis_ml.ldf`](https://docs.luxonis.com/software-v3/ai-inference/model-source/training/luxonis-ml/luxonis-ml-api-reference/ldf) module for every annotation
schema.

<a name="converting-a-dataset"></a>

### 🔄 Converting an Existing Dataset

`LuxonisParser` reads the common dataset formats. It detects the format from
the directory structure when you do not name one.

```python
from luxonis_ml.data import LuxonisParser

dataset = LuxonisParser(
    "path/to/coco_dataset",
    dataset_name="coco",
).parse()
```

**The dataset path can be one of the following:**

- a local directory, or a `ZIP` archive
- `s3://bucket/path/to/directory` for **AWS S3**
- `gs://bucket/path/to/directory` for **Google Cloud Storage**
- `roboflow://{workspace}/{project}/{version}/{format}` for **RoboFlow**
- `ultralytics://{username}/datasets/{slug}` for **Ultralytics**

See
[`luxonis_ml.data.parsers`](https://docs.luxonis.com/software-v3/ai-inference/model-source/training/luxonis-ml/luxonis-ml-api-reference/data/parsers)
for the supported formats and their expected layouts.

<a name="loading-the-data"></a>

### 📤 Loading the Data

`LuxonisLoader` reads one or more splits. It resizes the images, applies the
augmentations, and returns the labels for each task.

```python
from luxonis_ml.data import LuxonisLoader

loader = LuxonisLoader(dataset, view="train", height=640, width=640)

for sample in loader:
    images = sample.images
    labels = sample.labels
```

Labels use `"task_name/task_type"` keys, such as `"detection/boundingbox"`. See
[`luxonis_ml.data.loaders`](https://docs.luxonis.com/software-v3/ai-inference/model-source/training/luxonis-ml/luxonis-ml-api-reference/data/loaders)
for the output shapes, and
[`luxonis_ml.data.augmentations`](https://docs.luxonis.com/software-v3/ai-inference/model-source/training/luxonis-ml/luxonis-ml-api-reference/data/augmentations)
for the augmentation configuration.

<a name="tracking-an-experiment"></a>

### 📈 Tracking an Experiment

```python
from luxonis_ml.tracker import LuxonisTracker

tracker = LuxonisTracker(
    project_name="parking_lot",
    run_name="baseline",
    is_tensorboard=True,
)

tracker.log_metric("loss", 0.42, step=1)
tracker.close()
```

> [!NOTE]
> The `tracker` extra does not install every dependency of `luxonis_ml.tracker`. The module imports `mlflow` and `cv2` at import time, so install `mlflow` and `opencv-python` as well: `pip install "luxonis-ml[tracker,mlflow]" opencv-python`. Install `torch` for TensorBoard and `wandb` for Weights & Biases.

<a name="cli"></a>

## 💻 CLI

The package installs the `luxonis_ml` executable.

**Available commands:**

- `data` - Parse, inspect, export, merge, push, pull, and delete datasets
- `archive` - Inspect and extract `NN Archive` files
- `fs` - Copy and list files across the supported storage backends
- `checkhealth` - Report whether the `ldf`, `data`, `utils`, and `nn_archive` modules import correctly

**To get help on any command:**

```bash
luxonis_ml <command> --help
```

**Examples:**

```bash
luxonis_ml data parse ./coco_dataset --name coco
luxonis_ml data ls
luxonis_ml data health coco
luxonis_ml data export coco --type ultralyticsndjson
```

<a name="credentials"></a>

## 🔑 Credentials

When using cloud services, avoid hard-coding credentials or placing them
directly in your configuration files. Instead:

- Use environment variables to store sensitive information.
- Use a `.env` file and load it securely, ensuring it's excluded from version control.

**Supported Cloud Services:**

- **AWS S3**, requires:
  - `AWS_ACCESS_KEY_ID`
  - `AWS_SECRET_ACCESS_KEY`
  - `AWS_S3_ENDPOINT_URL`
- **Google Cloud Storage**, requires:
  - `GOOGLE_APPLICATION_CREDENTIALS`
- **RoboFlow**, requires:
  - `ROBOFLOW_API_KEY`
- **Ultralytics**, requires:
  - `ULTRALYTICS_API_KEY`

**For logging and tracking, we support:**

- **MLFlow**, requires:
  - `MLFLOW_S3_BUCKET`
  - `MLFLOW_S3_ENDPOINT_URL`
  - `MLFLOW_TRACKING_URI`

**Dataset storage is configured with:**

- `LUXONISML_BASE_PATH` - local base path for datasets and cache files, `~/luxonis_ml` by default
- `LUXONISML_TEAM_ID` - team identifier used by dataset storage, `offline` by default
- `LUXONISML_BUCKET` - the cloud bucket that holds remote datasets

> [!NOTE]
> `LuxonisML` sends no telemetry. It only provides a telemetry client that other Luxonis packages can use. To turn that client off, set `LUXONIS_TELEMETRY_ENABLED=false`. See the [`luxonis_ml.telemetry`](https://docs.luxonis.com/software-v3/ai-inference/model-source/training/luxonis-ml/luxonis-ml-api-reference/telemetry) documentation for the data an event carries.

<a name="documentation"></a>

## 📚 Documentation

The API documentation is generated from the docstrings in the source code, and
it is published on the Luxonis documentation portal:

**<https://docs.luxonis.com/software-v3/ai-inference/model-source/training/luxonis-ml/luxonis-ml-api-reference/>**

To build the documentation locally, run:

```bash
uv run pydoctor luxonis_ml
```

The command writes the site to `apidocs/index.html`.

<a name="contributing"></a>

## 🤝 Contributing

We welcome contributions! Please read our
[Contribution Guide](https://github.com/luxonis/luxonis-ml/blob/main/CONTRIBUTING.md)
to get started. Whether it's reporting bugs, improving documentation, or adding
new features, your help is appreciated.

## 📄 License

This project is licensed under the
[Apache 2.0 License](https://github.com/luxonis/luxonis-ml/blob/main/LICENSE).
