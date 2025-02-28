# LuxonisML

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![PyBadge](https://img.shields.io/pypi/pyversions/luxonis-ml?logo=data:image/svg+xml%3Bbase64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAxMDAgMTAwIj4KICA8ZGVmcz4KICAgIDxsaW5lYXJHcmFkaWVudCBpZD0icHlZZWxsb3ciIGdyYWRpZW50VHJhbnNmb3JtPSJyb3RhdGUoNDUpIj4KICAgICAgPHN0b3Agc3RvcC1jb2xvcj0iI2ZlNSIgb2Zmc2V0PSIwLjYiLz4KICAgICAgPHN0b3Agc3RvcC1jb2xvcj0iI2RhMSIgb2Zmc2V0PSIxIi8+CiAgICA8L2xpbmVhckdyYWRpZW50PgogICAgPGxpbmVhckdyYWRpZW50IGlkPSJweUJsdWUiIGdyYWRpZW50VHJhbnNmb3JtPSJyb3RhdGUoNDUpIj4KICAgICAgPHN0b3Agc3RvcC1jb2xvcj0iIzY5ZiIgb2Zmc2V0PSIwLjQiLz4KICAgICAgPHN0b3Agc3RvcC1jb2xvcj0iIzQ2OCIgb2Zmc2V0PSIxIi8+CiAgICA8L2xpbmVhckdyYWRpZW50PgogIDwvZGVmcz4KCiAgPHBhdGggZD0iTTI3LDE2YzAtNyw5LTEzLDI0LTEzYzE1LDAsMjMsNiwyMywxM2wwLDIyYzAsNy01LDEyLTExLDEybC0yNCwwYy04LDAtMTQsNi0xNCwxNWwwLDEwbC05LDBjLTgsMC0xMy05LTEzLTI0YzAtMTQsNS0yMywxMy0yM2wzNSwwbDAtM2wtMjQsMGwwLTlsMCwweiBNODgsNTB2MSIgZmlsbD0idXJsKCNweUJsdWUpIi8+CiAgPHBhdGggZD0iTTc0LDg3YzAsNy04LDEzLTIzLDEzYy0xNSwwLTI0LTYtMjQtMTNsMC0yMmMwLTcsNi0xMiwxMi0xMmwyNCwwYzgsMCwxNC03LDE0LTE1bDAtMTBsOSwwYzcsMCwxMyw5LDEzLDIzYzAsMTUtNiwyNC0xMywyNGwtMzUsMGwwLDNsMjMsMGwwLDlsMCwweiBNMTQwLDUwdjEiIGZpbGw9InVybCgjcHlZZWxsb3cpIi8+CgogIDxjaXJjbGUgcj0iNCIgY3g9IjY0IiBjeT0iODgiIGZpbGw9IiNGRkYiLz4KICA8Y2lyY2xlIHI9IjQiIGN4PSIzNyIgY3k9IjE1IiBmaWxsPSIjRkZGIi8+Cjwvc3ZnPgo=)
[![PyPI](https://img.shields.io/pypi/v/luxonis-ml?label=pypi%20package)](https://pypi.org/project/luxonis-ml/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/luxonis-ml)](https://pypi.org/project/luxonis-ml/)

![CI](https://github.com/luxonis/luxonis-ml/actions/workflows/ci.yaml/badge.svg)
[![codecov](https://codecov.io/gh/luxonis/luxonis-ml/graph/badge.svg?token=01E7QTYXWU)](https://codecov.io/gh/luxonis/luxonis-ml)

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Docformatter](https://img.shields.io/badge/%20formatter-docformatter-fedcba.svg)](https://github.com/PyCQA/docformatter)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This library includes a collection of helper functions and utilities for the Luxonis MLOps stack. This includes the following submodules:

- **Dataset Management**: Creating computer vision datasets focused around Luxonis hardware and to be used with our [LuxonisTrain](https://pypi.org/project/luxonis-train/) framework. Additional documentation can be found [here](https://github.com/luxonis/luxonis-ml/blob/main/luxonis_ml/data/README.md).
- **Embeddings**: Methods to compute image embeddings.
- **Tracking**: Our implementation of a logger for use with PyTorch Lightning or in [LuxonisTrain](https://pypi.org/project/luxonis-train/)
- **Utils**: Miscellaneous utils for developers.

**NOTE**:
The project is in a beta state, it might be missing certain features or contain bugs - please report any feedback!

## Table of Contents

- [Installation](#installation)
- [CLI](#cli)
- [Contributing](#contributing)

## Installation

The `luxonis_ml` package is hosted on PyPI, so you can install it with `pip`.

We offer several versions of the package:

- `luxonis-ml[data]`: installs necessary dependencies for using `luxonis_ml.data` module
- `luxonis-ml[utils]`: installs necessary dependencies for using `luxonis_ml.utils` module
- `luxonis-ml[embedd]`: installs necessary dependencies for using `luxonis_ml.embeddings` module
- `luxonis-ml[tracker]`: installs necessary dependencies for using `luxonis_ml.tracker` module
- `luxonis-ml[all]`: installs all dependencies
- `luxonis-ml[dev]`: installs all dependencies, including development dependencies

### Additional dependencies

Additional dependencies for working with specific cloud services can be installed using the following extras:

- `gcs`: Dependencies for working with Google Cloud Storage
- `s3`: Dependencies for working with AWS S3
- `roboflow`: Dependencies for downloading datasets from Roboflow
- `mlflow`: Dependencies for working with MLFlow

> \[!NOTE\]
> If some of the additional dependencies are required but not installed (_e.g._ attempting to use Google Cloud Storage without installing the `gcs` extra), then the missing dependencies will be installed automatically.

**Example**:

Installing the package with the `data` extra and dependencies for `gcs` and `roboflow`:

```bash
pip install luxonis-ml[data,gcs,roboflow]
```

Installing the package with all dependencies:

```bash
pip install luxonis-ml[all]
```

## CLI

The `luxonis-ml` package comes with a CLI that can be used to interact with the library.

To see the available commands, run:

```bash
luxonis_ml --help
```

## Contributing

If you want to contribute to this project, read the instructions in [CONTRIBUTING.md](https://github.com/luxonis/luxonis-ml/blob/main/CONTRIBUTING.md)
