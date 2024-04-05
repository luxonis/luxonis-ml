# LuxonisML

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![PyBadge](https://github.com/luxonis/luxonis-ml/blob/116262ee55c36433861689077a522faeb32b3967/media/pybadge.svg)
![PyPI](https://img.shields.io/pypi/v/luxonis-ml?label=pypi%20package)
![PyPI - Downloads](https://img.shields.io/pypi/dm/luxonis-ml)

![CI](https://github.com/luxonis/luxonis-ml/actions/workflows/ci.yaml/badge.svg)
![Coverage](https://github.com/luxonis/luxonis-ml/blob/dev/media/coverage_badge.svg)

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Docformatter](https://img.shields.io/badge/%20formatter-docformatter-fedcba.svg)](https://github.com/PyCQA/docformatter)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This library includes a collection of helper functions and utilities for the Luxonis MLOps stack. This includes the following submodules:

- **Dataset Management**: Creating computer vision datasets focused around Luxonis hardware and loading data into our training library.
- **Embeddings**: Methods to compute image embeddings.
- **Tracking**: Our implementation of a logger for use with PyTorch Lightning or in our training library.
- **Utils**: Miscellaneous utils for developers. See this README for details on the different utils.

**NOTE**:
The project is in an alpha state, so it may be missing some critical features or contain bugs - please report any feedback!

## Table of Contents

- [Installation](#installation)
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

To install the package with all dependecies, run:

```bash
pip install luxonis-ml[all]
```

## Contributing

If you want to contribute to this project, read the instructions in [CONTRIBUTING.md](https://github.com/luxonis/luxonis-ml/blob/main/CONTRIBUTING.md)
