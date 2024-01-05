# LuxonisML

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![PyBadge](media/pybadge.svg)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![CI](https://github.com/luxonis/luxonis-ml/actions/workflows/ci.yaml/badge.svg)
![Docs](https://github.com/luxonis/luxonis-ml/actions/workflows/docs.yaml/badge.svg)
![Coverage](media/coverage_badge.svg)

A collection of helper function and utilities.

**NOTE**:
The project is in an alpha state, so it may be missing some critical features or contain bugs - please report any feedback!

## Table of Contents

- [Installation](#installation)
- [Setup](#setup)
- [Contributing](#contributing)

## Installation

The `luxonis_ml` package is hosted on PyPI, so you can install it with `pip`.

We offer several version of the package:

- [`luxonis-ml[data]`](./luxonis_ml/data/README.md): installs necessary dependencies for using `luxonis_ml.data` module
- [`luxonis-ml[utils]`](./luxonis_ml/utils/README.md): installs necessary dependencies for using `luxonis_ml.utils` module
- [`luxonis-ml[embedd]`](./luxonis_ml/embeddings/README.md): installs necessary dependencies for using `luxonis_ml.embeddings` module
- [`luxonis-ml[tracker]`](./luxonis_ml/tracker/README.md): installs necessary dependencies for using `luxonis_ml.tracker` module
- `luxonis-ml[all]`: installs all dependencies
- `luxonis-ml[dev]`: installs all dependencies, including development dependencies

To install the package with all dependecies, run:

```bash
pip install luxonis-ml[all]
```

## Contributing

If you want to contribute to this project, read the instructions in [CONTRIBUTING.md](CONTRIBUTING.md)
