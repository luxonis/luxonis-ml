# Contributing to LuxonisML

**This guide is intended for our internal development team.**
It outlines our workflow and standards for contributing to this project.

## Table of Contents

- [Pre-requisites](#pre-requisites)
- [Pre-commit Hooks](#pre-commit-hooks)
- [Documentation](#documentation)
  - [Editor Support](#editor-support)
- [Tests](#tests)
- [GitHub Actions](#github-actions)
- [Making and Reviewing Changes](#making-and-reviewing-changes)
- [Notes](#notes)

## Pre-requisites

Clone the repository and navigate to the root directory:

```bash
git clone git@github.com:luxonis/luxonis-ml.git
cd luxonis-train
```

Install the development dependencies by running `pip install -r requirements-dev.txt` or install the package with the `dev` extra flag:

```bash
pip install -e .[dev]
```

> \[!NOTE\]
> This will install the package in editable mode (`-e`),
> so you can make changes to the code and run them immediately.

## Pre-commit Hooks

We use pre-commit hooks to ensure code quality and consistency:

1. Install pre-commit (see [pre-commit.com](https://pre-commit.com/#install)).
1. Clone the repository and run `pre-commit install` in the root directory.
1. The pre-commit hook will now run automatically on `git commit`.
   - If the hook fails, it will print an error message and abort the commit.
   - It will also modify the files in-place to fix any issues it can.

## Documentation

We use the [Epytext](https://epydoc.sourceforge.net/epytext.html) markup language for documentation.
To verify that your documentation is formatted correctly, run the following command:

```bash
pydoctor --docformat=epytext luxonis_ml
```

### Editor Support

- **PyCharm** - built in support for generating `epytext` docstrings
- **Visual Studie Code** - [AI Docify](https://marketplace.visualstudio.com/items?itemName=AIC.docify) extension offers support for `epytext`
- **NeoVim** - [vim-python-docstring](https://github.com/pixelneo/vim-python-docstring) supports `epytext` style

## Tests

We use [pytest](https://docs.pytest.org/en/stable/) for testing with the `x-dist` and `coverage` plugins for parallel execution and coverage reporting.

The tests define multiple useful fixtures that make it easier
to achieve fully independent tests that can be run in parallel.

- `tempdir` - an empty temporary directory
- `dataset_name` - a unique name that can be used for instantiation of `LuxonisDataset`
- `randint` - a random integer between 0 and 100,000
- `python_version` - the version of Python running the tests
- `platform_name` - the name of the platform running the tests

For the full list, see the `conftest.py` file in the `tests` directory.

All tests and their files are located in the `tests` directory.

You can run the tests locally with:

```bash
pytest tests --cov=luxonis_ml --cov-report=html -n auto
```

This command will run all tests in parallel (`-n auto`) and will generate an HTML coverage report.

> \[!TIP\]
> The coverage report will be saved to `htmlcov` directory.
> If you want to inspect the coverage in more detail, open `htmlcov/index.html` in a browser.

> \[!IMPORTANT\]
> If a new feature is added, a new test should be added to cover it.
> There is no minimum coverage requirement for now, but minimal coverage will be enforced in the future.

> \[!IMPORTANT\]
> All tests must be passing using the `-n auto` flag before merging a PR.

## GitHub Actions

Our GitHub Actions workflow is run when a new PR is opened.

1. First, the [pre-commit](#pre-commit-hooks) hooks must pass and the [documentation](#documentation) must be built successfully.
1. If all previous checks pass, the [tests](#tests) are run.

> \[!TIP\]
> Review the GitHub Actions output if your PR fails.

> \[!IMPORTANT\]
> Successful completion of all the workflow checks is required for merging a PR.

## Making and Reviewing Changes

1. Make changes in a new branch.
1. Test your changes locally.
1. Commit your changes (pre-commit hooks will run).
1. Push your branch and create a pull request.
1. The team will review and merge your PR.
