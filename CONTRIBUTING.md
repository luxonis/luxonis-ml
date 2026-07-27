# Contributing to LuxonisML

This guide is the starting point for local development and pull requests.
For API usage details, use the generated docs at
<https://luxonis.github.io/luxonis-ml/latest/>.

## First 10 Minutes

```bash
git clone git@github.com:luxonis/luxonis-ml.git
cd luxonis-ml

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e '.[dev]'

pre-commit install
luxonis_ml checkhealth
```

**Use Python 3.10 or newer.** CI currently runs the core checks on Python
3.10, so Python 3.10 is the safest local baseline.

The editable install (`-e`) keeps your checkout importable while you edit it.
The `dev` extra installs the package extras plus test, docs, and pre-commit
tools.

If dependency resolution behaves differently from CI, match the CI constraint:

```bash
printf 'setuptools<81\n' > /tmp/luxonis-ml-constraints.txt
PIP_CONSTRAINT=/tmp/luxonis-ml-constraints.txt python -m pip install -e '.[dev]'
```

## Repository Map

| Path                           | Purpose                                                                          |
| ------------------------------ | -------------------------------------------------------------------------------- |
| `luxonis_ml/data`              | Dataset creation, parsers, exporters, loaders, augmentations, and LDF utilities. |
| `luxonis_ml/utils`             | Shared config, filesystem, path, logging, graph, and registry helpers.           |
| `luxonis_ml/nn_archive`        | Model archive metadata and archive generation utilities.                         |
| `luxonis_ml/tracker`           | Experiment tracking integrations.                                                |
| `luxonis_ml/telemetry`         | Lightweight telemetry client, events, redaction, and backends.                   |
| `tests`                        | Pytest suite, fixtures, integration tests, and data workflow coverage.           |
| `tools/build_pydoctor_docs.py` | The local and CI entrypoint for generated API docs.                              |

Package dependencies are defined in `pyproject.toml` and loaded from the
module requirement files plus `extra_requirements/`.

## Daily Workflow

Run the same formatting and static checks before pushing:

```bash
pre-commit run --all-files
```

Pre-commit runs:

| Tool                   | What it enforces                                                                          |
| ---------------------- | ----------------------------------------------------------------------------------------- |
| `ruff` / `ruff-format` | Python linting, import sorting, and formatting.                                           |
| `typos`                | Common spelling mistakes.                                                                 |
| `mdformat`             | Markdown formatting.                                                                      |
| `prettier`             | YAML formatting.                                                                          |
| `taplo-format`         | TOML formatting.                                                                          |
| `pre-commit-hooks`     | File endings, JSON/YAML/TOML validity, private key detection, and main branch protection. |

**Do not commit directly to `main`.** The pre-commit hook blocks it, and PRs
are the expected review path.

## Tests

Run focused tests while developing, then broaden the run before opening a PR.

```bash
python -m pytest tests/test_utils/test_config.py -q
python -m pytest tests/test_data/test_loader.py --only-local -q
python -m pytest tests --only-local -n auto
```

Use `--only-local` for data tests that are parametrized over local and cloud
storage. Some tests still need credentials or downloaded fixtures because they
exercise explicit remote integrations.

Common fixtures live in `tests/conftest.py`:

| Fixture           | Use                                                       |
| ----------------- | --------------------------------------------------------- |
| `tempdir`         | Isolated temporary directory under the test workspace.    |
| `dataset_name`    | Unique `LuxonisDataset` name with cleanup.                |
| `randint`         | Fresh random integer for unique names.                    |
| `height`, `width` | Shared image dimensions.                                  |
| `bucket_storage`  | Parametrized storage backend; affected by `--only-local`. |

CI splits the full suite across Ubuntu and Windows and runs with `-n auto`.
When a change touches shared data loading, parsing, exporting, storage, or
annotation behavior, add or update tests near the affected workflow.

## Documentation

Public API docs are generated from docstrings with pydoctor using the
**Google** docstring format.

Build the current checkout locally:

```bash
python tools/build_pydoctor_docs.py --mode current --output apidocs
```

Open `apidocs/latest/index.html` to inspect the result.

For docs changes, prefer updating package and object docstrings that feed the
generated site. Several submodule `README.md` files are deprecated and point to
GitHub Pages; avoid expanding them unless the task specifically asks for it.

Good doc changes should:

- explain public behavior, accepted inputs, outputs, and failure modes;
- include compact examples for non-obvious data shapes;
- use readable spacing in JSON and Python snippets;
- avoid restating implementation details that users do not need.

## Type Checking and Security

CI runs Pyright in warning mode against `pyproject.toml` after installing
`.[dev]`.

```bash
pyright --project pyproject.toml
```

_Pyright is invoked through the GitHub Action in CI; install it locally through
your preferred Node or Python wrapper if you want the same feedback before
pushing._

CI also runs Semgrep with automatic rules and secret scanning. Treat those
findings as required review items unless the team explicitly accepts the risk.

## Pull Requests

1. Create a branch with a descriptive prefix such as `feature/`, `fix/`,
   `bugfix/`, `docs/`, or `ci/`.
1. Keep changes scoped. Update tests and generated-doc docstrings with behavior
   changes.
1. Run focused tests, then `pre-commit run --all-files`.
1. Build docs if public APIs, docstrings, or examples changed.
1. Open a PR and include the problem, solution, and verification commands.

Every file is owned by `@luxonis/ML-Reviewers` through CODEOWNERS. Labels are
applied automatically from branch names and changed paths.

PR CI runs:

| Check        | Notes                                                     |
| ------------ | --------------------------------------------------------- |
| `pre-commit` | Must pass before docs, type check, and tests proceed.     |
| `docs`       | Builds pydoctor docs with `tools/build_pydoctor_docs.py`. |
| `type-check` | Runs Pyright on Python 3.10.                              |
| `semgrep`    | Runs security and secret scanning.                        |
| `tests`      | Runs pytest on Ubuntu and Windows in six split groups.    |

Release branches named `release/*` also run extra package-install checks for
selected extras.

## Releases

Package publishing and documentation deployment are handled by GitHub Actions:

- `python-publish.yml` builds and publishes on release publication or manual
  dispatch.
- `docs-pages.yaml` publishes GitHub Pages docs on `main`, release publication,
  or manual dispatch.

Do not change release or dependency behavior without checking the relevant
workflow and package metadata in `pyproject.toml`.
