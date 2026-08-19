# Contributing to LuxonisML

This guide is the starting point for local development and pull requests.
For API usage details, use the generated docs at
<https://docs.luxonis.com/software-v3/ai-inference/model-source/training/luxonis-ml/luxonis-ml-api-reference/>.

## Preparation

Install [uv](https://docs.astral.sh/uv/), then:

```bash
git clone git@github.com:luxonis/luxonis-ml.git
cd luxonis-ml

uv sync

uv run prek install
uv run luxonis_ml checkhealth
```

`uv sync` creates `.venv` and installs the package in editable mode together
with the `dev` dependency group. That group depends on `luxonis-ml[all]`, so a
plain `uv sync` pulls in every extra as well — you do not need `--all-extras`.
There is no `source .venv/bin/activate` step: `uv run <cmd>` syncs the
environment and runs the command inside it. Use `uv run --no-sync <cmd>` when
you have hand-installed something into `.venv` that a sync would revert, such
as a git checkout of a dependency.

**The project targets Python 3.10 or newer.** `.python-version` pins the local
interpreter to 3.10, which is what CI runs, and `uv` downloads it for you if it
is missing.

The repository's requirements files are generated pip compatibility exports,
not inputs. Do not edit them by hand. Change dependencies in `pyproject.toml`
(or with `uv add`), then run:

```bash
tools/export_requirements.sh
```

A pre-commit hook runs it for you whenever the dependency files change, so
usually you only need to review and stage the refreshed `uv.lock` and
requirements files before committing again.

Only `requirements.txt` is hash pinned. The other exports retain their legacy
unhashed format; this is also required for `data`, because pip does not honour
the uv override that removes `opencv-python-headless`. See the comment in
`tools/export_requirements.sh`.

## Repository Map

| Path                           | Purpose                                                                          |
| ------------------------------ | -------------------------------------------------------------------------------- |
| `luxonis_ml/data`              | Dataset creation, parsers, exporters, loaders, augmentations, and LDF utilities. |
| `luxonis_ml/utils`             | Shared config, filesystem, path, logging, graph, and registry helpers.           |
| `luxonis_ml/nn_archive`        | Model archive metadata and archive generation utilities.                         |
| `luxonis_ml/tracker`           | Experiment tracking integrations.                                                |
| `luxonis_ml/telemetry`         | Lightweight telemetry client, events, redaction, and backends.                   |
| `tests`                        | Pytest suite, fixtures, integration tests, and data workflow coverage.           |
| `tools/export_requirements.sh` | Regenerates `uv.lock` and the `requirements*.txt` exports.                       |
| `tools/version.py`             | Reports the package version, or changes it for a release.                        |

Package dependencies are defined entirely in `pyproject.toml`: runtime
requirements in `[project.dependencies]`, the per-module and cloud extras in
`[project.optional-dependencies]`, and the `dev` and `docs` tooling in
`[dependency-groups]`.

## Pre-commit checks

Run the same formatting and static checks before pushing:

```bash
uv run prek run --all-files
```

The hooks are executed by [`prek`](https://github.com/j178/prek), a drop-in
replacement for `pre-commit` that reads the same `.pre-commit-config.yaml`.

Pre-commit runs:

| Tool                   | What it enforces                                                                          |
| ---------------------- | ----------------------------------------------------------------------------------------- |
| `ruff` / `ruff-format` | Python linting, import sorting, and formatting.                                           |
| `typos`                | Common spelling mistakes.                                                                 |
| `mdformat`             | Markdown formatting.                                                                      |
| `prettier`             | YAML formatting.                                                                          |
| `taplo-format`         | TOML formatting.                                                                          |
| `pre-commit-hooks`     | File endings, JSON/YAML/TOML validity, private key detection, and main branch protection. |
| `export-requirements`  | Keeps `uv.lock` and the pip requirements exports in sync with `pyproject.toml`.           |

**Do not commit directly to `main`.** The pre-commit hook blocks it, and PRs
are the expected review path.

## Tests

Run focused tests while developing, then broaden the run before opening a PR.

```bash
uv run pytest tests/test_utils/test_config.py -q
uv run pytest tests/test_data/test_loader.py --only-local -q
uv run pytest tests --only-local -n auto
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
uv run pydoctor luxonis_ml
```

Open `apidocs/index.html` to inspect the result. CI runs the same command. The
`[tool.pydoctor]` section of `pyproject.toml` sets the Google docstring format,
which pydoctor reads on its own.

> [!IMPORTANT]
> A malformed docstring is a syntax error. pydoctor reports it and exits
> non-zero, so the CI job fails on broken markup. pydoctor exits 0 for
> unresolved-reference and annotation warnings, and the build reports some of
> those today. Read the command output after you change a docstring, because
> those warnings do not stop the build.

The published reference lives on the Luxonis documentation portal. The
`Export pydoctor reference` workflow builds it from these docstrings with the
exporter that the `luxonis/docs-content` repository provides. The workflow
uploads the result as the `luxonis-ml-api-reference` artifact. It runs on
`release/*` pull requests and on manual dispatch, so a docstring change reaches
the portal with the next release.

For docs changes, update the package and object docstrings that feed the
generated site. The submodule `README.md` files were removed, so the docstrings
are the only home for API documentation. Keep the root `README.md` a user
guide: it introduces the library, shows the main workflows, and links to the
generated site. Document a new behavior next to the code that implements it.

Good doc changes should:

- explain public behavior, accepted inputs, outputs, and failure modes;
- include compact examples for non-obvious data shapes;
- use readable spacing in JSON and Python snippets;
- avoid restating implementation details that users do not need.

> [!IMPORTANT]
> Use direct names in docstrings rather than Sphinx cross-reference roles (`:func:`, `:mod:`, `:attr:`, *etc.*);
> pydoctor resolves supported references automatically.

## Type Checking and Security

CI runs Pyright in warning mode against `pyproject.toml`. Pyright is pinned in
the `dev` dependency group, so the local run matches CI:

```bash
uv run pyright --warnings --level warning --project pyproject.toml
```

CI also runs Semgrep with automatic rules and secret scanning. Treat those
findings as required review items unless the team explicitly accepts the risk.

## Pull Requests

1. Create a branch with a descriptive prefix such as `feature/`, `fix/`,
   `bugfix/`, `docs/`, or `ci/`.
1. Keep changes scoped. Update tests and generated-doc docstrings with behavior
   changes.
1. Run focused tests, then `uv run prek run --all-files`.
1. Build docs if public APIs, docstrings, or examples changed.
1. Open a PR and include the problem, solution, and verification commands.

Every file is owned by `@luxonis/ML-Reviewers` through CODEOWNERS. Labels are
applied automatically from branch names and changed paths.

PR CI runs:

| Check        | Notes                                                  |
| ------------ | ------------------------------------------------------ |
| `pre-commit` | Must pass before docs, type check, and tests proceed.  |
| `docs`       | Builds the pydoctor docs to check that they parse.     |
| `type-check` | Runs Pyright on Python 3.10.                           |
| `semgrep`    | Runs security and secret scanning.                     |
| `tests`      | Runs pytest on Ubuntu and Windows in six split groups. |

Release branches named `release/*` also run extra package-install checks for
selected extras.

## Releases

A release needs two manual steps: start the workflow, then merge the pull
request it opens.

1. Run the `Release PR` workflow from the Actions tab. Give it `major`,
   `minor`, `patch`, or an explicit version such as `1.2.3`.
1. The workflow changes `__version__` in `luxonis_ml/__init__.py` and opens a
   `release/vX.Y.Z` pull request that lists the changes after the last tag.
1. Review the pull request, wait for CI, and merge it.
1. The `Release Tag` workflow then tags `vX.Y.Z-beta` on the merge commit and
   creates the GitHub release with generated notes.
1. The release publication starts the PyPI upload.

Use `tools/version.py` for the same version change on your machine:

```bash
python3 tools/version.py --path luxonis_ml/__init__.py            # report
python3 tools/version.py --path luxonis_ml/__init__.py --set minor
```

`.github/release.yaml` groups the generated notes into categories from the pull
request labels. GitHub reads that file; the name is not ours to choose.

### Shared with the other repositories

The release logic lives in two reusable workflows, and every Luxonis
repository calls them. Only the two thin caller workflows and the label
configuration are per repository. To adopt them elsewhere, add a
`Release PR` caller:

```yaml
jobs:
  release-pr:
    uses: luxonis/luxonis-ml/.github/workflows/reusable-release-pr.yaml@main
    with:
      version: ${{ inputs.version }}
      version-file: luxonis_train/__init__.py
      project-name: LuxonisTrain
    secrets:
      WORKFLOW_SECRET: ${{ secrets.WORKFLOW_SECRET }}
```

and a `Release Tag` caller with the same `version-file`. The repository needs
the `WORKFLOW_SECRET` secret and a `release` label. The reusable workflows
check out `tools/version.py` from this repository, so the caller does not copy
it.

Package publishing, the API reference export, and dependency updates run in
GitHub Actions:

- `reusable-release-pr.yaml` and `reusable-release-tag.yaml` hold the shared
  release logic.
- `release-pr.yaml` opens the version bump pull request on manual dispatch.
- `release-tag.yaml` tags and releases a merged `release/*` pull request.
- `python-publish.yml` builds and publishes on release publication or manual
  dispatch.
- `export-pydoctor-reference.yaml` builds the API reference from the docstrings.
  It runs on a `release/*` pull request or on manual dispatch, and uploads the
  result as an artifact.
- `pip-install.yaml` installs from the `requirements*.txt` exports whenever they
  or their inputs change. Nothing else in CI uses those files, so this is what
  keeps the pip path working for users who do not have `uv`.
- `dependencies_autoupdate.yaml` runs `uv lock --upgrade` monthly and opens a PR
  with the refreshed lock and exports.

Do not change release or dependency behavior without checking the relevant
workflow and package metadata in `pyproject.toml`.
