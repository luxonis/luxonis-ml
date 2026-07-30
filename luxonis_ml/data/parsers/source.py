import subprocess
import sys
import zipfile
from importlib.util import find_spec
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import requests
from loguru import logger

from luxonis_ml.data.utils.remote_file_downloader import download_remote_file
from luxonis_ml.typing import PathType
from luxonis_ml.utils import LuxonisFileSystem, environ
from luxonis_ml.utils.filesystem import _pip_install


def prepare_source(
    source: PathType,
    save_dir: Path | str | None,
) -> tuple[Path, str]:
    """Acquire an import source locally and derive its dataset name."""
    source_string = str(source)
    local_dir = Path(save_dir) if save_dir is not None else None
    if source_string.startswith("roboflow://"):
        source_path, name = _download_roboflow_dataset(
            source_string,
            local_dir,
        )
    elif source_string.startswith("ultralytics://"):
        source_path, name = _download_ultralytics_dataset(
            source_string,
            local_dir,
        )
    else:
        # A trailing separator would otherwise yield an empty name, which in
        # turn makes both the dataset name and the download target empty.
        name = source_string.rstrip("/").rsplit("/", maxsplit=1)[-1]
        if not name:
            raise ValueError(
                f"Could not derive a dataset name from source '{source}'."
            )
        local_path = (local_dir or Path.cwd()) / name
        source_path = LuxonisFileSystem.download(source_string, local_path)

    if source_path.suffix == ".zip":
        with zipfile.ZipFile(source_path, "r") as archive:
            unzip_dir = source_path.parent / source_path.stem
            logger.info(f"Extracting '{source_path.name}' to '{unzip_dir}'")
            archive.extractall(unzip_dir)
        source_path = _resolve_extracted_zip_root(unzip_dir)

    return source_path, name


def _resolve_extracted_zip_root(unzip_dir: Path) -> Path:
    from .classification_directory_parser import (
        ClassificationDirectoryParser,
    )
    from .parser_plugin import PARSERS_REGISTRY

    ignored_entries = {"__MACOSX", "Thumbs.db", "desktop.ini"}
    visible_entries = [
        entry
        for entry in unzip_dir.iterdir()
        if entry.name not in ignored_entries and not entry.name.startswith(".")
    ]
    if len(visible_entries) != 1 or not visible_entries[0].is_dir():
        return unzip_dir

    only_entry = visible_entries[0]
    marker_dirs = {
        "train",
        "valid",
        "val",
        "validation",
        "test",
        "images",
        "labels",
        "data",
        "raw",
        "masks",
    }
    marker_files = {
        "annotations.json",
        "labels.json",
        "data.yaml",
        "dataset.yaml",
        "dataset.yml",
    }
    child_dirs = {path.name for path in only_entry.iterdir() if path.is_dir()}
    child_files = {
        path.name for path in only_entry.iterdir() if path.is_file()
    }
    has_markers = bool(child_dirs & marker_dirs or child_files & marker_files)
    if not has_markers:
        recognized = any(
            plugin is not ClassificationDirectoryParser
            and plugin.supports(only_entry)
            for plugin in dict.fromkeys(PARSERS_REGISTRY.values())
        )
        if not recognized:
            return unzip_dir

    logger.info(
        f"Detected top-level folder '{only_entry.name}' in extracted zip. "
        "Using it as dataset root."
    )
    return only_entry


def _download_roboflow_dataset(
    source: str,
    local_path: Path | None,
) -> tuple[Path, str]:
    if environ.ROBOFLOW_API_KEY is None:
        raise RuntimeError(
            "ROBOFLOW_API_KEY environment variable is not set. "
            "Please set it to your Roboflow API key."
        )

    if find_spec("roboflow") is None:  # pragma: no cover
        _pip_install("roboflow", "roboflow~=1.4")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "uninstall",
                "-y",
                "opencv-python",
                "opencv-python-headless",
            ],
            check=False,
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "opencv-python~=4.10.0",
            ],
            check=False,
        )

    from roboflow import Roboflow

    parts = source.removeprefix("roboflow://").split("/")
    if len(parts) != 4:
        raise ValueError(
            f"Incorrect Roboflow dataset URL: `{source}`. Expected format: "
            "`roboflow://workspace/project/version/format`."
        )
    workspace, project, raw_version, export_format = parts
    try:
        version = int(raw_version)
    except ValueError as error:
        raise ValueError(
            f"Roboflow version must be an integer, got `{raw_version}`."
        ) from error

    destination = local_path or Path.cwd() / f"{project}_{export_format}"
    dataset = (
        Roboflow(api_key=environ.ROBOFLOW_API_KEY.get_secret_value())
        .workspace(workspace)
        .project(project)
        .version(version)
        .download(export_format, str(destination / project))
    )
    return Path(dataset.location), project


def _download_ultralytics_dataset(
    source: str,
    local_path: Path | None,
) -> tuple[Path, str]:
    if environ.ULTRALYTICS_API_KEY is None:
        raise RuntimeError(
            "ULTRALYTICS_API_KEY environment variable is not set. "
            "Please set it to your Ultralytics API key."
        )

    api_base_url = "https://platform.ultralytics.com/api"
    headers = {
        "Authorization": (
            f"Bearer {environ.ULTRALYTICS_API_KEY.get_secret_value()}"
        ),
        "User-Agent": "luxonis-ml",
    }
    parsed = urlsplit(source)
    raw_version = parse_qs(parsed.query).get("v", [None])[0]
    version: int | None = None
    if raw_version is not None:
        try:
            version = int(raw_version)
        except ValueError as error:
            raise ValueError(
                "Ultralytics dataset export version must be an integer, "
                f"got `{raw_version}`."
            ) from error
        if version < 1:
            raise ValueError(
                "Ultralytics dataset export version must be >= 1, "
                f"got `{version}`."
            )

    path_parts = [part for part in parsed.path.split("/") if part]
    if not (
        parsed.scheme == "ultralytics"
        and parsed.netloc
        and len(path_parts) == 2
        and path_parts[0] == "datasets"
    ):
        raise ValueError(
            f"Incorrect Ultralytics dataset reference: `{source}`. Expected "
            "`ultralytics://username/datasets/slug`."
        )

    dataset_response = requests.get(
        f"{api_base_url}/datasets",
        headers=headers,
        params={"username": parsed.netloc, "slug": path_parts[1]},
        timeout=30.0,
    )
    if not dataset_response.ok:
        try:
            error = dataset_response.json().get("error")
        except ValueError:
            error = dataset_response.text
        raise RuntimeError(
            "Ultralytics API request failed "
            f"({dataset_response.status_code}): {error}"
        )

    dataset = dataset_response.json()["dataset"]
    export_response = requests.get(
        f"{api_base_url}/datasets/{dataset['_id']}/export",
        headers=headers,
        params={"v": version} if version is not None else None,
        timeout=120.0,
    )
    if not export_response.ok:
        try:
            error = export_response.json().get("error")
        except ValueError:
            error = export_response.text.strip() or export_response.reason
        detail = f"{export_response.status_code} {export_response.reason}" + (
            f": {error}" if error else ""
        )
        raise RuntimeError(f"Ultralytics API request failed: {detail}")

    destination_root = local_path or Path.cwd()
    destination = (
        destination_root / f"{dataset['slug']}.v{version}.ndjson"
        if version is not None
        else destination_root / f"{dataset['slug']}.ndjson"
    )
    download_remote_file(
        export_response.json()["downloadUrl"],
        destination,
        timeout=120.0,
    )
    return destination, dataset["name"]
