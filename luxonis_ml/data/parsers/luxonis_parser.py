import subprocess
import sys
import zipfile
from enum import Enum
from importlib.util import find_spec
from pathlib import Path
from typing import Generic, TypeVar, overload
from urllib.parse import parse_qs, urlsplit

import requests
from loguru import logger

from luxonis_ml.data import DATASETS_REGISTRY, BaseDataset, LuxonisDataset
from luxonis_ml.data.utils.enums import ParserIssueMessage
from luxonis_ml.data.utils.remote_file_downloader import download_remote_file
from luxonis_ml.enums import DatasetType
from luxonis_ml.utils import LuxonisFileSystem, environ
from luxonis_ml.utils.filesystem import _pip_install

from .base_parser import BaseParser
from .classification_directory_parser import ClassificationDirectoryParser
from .coco_parser import COCOParser
from .create_ml_parser import CreateMLParser
from .darknet_parser import DarknetParser
from .fiftyone_classification_parser import FiftyOneClassificationParser
from .native_parser import NativeParser
from .segmentation_mask_directory_parser import SegmentationMaskDirectoryParser
from .solo_parser import SOLOParser
from .tensorflow_csv_parser import TensorflowCSVParser
from .ultralytics_ndjson_parser import UltralyticsNDJSONParser
from .voc_parser import VOCParser
from .yolov4_parser import YoloV4Parser
from .yolov6_parser import YoloV6Parser
from .yolov8_parser import YOLOv8Parser


class ParserType(Enum):
    """Recognized parser entry-point type.

    Attributes:
        DIR: Dataset directory with multiple splits.
        SPLIT: Directory or file representing one parser input split.

    """

    DIR = "dir"
    SPLIT = "split"


T = TypeVar("T", str, None)


class LuxonisParser(Generic[T]):
    """Detect a dataset format and dispatch to the matching parser.

    The parser accepts local paths, remote paths supported by
    `LuxonisFileSystem`, ZIP archives, and Roboflow dataset URLs in
    ``roboflow://workspace/project/version/format`` form. If
    ``dataset_type`` is omitted, parsers are tried until one validates the
    dataset directory as either a full directory or a single split.

    Attributes:
        parsers: Dataset types mapped to concrete parser classes.
        dataset_dir: Local dataset directory used for parsing.
        dataset_type: Recognized or user-provided dataset type.
        parser_type: Whether ``dataset_dir`` represents a full directory or
            a single split.
        dataset_constructor: Dataset class resolved from the plugin
            registry, or `LuxonisDataset`.
        dataset: Dataset instance populated by parsing.
        parser: Concrete parser instance selected for ``dataset_type``.

    """

    parsers: dict[DatasetType, type[BaseParser]] = {
        DatasetType.ULTRALYTICSNDJSON: UltralyticsNDJSONParser,
        DatasetType.ULTRALYTICSNDJSONINSTANCESEGMENTATION: (
            UltralyticsNDJSONParser
        ),
        DatasetType.ULTRALYTICSNDJSONKEYPOINTS: UltralyticsNDJSONParser,
        DatasetType.COCO: COCOParser,
        DatasetType.FIFTYONECLS: FiftyOneClassificationParser,
        DatasetType.VOC: VOCParser,
        DatasetType.DARKNET: DarknetParser,
        DatasetType.YOLOV6: YoloV6Parser,
        DatasetType.YOLOV4: YoloV4Parser,
        DatasetType.CREATEML: CreateMLParser,
        DatasetType.TFCSV: TensorflowCSVParser,
        DatasetType.CLSDIR: ClassificationDirectoryParser,
        DatasetType.SEGMASK: SegmentationMaskDirectoryParser,
        DatasetType.SOLO: SOLOParser,
        DatasetType.NATIVE: NativeParser,
        DatasetType.YOLOV8BOUNDINGBOX: YOLOv8Parser,
        DatasetType.YOLOV8INSTANCESEGMENTATION: YOLOv8Parser,
        DatasetType.YOLOV8KEYPOINTS: YOLOv8Parser,
    }

    def __init__(
        self,
        dataset_dir: str,
        *,
        dataset_name: str | None = None,
        save_dir: Path | str | None = None,
        dataset_plugin: T = None,
        dataset_type: DatasetType | None = None,
        task_name: str | dict[str, str] | None = None,
        **kwargs,
    ):
        """High-level abstraction over various parsers.

        Automatically recognizes the dataset format and uses the
        appropriate parser.

        Args:
            dataset_dir: Dataset directory identifier.
                Can be one of:

                    - Local path to the dataset directory.

                    - Remote URL supported by L{LuxonisFileSystem}.
                        - ``gcs://`` for Google Cloud Storage
                        - ``s3://`` for Amazon S3

                    - ``roboflow://`` for Roboflow datasets.
                        - Expected format: ``roboflow://workspace/project/version/format``.

                    - ``ultralytics://`` for Ultralytics Platform datasets.
                        - Expected format: ``ultralytics://username/datasets/slug``
                        - Optional version: append ``?v=<version>`` to export a specific dataset version.

            dataset_name: Optional output dataset name. If omitted, the
                name is derived from ``dataset_dir``.
            save_dir: Optional directory used when downloading remote
                datasets. If omitted, the current working directory is used.
            dataset_plugin: Optional dataset plugin registry name. If
                omitted, ``LuxonisDataset`` is used.
            dataset_type: Optional dataset type. If provided, automatic
                format recognition is skipped.
            task_name: Optional task naming rule. A string is used for all
                records. A mapping uses class names as keys and task names
                as values.
            kwargs: Additional arguments passed to the selected dataset
                constructor.

        Raises:
            RuntimeError: If a Roboflow URL is used and
                ``ROBOFLOW_API_KEY`` is not configured.
            ValueError: If a Roboflow URL is malformed, its version is not
                an integer, or the dataset format cannot be recognized.

        """
        save_dir = Path(save_dir) if save_dir else None
        if dataset_dir.startswith("roboflow://"):
            self.dataset_dir, name = self._download_roboflow_dataset(
                dataset_dir, save_dir
            )
        elif dataset_dir.startswith("ultralytics://"):
            self.dataset_dir, name = self._download_ultralytics_dataset(
                dataset_dir, save_dir
            )
        else:
            name = dataset_dir.rsplit("/", maxsplit=1)[-1]
            local_path = (save_dir or Path.cwd()) / name
            self.dataset_dir = LuxonisFileSystem.download(
                dataset_dir, local_path
            )
        if self.dataset_dir.suffix == ".zip":
            with zipfile.ZipFile(self.dataset_dir, "r") as zip_ref:
                unzip_dir = self.dataset_dir.parent / self.dataset_dir.stem
                logger.info(
                    f"Extracting '{self.dataset_dir.name}' to '{unzip_dir}'"
                )
                zip_ref.extractall(unzip_dir)
                self.dataset_dir = self._resolve_extracted_zip_root(unzip_dir)

        if dataset_type:
            self.dataset_type = dataset_type
            self.parser_type = self._infer_parser_type_for_explicit_type(
                self.dataset_type
            )
        else:
            self.dataset_type, self.parser_type = self._recognize_dataset()

        if dataset_plugin is not None:
            self.dataset_constructor = DATASETS_REGISTRY.get(dataset_plugin)
        else:
            self.dataset_constructor = LuxonisDataset

        dataset_name = dataset_name or name.replace(" ", "_").split(".")[0]

        self.dataset = self.dataset_constructor(
            dataset_name=dataset_name,  # type: ignore
            **kwargs,
        )
        self.parser = self.parsers[self.dataset_type](
            self.dataset, self.dataset_type, task_name
        )

    @staticmethod
    def _resolve_extracted_zip_root(unzip_dir: Path) -> Path:
        ignored_entries = {"__MACOSX", "Thumbs.db", "desktop.ini"}
        visible_entries = [
            entry
            for entry in unzip_dir.iterdir()
            if entry.name not in ignored_entries
            and not entry.name.startswith(".")
        ]
        if len(visible_entries) != 1:
            return unzip_dir

        only_entry = visible_entries[0]
        if not only_entry.is_dir():
            return unzip_dir

        # Only unwrap when the inner directory clearly looks like a
        # dataset root, not an arbitrary folder.
        # ClassificationDirectoryParser is excluded from parser-based
        # checks because a single class folder can look like a wrapper.
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
        child_dirs = {d.name for d in only_entry.iterdir() if d.is_dir()}
        child_files = {f.name for f in only_entry.iterdir() if f.is_file()}
        has_markers = bool(
            child_dirs & marker_dirs or child_files & marker_files
        )
        if not has_markers:
            recognized_by_non_clsdir = any(
                (
                    typ != DatasetType.CLSDIR
                    and (
                        parser.validate(only_entry)
                        or parser.validate_split(only_entry)
                    )
                )
                for typ, parser in LuxonisParser.parsers.items()
            )
            if not recognized_by_non_clsdir:
                return unzip_dir

        logger.info(
            f"Detected top-level folder '{only_entry.name}' in extracted zip. "
            "Using it as dataset root."
        )
        return only_entry

    @overload
    def parse(self: "LuxonisParser[str]", **kwargs) -> BaseDataset: ...

    @overload
    def parse(self: "LuxonisParser[None]", **kwargs) -> LuxonisDataset: ...

    def parse(self, **kwargs) -> BaseDataset:
        """Parse the dataset and return it in LDF format.

        If the dataset already exists, parsing will be skipped and the
        existing dataset will be returned instead.

        Args:
            kwargs: Parser-specific arguments.

        Returns:
            Parsed dataset.

        Raises:
            ValueError: If the selected parser rejects the dataset
                structure or split arguments.

        """
        if self.parser_type == ParserType.DIR:
            dataset = self._parse_dir(**kwargs)
        else:
            dataset = self._parse_split(**kwargs)

        logger.info("Dataset parsed successfully.")
        return dataset

    def get_parser_issue_messages(self) -> list[ParserIssueMessage]:
        """Return parser issue messages collected during the last
        parse.
        """
        return self.parser.get_parser_issue_messages()

    def _recognize_dataset(self) -> tuple[DatasetType, ParserType]:
        """Recognize the dataset format and parser type.

        Returns:
            Dataset type and parser type.

        """
        for dataset_type, parser in self.parsers.items():
            if parser.validate(self.dataset_dir):
                logger.info(
                    f"Recognized dataset format as <{dataset_type.name}>"
                )
                return dataset_type, ParserType.DIR

        subset_matches: dict[type[BaseParser], DatasetType] = {}
        for dataset_type, parser in self.parsers.items():
            # The same YoloV8 or UltralyticsNDJSON parser can correspond to multiple dataset types.
            if parser in subset_matches:
                continue
            if parser.discover_dir_splits(self.dataset_dir):
                subset_matches[parser] = dataset_type

        if len(subset_matches) == 1:
            dataset_type = next(iter(subset_matches.values()))
            logger.info(
                f"Recognized dataset format as <{dataset_type.name}> from partial splits"
            )
            return dataset_type, ParserType.DIR

        if len(subset_matches) > 1:
            matched_parsers = set(subset_matches)
            if matched_parsers == {YoloV6Parser, YOLOv8Parser}:
                raise ValueError(
                    "Dataset layout is compatible with multiple parsers when "
                    "only a subset of splits is present. This layout is "
                    "ambiguous between YOLOv6 and YOLOv8. Please specify "
                    "dataset_type."
                )
            raise ValueError(
                "Dataset layout is compatible with multiple parsers when "
                "only a subset of splits is present. Please specify "
                "dataset_type."
            )

        for dataset_type, parser in self.parsers.items():
            if parser.validate_split(self.dataset_dir):
                logger.info(
                    f"Recognized dataset format as a split of <{dataset_type.name}>"
                )
                return dataset_type, ParserType.SPLIT

        raise ValueError(
            f"Dataset {self.dataset_dir} is not in expected format for any of the parsers."
        )

    def _infer_parser_type_for_explicit_type(
        self, dataset_type: DatasetType
    ) -> ParserType:
        parser = self.parsers[dataset_type]
        if parser.validate(self.dataset_dir) or parser.discover_dir_splits(
            self.dataset_dir
        ):
            return ParserType.DIR
        if parser.validate_split(self.dataset_dir):
            return ParserType.SPLIT
        raise ValueError(
            f"Dataset {self.dataset_dir} is not in expected format for the "
            f"{dataset_type.name} parser."
        )

    def _parse_dir(self, **kwargs) -> BaseDataset:
        """Parse all data in the recognized dataset directory.

        Check under each parser for the expected directory structure.

        Args:
            kwargs: Parser-specific arguments.

        Returns:
            Parsed dataset.

        """
        return self.parser.parse_dir(self.dataset_dir, **kwargs)

    def _parse_split(
        self,
        split: str | None = None,
        random_split: bool = True,
        split_ratios: dict[str, float | int] | None = None,
        **kwargs,
    ) -> BaseDataset:
        """Parse data from a directory representing a single split.

        Should be used if adding/changing only specific split. Check
        under each parser for expected directory structure.

        Args:
            split: Optional split name to assign to parsed data. When set,
                ``split_ratios`` and ``random_split`` are ignored.
            random_split: Whether to create random splits using
                ``split_ratios``.
            split_ratios: Optional split ratios or counts.
            kwargs: Parser-specific arguments.

        Returns:
            Parsed dataset.

        """
        parsed_kwargs = self.parser.validate_split(self.dataset_dir)
        if parsed_kwargs is None:
            raise ValueError(
                f"Dataset {self.dataset_dir} is not in the expected format for {self.dataset_type} parser."
            )
        return self.parser.parse_split(
            split, random_split, split_ratios, **parsed_kwargs, **kwargs
        )

    def _download_roboflow_dataset(
        self, dataset_dir: str, local_path: Path | None
    ) -> tuple[Path, str]:
        if environ.ROBOFLOW_API_KEY is None:
            raise RuntimeError(
                "ROBOFLOW_API_KEY environment variable is not set. "
                "Please set it to your Roboflow API key."
            )

        if find_spec("roboflow") is None:  # pragma: no cover
            _pip_install("roboflow", "roboflow~=1.1.0")
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

        rf = Roboflow(api_key=environ.ROBOFLOW_API_KEY.get_secret_value())
        parts = dataset_dir.split("roboflow://")[1].split("/")
        if len(parts) != 4:
            raise ValueError(
                f"Incorrect Roboflow dataset URL: `{dataset_dir}`. "
                "Expected format: `roboflow://workspace/project/version/format`."
            )
        workspace, project, version, format = dataset_dir.split("roboflow://")[
            1
        ].split("/")
        try:
            version = int(version)
        except ValueError as e:
            raise ValueError(
                f"Roboflow version must be an integer, got `{version}`."
            ) from e

        local_path = local_path or Path.cwd() / f"{project}_{format}"
        dataset = (
            rf.workspace(workspace)
            .project(project)
            .version(int(version))
            .download(format, str(local_path / project))
        )
        return Path(dataset.location), project

    def _download_ultralytics_dataset(
        self,
        dataset_dir: str,
        local_path: Path | None,
    ) -> tuple[Path, str]:
        if environ.ULTRALYTICS_API_KEY is None:
            raise RuntimeError(
                "ULTRALYTICS_API_KEY environment variable is not set. "
                "Please set it to your Ultralytics API key."
            )

        ultralytics_api_base_url = "https://platform.ultralytics.com/api"
        headers = {
            "Authorization": (
                f"Bearer {environ.ULTRALYTICS_API_KEY.get_secret_value()}"
            ),
            "User-Agent": "luxonis-ml",
        }

        parsed = urlsplit(dataset_dir)
        raw_version = parse_qs(parsed.query).get("v", [None])[0]
        version: int | None = None
        if raw_version is not None:
            try:
                version = int(raw_version)
            except ValueError as e:
                raise ValueError(
                    "Ultralytics dataset export version must be an integer, "
                    f"got `{raw_version}`."
                ) from e
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
                f"Incorrect Ultralytics dataset reference: `{dataset_dir}`. "
                "Expected `ultralytics://username/datasets/slug`."
            )

        dataset_response = requests.get(
            f"{ultralytics_api_base_url}/datasets",
            headers=headers,
            params={
                "username": parsed.netloc,
                "slug": path_parts[1],
            },
            timeout=30.0,
        )
        if not dataset_response.ok:
            try:
                error = dataset_response.json().get("error")
            except ValueError:
                error = dataset_response.text

            raise RuntimeError(
                f"Ultralytics API request failed "
                f"({dataset_response.status_code}): {error}"
            )

        dataset = dataset_response.json()["dataset"]

        dataset_id = dataset["_id"]

        export_params = {"v": version} if version is not None else None
        export_response = requests.get(
            f"{ultralytics_api_base_url}/datasets/{dataset_id}/export",
            headers=headers,
            params=export_params,
            timeout=120.0,
        )
        if not export_response.ok:
            error = None
            try:
                payload = export_response.json()
                error = payload.get("error")
            except ValueError:
                error = export_response.text.strip() or export_response.reason

            detail = f"{export_response.status_code} {export_response.reason}"
            if error:
                detail = f"{detail}: {error}"
            raise RuntimeError(f"Ultralytics API request failed: {detail}")

        download_url = export_response.json()["downloadUrl"]

        file_stem = dataset["slug"]
        dataset_name = dataset["name"]
        local_path = local_path or Path.cwd()
        destination = (
            local_path / f"{file_stem}.v{version}.ndjson"
            if version is not None
            else local_path / f"{file_stem}.ndjson"
        )
        download_remote_file(download_url, destination, timeout=120.0)

        return destination, dataset_name
