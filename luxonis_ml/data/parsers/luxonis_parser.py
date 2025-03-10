import subprocess
import sys
import zipfile
from enum import Enum
from importlib.util import find_spec
from pathlib import Path
from typing import (
    Dict,
    Generic,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)

from loguru import logger

from luxonis_ml.data import DATASETS_REGISTRY, BaseDataset, LuxonisDataset
from luxonis_ml.enums import DatasetType
from luxonis_ml.utils import LuxonisFileSystem, environ
from luxonis_ml.utils.filesystem import _pip_install

from .base_parser import BaseParser
from .classification_directory_parser import ClassificationDirectoryParser
from .coco_parser import COCOParser
from .create_ml_parser import CreateMLParser
from .darknet_parser import DarknetParser
from .native_parser import NativeParser
from .segmentation_mask_directory_parser import SegmentationMaskDirectoryParser
from .solo_parser import SOLOParser
from .tensorflow_csv_parser import TensorflowCSVParser
from .voc_parser import VOCParser
from .yolov4_parser import YoloV4Parser
from .yolov6_parser import YoloV6Parser


class ParserType(Enum):
    DIR = "dir"
    SPLIT = "split"


T = TypeVar("T", str, None)


class LuxonisParser(Generic[T]):
    parsers: Dict[DatasetType, Type[BaseParser]] = {
        DatasetType.COCO: COCOParser,
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
    }

    def __init__(
        self,
        dataset_dir: str,
        *,
        dataset_name: Optional[str] = None,
        save_dir: Optional[Union[Path, str]] = None,
        dataset_plugin: T = None,
        dataset_type: Optional[DatasetType] = None,
        task_name: Optional[Union[str, Dict[str, str]]] = None,
        **kwargs,
    ):
        """High-level abstraction over various parsers.

        Automatically recognizes the dataset format and uses the
        appropriate parser.

        @type dataset_dir: str
        @param dataset_dir: Identifier of the dataset directory.
            Can be one of:
                - Local path to the dataset directory.
                - Remote URL supported by L{LuxonisFileSystem}.
                  - C{gcs://} for Google Cloud Storage
                  - C{s3://} for Amazon S3
                - C{roboflow://} for Roboflow datasets.
                  - Expected format: C{roboflow://workspace/project/version/format}.
            Can be a remote URL supported by L{LuxonisFileSystem}.
        @type dataset_name: Optional[str]
        @param dataset_name: Name of the dataset. If C{None}, the name
            is derived from the name of the dataset directory.
        @type save_dir: Optional[Union[Path, str]]
        @param save_dir: If a remote URL is provided in C{dataset_dir},
            the dataset will be downloaded to this directory. If
            C{None}, the dataset will be downloaded to the current
            working directory.
        @type dataset_plugin: Optional[str]
        @param dataset_plugin: Name of the dataset plugin to use. If
            C{None}, C{LuxonisDataset} is used.
        @type dataset_type: Optional[DatasetType]
        @param dataset_type: If provided, the parser will use this
            dataset type instead of trying to recognize it
            automatically.
        @type task_name: Optional[Union[str, Dict[str, str]]]
        @param task_name: Optional task name(s) for the dataset.
            Can be either a single string, in which case all the records
            added to the dataset will use this value as `task_name`, or
            a dictionary with class names as keys and task names as values.
            In the latter case, the task name for a record with a given
            class name will be taken from the dictionary.
        @type kwargs: Dict[str, Any]
        @param kwargs: Additional C{kwargs} to be passed to the
            constructor of specific L{BaseDataset} implementation.
        """
        save_dir = Path(save_dir) if save_dir else None
        if dataset_dir.startswith("roboflow://"):
            self.dataset_dir, name = self._download_roboflow_dataset(
                dataset_dir, save_dir
            )
        else:
            name = dataset_dir.split("/")[-1]
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
                self.dataset_dir = unzip_dir

        if dataset_type:
            self.dataset_type = dataset_type
            self.parser_type = (
                ParserType.DIR
                if Path(self.dataset_dir / "train").exists()
                else ParserType.SPLIT
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

    @overload
    def parse(self: "LuxonisParser[str]", **kwargs) -> BaseDataset: ...

    @overload
    def parse(self: "LuxonisParser[None]", **kwargs) -> LuxonisDataset: ...

    def parse(self, **kwargs) -> BaseDataset:
        """Parses the dataset and returns it in LuxonisDataset format.

        If the dataset already exists, parsing will be skipped and the
        existing dataset will be returned instead.

        @type kwargs: Dict[str, Any]
        @param kwargs: Additional C{kwargs} for specific parser
            implementation.
        @rtype: LuxonisDataset
        @return: Parsed dataset in L{LuxonisDataset} format.
        """
        if self.parser_type == ParserType.DIR:
            dataset = self._parse_dir(**kwargs)
        else:
            dataset = self._parse_split(**kwargs)

        logger.info("Dataset parsed successfully.")
        return dataset

    def _recognize_dataset(self) -> Tuple[DatasetType, ParserType]:
        """Recognizes the dataset format and parser type.

        @rtype: Tuple[DatasetType, ParserType]
        @return: Tuple of dataset type and parser type.
        """
        for typ, parser in self.parsers.items():
            if parser.validate(self.dataset_dir):
                logger.info(f"Recognized dataset format as <{typ.name}>")
                return typ, ParserType.DIR

        for typ, parser in self.parsers.items():
            if parser.validate_split(self.dataset_dir):
                logger.info(
                    f"Recognized dataset format as a split of <{typ.name}>"
                )
                return typ, ParserType.SPLIT

        raise ValueError(
            f"Dataset {self.dataset_dir} is not in expected format for any of the parsers."
        )

    def _parse_dir(self, **kwargs) -> BaseDataset:
        """Parses all present data in LuxonisDataset format.

        Check under each parser for the expected directory structure.

        @type kwargs: Dict[str, Any]
        @param kwargs: Additional C{kwargs} for specific parser
            function.
        @rtype: LuxonisDataset
        @return: C{LDF} with all the images and annotations parsed.
        """

        return self.parser.parse_dir(self.dataset_dir, **kwargs)

    def _parse_split(
        self,
        split: Optional[str] = None,
        random_split: bool = True,
        split_ratios: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> BaseDataset:
        """Parses data from a subdirectory representing a single split.

        Should be used if adding/changing only specific split. Check
        under each parser for expected directory structure.

        @type split: Optional[Literal["train", "val", "test"]]
        @param split: As what split the data will be added to LDF. If
            set, C{split_ratios} and C{random_split} are ignored.
        @type random_split: bool
        @param random_split: If random splits should be made. If
            C{True}, C{split_ratios} are used.
        @type split_ratios: Optional[Dict[str, float]]
        @param split_ratios: Ratios for random splits. Only used if
            C{random_split} is C{True}. Defaults to C{{"train": 0.8,
            "val": 0.1, "test": 0.1}}.
        @type kwargs: Dict[str, Any]
        @param kwargs: Additional kwargs for specific parser
            implementation.
        @rtype: LuxonisDataset
        @return: C{LDF} with all the images and annotations parsed.
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
        self, dataset_dir: str, local_path: Optional[Path]
    ) -> Tuple[Path, str]:
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

        if environ.ROBOFLOW_API_KEY is None:
            raise RuntimeError(
                "ROBOFLOW_API_KEY environment variable is not set. "
                "Please set it to your Roboflow API key."
            )

        rf = Roboflow(api_key=environ.ROBOFLOW_API_KEY)
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
