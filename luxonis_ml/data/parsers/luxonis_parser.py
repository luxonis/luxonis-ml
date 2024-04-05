import logging
import zipfile
from enum import Enum
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, Type, Union

from luxonis_ml.data import DATASETS_REGISTRY, BaseDataset, LuxonisDataset
from luxonis_ml.enums import DatasetType
from luxonis_ml.utils import LuxonisFileSystem

from .base_parser import BaseParser
from .classification_directory_parser import ClassificationDirectoryParser
from .coco_parser import COCOParser
from .create_ml_parser import CreateMLParser
from .darknet_parser import DarknetParser
from .segmentation_mask_directory_parser import SegmentationMaskDirectoryParser
from .solo_parser import SOLOParser
from .tensorflow_csv_parser import TensorflowCSVParser
from .voc_parser import VOCParser
from .yolov4_parser import YoloV4Parser
from .yolov6_parser import YoloV6Parser

logger = logging.getLogger(__name__)


class ParserType(Enum):
    DIR = "dir"
    SPLIT = "split"


class LuxonisParser:
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
    }

    def __init__(
        self,
        dataset_dir: str,
        *,
        dataset_name: Optional[str] = None,
        delete_existing: bool = False,
        save_dir: Optional[Union[Path, str]] = None,
        dataset_plugin: Optional[str] = None,
        dataset_type: Optional[DatasetType] = None,
        **kwargs,
    ):
        """High-level abstraction over various parsers.

        Automatically recognizes the dataset format and uses the appropriate parser.

        @type dataset_dir: str
        @param dataset_dir: Path to the dataset directory or zip file. Can also be a
            remote URL supported by L{LuxonisFileSystem}.
        @type dataset_name: Optional[str]
        @param dataset_name: Name of the dataset. If C{None}, the name is derived from
            the name of the dataset directory.
        @type delete_existing: bool
        @param delete_existing: If existing dataset with the same name should be deleted
            before parsing.
        @type save_dir: Optional[Union[Path, str]]
        @param save_dir: If a remote URL is provided in C{dataset_dir}, the dataset will
            be downloaded to this directory. If C{None}, the dataset will be downloaded
            to the current working directory.
        @type dataset_plugin: Optional[str]
        @param dataset_plugin: Name of the dataset plugin to use. If C{None},
            C{LuxonisDataset} is used.
        @type dataset_type: Optional[DatasetType]
        @param dataset_type: If provided, the parser will use this dataset type instead
            of trying to recognize it automatically.
        """
        save_dir = Path(save_dir) if save_dir else None
        name = Path(dataset_dir).name
        local_path = (save_dir or Path.cwd()) / name
        self.dataset_dir = LuxonisFileSystem.download(dataset_dir, local_path)
        if self.dataset_dir.suffix == ".zip":
            with zipfile.ZipFile(self.dataset_dir, "r") as zip_ref:
                unzip_dir = self.dataset_dir.parent / self.dataset_dir.stem
                logger.info(f"Extracting '{self.dataset_dir.name}' to '{unzip_dir}'")
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

        if dataset_plugin:
            self.dataset_constructor = DATASETS_REGISTRY.get(dataset_plugin)
        else:
            self.dataset_constructor = LuxonisDataset

        dataset_name = dataset_name or name.replace(" ", "_").split(".")[0]
        self.dataset_exists = self.dataset_constructor.exists(dataset_name=dataset_name)
        if delete_existing and self.dataset_exists:
            logger.warning(f"Deleting existing dataset '{dataset_name}'")
            self.dataset = self.dataset_constructor(dataset_name=dataset_name, **kwargs)
            self.dataset.delete_dataset()
            self.dataset_exists = False

        self.dataset = self.dataset_constructor(dataset_name=dataset_name, **kwargs)
        self.parser = self.parsers[self.dataset_type](self.dataset)

    def parse(self, **kwargs) -> BaseDataset:
        """Parses the dataset and returns it in BaseDataset format.

        If the dataset already exists, parsing will be skipped and the existing dataset
        will be returned instead.

        @type kwargs: Dict[str, Any]
        @param kwargs: Additional C{kwargs} for specific parser implementation.
        @rtype: BaseDataset
        @return: Parsed dataset in L{BaseDataset} format.
        """
        if self.dataset_exists:
            logger.warning(
                "There already exists an LDF dataset with this name. "
                "Parsing will be skipped and existing dataset will be used instead."
            )
            return self.dataset
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
                logger.info(
                    f"[cyan]Recognized dataset format as [red]<{typ.name}>",
                    extra={"markup": True},
                )
                return typ, ParserType.DIR

        for typ, parser in self.parsers.items():
            if parser.validate_split(self.dataset_dir):
                logger.info(
                    f"[cyan]Recognized dataset format as a split of [red]<{typ.name}>",
                    extra={"markup": True},
                )
                return typ, ParserType.SPLIT

        raise ValueError(
            f"Dataset {self.dataset_dir} is not in expected format for any of the parsers."
        )

    def _parse_dir(self, **kwargs) -> BaseDataset:
        """Parses all present data in BaseDataset format.

        Check under each parser for the expected directory structure.

        @type kwargs: Dict[str, Any]
        @param kwargs: Additional C{kwargs} for specific parser function.
        @rtype: BaseDataset
        @return: C{LDF} with all the images and annotations parsed.
        """

        return self.parser.parse_dir(self.dataset_dir, **kwargs)

    def _parse_split(
        self,
        split: Optional[Literal["train", "val", "test"]] = None,
        random_split: bool = True,
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        **kwargs,
    ) -> BaseDataset:
        """Parses data from a subdirectory representing a single split.

        Should be used if adding/changing only specific split. Check under each parser
        for expected directory structure.

        @type split: Optional[Literal["train", "val", "test"]]
        @param split: As what split the data will be added to LDF. If set,
            C{split_ratios} and C{random_split} are ignored.
        @type random_split: bool
        @param random_split: If random splits should be made. If C{True},
            C{split_ratios} are used.
        @type split_ratios: Optional[Tuple[float, float, float]]
        @param split_ratios: Ratios for random splits. Only used if C{random_split} is
            C{True}. Defaults to C{(0.8, 0.1, 0.1)}.
        @type kwargs: Dict[str, Any]
        @param kwargs: Additional kwargs for specific parser implementation.
        @rtype: BaseDataset
        @return: C{LDF} with all the images and annotations parsed.
        """
        parsed_kwargs = self.parser.validate_split(self.dataset_dir)
        if parsed_kwargs is None:
            raise ValueError(
                f"Dataset {self.dataset_dir} is not in the expected format for {self.dataset_type} parser."
            )
        kwargs = {**parsed_kwargs, **kwargs}
        return self.parser.parse_split(split, random_split, split_ratios, **kwargs)
