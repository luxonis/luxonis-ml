import logging
import zipfile
from enum import Enum
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, Type, Union

from luxonis_ml.data import LuxonisDataset
from luxonis_ml.enums import DatasetType

from .base_parser import BaseParser
from .classification_directory_parser import ClassificationDirectoryParser
from .coco_parser import COCOParser
from .create_ml_parser import CreateMLParser
from .darknet_parser import DarknetParser
from .segmentation_mask_directory_parser import SegmentationMaskDirectoryParser
from .tensorflow_csv_parser import TensorflowCSVParser
from .voc_parser import VOCParser
from .yolov4_parser import YoloV4Parser
from .yolov6_parser import YoloV6Parser

logger = logging.getLogger(__name__)


class ParserType(Enum):
    DIR = "dir"
    SPLIT = "split"


class LuxonisParser:
    def __init__(
        self,
        dataset_dir: Union[str, Path],
        *,
        dataset_name: Optional[str] = None,
        delete_existing: bool = False,
        **kwargs,
    ):
        self.dataset_dir = Path(dataset_dir)
        if self.dataset_dir.suffix == ".zip":
            with zipfile.ZipFile(self.dataset_dir, "r") as zip_ref:
                unzip_dir = self.dataset_dir.parent / self.dataset_dir.stem
                logger.info(f"Extracting {self.dataset_dir.name} to {unzip_dir}")
                zip_ref.extractall(unzip_dir)
                self.dataset_dir = unzip_dir

        dataset_name = dataset_name or self.dataset_dir.name
        self.dataset_exists = LuxonisDataset.exists(dataset_name=dataset_name)
        if delete_existing and self.dataset_exists:
            logger.warning(f"Deleting existing dataset {dataset_name}")
            self.dataset = LuxonisDataset(dataset_name=dataset_name, **kwargs)
            self.dataset.delete_dataset()
            self.dataset_exists = False

        self.dataset = LuxonisDataset(dataset_name=dataset_name, **kwargs)
        self.parsers: Dict[DatasetType, Type[BaseParser]] = {
            DatasetType.COCO: COCOParser,
            DatasetType.VOC: VOCParser,
            DatasetType.DARKNET: DarknetParser,
            DatasetType.YOLOV6: YoloV6Parser,
            DatasetType.YOLOV4: YoloV4Parser,
            DatasetType.CREATEML: CreateMLParser,
            DatasetType.TFCSV: TensorflowCSVParser,
            DatasetType.CLSDIR: ClassificationDirectoryParser,
            DatasetType.SEGMASK: SegmentationMaskDirectoryParser,
        }
        self.dataset_type, self.parser_type = self._recognize_dataset()
        self.parser = self.parsers[self.dataset_type]()

    def _recognize_dataset(self) -> Tuple[DatasetType, ParserType]:
        for typ, parser in self.parsers.items():
            if parser.validate(self.dataset_dir):
                logger.info(f"Recognized entire dataset as {typ}")
                return typ, ParserType.DIR
            if parser.validate_split(self.dataset_dir):
                logger.info(f"Recognized dataset content as one split of '{typ}'")
                return typ, ParserType.SPLIT
        raise ValueError(
            f"Dataset {self.dataset_dir} is not in expected format for any of the parsers."
        )

    def parse(self, **kwargs) -> LuxonisDataset:
        if self.dataset_exists:
            logger.warning(
                "There already exists an LDF dataset with this name. "
                "Parsing will be skipped and existing dataset will be used instead."
            )
            return self.dataset
        if self.parser_type == ParserType.DIR:
            return self._parse_dir(**kwargs)
        else:
            return self._parse_split(**kwargs)

    def _parse_dir(self, **kwargs) -> LuxonisDataset:
        """Parses all present data in LuxonisDataset format. Check under selected parser
        function for expected directory structure.

        @type kwargs: Dict[str, Any]
        @param kwargs: Additional kwargs for specific parser function.
        @rtype: LuxonisDataset
        @return: Output LDF with all images and annotations parsed.
        """

        return self.parser.parse_dir(self.dataset, self.dataset_dir, **kwargs)

    def _parse_split(
        self,
        split: Optional[Literal["train", "val", "test"]] = None,
        random_split: bool = False,
        split_ratios: Optional[Tuple[float, float, float]] = None,
        **kwargs,
    ) -> LuxonisDataset:
        """Parses data in specific directory, should be used if adding/changing only
        specific split. Check under selected parser function for expected directory
        structure.

        @type dataset_type: DatasetType
        @param dataset_type: Source dataset type
        @type split: Optional[Literal["train", "val", "test"]]
        @param split: Split under which data will be added.
        @type random_split: bool
        @param random_split: If random splits should be made.
        @type split_ratios: Optional[Tuple[float, float, float]]
        @param split_ratios: Ratios for random splits.
        @type parser_kwargs: Dict[str, Any]
        @param parser_kwargs: Additional kwargs for specific parser function.
        @rtype: LuxonisDataset
        @return: Output LDF with all images and annotations parsed.
        """
        parsed_kwargs = self.parser.validate_split(self.dataset_dir)
        if parsed_kwargs is None:
            raise ValueError(
                f"Dataset {self.dataset_dir} is not in expected format for {self.dataset_type} parser."
            )
        kwargs = {**parsed_kwargs, **kwargs}
        return self.parser.parse_split(
            self.dataset, split, random_split, split_ratios, **kwargs
        )
