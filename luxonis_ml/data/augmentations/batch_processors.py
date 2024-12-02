import copy
from typing import Dict, Optional, Tuple, Type

import numpy as np
from albumentations.core.bbox_utils import BboxParams, BboxProcessor
from albumentations.core.keypoints_utils import (
    KeypointParams,
    KeypointsProcessor,
)
from albumentations.core.utils import DataProcessor, Params
from typing_extensions import override

from .batch_utils import to_unbatched_name


class BatchProcessor(DataProcessor):
    def __init__(
        self,
        params: Params,
        processor: Type[DataProcessor],
        default_data_name: str,
        additional_targets: Optional[Dict[str, str]] = None,
    ):
        """Data processor class to process data in batches.

        @type params: Params
        @param params: Parameters
        @type additional_targets: Optional[Dict[str, str]]
        @param additional_targets: Additional targets of the transform.
            Defaults to None.
        """
        self._default_data_name = default_data_name

        super().__init__(params, additional_targets)

        item_params = copy.deepcopy(params)
        if item_params.label_fields is not None:
            label_fields = item_params.label_fields
            item_params.label_fields = [
                to_unbatched_name(field) for field in label_fields
            ]
        self.item_processor = processor(item_params, additional_targets)

    @property
    @override
    def default_data_name(self) -> str:
        return self._default_data_name

    @override
    def filter(
        self, data: np.ndarray, image_shape: Tuple[int, int]
    ) -> np.ndarray:
        return self.item_processor.filter(data, image_shape)

    @override
    def check(self, data: np.ndarray, image_shape: Tuple[int, int]) -> None:
        return self.item_processor.check(data, image_shape)

    @override
    def convert_to_albumentations(
        self, data: np.ndarray, image_shape: Tuple[int, int]
    ) -> np.ndarray:
        return self.item_processor.convert_to_albumentations(data, image_shape)

    @override
    def convert_from_albumentations(
        self, data: np.ndarray, image_shape: Tuple[int, int]
    ) -> np.ndarray:
        return self.item_processor.convert_from_albumentations(
            data, image_shape
        )


class BboxBatchProcessor(BatchProcessor):
    def __init__(
        self,
        params: BboxParams,
        additional_targets: Optional[Dict[str, str]] = None,
    ):
        super().__init__(
            params, BboxProcessor, "bboxes_batch", additional_targets
        )


class KeypointsBatchProcessor(BatchProcessor):
    def __init__(
        self,
        params: KeypointParams,
        additional_targets: Optional[Dict[str, str]] = None,
    ):
        super().__init__(
            params, KeypointsProcessor, "keypoints_batch", additional_targets
        )
