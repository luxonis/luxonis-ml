import copy
from typing import Any, Dict, Optional, Sequence

from albumentations.core.bbox_utils import BboxParams, BboxProcessor
from albumentations.core.keypoints_utils import KeypointParams, KeypointsProcessor
from albumentations.core.utils import DataProcessor

from .batch_utils import batch2list, list2batch, to_unbatched_name


class BboxBatchProcessor(DataProcessor):
    def __init__(
        self, params: BboxParams, additional_targets: Optional[Dict[str, str]] = None
    ):
        """Data processor class to process bbox data in batches.

        @param params: Bbox parameters
        @type params: BboxParams
        @param additional_targets: Additional targets of the transform. Defaults to
            None.
        @type additional_targets: Optional[Dict[str, str]]
        """
        super().__init__(params, additional_targets)
        item_params = copy.deepcopy(params)
        if item_params.label_fields is not None:
            label_fields = item_params.label_fields
            item_params.label_fields = [
                to_unbatched_name(field) for field in label_fields
            ]
        self.item_processor = BboxProcessor(item_params, additional_targets)

    @property
    def default_data_name(self) -> str:
        return "bboxes_batch"

    def ensure_data_valid(self, data: Dict[str, Any]) -> None:
        for item in batch2list(data):
            self.item_processor.ensure_data_valid(item)

    def postprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        processed = [self.item_processor.postprocess(item) for item in batch2list(data)]
        procesed_data = list2batch(processed)
        for k in data.keys():
            data[k] = procesed_data[k]
        return data

    def preprocess(self, data: Dict[str, Any]) -> None:
        processed = batch2list(data)
        for item in processed:
            self.item_processor.preprocess(item)
        procesed_data = list2batch(processed)
        for k in data.keys():
            data[k] = procesed_data[k]

    def filter_batch(self, batched_data: Dict[str, Any]) -> Dict[str, Any]:
        processed = []
        for data in batch2list(batched_data):
            rows, cols = data["image"][:2]
            for data_name in self.item_processor.data_fields:
                data[data_name] = self.item_processor.filter(
                    data[data_name], rows, cols
                )
            processed.append(data)
        return list2batch(processed)

    def filter(self, data: Sequence, rows: int, cols: int) -> Sequence:
        return self.item_processor.filter(data, rows, cols)

    def check(self, data: Sequence, rows: int, cols: int) -> None:
        return self.item_processor.check(data, rows, cols)

    def convert_to_albumentations(
        self, data: Sequence, rows: int, cols: int
    ) -> Sequence:
        return self.item_processor.convert_to_albumentations(data, rows, cols)

    def convert_from_albumentations(
        self, data: Sequence, rows: int, cols: int
    ) -> Sequence:
        return self.item_processor.convert_from_albumentations(data, rows, cols)


class KeypointsBatchProcessor(DataProcessor):
    def __init__(
        self,
        params: KeypointParams,
        additional_targets: Optional[Dict[str, str]] = None,
    ):
        """Data processor class to process keypoint data in batches.

        @param params: Keypoint parameters
        @type params: KeypointParams
        @param additional_targets: Additional targets of the transform. Defaults to
            None.
        @type additional_targets: Optional[Dict[str, str]]
        """
        super().__init__(params, additional_targets)
        item_params = copy.deepcopy(params)
        if item_params.label_fields is not None:
            label_fields = item_params.label_fields
            item_params.label_fields = [
                to_unbatched_name(field) for field in label_fields
            ]
        self.item_processor = KeypointsProcessor(item_params, additional_targets)

    @property
    def default_data_name(self) -> str:
        return "keypoints_batch"

    def ensure_data_valid(self, data: Dict[str, Any]) -> None:
        for item in batch2list(data):
            self.item_processor.ensure_data_valid(item)

    def postprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        processed = [self.item_processor.postprocess(item) for item in batch2list(data)]
        procesed_data = list2batch(processed)
        for k in data.keys():
            data[k] = procesed_data[k]
        return data

    def preprocess(self, data: Dict[str, Any]) -> None:
        processed = batch2list(data)
        for item in processed:
            self.item_processor.preprocess(item)
        procesed_data = list2batch(processed)
        for k in data.keys():
            data[k] = procesed_data[k]

    def filter_batch(self, batched_data: Dict[str, Any]) -> Dict[str, Any]:
        processed = []
        for data in batch2list(batched_data):
            rows, cols = data["image"][:2]
            for data_name in self.item_processor.data_fields:
                data[data_name] = self.item_processor.filter(
                    data[data_name], rows, cols
                )
            processed.append(data)
        return list2batch(processed)

    def filter(self, data: Sequence, rows: int, cols: int) -> Sequence:
        return self.item_processor.filter(data, rows, cols)

    def check(self, data: Sequence, rows: int, cols: int) -> None:
        return self.item_processor.check(data, rows, cols)

    def convert_to_albumentations(
        self, data: Sequence, rows: int, cols: int
    ) -> Sequence:
        return self.item_processor.convert_to_albumentations(data, rows, cols)

    def convert_from_albumentations(
        self, data: Sequence, rows: int, cols: int
    ) -> Sequence:
        return self.item_processor.convert_from_albumentations(data, rows, cols)
