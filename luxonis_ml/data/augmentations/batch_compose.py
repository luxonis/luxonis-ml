from typing import Any, Dict, List, Optional

import numpy as np
from albumentations.core.bbox_utils import BboxParams
from albumentations.core.composition import BaseCompose, TransformsSeqType
from albumentations.core.keypoints_utils import KeypointParams
from albumentations.core.utils import get_shape
from typing_extensions import override

from .batch_processors import (
    BatchProcessor,
    BboxBatchProcessor,
    KeypointsBatchProcessor,
)
from .batch_transform import BatchBasedTransform
from .batch_utils import (
    batch_all,
    unbatch_all,
    yield_batches,
)


class BatchCompose(BaseCompose):
    def __init__(
        self,
        transforms: TransformsSeqType,
        bbox_params: Optional[BboxParams] = None,
        keypoint_params: Optional[KeypointParams] = None,
        additional_targets: Optional[Dict[str, str]] = None,
        p: float = 1.0,
    ):
        """Compose transforms and handle all transformations regarding
        bounding boxes.

        @param transforms: List of transformations to compose
        @type transforms: TransformsSeqType
        @param bbox_params: Parameters for bounding boxes transforms. Defaults to None.
        @type bbox_params: Optional[Union[dict, BboxParams]]
        @param keypoint_params: Parameters for keypoint transforms. Defaults to None.
        @type keypoint_params: Optional[Union[dict, KeypointParams]]
        @param additional_targets: Dict with keys - new target name, values - old target
        name. ex: {'image2': 'image'}. Defaults to None.
        @type additional_targets: Optional[Dict[str, str]]
        @param p: Probability of applying all list of transforms. Defaults to 1.0.
        @type p: float
        """
        super().__init__(transforms, p)

        self.processors: Dict[str, BatchProcessor] = {}

        self.batch_size = 1
        for transform in self.transforms:
            if isinstance(transform, BatchBasedTransform):
                self.batch_size *= transform.batch_size

        if bbox_params is not None:
            self.processors["bboxes"] = BboxBatchProcessor(
                bbox_params, additional_targets
            )

        if keypoint_params is not None:
            self.processors["keypoints"] = KeypointsBatchProcessor(
                keypoint_params, additional_targets
            )

        self._additional_targets = additional_targets or {}

        for proc in self.processors.values():
            proc.ensure_transforms_valid(self.transforms)

        self.add_targets(additional_targets)

    @override
    def __call__(self, batch_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(batch_data) != self.batch_size:
            raise ValueError(
                f"Batch size must be equal to {self.batch_size}, "
                f"but got {len(batch_data)}."
            )

        for p in self.processors.values():
            for data in batch_data:
                p.item_processor.ensure_data_valid(data)
                p.item_processor.preprocess(data)

        check_each_transform = any(
            getattr(proc.params, "check_each_transform", False)
            for proc in self.processors.values()
        )

        for transform in self.transforms:
            assert isinstance(transform, BatchBasedTransform)
            new_batch = []
            for data in yield_batches(batch_data, transform.batch_size):
                data = transform(**data, force_apply=False)

                if check_each_transform:
                    data = self._check_data_post_transform(data)
                new_batch.append(unbatch_all(data))
            batch_data = new_batch

        # TODO: why?
        self._make_targets_contiguous(data)

        data = unbatch_all(data)
        for p in self.processors.values():
            data = p.item_processor.postprocess(data)
        return data

    def _make_targets_contiguous(self, data: Dict[str, Any]):
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                value = np.ascontiguousarray(value)
            data[key] = value

    def _check_data_post_transform(
        self, data: Dict[str, Any]
    ) -> Dict[str, Any]:
        data = unbatch_all(data)
        shape = get_shape(data["image"])
        for p in self.processors.values():
            if not getattr(p.params, "check_each_transform", False):
                continue
            for data_name in p.item_processor.data_fields:
                data[data_name] = p.filter(data[data_name], shape)
        return batch_all(data)
