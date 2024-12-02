from typing import Any, Dict, Optional

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
    batch2list,
    batch_all,
    to_unbatched_name,
    unbatch_all,
)


class BatchCompose(BaseCompose):
    def __init__(
        self,
        transforms: TransformsSeqType,
        bbox_params: Optional[BboxParams] = None,
        keypoint_params: Optional[KeypointParams] = None,
        additional_targets: Optional[Dict[str, str]] = None,
        p: float = 1.0,
        is_check_shapes: bool = True,
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
        @param is_check_shapes: If True shapes consistency of images/mask/masks would be checked on
        each call. If you would like to disable this check - pass False (do it only if you are sure
        in your data consistency). Defaults to True.
        @type is_check_shapes: bool
        """
        super().__init__(transforms, p)

        self.processors: Dict[str, BatchProcessor] = {}

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

        self.is_check_args = True
        self._disable_check_args_for_transforms(self.transforms)

        self.is_check_shapes = is_check_shapes

    @override
    def __call__(self, *, force_apply: bool = False, **data) -> Dict[str, Any]:
        if self.is_check_args:
            self.check_args(self.additional_targets, **data)

        for p in self.processors.values():
            p.ensure_data_valid(data)

        check_each_transform = any(
            getattr(item.params, "check_each_transform", False)
            for item in self.processors.values()
        )

        for p in self.processors.values():
            p.preprocess(data)

        for transform in self.transforms:
            assert isinstance(transform, BatchBasedTransform)
            data = transform(**data)
            if check_each_transform:
                data = self._check_data_post_transform(data)

        # TODO: why?
        data = self._make_targets_contiguous(data)

        for p in self.processors.values():
            data = p.postprocess(data)
        return data

    def check_args(self, additional_targets, **kwargs) -> None:
        datalist = batch2list(kwargs)
        unbatched_targets = {
            to_unbatched_name(k): to_unbatched_name(v)
            for k, v in additional_targets.items()
        }
        for data in datalist:
            self._check_args(unbatched_targets, **data)

    def _check_args(self, additional_targets, **kwargs) -> None:
        checked_single = ["image", "mask"]
        checked_multi = ["masks"]
        check_bbox_param = ["bboxes"]
        shapes = []
        for data_name, data in kwargs.items():
            internal_data_name = additional_targets.get(data_name, data_name)
            if internal_data_name in checked_single:
                if not isinstance(data, np.ndarray):
                    raise TypeError(f"{data_name} must be numpy array type")
                shapes.append(data.shape[:2])
            if internal_data_name in checked_multi:
                if data is not None:
                    if not isinstance(data[0], np.ndarray):
                        raise TypeError(
                            f"{data_name} must be list of numpy arrays"
                        )
                    shapes.append(data[0].shape[:2])
            if (
                internal_data_name in check_bbox_param
                and self.processors.get("bboxes") is None
            ):
                raise ValueError(
                    "bbox_params must be specified for bbox transformations"
                )

        if (
            self.is_check_shapes
            and shapes
            and shapes.count(shapes[0]) != len(shapes)
        ):
            raise ValueError(
                "Height and Width of image, mask or masks should be equal. You can disable shapes check "
                "by setting a parameter is_check_shapes=False of Compose class (do it only if you are sure "
                "about your data consistency)."
            )

    def _disable_check_args_for_transforms(
        self,
        transforms: TransformsSeqType,
    ) -> None:
        for transform in transforms:
            if isinstance(transform, BaseCompose):
                self._disable_check_args_for_transforms(transform.transforms)
            elif isinstance(transform, BatchCompose):
                self.is_check_args = False

    def _make_targets_contiguous(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data = unbatch_all(data)
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                value = np.ascontiguousarray(value)
            data[key] = value
        return batch_all(data)

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
