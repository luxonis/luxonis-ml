import random
from typing import Any, Dict, List, Optional, Union, cast

import numpy as np
from albumentations.core.bbox_utils import (
    BboxParams,
    BboxProcessor,
    DataProcessor,
)
from albumentations.core.composition import (
    BaseCompose,
    TransformsSeqType,
    get_always_apply,
)
from albumentations.core.keypoints_utils import KeypointParams, KeypointsProcessor
from albumentations.core.utils import get_shape

from .batch_processors import BboxBatchProcessor, KeypointsBatchProcessor
from .batch_utils import batch2list, concat_batches, list2batch, to_unbatched_name


class Compose(BaseCompose):
    def __init__(
        self,
        transforms: TransformsSeqType,
        bbox_params: Optional[Union[dict, BboxParams]] = None,
        keypoint_params: Optional[Union[dict, KeypointParams]] = None,
        additional_targets: Optional[Dict[str, str]] = None,
        p: float = 1.0,
        is_check_shapes: bool = True,
    ):
        """Compose transforms and handle all transformations regarding bounding boxes.

        @param transforms: List of transformations to compose
        @type transforms: TransformsSeqType
        @param bboxparams: Parameters for bounding boxes transforms. Defaults to None.
        @type bboxparams: Optional[Union[dict, BboxParams]]
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
        super(Compose, self).__init__(transforms, p)

        self.processors: Dict[str, DataProcessor] = {}

        if bbox_params:
            if isinstance(bbox_params, dict):
                b_params = BboxParams(**bbox_params)
            elif isinstance(bbox_params, BboxParams):
                b_params = bbox_params
            else:
                raise ValueError(
                    "unknown format of bbox_params, please use `dict` or `BboxParams`"
                )
            self.processors["bboxes"] = self._get_bbox_processor(
                b_params, additional_targets
            )

        if keypoint_params:
            if isinstance(keypoint_params, dict):
                k_params = KeypointParams(**keypoint_params)
            elif isinstance(keypoint_params, KeypointParams):
                k_params = keypoint_params
            else:
                raise ValueError(
                    "unknown format of keypoint_params, please use `dict` or `KeypointParams`"
                )
            self.processors["keypoints"] = self._get_keypoints_processor(
                k_params, additional_targets
            )

        if additional_targets is None:
            additional_targets = {}

        self.additional_targets = additional_targets

        for proc in self.processors.values():
            proc.ensure_transforms_valid(self.transforms)

        self.add_targets(additional_targets)

        self.is_check_args = True
        self._disable_check_args_for_transforms(self.transforms)

        self.is_check_shapes = is_check_shapes

    def _get_bbox_processor(self, b_params, additional_targets):
        return BboxProcessor(b_params, additional_targets)

    def _get_keypoints_processor(self, k_params, additional_targets):
        return KeypointsProcessor(k_params, additional_targets)

    @staticmethod
    def _disable_check_args_for_transforms(transforms: TransformsSeqType) -> None:
        for transform in transforms:
            if isinstance(transform, BaseCompose):
                Compose._disable_check_args_for_transforms(transform.transforms)
            if isinstance(transform, Compose):
                transform._disable_check_args()

    def _disable_check_args(self) -> None:
        self.is_check_args = False

    def __call__(self, *args, force_apply: bool = False, **data) -> Dict[str, Any]:
        if args:
            raise KeyError(
                "You have to pass data to augmentations as named arguments, for example: aug(image=image)"
            )
        if self.is_check_args:
            self._check_args(self.additional_targets, **data)
        assert isinstance(
            force_apply, (bool, int)
        ), "force_apply must have bool or int type"
        need_to_run = force_apply or random.random() < self.p
        for p in self.processors.values():
            p.ensure_data_valid(data)
        transforms = (
            self.transforms if need_to_run else get_always_apply(self.transforms)
        )

        check_each_transform = any(
            getattr(item.params, "check_each_transform", False)
            for item in self.processors.values()
        )

        for p in self.processors.values():
            p.preprocess(data)

        for _, t in enumerate(transforms):
            data = t(**data)
            if check_each_transform:
                data = self._check_data_post_transform(data)

        data = self._make_targets_contiguous(
            data
        )  # ensure output targets are contiguous

        for p in self.processors.values():
            p.postprocess(data)

        return data

    def _check_data_post_transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        rows, cols = get_shape(data["image"])

        for p in self.processors.values():
            if not getattr(p.params, "check_each_transform", False):
                continue

            for data_name in p.data_fields:
                data[data_name] = p.filter(data[data_name], rows, cols)
        return data

    def _to_dict(self) -> Dict[str, Any]:
        dictionary = super(Compose, self)._to_dict()
        bbox_processor = self.processors.get("bboxes")
        keypoints_processor = self.processors.get("keypoints")
        dictionary.update(
            {
                "bbox_params": bbox_processor.params._to_dict()
                if bbox_processor
                else None,  # skipcq: PYL-W0212
                "keypoint_params": keypoints_processor.params._to_dict()  # skipcq: PYL-W0212
                if keypoints_processor
                else None,
                "additional_targets": self.additional_targets,
                "is_check_shapes": self.is_check_shapes,
            }
        )
        return dictionary

    def get_dict_with_id(self) -> Dict[str, Any]:
        dictionary = super().get_dict_with_id()
        bbox_processor = self.processors.get("bboxes")
        keypoints_processor = self.processors.get("keypoints")
        dictionary.update(
            {
                "bbox_params": bbox_processor.params._to_dict()
                if bbox_processor
                else None,  # skipcq: PYL-W0212
                "keypoint_params": keypoints_processor.params._to_dict()  # skipcq: PYL-W0212
                if keypoints_processor
                else None,
                "additional_targets": self.additional_targets,
                "params": None,
                "is_check_shapes": self.is_check_shapes,
            }
        )
        return dictionary

    def _check_args(self, additional_targets, **kwargs) -> None:
        checked_single = ["image", "mask"]
        checked_multi = ["masks"]
        check_bbox_param = ["bboxes"]
        # ["bboxes", "keypoints"] could be almost any type, no need to check them
        shapes = []
        for data_name, data in kwargs.items():
            internal_data_name = additional_targets.get(data_name, data_name)
            if internal_data_name in checked_single:
                if not isinstance(data, np.ndarray):
                    raise TypeError("{} must be numpy array type".format(data_name))
                shapes.append(data.shape[:2])
            if internal_data_name in checked_multi:
                if data is not None:
                    if not isinstance(data[0], np.ndarray):
                        raise TypeError(
                            "{} must be list of numpy arrays".format(data_name)
                        )
                    shapes.append(data[0].shape[:2])
            if (
                internal_data_name in check_bbox_param
                and self.processors.get("bboxes") is None
            ):
                raise ValueError(
                    "bbox_params must be specified for bbox transformations"
                )

        if self.is_check_shapes and shapes and shapes.count(shapes[0]) != len(shapes):
            raise ValueError(
                "Height and Width of image, mask or masks should be equal. You can disable shapes check "
                "by setting a parameter is_check_shapes=False of Compose class (do it only if you are sure "
                "about your data consistency)."
            )

    def _make_targets_contiguous(self, data: Dict[str, Any]) -> Dict[str, Any]:
        result = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                value = np.ascontiguousarray(value)
            result[key] = value
        return result


class BatchCompose(Compose):
    def __init__(
        self,
        transforms: TransformsSeqType,
        bbox_params: Optional[Union[dict, BboxParams]] = None,
        keypoint_params: Optional[Union[dict, KeypointParams]] = None,
        additional_targets: Optional[Dict[str, str]] = None,
        p: float = 1.0,
        is_check_shapes: bool = True,
    ):
        """Compose designed to handle the multi-image transforms The contents can be a
        subclass of `BatchBasedTransform` or other transforms enclosed by ForEach
        container. All targets' names should have the suffix "_batch", ex
        ("image_batch", "bboxes_batch"). Note this nameing rule is applied to the
        `label_fields` of the `BboxParams` and the `KeypointsParams`.

        @param transforms: List of transformations to compose
        @type transforms: TransformsSeqType
        @param bboxparams: Parameters for bounding boxes transforms. Defaults to None.
        @type bboxparams: Optional[Union[dict, BboxParams]]
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
        super(BatchCompose, self).__init__(
            transforms=transforms,
            bbox_params=bbox_params,
            keypoint_params=keypoint_params,
            additional_targets=additional_targets,
            p=p,
            is_check_shapes=is_check_shapes,
        )

    def _get_bbox_processor(self, b_params, additional_targets):
        return BboxBatchProcessor(b_params, additional_targets)

    def _get_keypoints_processor(self, k_params, additional_targets):
        return KeypointsBatchProcessor(k_params, additional_targets)

    def _check_data_post_transform(
        self, batched_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        datalist = batch2list(batched_data)
        processed = []
        for data in datalist:
            rows, cols = get_shape(data["image"])
            for p in self.processors.values():
                if not getattr(p.params, "check_each_transform", False):
                    continue
                p = cast(Union[BboxBatchProcessor, KeypointsBatchProcessor], p)
                for data_name in p.item_processor.data_fields:
                    data[data_name] = p.filter(data[data_name], rows, cols)
            processed.append(data)
        return list2batch(processed)

    def _check_args(self, additional_targets, **kwargs) -> None:
        datalist = batch2list(kwargs)
        unbatched_targets = {
            to_unbatched_name(k): to_unbatched_name(v)
            for k, v in additional_targets.items()
        }
        for data in datalist:
            super(BatchCompose, self)._check_args(unbatched_targets, **data)

    def _make_targets_contiguous(self, batched_data: Dict[str, Any]) -> Dict[str, Any]:
        datalist = batch2list(batched_data)
        if len(datalist) == 0:
            return batched_data
        processed = []
        for data in datalist:
            data = super(BatchCompose, self)._make_targets_contiguous(data)
            processed.append(data)
        return list2batch(processed)


class ForEach(BaseCompose):
    """Apply transforms for each batch element This expects batched input and can be
    contained by the `BatchCompose`."""

    def __init__(self, transforms: TransformsSeqType, p: float = 0.5):
        super().__init__(transforms, p)

    def __call__(
        self, *args, force_apply: bool = False, **batched_data
    ) -> Dict[str, List]:
        datalist = batch2list(batched_data)
        processed = []
        for data in datalist:
            for t in self.transforms:
                data = t(force_apply=force_apply, **data)
            processed.append(data)
        batched_data = list2batch(processed)
        return batched_data

    def add_targets(self, additional_targets: Optional[Dict[str, str]]) -> None:
        if additional_targets:
            unbatched_targets = {
                to_unbatched_name(k): to_unbatched_name(v)
                for k, v in additional_targets.items()
            }
            for t in self.transforms:
                t.add_targets(unbatched_targets)


class Repeat(BaseCompose):
    """Apply transforms repeatedly and concatenates the output batches.

    This expects batched input and can be contained by the `BatchCompose`.
    The contained transforms should be a subbclass of the `BatchBasedTransform`.
    Internally, this container works as the following way:
    Note: This class assumes that each transform does not modify the input data.
    """

    def __init__(self, transforms: TransformsSeqType, n: int, p: float = 0.5):
        super().__init__(transforms, p)
        if n <= 0:
            raise ValueError("Repetition `n` should be larger than 0")
        self.n = n

    def __call__(
        self, *args, force_apply: bool = False, **batched_data
    ) -> Dict[str, List]:
        processed = []
        for _ in range(self.n):
            image = batched_data["image_batch"][0].copy()
            data = batched_data
            for t in self.transforms:
                data = t(force_apply=force_apply, **data)
            processed.append(data)
            assert np.all(batched_data["image_batch"][0] == image)
        return concat_batches(processed)
