import random

import albumentations as A
import numpy as np
from albumentations.core.composition import TransformsSeqType
from typing_extensions import override

from .batch_transform import BatchTransform
from .utils import yield_batches


class BatchCompose(A.Compose):
    r"""Compose batch-aware Albumentations transforms.

    Attributes:
        transforms: Batch transformations in composition order.
        batch_size: Product of nested transform batch sizes,
            :math:`\prod_i b_i`.

    """

    transforms: list[BatchTransform]

    def __init__(self, transforms: TransformsSeqType, **kwargs):
        """Compose batch transforms.

        Args:
            transforms: Transformations to compose.
            **kwargs: Additional arguments passed to `A.Compose`_.

        .. _A.Compose:
            https://github.com/albumentations-team/albumentations/blob/66212d77a44927a29d6a0e81621d3c27afbd929c/albumentations/core/composition.py#L609

        """
        super().__init__(transforms, is_check_shapes=False, **kwargs)

        random.seed(self.seed)
        np.random.seed(self.seed)

        self.batch_size = 1
        for transform in self.transforms:
            self.batch_size *= transform.batch_size

    @override
    def __call__(
        self, data_batch: list[dict[str, np.ndarray]]
    ) -> dict[str, np.ndarray]:
        """Apply the composed transforms to a batch.

        Args:
            data_batch: Batch of Albumentations data dictionaries. Its
                length must equal ``batch_size``.

        Returns:
            Single transformed data dictionary.

        Raises:
            ValueError: If ``len(data_batch)`` does not match
                ``batch_size``.

        """
        if len(data_batch) != self.batch_size:
            raise ValueError(
                f"Batch size must be equal to {self.batch_size}, "
                f"but got {len(data_batch)}."
            )

        if not self.transforms:
            return data_batch[0]

        for data in data_batch:
            original_image_key = data.pop("_original_image_key", None)
            self.preprocess(data)

        for transform in self.transforms:
            new_batch = []
            for batch in yield_batches(data_batch, transform.batch_size):
                data = transform(**batch)  # type: ignore

                if isinstance(next(iter(data.values())), list):
                    data = {key: value[0] for key, value in batch.items()}

                self._reindex_bboxes(data)
                data = self.check_data_post_transform(data)
                new_batch.append(data)
            data_batch = new_batch

        assert len(data_batch) == 1
        data = data_batch[0]

        data = self._make_contiguous(data)

        data = self.postprocess(data)

        data["_original_image_key"] = original_image_key

        return data

    def _reindex_bboxes(self, data: dict[str, np.ndarray]) -> None:
        """Reindex boxes to match their associated batched labels.

        Batch transforms can reduce several input samples to one output. In
        particular, when a transform is skipped, the retained sample can keep
        bbox indices assigned for its position in the original input batch.
        Instance masks and other bbox-associated labels are compacted with the
        output, so subsequent batch transforms require contiguous indices.

        The last bbox column is the stable index that ``_postprocess`` uses to
        pair each surviving box with its instance mask channel (and keypoints,
        arrays, metadata). After a batch transform those labels are
        concatenated in the same order as the boxes, so position ``i`` in the
        box array lines up with channel ``i`` in the concatenated labels.
        Assigning ``arange`` restores that alignment only while the two are
        still in lockstep, so this must run on the *complete* set of boxes,
        before ``check_data_post_transform`` drops any of them. Reindexing
        after a drop would renumber the survivors ``0..M-1`` while the label
        channels stay un-dropped, silently pairing boxes with the wrong
        instances.
        """
        for field_name in self.processors["bboxes"].data_fields:
            bboxes = data.get(field_name)
            if bboxes is None or bboxes.size == 0:
                continue
            bboxes[:, -1] = np.arange(len(bboxes), dtype=bboxes.dtype)

    @staticmethod
    def _make_contiguous(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                value = np.ascontiguousarray(value)
            data[key] = value
        return data
