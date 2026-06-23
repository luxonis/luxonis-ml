import random
from typing import Any

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
    def __call__(self, data_batch: list[dict[str, Any]]) -> dict[str, Any]:
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
            data = data_batch[0]
            data["_luxonis_contributor_indices"] = [0]
            return data

        original_image_key = data_batch[0].pop("_original_image_key", None)
        for data in data_batch[1:]:
            data.pop("_original_image_key", None)

        contributor_batches = [[i] for i in range(len(data_batch))]

        for data in data_batch:
            self.preprocess(data)

        for transform in self.transforms:
            new_batch = []
            new_contributor_batches = []
            for batch_idx, batch in enumerate(
                yield_batches(data_batch, transform.batch_size)
            ):
                i = batch_idx * transform.batch_size
                contributor_batch = contributor_batches[
                    i : i + transform.batch_size
                ]
                data = transform(**batch)  # type: ignore

                if isinstance(next(iter(data.values())), list):
                    data = {key: value[0] for key, value in batch.items()}
                    contributor_indices = contributor_batch[0]
                else:
                    contributor_indices = [
                        index
                        for contributors in contributor_batch
                        for index in contributors
                    ]

                data = self.check_data_post_transform(data)
                new_batch.append(data)
                new_contributor_batches.append(contributor_indices)
            data_batch = new_batch
            contributor_batches = new_contributor_batches

        assert len(data_batch) == 1
        data = data_batch[0]

        data = self._make_contiguous(data)

        data = self.postprocess(data)

        data["_original_image_key"] = original_image_key
        data["_luxonis_contributor_indices"] = contributor_batches[0]

        return data

    @staticmethod
    def _make_contiguous(data: dict[str, Any]) -> dict[str, Any]:
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                value = np.ascontiguousarray(value)
            data[key] = value
        return data
