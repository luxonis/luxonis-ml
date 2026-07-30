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
        applied_params: Runtime parameters of the transformations that were
            applied during the latest call, keyed by transformation
            identity. A transformation is invoked once per sub-batch, so
            this accumulates across invocations instead of describing only
            the last one.

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
        self.batch_augmentation_indices = [0]
        self.applied_params: dict[int, dict[str, Any]] = {}
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

        input_indices = [[i] for i in range(len(data_batch))]
        self.applied_params = {}
        if not self.transforms:
            return data_batch[0]

        for data in data_batch:
            original_image_key = data.pop("_original_image_key", None)
            self.preprocess(data)

        for transform in self.transforms:
            new_batch = []
            new_indices = []
            for i, batch in enumerate(
                yield_batches(data_batch, transform.batch_size)
            ):
                data = transform(**batch)  # type: ignore
                batch_indices = input_indices[
                    i * transform.batch_size : (i + 1) * transform.batch_size
                ]

                # A transform that applied collapsed the batched lists
                # `yield_batches` produced into single values; one that did
                # not returns them untouched. Empty ``params`` does not tell
                # the two apart, as a transform can apply while sampling no
                # parameters at all.
                if isinstance(next(iter(data.values())), list):
                    data = {key: value[0] for key, value in batch.items()}
                    new_indices.append(batch_indices[0])
                else:
                    self.applied_params[id(transform)] = dict(transform.params)
                    new_indices.append(
                        [
                            index
                            for indices in batch_indices
                            for index in indices
                        ]
                    )

                data = self.check_data_post_transform(data)
                new_batch.append(data)
            data_batch = new_batch
            input_indices = new_indices

        assert len(data_batch) == 1
        self.batch_augmentation_indices = input_indices[0]
        data = data_batch[0]

        data = self._make_contiguous(data)

        data = self.postprocess(data)

        data["_original_image_key"] = original_image_key

        return data

    @staticmethod
    def _make_contiguous(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                value = np.ascontiguousarray(value)
            data[key] = value
        return data
