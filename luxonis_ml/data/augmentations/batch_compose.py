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

    def __init__(
        self,
        transforms: TransformsSeqType,
        bbox_associations: dict[str, dict[str, str]] | None = None,
        **kwargs,
    ):
        """Compose batch transforms.

        Args:
            transforms: Transformations to compose.
            bbox_associations: Bbox fields mapped to their associated target
                fields and target types.
            **kwargs: Additional arguments passed to `A.Compose`_.

        .. _A.Compose:
            https://github.com/albumentations-team/albumentations/blob/66212d77a44927a29d6a0e81621d3c27afbd929c/albumentations/core/composition.py#L609

        """
        super().__init__(transforms, is_check_shapes=False, **kwargs)
        self._bbox_associations = bbox_associations or {}

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

                bbox_counts = self._reindex_bboxes(data)
                data = self.check_data_post_transform(data)
                self._compact_bbox_associated_labels(data, bbox_counts)
                self._reindex_bboxes(data)
                new_batch.append(data)
            data_batch = new_batch

        assert len(data_batch) == 1
        data = data_batch[0]

        data = self._make_contiguous(data)

        data = self.postprocess(data)

        data["_original_image_key"] = original_image_key

        return data

    def _reindex_bboxes(self, data: dict[str, np.ndarray]) -> dict[str, int]:
        """Give each bbox field contiguous indices and return its size.

        The indices are local positions into bbox-associated labels. They are
        assigned before bbox filtering so the surviving indices can be used to
        compact those labels in lockstep.
        """
        bbox_counts = {}
        bbox_processor = self.processors.get("bboxes")
        if bbox_processor is None:
            return bbox_counts

        for field_name in bbox_processor.data_fields:
            bboxes = data.get(field_name)
            if bboxes is None:
                continue
            bbox_counts[field_name] = len(bboxes)
            if bboxes.size == 0:
                continue
            if bboxes.ndim < 2:
                raise ValueError(
                    f"Bbox field '{field_name}' must be a 2D array."
                )
            bboxes[:, -1] = np.arange(len(bboxes), dtype=bboxes.dtype)
        return bbox_counts

    def _compact_bbox_associated_labels(
        self, data: dict[str, np.ndarray], bbox_counts: dict[str, int]
    ) -> None:
        """Drop labels associated with bboxes filtered from the current stage."""
        for bbox_field, associations in self._bbox_associations.items():
            bboxes = data.get(bbox_field)
            if bboxes is None:
                continue

            bbox_count = bbox_counts.get(bbox_field, 0)
            if bboxes.size == 0:
                indices = np.array([], dtype=int)
            elif bboxes.ndim < 2:
                raise ValueError(
                    f"Bbox field '{bbox_field}' must be a 2D array."
                )
            else:
                indices = bboxes[:, -1].astype(int)

            for field_name, target_type in associations.items():
                value = data.get(field_name)
                if value is None:
                    continue

                if bbox_count == 0:
                    if target_type == "instance_mask" and value.ndim > 1:
                        data[field_name] = value[..., :0]
                    else:
                        data[field_name] = value[:0]
                    continue

                if value.size == 0:
                    continue

                if target_type == "instance_mask":
                    if value.shape[-1] != bbox_count:
                        raise ValueError(
                            f"Instance-mask field '{field_name}' has "
                            f"{value.shape[-1]} instances for {bbox_count} "
                            f"bboxes in '{bbox_field}'."
                        )
                    data[field_name] = value[..., indices]
                elif target_type == "keypoints":
                    if value.shape[0] % bbox_count:
                        raise ValueError(
                            f"Keypoint field '{field_name}' has "
                            f"{value.shape[0]} rows, which cannot be grouped "
                            f"across {bbox_count} bboxes in '{bbox_field}'."
                        )
                    keypoints_per_bbox = value.shape[0] // bbox_count
                    grouped = value.reshape(
                        bbox_count, keypoints_per_bbox, *value.shape[1:]
                    )
                    data[field_name] = grouped[indices].reshape(
                        -1, *value.shape[1:]
                    )
                else:
                    if len(value) != bbox_count:
                        raise ValueError(
                            f"Field '{field_name}' has {len(value)} values for "
                            f"{bbox_count} bboxes in '{bbox_field}'."
                        )
                    data[field_name] = value[indices]

    @staticmethod
    def _make_contiguous(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                value = np.ascontiguousarray(value)
            data[key] = value
        return data
