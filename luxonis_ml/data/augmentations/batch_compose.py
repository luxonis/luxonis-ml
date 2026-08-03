import random
from typing import Any

import albumentations as A
import numpy as np
from albumentations.core.composition import TransformsSeqType
from loguru import logger
from typing_extensions import override

from .batch_transform import BatchTransform
from .utils import instance_count, yield_batches


class BatchCompose(A.Compose):
    r"""Compose batch-aware Albumentations transforms.

    Attributes:
        transforms: Batch transformations in composition order.
        batch_size: Product of nested transform batch sizes,
            :math:`\prod_i b_i`.
        applied_params: Runtime parameters of the transformations that
            shaped the sample returned by the latest call, keyed by
            transformation identity. Invocations on sub-batches that did
            not survive into that sample are not reported; of those that
            did, the last one wins.

    """

    transforms: list[BatchTransform]

    def __init__(
        self,
        transforms: TransformsSeqType,
        *,
        bbox_associations: dict[str, dict[str, str]] | None = None,
        **kwargs,
    ):
        """Compose batch transforms.

        Args:
            transforms: Transformations to compose.
            bbox_associations: Bbox fields mapped to their associated target
                fields and target types. Fields listed here must carry the
                index column appended by
                `luxonis_ml.data.augmentations.utils.preprocess_bboxes`;
                any other bbox field is left untouched.
            **kwargs: Additional arguments passed to `A.Compose`_.

        .. _A.Compose:
            https://github.com/albumentations-team/albumentations/blob/66212d77a44927a29d6a0e81621d3c27afbd929c/albumentations/core/composition.py#L609

        """
        super().__init__(transforms, is_check_shapes=False, **kwargs)
        self._bbox_associations = bbox_associations or {}
        self._mismatched_fields: set[str] = set()
        self._index_column = self._locate_index_column()

        random.seed(self.seed)
        np.random.seed(self.seed)

        self.batch_size = 1
        self.batch_augmentation_indices = [0]
        self.applied_params: dict[int, dict[str, Any]] = {}
        for transform in self.transforms:
            self.batch_size *= transform.batch_size

    @override
    def __call__(
        self,
        data_batch: list[dict[str, np.ndarray]],
        *,
        keypoints_per_instance: dict[str, int] | None = None,
    ) -> dict[str, np.ndarray]:
        """Apply the composed transforms to a batch.

        Args:
            data_batch: Batch of Albumentations data dictionaries. Its
                length must equal ``batch_size``.
            keypoints_per_instance: Number of keypoints each instance of a
                keypoint field carries, used to regroup keypoint rows by
                instance. Fields missing from this mapping are left
                untouched.

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

        keypoints_per_instance = keypoints_per_instance or {}
        self._mismatched_fields = set()

        input_indices = [[i] for i in range(len(data_batch))]
        self.applied_params = {}
        # Every invocation that applied, with the inputs it merged. Which of
        # them shaped the returned sample is only known at the very end.
        invocations: list[tuple[int, list[int], dict[str, Any]]] = []
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
                # Captured before padding so that a transform which does not
                # fire returns the first sample exactly as it came in.
                first_sample = {key: value[0] for key, value in batch.items()}
                self._pad_absent_instances(batch, keypoints_per_instance)

                data = transform(**batch)  # type: ignore
                batch_indices = input_indices[
                    i * transform.batch_size : (i + 1) * transform.batch_size
                ]

                # A transform that applied collapsed the batched lists
                # `yield_batches` produced into single values. Empty
                # ``params`` does not tell the two apart, as a transform can
                # apply while sampling no parameters at all.
                if isinstance(next(iter(data.values())), list):
                    data = first_sample
                    new_indices.append(batch_indices[0])
                else:
                    merged_indices = [
                        index for indices in batch_indices for index in indices
                    ]
                    invocations.append(
                        (id(transform), merged_indices, dict(transform.params))
                    )
                    new_indices.append(merged_indices)

                bbox_counts = self._reindex_bboxes(data)
                data = self.check_data_post_transform(data)
                self._compact_bbox_associated_labels(
                    data, bbox_counts, keypoints_per_instance
                )
                # Filtering left gaps in the index column; close them so the
                # next transform starts from contiguous indices again.
                self._reindex_bboxes(data)
                new_batch.append(data)
            data_batch = new_batch
            input_indices = new_indices

        assert len(data_batch) == 1
        self.batch_augmentation_indices = input_indices[0]

        # A sub-batch a transform applied to can still be dropped by a later
        # one that did not apply and passed its first input through instead.
        surviving = set(self.batch_augmentation_indices)
        self.applied_params = {
            key: params
            for key, indices, params in invocations
            if not surviving.isdisjoint(indices)
        }

        data = data_batch[0]

        data = self._make_contiguous(data)

        data = self.postprocess(data)

        data["_original_image_key"] = original_image_key

        return data

    def _locate_index_column(self) -> int:
        """Position of the instance-index column, counted from the right.

        `luxonis_ml.data.augmentations.utils.preprocess_bboxes` appends the
        index as the last column, but the wrapped bbox processor appends one
        more column for every configured label field, pushing the index left.
        """
        processor = self.processors.get("bboxes")
        if processor is None:
            return -1
        return -1 - len(processor.params.label_fields or [])

    def _pad_absent_instances(
        self,
        batch: dict[str, list[np.ndarray]],
        keypoints_per_instance: dict[str, int],
    ) -> None:
        """Give samples missing an associated label one empty instance per box.

        Batch transforms concatenate only the samples that carry a field, so
        in a partially annotated batch the merged labels would come back
        shorter than the merged boxes with no way to tell which boxes they
        belong to. Filling the gaps keeps the two one-to-one.
        """
        image_shapes = [image.shape[:2] for image in batch.get("image", [])]

        for bbox_field, associations in self._bbox_associations.items():
            if bbox_field not in batch:
                continue

            bbox_counts = [len(bboxes) for bboxes in batch[bbox_field]]

            for field_name, target_type in associations.items():
                if field_name not in batch:
                    continue
                self._fill_empty_entries(
                    batch[field_name],
                    target_type,
                    bbox_counts,
                    image_shapes,
                    keypoints_per_instance.get(field_name, 0),
                )

    @staticmethod
    def _fill_empty_entries(
        values: list[np.ndarray],
        target_type: str,
        bbox_counts: list[int],
        image_shapes: list[tuple[int, ...]],
        n_keypoints: int,
    ) -> None:
        """Replace each empty entry with zeroed instances, one per box.

        The layout is copied from a populated entry, so a field no sample in
        the batch carries is left empty rather than invented.
        """
        template = next((value for value in values if value.size), None)
        if template is None:
            return
        if target_type == "keypoints" and n_keypoints < 1:
            return

        for i, value in enumerate(values):
            if value.size or not bbox_counts[i]:
                continue

            if target_type == "instance_mask":
                if i >= len(image_shapes):
                    continue
                shape = (*image_shapes[i], bbox_counts[i])
            elif target_type == "keypoints":
                shape = (bbox_counts[i] * n_keypoints, *template.shape[1:])
            else:
                shape = (bbox_counts[i], *template.shape[1:])

            values[i] = np.zeros(shape, dtype=template.dtype)

    def _reindex_bboxes(self, data: dict[str, np.ndarray]) -> dict[str, int]:
        """Give each bbox field contiguous indices and return its size.

        Only fields declared in ``bbox_associations`` are touched; for any
        other bbox field the last column is the class label. The indices are
        positions into the bbox-associated labels, stamped before filtering so
        that the survivors can be used to compact those labels in lockstep.
        """
        bbox_counts = {}

        for field_name in self._bbox_associations:
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
            # The caller's array may be read-only, and is not ours to modify.
            bboxes = bboxes.copy()
            bboxes[:, self._index_column] = np.arange(
                len(bboxes), dtype=bboxes.dtype
            )
            data[field_name] = bboxes
        return bbox_counts

    def _compact_bbox_associated_labels(
        self,
        data: dict[str, np.ndarray],
        bbox_counts: dict[str, int],
        keypoints_per_instance: dict[str, int],
    ) -> None:
        """Drop labels associated with bboxes filtered from the current stage.

        A field whose instance count disagrees with the bbox count cannot be
        matched up instance by instance, so it is left untouched rather than
        dropped or regrouped on a guess. The counts are compared even when
        nothing was filtered, because a transform that drops boxes itself
        leaves the surviving indices looking untouched.
        """
        for bbox_field, associations in self._bbox_associations.items():
            if bbox_field not in bbox_counts:
                continue

            bbox_count = bbox_counts[bbox_field]
            bboxes = data[bbox_field]
            indices = (
                bboxes[:, self._index_column].astype(int)
                if bboxes.size
                else np.array([], dtype=int)
            )
            unfiltered = bool(
                bbox_count and np.array_equal(indices, np.arange(bbox_count))
            )

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

                # Keypoint rows can only be counted by instance once it is
                # known how many keypoints each instance carries.
                n_keypoints = keypoints_per_instance.get(field_name, 0)
                countable = target_type != "keypoints" or n_keypoints > 0

                if not countable or (
                    instance_count(value, target_type, n_keypoints)
                    != bbox_count
                ):
                    self._warn_unmatched(
                        field_name, value, bbox_count, bbox_field
                    )
                elif not unfiltered:
                    data[field_name] = self._select_instances(
                        value, target_type, indices, bbox_count, n_keypoints
                    )

    @staticmethod
    def _select_instances(
        value: np.ndarray,
        target_type: str,
        indices: np.ndarray,
        bbox_count: int,
        n_keypoints: int,
    ) -> np.ndarray:
        """Keep only the instances at ``indices``, in that order."""
        if target_type == "instance_mask":
            # Instance masks are (H, W, N) at this point.
            return value[..., indices]

        if target_type == "keypoints":
            grouped = value.reshape(bbox_count, n_keypoints, *value.shape[1:])
            return grouped[indices].reshape(-1, *value.shape[1:])

        return value[indices]

    def _warn_unmatched(
        self,
        field_name: str,
        value: np.ndarray,
        bbox_count: int,
        bbox_field: str,
    ) -> None:
        """Warn once per field per sample that a field could not be matched."""
        if field_name in self._mismatched_fields:
            return
        self._mismatched_fields.add(field_name)
        logger.warning(
            f"Field '{field_name}' with shape {value.shape} cannot be matched "
            f"to the {bbox_count} bounding boxes in '{bbox_field}'; leaving "
            f"it as is. It may end up misaligned with the boxes that survive "
            f"augmentation."
        )

    @staticmethod
    def _make_contiguous(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                value = np.ascontiguousarray(value)
            data[key] = value
        return data
