"""Composition that runs `BatchTransform` steps on a batch of samples.

`BatchCompose` merges the batch into one sample and drops the labels
of bounding boxes filtered out after each transform.
"""

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
    ) -> None:
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
            self.preprocess(data)

        for transform in self.transforms:
            new_batch = []
            new_indices = []
            for i, batch in enumerate(
                yield_batches(data_batch, transform.batch_size)
            ):
                batch_indices = input_indices[
                    i * transform.batch_size : (i + 1) * transform.batch_size
                ]
                data, indices, applied = self._apply_once(
                    transform, batch, batch_indices, keypoints_per_instance
                )
                if applied:
                    invocations.append(
                        (id(transform), indices, dict(transform.params))
                    )
                new_batch.append(data)
                new_indices.append(indices)
            data_batch = new_batch
            input_indices = new_indices

        # The batch size is the product of the transform batch sizes, so
        # exactly one sample is left.
        [self.batch_augmentation_indices] = input_indices

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

        return self.postprocess(data)

    def _apply_once(
        self,
        transform: BatchTransform,
        batch: dict[str, list[np.ndarray]],
        batch_indices: list[list[int]],
        keypoints_per_instance: dict[str, int],
    ) -> tuple[dict[str, np.ndarray], list[int], bool]:
        """Apply a batch transform to one sub-batch.

        A transform that did not apply passes its first input through
        unchanged.

        Returns:
            The output sample, the indices of the batch inputs it is made
            of, and whether the transform applied. The sample has its
            filtered boxes and their associated labels removed. The
            indices are those of all the inputs if the transform applied,
            else those of the first input.

        """
        # Captured before padding so that a transform which does not
        # fire returns the first sample exactly as it came in.
        first_sample = {key: value[0] for key, value in batch.items()}
        self._pad_absent_instances(batch, keypoints_per_instance)

        data = transform(**batch)  # pyright: ignore[reportArgumentType]

        # A transform that applied collapsed the batched lists
        # `yield_batches` produced into single values. Empty
        # ``params`` does not tell the two apart, as a transform can
        # apply while sampling no parameters at all.
        applied = not isinstance(next(iter(data.values())), list)
        if applied:
            indices = [index for group in batch_indices for index in group]
        else:
            data = first_sample
            indices = batch_indices[0]

        bbox_counts = self._reindex_bboxes(data)
        data = self.check_data_post_transform(data)
        self._compact_bbox_associated_labels(
            data, bbox_counts, keypoints_per_instance
        )
        # Filtering left gaps in the index column; close them so the
        # next transform starts from contiguous indices again.
        self._reindex_bboxes(data)
        return data, indices, applied

    def _locate_index_column(self) -> int:
        """Position of the instance-index column, counted from the right.

        `luxonis_ml.data.augmentations.utils.preprocess_bboxes` appends the
        index as the last column, but the wrapped bbox processor appends one
        more column for every configured label field, pushing the index left.

        Returns:
            The position as a negative index, ``-1`` for the last column.

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
            image_shape = image_shapes[i] if i < len(image_shapes) else None
            shape = BatchCompose._instances_shape(
                target_type,
                bbox_counts[i],
                template.shape[1:],
                image_shape,
                n_keypoints,
            )
            if shape is not None:
                values[i] = np.zeros(shape, dtype=template.dtype)

    @staticmethod
    def _instances_shape(
        target_type: str,
        count: int,
        trailing: tuple[int, ...],
        image_shape: tuple[int, ...] | None,
        n_keypoints: int,
    ) -> tuple[int, ...] | None:
        """Return the shape of ``count`` zeroed instances of a target.

        Instance masks are ``(H, W, N)``, so they need the image shape;
        without one there is nothing to fill.

        Returns:
            The shape, or ``None`` for an instance mask without
            ``image_shape``.

        Examples:
            >>> shape = BatchCompose._instances_shape
            >>> shape(
            ...     "instance_mask", 2, (), image_shape=(5, 6), n_keypoints=0
            ... )
            (5, 6, 2)
            >>> shape(
            ...     "instance_mask", 2, (), image_shape=None, n_keypoints=0
            ... ) is None
            True
            >>> shape("keypoints", 2, (3,), image_shape=None, n_keypoints=4)
            (8, 3)
            >>> shape("array", 2, (7,), image_shape=None, n_keypoints=0)
            (2, 7)

        """
        if target_type == "instance_mask":
            if image_shape is None:
                return None
            return (*image_shape, count)
        if target_type == "keypoints":
            return (count * n_keypoints, *trailing)
        return (count, *trailing)

    def _reindex_bboxes(self, data: dict[str, np.ndarray]) -> dict[str, int]:
        """Give each bbox field contiguous indices and return its size.

        Only fields declared in ``bbox_associations`` are touched; for any
        other bbox field the last column is the class label. The indices are
        positions into the bbox-associated labels, stamped before filtering so
        that the survivors can be used to compact those labels in lockstep.

        Returns:
            Number of boxes in each field of ``bbox_associations`` that
            ``data`` holds, keyed by field name.

        Raises:
            ValueError: If a non-empty field of ``bbox_associations`` is
                one-dimensional.

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
                data[field_name] = self._compact_field(
                    field_name,
                    value,
                    target_type,
                    keypoints_per_instance.get(field_name, 0),
                    bbox_field=bbox_field,
                    bbox_count=bbox_count,
                    indices=indices,
                    unfiltered=unfiltered,
                )

    def _compact_field(
        self,
        field_name: str,
        value: np.ndarray,
        target_type: str,
        n_keypoints: int,
        *,
        bbox_field: str,
        bbox_count: int,
        indices: np.ndarray,
        unfiltered: bool,
    ) -> np.ndarray:
        """Keep the instances of one field whose boxes survived.

        Returns:
            The instances of ``value`` at ``indices``, in that order. If
            ``bbox_count`` is ``0``, an empty slice of ``value``.
            ``value`` itself if it is empty, if no box was filtered, or
            if the field cannot be matched.

        """
        if bbox_count == 0:
            if target_type == "instance_mask" and value.ndim > 1:
                return value[..., :0]
            return value[:0]

        if value.size == 0:
            return value

        # Keypoint rows can only be counted by instance once it is
        # known how many keypoints each instance carries.
        countable = target_type != "keypoints" or n_keypoints > 0
        if (
            not countable
            or instance_count(value, target_type, n_keypoints) != bbox_count
        ):
            self._warn_unmatched(field_name, value, bbox_count, bbox_field)
            return value

        if unfiltered:
            return value
        return self._select_instances(
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
        """Return the arrays of ``data`` in C-contiguous memory."""
        return {
            key: np.ascontiguousarray(value) for key, value in data.items()
        }
