"""Loader for the samples and labels of a `LuxonisDataset`."""

import inspect
import json
import random
import warnings
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import cv2
import numpy as np
import polars as pl
import pycocotools.mask
import yaml
from loguru import logger
from typeguard import typechecked
from typing_extensions import override

from luxonis_ml.data.augmentations import (
    AUGMENTATION_ENGINES,
    AugmentationEngine,
)
from luxonis_ml.data.datasets import (
    Annotation,
    Category,
    LuxonisDataset,
    UpdateMode,
    load_annotation,
)
from luxonis_ml.data.loaders.base_loader import BaseLoader
from luxonis_ml.data.utils import get_task_type, split_task
from luxonis_ml.data.utils.task_utils import task_is_metadata
from luxonis_ml.ldf import DatasetRecord
from luxonis_ml.typing import (
    Labels,
    LoaderOutput,
    Params,
    PathType,
    TrackedAugmentations,
)

if TYPE_CHECKING:
    from pycocotools import _EncodedRLE


class LuxonisLoader(BaseLoader):
    r"""Indexed loader for `LuxonisDataset` samples.

    `LuxonisLoader` reads one split or multiple splits, loads image
    sources, assembles labels by task key, optionally applies
    augmentations, and returns `LoaderOutput` values with images, labels,
    and sample metadata.

    Iterating a `LoaderOutput` preserves the legacy tuple behavior:
    for a single-source dataset it yields ``(image, labels)``. For a
    multi-source dataset it yields ``(images, labels)``, where ``images``
    maps source names to arrays.

    Prefer attribute access for new code:

    .. python::

        sample = loader[0]

        images = sample.images
        labels = sample.labels
        metadata = sample.metadata

    ``sample.metadata`` contains **record-level metadata** from
    `DatasetRecord.sample_metadata`. When ``autopopulate_metadata`` is enabled,
    the loader also adds a ``"filenames"`` mapping from source name to file
    basename. Augmented outputs additionally include an ``"augmentations"``
    mapping from configured transformation paths to the runtime parameters
    that were selected. Both are reserved keys that never overwrite metadata
    stored on the record.

    Label keys use ``"task_name/task_type"``. If a dataset was created
    without a task name, the default task name is empty and keys look like
    ``"/boundingbox"`` or ``"/segmentation"``.

    Attributes:
        dataset: Dataset being loaded.

    """

    @typechecked
    def __init__(
        self,
        dataset: LuxonisDataset,
        view: str | list[str] = "train",
        *,
        augmentation_engine: Literal["albumentations"]
        | str = "albumentations",
        augmentation_config: list[Params] | PathType | None = None,
        height: int | None = None,
        width: int | None = None,
        keep_aspect_ratio: bool = True,
        exclude_empty_annotations: bool = False,
        color_space: dict[str, Literal["RGB", "BGR", "GRAY"]]
        | Literal["RGB", "BGR", "GRAY"]
        | None = None,
        seed: int | None = None,
        min_bbox_visibility: float = 0.0,
        bbox_area_threshold: float = 0.0004,
        keep_categorical_as_strings: bool = False,
        update_mode: UpdateMode | Literal["all", "missing"] = UpdateMode.ALL,
        filter_task_names: list[str] | None = None,
        autopopulate_metadata: bool = True,
    ) -> None:
        """Create a loader for a Luxonis dataset.

        Args:
            dataset: Dataset to load from.
            view: Split name or split names to load.
            augmentation_engine: Augmentation engine registry name.
            augmentation_config: Optional augmentation configuration or path
                to a configuration file. Each configuration item contains
                ``name`` and optional ``params`` keys.
            height: Optional output image height. Required when
                augmentations are enabled.
            width: Optional output image width. Required when augmentations
                are enabled.
            keep_aspect_ratio: Whether resizing should preserve image aspect
                ratio.
            exclude_empty_annotations: Whether to omit empty annotations
                from the returned label dictionary.
            color_space: Optional color space for each source. A single
                value applies to all sources; if omitted, all sources use
                ``"RGB"``.
            seed: Optional random seed for augmentations.
            min_bbox_visibility: Minimum fraction of the original bounding
                box that must remain visible after augmentation.
            bbox_area_threshold: Minimum normalized area for bounding boxes
                to remain valid. The default removes very small boxes and
                their associated keypoints.
            keep_categorical_as_strings: Whether to keep categorical
                metadata labels as strings instead of converting them to
                integers.
            update_mode: Sync mode for media files in remote datasets.
                Annotations and metadata are always overwritten.
            filter_task_names: Optional task names to include. If omitted,
                all tasks are included.
            autopopulate_metadata: Whether to add automatic sample metadata
                such as source filenames. When enabled, returned metadata
                includes a ``"filenames"`` dictionary keyed by source name.
                Set to ``False`` to return only metadata stored in
                `DatasetRecord.sample_metadata`. Augmentation provenance
                is reported either way, as it describes the returned
                arrays rather than the record.

        Raises:
            ValueError: If ``filter_task_names`` contains task names not
                present in the dataset.
            ValueError: If ``augmentation_config`` is set without
                ``height`` and ``width``.
            RuntimeError: If split metadata is missing.

        Example:
            .. python::

                augmentation_config = [
                    {
                        "name": "Defocus",
                        "params": {"p": 1},
                    },
                    {
                        "name": "RandomCrop",
                        "params": {"height": 512, "width": 512, "p": 1},
                    },
                    {
                        "name": "Mosaic4",
                        "params": {
                            "height": 256,
                            "width": 256,
                            "p": 1.0,
                        },
                    },
                ]

                loader = LuxonisLoader(
                    dataset,
                    view="train",
                    augmentation_config=augmentation_config,
                    height=640,
                    width=640,
                )

        """

        self.dataset = dataset

        if self.dataset.is_remote:
            self.dataset.pull_from_cloud(update_mode=UpdateMode(update_mode))

        self._height = height
        self._width = width
        self._view = [view] if isinstance(view, str) else view

        self._autopopulate_metadata = autopopulate_metadata
        self._exclude_empty_annotations = exclude_empty_annotations
        self._keep_categorical_as_strings = keep_categorical_as_strings
        self._filter_task_names = filter_task_names

        self._source_names = self.dataset.get_source_names()
        self._color_space = self._resolve_color_space(color_space)

        self._df = self.dataset._load_df_offline(raise_when_empty=True)
        self._classes = self.dataset.get_classes()
        if self._filter_task_names is not None:
            self._filter_tasks(self._filter_task_names)

        self._df_col_indices = {
            column: i for i, column in enumerate(self._df.columns)
        }

        view_instances = self._view_instances()
        group_ids = self._df["group_id"].to_list()
        group_id_set = set(group_ids)
        self._instances = [g for g in view_instances if g in group_id_set]

        idx_map: dict[str, list[int]] = defaultdict(list)
        for i, group_id in enumerate(group_ids):
            idx_map[group_id].append(i)
        self._idx_to_df_row = [idx_map[uuid] for uuid in self._instances]

        self._idx_to_img_paths = self._precompute_image_paths()

        self._tasks_without_background: set[str] = set()
        for task_name in self._uncovered_tasks():
            self._add_background_class(task_name)

        self._augmentations = self._init_augmentations(
            augmentation_engine,
            augmentation_config or [],
            height=height,
            width=width,
            keep_aspect_ratio=keep_aspect_ratio,
            seed=seed,
            min_bbox_visibility=min_bbox_visibility,
            bbox_area_threshold=bbox_area_threshold,
        )

    def _resolve_color_space(
        self,
        color_space: dict[str, Literal["RGB", "BGR", "GRAY"]]
        | Literal["RGB", "BGR", "GRAY"]
        | None,
    ) -> dict[str, str]:
        """Return the color space of each source; ``"RGB"`` by default."""
        if isinstance(color_space, dict):
            return {
                source_name: color_space.get(source_name, "RGB")
                for source_name in self._source_names
            }
        return dict.fromkeys(self._source_names, color_space or "RGB")

    def _filter_tasks(self, task_names: list[str]) -> None:
        """Keep only the rows and the classes of the given task names."""
        if self.dataset.metadata.tasks:
            df_task_names = set(self.dataset.metadata.tasks)
        else:
            df_task_names = set(self._df["task_name"].to_list())
        if extras := set(task_names) - df_task_names:
            raise ValueError(
                f"`filter_task_names` contains task names that "
                f"are not in the dataset: {extras}"
            )
        self._df = self._df.filter(pl.col("task_name").is_in(task_names))
        self._classes = {
            task_name: self._classes[task_name]
            for task_name in task_names
            if task_name in self._classes
        }

    def _view_instances(self) -> list[str]:
        """List the group IDs of the splits in the view."""
        splits_path = self.dataset._metadata_path / "splits.json"
        if not splits_path.exists():
            raise RuntimeError(
                "Cannot find splits! Ensure you call dataset.make_splits()"
            )
        with open(splits_path) as file:
            splits = json.load(file)

        instances: list[str] = []
        for v in self._view:
            instances.extend(splits[v])
        return instances

    def _add_background_class(self, task_name: str) -> None:
        """Give a segmentation task with unassigned pixels a background class.

        The class gets index 0; the other classes move up by one.
        """
        classes = self._classes[task_name]
        if len(classes) <= 1 or classes.get("background") == 0:
            return
        logger.warning(
            "Found unassigned pixels in the segmentation masks of task "
            f"'{task_name}'. Assigning them to `background` class (class "
            "index 0). If this is not desired then make sure all pixels "
            "are assigned to one class or rename your background class."
        )
        self._tasks_without_background.add(f"{task_name}/segmentation")
        if "background" not in classes:
            self._classes[task_name] = {
                "background": 0,
                **{name: i + 1 for name, i in classes.items()},
            }

    @override
    def __len__(self) -> int:
        """Return the number of samples in the loader.

        Returns:
            Number of samples.

        """
        return len(self._instances)

    @override
    def __getitem__(self, idx: int) -> LoaderOutput:
        """Load a sample and its annotations.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            `LoaderOutput` with image arrays, annotation labels, and
            record-level metadata.

        Example:
            .. python::

                sample = loader[0]

                images = sample.images
                labels = sample.labels
                metadata = sample.metadata

                image_or_images, labels = sample  # Legacy compatibility.

        Raises:
            ValueError: If the selected views contain no records or a
                grayscale source has an unsupported image shape.
            FileNotFoundError: If an image path cannot be found or read.

        """
        if self._augmentations is None:
            img_dict, labels, metadata = self._load_data(idx)
        else:
            img_dict, labels, metadata = self._load_with_augmentations(idx)

        if not self._exclude_empty_annotations:
            img_dict, labels = self._add_empty_annotations(img_dict, labels)

        # Albumentations needs RGB
        bgr_sources = [k for k, v in self._color_space.items() if v == "BGR"]
        for source_name in bgr_sources:
            img_dict[source_name] = cv2.cvtColor(
                img_dict[source_name], cv2.COLOR_RGB2BGR
            )
        return LoaderOutput(img_dict, labels, metadata)

    def _uncovered_tasks(self) -> set[str]:
        """Return the segmentation tasks that leave pixels unassigned.

        A task is uncovered when the union of one sample's masks does
        not fill the image. The union counts a pixel once, as
        `SegmentationAnnotation.combine_to_numpy` does, and
        ``pycocotools`` measures it without a decode.

        A sample with no mask gives no verdict on its task.

        Returns:
            The names of the uncovered segmentation tasks.

        """
        samples = (
            self._df.filter(
                (pl.col("task_type") == "segmentation")
                & pl.col("group_id").is_in(self._instances)
            )
            .group_by(["group_id", "task_name"])
            .agg(pl.col("annotation"))
        )

        uncovered: set[str] = set()
        for _, task_name, annotations in samples.iter_rows():
            # One uncovered sample settles the task.
            if task_name in uncovered:
                continue
            union = pycocotools.mask.merge(self._sample_masks(annotations))
            height, width = union["size"]
            if pycocotools.mask.area(union) < height * width:
                uncovered.add(task_name)
        return uncovered

    def _add_empty_annotations(
        self, img_dict: dict[str, np.ndarray], labels: Labels
    ) -> tuple[dict[str, np.ndarray], Labels]:
        image_size = next(iter(img_dict.values())).shape[:2]
        for task_name, task_type in self._loaded_tasks():
            task = f"{task_name}/{task_type}"
            if task in labels:
                continue
            empty = self._empty_label(task_name, task_type, image_size)
            if empty is not None:
                labels[task] = empty

        return img_dict, labels

    def _loaded_tasks(self) -> list[tuple[str, str]]:
        """List the ``(task_name, task_type)`` pairs the loader returns."""
        return [
            (task_name, task_type)
            for task_name, task_types in self.dataset.get_tasks().items()
            if self._filter_task_names is None
            or task_name in self._filter_task_names
            for task_type in task_types
        ]

    def _empty_label(
        self, task_name: str, task_type: str, image_size: tuple[int, int]
    ) -> np.ndarray | None:
        """Return the label array of a task with no annotations."""
        image_height, image_width = image_size
        if task_type == "boundingbox":
            return np.zeros((0, 5))
        if task_type == "keypoints":
            n_keypoints = self.dataset.get_n_keypoints()[task_name]
            return np.zeros((0, n_keypoints * 3))
        if task_type == "instance_segmentation":
            return np.zeros((0, image_height, image_width))
        if task_type == "segmentation":
            classes = self._classes[task_name]
            mask = np.zeros((len(classes), image_height, image_width))
            if f"{task_name}/{task_type}" in self._tasks_without_background:
                mask[classes["background"]] = 1
            return mask
        if task_type == "classification" or task_is_metadata(
            f"{task_name}/{task_type}"
        ):
            return np.zeros((len(self._classes[task_name]),))
        return None

    def _load_data(
        self, idx: int
    ) -> tuple[dict[str, np.ndarray], Labels, Params]:
        """Load image data and annotations by index.

        Args:
            idx: Index of the image.

        Returns:
            Images, labels, and sample metadata present for the sample.

        Raises:
            ValueError: If the selected views contain no records or a
                grayscale source has an unsupported image shape.
            FileNotFoundError: If an image path cannot be found or read.

        """
        if not self._idx_to_df_row:
            raise ValueError(
                f"No data found in dataset '{self.dataset.identifier}' "
                f"for {self._view} views"
            )

        ann_rows = [self._df.row(row) for row in self._idx_to_df_row[idx]]
        source_to_path = self._idx_to_img_paths[idx]
        sample_metadata = self._sample_metadata(ann_rows, source_to_path)

        img_dict = {
            source_name: _read_image(
                path, self._color_space.get(source_name, "RGB")
            )
            for source_name, path in source_to_path.items()
        }
        image_shape = next(iter(img_dict.values())).shape

        annotations_by_task: dict[str, list[tuple[int, Annotation, int]]] = (
            defaultdict(list)
        )
        metadata_by_task: dict[
            str, list[tuple[int, str | int | float | Category]]
        ] = defaultdict(list)
        for row in ann_rows:
            parsed = self._parse_annotation_row(row, image_shape)
            if parsed is None:
                continue
            task, instance_id, value = parsed
            if isinstance(value, Annotation):
                class_id = self._class_id(row)
                annotations_by_task[task].append(
                    (instance_id, value, class_id)
                )
            else:
                metadata_by_task[task].append((instance_id, value))

        # Both helpers sort by instance ID, so that index i is the same object
        # in every label array of a task: boxes, keypoints, masks and metadata.
        labels = self._metadata_labels(metadata_by_task)
        for task, instances in annotations_by_task.items():
            labels[task] = self._annotation_label(task, instances)

        return img_dict, labels, sample_metadata

    def _sample_metadata(
        self, ann_rows: list[tuple], source_to_path: dict[str, Path]
    ) -> Params:
        """Decode the sample metadata of the rows, plus the filenames."""
        col = self._df_col_indices
        sample_metadata: Params = {}
        metadata_idx = col.get("sample_metadata")
        for row in ann_rows:
            sample_metadata = DatasetRecord.decode_metadata(
                row[metadata_idx] if metadata_idx is not None else None
            )

        if self._autopopulate_metadata and self._reserve_key(
            sample_metadata, "filenames"
        ):
            sample_metadata = {
                "filenames": {
                    source_name: path.name
                    for source_name, path in source_to_path.items()
                },
                **sample_metadata,
            }
        return sample_metadata

    def _parse_annotation_row(
        self, row: tuple, image_shape: tuple[int, ...]
    ) -> tuple[str, int, Annotation | str | int | float | Category] | None:
        """Parse one dataframe row into its task, instance ID and value.

        A metadata row gives its raw value, any other row an `Annotation`.

        Returns:
            The ``"task_name/task_type"`` key of the row, its instance
            ID and its value. ``None`` if the row has no annotation.

        """
        col = self._df_col_indices
        ann_str: str | None = row[col["annotation"]]
        if ann_str is None:
            return None

        task_name: str = row[col["task_name"]]
        instance_id: int = row[col["instance_id"]]
        task = f"{task_name}/{row[col['task_type']]}"
        task_type = get_task_type(task)

        data = json.loads(ann_str)
        if task_type == "array" and self.dataset.is_remote:
            data["path"] = self.dataset._arrays_path / data["path"]

        if task_type.startswith("metadata/"):
            return task, instance_id, data

        # Conversion from LDF v1.0
        if "points" in data and "width" not in data:  # pragma: no cover
            data["height"], data["width"] = image_shape[:2]
            data["points"] = [tuple(p) for p in data["points"]]

        return task, instance_id, load_annotation(task_type, data)  # type: ignore

    def _class_id(self, row: tuple) -> int:
        """Return the class index of a row, or 0 for a row without a class."""
        col = self._df_col_indices
        class_name: str | None = row[col["class_name"]]
        if class_name is None:
            return 0
        task_name: str = row[col["task_name"]]
        return self._classes[task_name][class_name]

    def _metadata_labels(
        self,
        metadata_by_task: dict[
            str, list[tuple[int, str | int | float | Category]]
        ],
    ) -> Labels:
        """Build one array per metadata task, in instance ID order."""
        labels: Labels = {}
        encodings = self.dataset.get_categorical_encodings()
        for task, values in metadata_by_task.items():
            values.sort(key=lambda x: x[0])
            metadata = [value for _, value in values]
            if not self._keep_categorical_as_strings and task in encodings:
                metadata = [encodings[task][m] for m in metadata]  # type: ignore
            labels[task] = np.array(metadata)
        return labels

    def _annotation_label(
        self, task: str, instances: list[tuple[int, Annotation, int]]
    ) -> np.ndarray:
        """Combine the annotations of a task into one array, in instance ID order."""
        instances.sort(key=lambda x: x[0])
        anns = [ann for _, ann, _ in instances]
        class_ids = [class_id for _, _, class_id in instances]

        task_name, _ = split_task(task)
        array = anns[0].combine_to_numpy(
            anns,
            class_ids,
            len(self._classes[task_name]),
        )
        if task in self._tasks_without_background:
            unassigned_pixels = ~np.any(array, axis=0)
            background_idx = self._classes[task_name]["background"]
            array[background_idx, unassigned_pixels] = 1
        return array

    def _load_with_augmentations(
        self, idx: int
    ) -> tuple[dict[str, np.ndarray], Labels, Params]:
        assert self._augmentations is not None
        indices = [idx, *self._pick_batch_members(idx)]

        loaded_anns: list[tuple[dict[str, np.ndarray], Labels]] = []
        sample_metadata_list: list[Params] = []
        for i in indices:
            img_dict, labels, sample_metadata = self._load_data(i)
            loaded_anns.append((img_dict, labels))
            sample_metadata_list.append(sample_metadata)

        img_dict, labels = self._augmentations.apply(loaded_anns)
        metadata_indices = self._augmentations.batch_augmentation_indices
        if len(metadata_indices) > 1:
            sample_metadata = self._merge_sample_metadata(
                [sample_metadata_list[i] for i in metadata_indices],
                metadata_indices,
            )
        else:
            sample_metadata = sample_metadata_list[metadata_indices[0]]

        if self._reserve_key(sample_metadata, "augmentations"):
            sample_metadata["augmentations"] = TrackedAugmentations(
                self._augmentations.applied_augmentations
            )
        return img_dict, labels, sample_metadata

    def _pick_batch_members(self, idx: int) -> list[int]:
        """Draw the other samples that a batch augmentation of ``idx`` needs.

        The samples are distinct from ``idx`` and from each other, unless
        the batch is larger than the dataset.

        Returns:
            The indices of the ``batch_size - 1`` other samples. An
            empty list if the batch has one sample only.

        """
        assert self._augmentations is not None
        n_others = self._augmentations.batch_size - 1
        if n_others < 1:
            return []

        if self._augmentations.batch_size > len(self):
            warnings.warn(
                f"Augmentations batch_size ({self._augmentations.batch_size}) "
                f"is larger than dataset size ({len(self)}). "
                "Samples will include repetitions.",
                stacklevel=3,
            )
            other_indices = [i for i in range(len(self)) if i != idx] or [idx]
            return random.choices(other_indices, k=n_others)

        picked: set[int] = set()
        while len(picked) < n_others:
            rand_idx = random.randint(0, len(self) - 1)
            if rand_idx != idx and rand_idx not in picked:
                picked.add(rand_idx)
        return list(picked)

    @staticmethod
    def _reserve_key(metadata: Params, key: str) -> bool:
        """Whether the loader may fill in one of its reserved keys."""
        if key not in metadata:
            return True
        warnings.warn(
            f"Record metadata already defines the reserved '{key}' key. "
            "Keeping the stored value.",
            stacklevel=3,
        )
        return False

    @staticmethod
    def _merge_sample_metadata(
        metadata_batch: list[Params], input_indices: list[int] | None = None
    ) -> Params:
        """Merge metadata from samples used by a batch augmentation.

        The first sample's metadata remains at the top level. Metadata from
        every contributing input sample is also preserved in
        ``"batch_augmentation_metadata"`` so consumers can inspect which
        records contributed to the augmented output. ``input_index`` is the
        source sample's position in the original augmentation batch.

        Returns:
            The merged metadata, built from deep copies of the inputs.
            An empty dict if ``metadata_batch`` is empty.

        Example:
            .. python::

                {
                    "record_id": 123,

                    "batch_augmentation_metadata": [
                        {
                            "input_index": 0,
                            "sample_metadata": {"record_id": 123},
                        },
                        {
                            "input_index": 1,
                            "sample_metadata": {"record_id": 456},
                        },
                    ],
                }

        """
        if not metadata_batch:
            return {}

        if input_indices is None:
            input_indices = list(range(len(metadata_batch)))

        metadata = deepcopy(metadata_batch[0])

        metadata["batch_augmentation_metadata"] = [  # type: ignore
            {
                "input_index": i,
                "sample_metadata": deepcopy(m),
            }
            for i, m in zip(input_indices, metadata_batch, strict=True)
        ]
        return metadata

    def _init_augmentations(
        self,
        augmentation_engine: Literal["albumentations"] | str,
        augmentation_config: list[Params] | PathType,
        *,
        height: int | None,
        width: int | None,
        keep_aspect_ratio: bool,
        seed: int | None = None,
        min_bbox_visibility: float = 0.0,
        bbox_area_threshold: float = 0.0004,
    ) -> AugmentationEngine | None:
        if isinstance(augmentation_config, PathType):
            with open(augmentation_config) as file:
                augmentation_config = cast(
                    list[Params], yaml.safe_load(file) or []
                )
        if augmentation_config and (width is None or height is None):
            raise ValueError(
                "Height and width must be provided when using augmentations"
            )

        if height is None or width is None:
            return None

        loaded_tasks = self._loaded_tasks()
        targets = {
            f"{task_name}/{task_type}": task_type
            for task_name, task_type in loaded_tasks
        }
        dataset_n_classes = self.dataset.get_n_classes()
        n_classes = {
            f"{task_name}/{task_type}": dataset_n_classes[task_name]
            for task_name, task_type in loaded_tasks
        }
        pipeline_stage = self._get_augmentation_pipeline_stage()

        engine_cls = AUGMENTATION_ENGINES.get(augmentation_engine)
        init_kwargs = {
            "height": height,
            "width": width,
            "config": augmentation_config,
            "targets": targets,
            "n_classes": n_classes,
            "source_names": self._source_names,
            "keep_aspect_ratio": keep_aspect_ratio,
            "seed": seed,
            "min_bbox_visibility": min_bbox_visibility,
            "bbox_area_threshold": bbox_area_threshold,
        }
        if (
            "pipeline_stage"
            in inspect.signature(engine_cls.__init__).parameters
        ):
            init_kwargs["pipeline_stage"] = pipeline_stage
        elif (
            "is_validation_pipeline"
            in inspect.signature(engine_cls.__init__).parameters
        ):
            # Backward compatibility for custom engines still using
            # the older `is_validation_pipeline` train-vs-eval boolean API.
            init_kwargs["is_validation_pipeline"] = pipeline_stage != "train"

        return engine_cls(**init_kwargs)

    def _get_augmentation_pipeline_stage(self) -> str:
        if "train" in self._view:
            return "train"
        if "val" in self._view:
            return "val"
        if "test" in self._view:
            return "test"
        # preserve the old `is_validation_pipeline` behavior:
        # any non-train split is treated as an evaluation pipeline
        return "val"

    def _precompute_image_paths(self) -> dict[int, dict[str, Path]]:
        idx_to_img_paths = {}
        col = self._df_col_indices

        for idx, ann_indices in enumerate(self._idx_to_df_row):
            ann_rows = [self._df.row(row) for row in ann_indices]

            source_to_path = {}

            for row in ann_rows:
                path = Path(row[col["file"]])
                source_name = row[col["source_name"]]
                uuid = row[col["uuid"]]

                if source_name in source_to_path:
                    continue

                if not path.exists():
                    path = (self.dataset._media_path / uuid).with_suffix(
                        path.suffix
                    )
                    if not path.exists():
                        raise FileNotFoundError(
                            f"Cannot find image for uuid '{uuid}' and source '{source_name}'"
                        )

                source_to_path[source_name] = path

            idx_to_img_paths[idx] = dict(sorted(source_to_path.items()))

        return idx_to_img_paths

    @staticmethod
    def _sample_masks(annotations: list[str]) -> list["_EncodedRLE"]:
        """Return the masks of one sample as run-length encodings.

        An LDF 1.0 polyline stores normalized points and no size, so it
        is rasterized on a grid and no image is read.

        Returns:
            One run-length encoding for each mask and each polyline.
            All have the size of the first encoded mask, or of a fixed
            grid if the sample has polylines only.

        """
        parsed = [json.loads(annotation) for annotation in annotations]
        encoded = [data for data in parsed if "counts" in data]
        polylines = [data["points"] for data in parsed if "counts" not in data]

        # Any grid measures a normalized polyline.
        height, width = 1000, 1000
        if encoded:
            height, width = encoded[0]["height"], encoded[0]["width"]

        masks: list[_EncodedRLE] = [
            {"counts": data["counts"].encode(), "size": [height, width]}
            for data in encoded
        ]
        for points in polylines:
            polygon = []
            for x, y in points:
                polygon += [round(x * width), round(y * height)]
            masks += pycocotools.mask.frPyObjects([polygon], height, width)
        return masks


def _read_image(path: Path, color_space: str) -> np.ndarray:
    """Read an image in the given color space, with a channel axis."""
    if color_space == "GRAY":
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    else:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image at path: {path}")
    if color_space != "GRAY":
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.ndim == 2:
        return img[..., np.newaxis]
    if img.ndim == 3 and img.shape[2] == 3:
        return img[..., 0:1]
    raise ValueError(f"Unsupported image format: shape {img.shape}")
