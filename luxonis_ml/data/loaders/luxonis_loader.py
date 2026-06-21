import inspect
import json
import random
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Literal, cast

import cv2
import numpy as np
import polars as pl
import yaml
from loguru import logger
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
from luxonis_ml.data.utils import (
    get_task_name,
    get_task_type,
    split_task,
    task_type_iterator,
)
from luxonis_ml.data.utils.task_utils import task_is_metadata
from luxonis_ml.typing import (
    Labels,
    LoaderMultiOutput,
    LoaderOutput,
    LoaderSingleOutput,
    Params,
    PathType,
)


class LuxonisLoader(BaseLoader):
    r"""Indexed loader for `LuxonisDataset` samples.

    `LuxonisLoader` reads one split or multiple splits, loads image
    sources, assembles labels by task key, optionally applies
    augmentations, and returns a tuple of image data and labels.

    For a single-source dataset, ``loader[i]`` returns
    ``(image, labels)``. For a multi-source dataset it returns
    ``(images, labels)``, where ``images`` maps source names to arrays.

    Label keys use ``"task_name/task_type"``. If a dataset was created
    without a task name, the default task name is empty and keys look like
    ``"/boundingbox"`` or ``"/segmentation"``.

    Attributes:
        dataset: Dataset being loaded.
        view: Split names loaded by this loader.
        df: Dataframe with records used by the loader.
        classes: Class-name mappings per task.
        source_names: Source names expected in each sample.
        instances: Group IDs included in the selected views.
        idx_to_df_row: Mapping from loader index to dataframe row indices.
        color_space: Output color space per source.
        height: Optional output image height.
        width: Optional output image width.
        augmentations: Optional augmentation engine.
        exclude_empty_annotations: Whether empty annotations are omitted.
        sync_mode: Whether the dataset is remote and pulled before loading.
        keep_categorical_as_strings: Whether categorical metadata remains
            as strings.
        filter_task_names: Optional task-name allowlist.
        tasks_without_background: Segmentation tasks where unassigned
            pixels are mapped to background class :math:`0`.

    """

    def __init__(
        self,
        dataset: LuxonisDataset,
        view: str | list[str] = "train",
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
        *,
        keep_categorical_as_strings: bool = False,
        update_mode: UpdateMode | Literal["all", "missing"] = UpdateMode.ALL,
        filter_task_names: list[str] | None = None,
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

        Raises:
            ValueError: If `color_space` is neither a string nor a
                dictionary.
            ValueError: If `filter_task_names` contains task names not
                present in the dataset.
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
        self.exclude_empty_annotations = exclude_empty_annotations
        self.height = height
        self.width = width

        self.dataset = dataset

        self.sync_mode = self.dataset.is_remote
        self.keep_categorical_as_strings = keep_categorical_as_strings
        self.filter_task_names = filter_task_names

        if self.sync_mode:
            self.dataset.pull_from_cloud(update_mode=UpdateMode(update_mode))

        if isinstance(view, str):
            view = [view]
        self.view = view

        self.df = self.dataset._load_df_offline(raise_when_empty=True)
        self.classes = self.dataset.get_classes()
        self.source_names = self.dataset.get_source_names()

        if color_space is None:
            color_space = dict.fromkeys(self.source_names, "RGB")
        elif isinstance(color_space, str):
            color_space = dict.fromkeys(self.source_names, color_space)
        elif not isinstance(color_space, dict):
            raise ValueError(
                "color_space must be either a string or a dictionary"
            )
        self.color_space = color_space

        if self.filter_task_names is not None:
            if self.dataset.metadata.tasks:
                df_task_names = set(self.dataset.metadata.tasks)
            else:
                df_task_names = set(self.df["task_name"].to_list())
            if extras := set(self.filter_task_names) - df_task_names:
                raise ValueError(
                    f"filter_task_names contains task names that "
                    f"are not in the dataset: {extras}"
                )
            self.df = self.df.filter(
                pl.col("task_name").is_in(self.filter_task_names)
            )
            self.classes = {
                task_name: self.classes[task_name]
                for task_name in self.filter_task_names
                if task_name in self.classes
            }

        self.classes = self.dataset.get_classes()
        self.instances: list[str] = []
        splits_path = self.dataset._metadata_path / "splits.json"
        if not splits_path.exists():
            raise RuntimeError(
                "Cannot find splits! Ensure you call dataset.make_splits()"
            )
        with open(splits_path) as file:
            splits = json.load(file)

        for view in self.view:
            self.instances.extend(splits[view])

        self.idx_to_df_row: list[list[int]] = []
        group_id_list = self.df["group_id"].to_list()
        group_id_set = set(group_id_list)
        self.instances = [
            group_id for group_id in self.instances if group_id in group_id_set
        ]

        idx_map: dict[str, list[int]] = defaultdict(list)
        for i, group_id in enumerate(group_id_list):
            idx_map[group_id].append(i)

        self.idx_to_df_row = [idx_map[uid] for uid in self.instances]

        self.tasks_without_background = set()

        self._precompute_image_paths()

        _, test_labels = self._load_data(0)
        for task, seg_masks in task_type_iterator(test_labels, "segmentation"):
            task_name = get_task_name(task)
            if seg_masks.shape[0] > 1 and (
                "background" not in self.classes[task_name]
                or self.classes[task_name]["background"] != 0
            ):
                unassigned_pixels = np.sum(seg_masks, axis=0) == 0

                if np.any(unassigned_pixels):
                    logger.warning(
                        "Found unassigned pixels in segmentation masks. "
                        "Assigning them to `background` class (class index 0). "
                        "If this is not desired then make sure all pixels are "
                        "assigned to one class or rename your background class."
                    )
                    self.tasks_without_background.add(task)
                    if "background" not in self.classes[task_name]:
                        self.classes[task_name] = {
                            "background": 0,
                            **{
                                class_name: i + 1
                                for class_name, i in self.classes[
                                    task_name
                                ].items()
                            },
                        }

        self.augmentations = self._init_augmentations(
            augmentation_engine,
            augmentation_config or [],
            height,
            width,
            keep_aspect_ratio,
            seed,
            min_bbox_visibility,
            bbox_area_threshold,
        )

    @override
    def __len__(self) -> int:
        """Return the number of samples in the loader.

        Returns:
            Number of samples.

        """
        return len(self.instances)

    @override
    def __getitem__(self, idx: int) -> LoaderOutput:
        """Load a sample and its annotations.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Image data and annotation labels.

        Raises:
            ValueError: If the selected views contain no records or a
                grayscale source has an unsupported image shape.
            FileNotFoundError: If an image path cannot be found or read.

        """
        if self.augmentations is None:
            img_dict, labels = self._load_data(idx)
        else:
            img_dict, labels = self._load_with_augmentations(idx)

        if not self.exclude_empty_annotations:
            img_dict, labels = self._add_empty_annotations(img_dict, labels)

        # Albumentations needs RGB
        bgr_sources = [k for k, v in self.color_space.items() if v == "BGR"]
        for source_name in bgr_sources:
            img_dict[source_name] = cv2.cvtColor(
                img_dict[source_name], cv2.COLOR_RGB2BGR
            )

        if len(self.source_names) == 1:
            img = next(iter(img_dict.values()))
            return cast(LoaderSingleOutput, (img, labels))
        return cast(LoaderMultiOutput, (img_dict, labels))

    def _add_empty_annotations(
        self, img_dict: dict[str, np.ndarray], labels: Labels
    ) -> LoaderMultiOutput:
        image_height, image_width = next(iter(img_dict.values())).shape[:2]
        for task_name, task_types in self.dataset.get_tasks().items():
            if (
                self.filter_task_names is not None
                and task_name not in self.filter_task_names
            ):
                continue
            for task_type in task_types:
                task = f"{task_name}/{task_type}"
                if task not in labels:
                    if task_type == "boundingbox":
                        labels[task] = np.zeros((0, 5))
                    elif task_type == "keypoints":
                        n_keypoints = self.dataset.get_n_keypoints()[task_name]
                        labels[task] = np.zeros((0, n_keypoints * 3))
                    elif task_type == "instance_segmentation":
                        labels[task] = np.zeros((0, image_height, image_width))
                    elif task_type == "segmentation":
                        labels[task] = np.zeros(
                            (
                                len(self.classes[task_name]),
                                image_height,
                                image_width,
                            )
                        )
                    elif task_type == "classification" or task_is_metadata(
                        task
                    ):
                        labels[task] = np.zeros(
                            (len(self.classes[task_name]),)
                        )

        return img_dict, labels

    def _load_data(self, idx: int) -> LoaderMultiOutput:
        """Load image data and annotations by index.

        Args:
            idx: Index of the image.

        Returns:
            Images and labels present for the sample.

        """
        if not self.idx_to_df_row:
            raise ValueError(
                f"No data found in dataset '{self.dataset.identifier}' "
                f"for {self.view} views"
            )

        ann_indices = self.idx_to_df_row[idx]
        ann_rows = [self.df.row(row) for row in ann_indices]

        img_dict: dict[str, np.ndarray] = {}
        source_to_path = self.idx_to_img_paths[idx]

        for source_name, path in source_to_path.items():
            color_space = self.color_space.get(source_name, "RGB")
            if color_space == "GRAY":
                img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            else:
                img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Cannot read image at path: {path}")
            if color_space == "GRAY":
                if img.ndim == 2:
                    img_gray = img[..., np.newaxis]
                elif img.ndim == 3 and img.shape[2] == 3:
                    img_gray = img[..., 0:1]
                else:
                    raise ValueError(
                        f"Unsupported image format: shape {img.shape}"
                    )

                img_dict[source_name] = img_gray
            else:
                img_dict[source_name] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        labels_by_task: dict[str, list[Annotation]] = defaultdict(list)
        class_ids_by_task: dict[str, list[int]] = defaultdict(list)
        instance_ids_by_task: dict[str, list[int]] = defaultdict(list)
        metadata_by_task: dict[str, list[str | int | float | Category]] = (
            defaultdict(list)
        )

        for annotation_data in ann_rows:
            task_name: str = annotation_data[2]
            class_name: str | None = annotation_data[3]
            instance_id: int = annotation_data[4]
            task_type: str = annotation_data[5]
            ann_str: str | None = annotation_data[6]

            if ann_str is None:
                continue

            data = json.loads(ann_str)
            full_task_name = f"{task_name}/{task_type}"
            task_type = get_task_type(full_task_name)
            if task_type == "array" and self.dataset.is_remote:
                data["path"] = self.dataset._arrays_path / data["path"]

            if task_type.startswith("metadata/"):
                metadata_by_task[full_task_name].append(data)
            else:  # pragma: no cover
                # Conversion from LDF v1.0
                if "points" in data and "width" not in data:
                    sample_img = next(iter(img_dict.values()))
                    data["width"] = sample_img.shape[1]
                    data["height"] = sample_img.shape[0]
                    data["points"] = [tuple(p) for p in data["points"]]

                annotation = load_annotation(task_type, data)  # type: ignore
                labels_by_task[full_task_name].append(annotation)
                if class_name is not None:
                    class_ids_by_task[full_task_name].append(
                        self.classes[task_name][class_name]
                    )
                else:
                    class_ids_by_task[full_task_name].append(0)
                instance_ids_by_task[full_task_name].append(instance_id)

        labels: Labels = {}
        encodings = self.dataset.get_categorical_encodings()
        for task, metadata in metadata_by_task.items():
            if not self.keep_categorical_as_strings and task in encodings:
                metadata = [encodings[task][m] for m in metadata]  # type: ignore
            labels[task] = np.array(metadata)

        for task, anns in labels_by_task.items():
            assert anns, f"No annotations found for task {task_name}"
            instance_ids = instance_ids_by_task[task]

            anns = [
                ann
                for _, ann in sorted(
                    zip(instance_ids, anns, strict=True), key=lambda x: x[0]
                )
            ]

            task_name, task_type = split_task(task)
            array = anns[0].combine_to_numpy(
                anns,
                class_ids_by_task[task],
                len(self.classes[task_name]),
            )
            if task in self.tasks_without_background:
                unassigned_pixels = ~np.any(array, axis=0)
                background_idx = self.classes[task_name]["background"]
                array[background_idx, unassigned_pixels] = 1

            labels[task] = array

        return img_dict, labels

    def _load_with_augmentations(self, idx: int) -> LoaderMultiOutput:
        indices = [idx]
        assert self.augmentations is not None
        if self.augmentations.batch_size > 1:
            if self.augmentations.batch_size > len(self):
                warnings.warn(
                    f"Augmentations batch_size ({self.augmentations.batch_size}) "
                    f"is larger than dataset size ({len(self)}). "
                    "Samples will include repetitions.",
                    stacklevel=2,
                )
                other_indices = [i for i in range(len(self)) if i != idx]
                picked_indices = random.choices(
                    other_indices, k=self.augmentations.batch_size - 1
                )
            else:
                picked_indices = set()
                max_val = len(self)
                while len(picked_indices) < self.augmentations.batch_size - 1:
                    rand_idx = random.randint(0, max_val - 1)
                    if rand_idx != idx and rand_idx not in picked_indices:
                        picked_indices.add(rand_idx)
                picked_indices = list(picked_indices)

            indices.extend(picked_indices)

        loaded_anns = [self._load_data(i) for i in indices]
        return self.augmentations.apply(loaded_anns)

    def _init_augmentations(
        self,
        augmentation_engine: Literal["albumentations"] | str,
        augmentation_config: list[Params] | PathType,
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

        dataset_tasks = self.dataset.get_tasks()

        targets = {
            f"{task_name}/{task_type}": task_type
            for task_name, task_types in dataset_tasks.items()
            if self.filter_task_names is None
            or task_name in self.filter_task_names
            for task_type in task_types
        }

        n_classes = {
            f"{task_name}/{task_type}": self.dataset.get_n_classes()[task_name]
            for task_name, task_types in dataset_tasks.items()
            if self.filter_task_names is None
            or task_name in self.filter_task_names
            for task_type in task_types
        }
        pipeline_stage = self._get_augmentation_pipeline_stage()

        engine_cls = AUGMENTATION_ENGINES.get(augmentation_engine)
        init_kwargs = {
            "height": height,
            "width": width,
            "config": augmentation_config,
            "targets": targets,
            "n_classes": n_classes,
            "source_names": self.source_names,
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
        if "train" in self.view:
            return "train"
        if "val" in self.view:
            return "val"
        if "test" in self.view:
            return "test"
        # preserve the old `is_validation_pipeline` behavior:
        # any non-train split is treated as an evaluation pipeline
        return "val"

    def _precompute_image_paths(self) -> None:
        self.idx_to_img_paths = {}

        for idx, ann_indices in enumerate(self.idx_to_df_row):
            ann_rows = [self.df.row(row) for row in ann_indices]

            source_to_path = {}

            for row in ann_rows:
                img_path = row[0]
                source_name = row[1]
                uuid = row[7]

                if source_name in source_to_path:
                    continue

                path = Path(img_path)
                if not path.exists():
                    file_extension = img_path.rsplit(".", 1)[-1]
                    path = (
                        self.dataset._media_path / f"{uuid}.{file_extension}"
                    )
                    if not path.exists():
                        raise FileNotFoundError(
                            f"Cannot find image for uuid {uuid} and source '{source_name}'"
                        )

                source_to_path[source_name] = path

            self.idx_to_img_paths[idx] = dict(sorted(source_to_path.items()))
