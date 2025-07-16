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
from luxonis_ml.typing import Labels, LoaderOutput, Params, PathType


class LuxonisLoader(BaseLoader):
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
        color_space: Literal["RGB", "BGR", "GRAY"] = "RGB",
        seed: int | None = None,
        min_bbox_visibility: float = 0.0,
        bbox_area_threshold: float = 0.0004,
        *,
        keep_categorical_as_strings: bool = False,
        update_mode: UpdateMode | Literal["all", "missing"] = UpdateMode.ALL,
        filter_task_names: list[str] | None = None,
    ) -> None:
        """A loader class used for loading data from L{LuxonisDataset}.

        @type dataset: LuxonisDataset
        @param dataset: Instance of C{LuxonisDataset} to use.
        @type view: Union[str, List[str]]
        @param view: What splits to use. Can be either a single split or
            a list of splits. Defaults to C{"train"}.
        @type augmentation_engine: Union[Literal["albumentations"], str]
        @param augmentation_engine: The augmentation engine to use.
            Defaults to C{"albumentations"}.
        @type augmentation_config: Optional[Union[List[Params],
            PathType]]
        @param augmentation_config: The configuration for the
            augmentations. This can be either a list of C{Dict[str, JsonValue]} or
            a path to a configuration file.
            The config member is a dictionary with two keys: C{name} and
            C{params}. C{name} is the name of the augmentation to
            instantiate and C{params} is an optional dictionary
            of parameters to pass to the augmentation.

            Example::

                [
                    {"name": "HorizontalFlip", "params": {"p": 0.5}},
                    {"name": "RandomBrightnessContrast", "params": {"p": 0.1}},
                    {"name": "Defocus"}
                ]

        @type height: Optional[int]
        @param height: The height of the output images. Defaults to
            C{None}.
        @type width: Optional[int]
        @param width: The width of the output images. Defaults to
            C{None}.
        @type keep_aspect_ratio: bool
        @param keep_aspect_ratio: Whether to keep the aspect ratio of the
            images. Defaults to C{True}.
        @type color_space: Literal["RGB", "BGR", "GRAY"]
        @param color_space: The color space of the output images. Defaults
            to C{"RGB"}.
        @type seed: Optional[int]
        @param seed: The random seed to use for the augmentations.
        @type min_bbox_visibility: float
        @param min_bbox_visibility: Minimum fraction of the original bounding box that must remain visible after augmentation.
        @type bbox_area_threshold: float
        @param bbox_area_threshold: Minimum area threshold for bounding boxes to be considered valid. In the range [0, 1].
            Default is 0.0004, which corresponds to a small area threshold to remove invalid bboxes and respective keypoints.
        @type exclude_empty_annotations: bool
        @param exclude_empty_annotations: Whether to exclude
            empty annotations from the final label dictionary.
            Defaults to C{False} (i.e. include empty annotations).

        @type keep_categorical_as_strings: bool
        @param keep_categorical_as_strings: Whether to keep categorical
            metadata labels as strings.
            Defaults to C{False} (i.e. convert categorical labels to integers).

        @type update_mode: UpdateMode
        @param update_mode: Enum that determines the sync mode for media files of the remote dataset (annotations and metadata are always overwritten):
            - UpdateMode.MISSING: Downloads only the missing media files for the dataset.
            - UpdateMode.ALL: Always downloads and overwrites all media files in the local dataset.

        @type filter_task_names: Optional[List[str]]
        @param filter_task_names: List of task names to filter the dataset by.
            If C{None}, all task names are included. Defaults to C{None}.
            This is useful for filtering out tasks that are not needed for a specific use case.
        """

        self.exclude_empty_annotations = exclude_empty_annotations
        self.color_space: Literal["RGB", "BGR", "GRAY"] = color_space
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

        if self.filter_task_names is not None:
            self.df = self.df.filter(
                pl.col("task_name").is_in(self.filter_task_names)
            )
            self.classes = {
                task_name: self.classes[task_name]
                for task_name in self.filter_task_names
                if task_name in self.classes
            }

        if not self.dataset.is_remote:
            file_index = self.dataset._get_index()
            if file_index is None:  # pragma: no cover
                raise FileNotFoundError("Cannot find file index")
            file_index = file_index.filter(
                pl.col("uuid").is_in(self.df["uuid"])
            )
            self.df = self.df.join(file_index, on="uuid").drop("file_right")

        self.classes = self.dataset.get_classes()
        self.instances: list[str] = []
        splits_path = self.dataset.metadata_path / "splits.json"
        if not splits_path.exists():
            raise RuntimeError(
                "Cannot find splits! Ensure you call dataset.make_splits()"
            )
        with open(splits_path) as file:
            splits = json.load(file)

        for view in self.view:
            self.instances.extend(splits[view])

        self.idx_to_df_row: list[list[int]] = []
        uuid_list = self.df["uuid"].to_list()
        uuid_set = set(uuid_list)
        self.instances = [uid for uid in self.instances if uid in uuid_set]

        idx_map: dict[str, list[int]] = defaultdict(list)
        for i, uid in enumerate(uuid_list):
            idx_map[uid].append(i)

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
        """Returns length of the dataset.

        @rtype: int
        @return: Length of the loader.
        """
        return len(self.instances)

    @override
    def __getitem__(self, idx: int) -> LoaderOutput:
        """Function to load a sample consisting of an image and its
        annotations.

        @type idx: int
        @param idx: The integer index of the sample to retrieve.
        @rtype: L{LuxonisLoaderOutput}
        @return: The loader ouput consisting of the image and a
            dictionary defining its annotations.
        """

        if self.augmentations is None:
            img, labels = self._load_data(idx)
        else:
            img, labels = self._load_with_augmentations(idx)

        if not self.exclude_empty_annotations:
            img, labels = self._add_empty_annotations(img, labels)

        # Albumentations needs RGB
        if self.color_space == "BGR":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return img, labels

    def _add_empty_annotations(
        self, img: np.ndarray, labels: Labels
    ) -> LoaderOutput:
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
                        labels[task] = np.zeros(
                            (0, img.shape[0], img.shape[1])
                        )
                    elif task_type == "segmentation":
                        labels[task] = np.zeros(
                            (
                                len(self.classes[task_name]),
                                img.shape[0],
                                img.shape[1],
                            )
                        )
                    elif task_type == "classification" or task_is_metadata(
                        task
                    ):
                        labels[task] = np.zeros(
                            (len(self.classes[task_name]),)
                        )

        return img, labels

    def _load_data(self, idx: int) -> tuple[np.ndarray, Labels]:
        """Loads image and its annotations based on index.

        @type idx: int
        @param idx: Index of the image
        @rtype: Tuple[np.ndarray, Labels]
        @return: Image as C{np.ndarray} in RGB format and a dictionary
            with all the present annotations
        """

        if not self.idx_to_df_row:
            raise ValueError(
                f"No data found in dataset '{self.dataset.identifier}' "
                f"for {self.view} views"
            )

        ann_indices = self.idx_to_df_row[idx]

        ann_rows = [self.df.row(row) for row in ann_indices]

        img_path = self.idx_to_img_path[idx]

        if self.color_space == "GRAY":
            img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_RGB2GRAY)[
                ..., np.newaxis
            ]
        else:
            img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)

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
                data["path"] = self.dataset.arrays_path / data["path"]

            if task_type.startswith("metadata/"):
                metadata_by_task[full_task_name].append(data)
            else:  # pragma: no cover
                # Conversion from LDF v1.0
                if "points" in data and "width" not in data:
                    data["width"] = img.shape[1]
                    data["height"] = img.shape[0]
                    data["points"] = [tuple(p) for p in data["points"]]

                annotation = load_annotation(task_type, data)
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

        return img, labels

    def _load_with_augmentations(self, idx: int) -> LoaderOutput:
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

        return AUGMENTATION_ENGINES.get(augmentation_engine)(
            height=height,
            width=width,
            config=augmentation_config,
            targets=targets,
            n_classes=n_classes,
            keep_aspect_ratio=keep_aspect_ratio,
            is_validation_pipeline="train" not in self.view,
            seed=seed,
            min_bbox_visibility=min_bbox_visibility,
            bbox_area_threshold=bbox_area_threshold,
        )

    def _precompute_image_paths(self) -> None:
        self.idx_to_img_path = {}

        for idx, ann_indices in enumerate(self.idx_to_df_row):
            ann_indices = self.idx_to_df_row[idx]

            ann_rows = [self.df.row(row) for row in ann_indices]
            img_path = ann_rows[0][0]
            if not Path(img_path).exists():
                uuid = ann_rows[0][7]
                file_extension = ann_rows[0][0].rsplit(".", 1)[-1]
                img_path = self.dataset.media_path / f"{uuid}.{file_extension}"
                if not img_path.exists():
                    raise FileNotFoundError(
                        f"Cannot find image for uuid {uuid}"
                    )

            self.idx_to_img_path[idx] = img_path
