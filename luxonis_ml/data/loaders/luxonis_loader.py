import json
import logging
import random
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, cast

import cv2
import numpy as np
import yaml
from typing_extensions import override

from luxonis_ml.data.augmentations import (
    AUGMENTATION_ENGINES,
    AugmentationEngine,
)
from luxonis_ml.data.datasets import (
    Annotation,
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
from luxonis_ml.typing import Labels, LoaderOutput, PathType

logger = logging.getLogger(__name__)


class LuxonisLoader(BaseLoader):
    def __init__(
        self,
        dataset: LuxonisDataset,
        view: Union[str, List[str]] = "train",
        augmentation_engine: Union[
            Literal["albumentations"], str
        ] = "albumentations",
        augmentation_config: Optional[
            Union[List[Dict[str, Any]], PathType]
        ] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        keep_aspect_ratio: bool = True,
        color_space: Literal["RGB", "BGR"] = "RGB",
        *,
        update_mode: UpdateMode = UpdateMode.ALWAYS,
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
        @type augmentation_config: Optional[Union[List[Dict[str, Any]],
            PathType]]
        @param augmentation_config: The configuration for the
            augmentations. This can be either a list of C{Dict[str, Any]} or
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
        @type color_space: Literal["RGB", "BGR"]
        @param color_space: The format of the output images. Defaults
            to C{"RGB"}.
        @type update_mode: UpdateMode
        @param update_mode: Enum that determines the sync mode:
            - UpdateMode.ALWAYS: Force a fresh download
            - UpdateMode.IF_EMPTY: Skip downloading if local data exists
        """

        self.logger = logging.getLogger(__name__)
        self.color_space = color_space

        self.dataset = dataset
        self.sync_mode = self.dataset.is_remote

        if self.sync_mode:
            self.dataset.sync_from_cloud(update_mode=update_mode)

        if isinstance(view, str):
            view = [view]
        self.view = view

        self.df = self.dataset._load_df_offline(raise_when_empty=True)

        if not self.dataset.is_remote:
            file_index = self.dataset._get_file_index()
            if file_index is None:  # pragma: no cover
                raise FileNotFoundError("Cannot find file index")
            self.df = self.df.join(file_index, on="uuid").drop("file_right")

        self.classes = self.dataset.get_classes()
        self.augmentations = self._init_augmentations(
            augmentation_engine,
            augmentation_config or [],
            height,
            width,
            keep_aspect_ratio,
        )
        self.instances: List[str] = []
        splits_path = self.dataset.metadata_path / "splits.json"
        if not splits_path.exists():
            raise RuntimeError(
                "Cannot find splits! Ensure you call dataset.make_splits()"
            )
        with open(splits_path, "r") as file:
            splits = json.load(file)

        for view in self.view:
            self.instances.extend(splits[view])

        self.idx_to_df_row: list[list[int]] = []
        for uuid in self.instances:
            boolean_mask = self.df["uuid"] == uuid
            row_indexes = boolean_mask.arg_true().to_list()
            self.idx_to_df_row.append(row_indexes)

        self.class_mappings: Dict[str, Dict[str, int]] = {}
        for task in self.df["task_name"].unique():
            class_mapping = {
                class_: i
                for i, class_ in enumerate(
                    sorted(
                        self.classes.get(task, []),
                        key=lambda x: {"background": -1}.get(x, 0),
                    )
                )
            }
            self.class_mappings[task] = class_mapping

        self.tasks_without_background = set()

        _, test_labels = self._load_data(0)
        for task, seg_masks in task_type_iterator(test_labels, "segmentation"):
            task_name = get_task_name(task)
            if seg_masks.shape[0] > 1 and (
                "background" not in self.class_mappings
                or self.class_mappings[task_name]["background"] != 0
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
                        self.classes[task_name].append("background")
                        self.class_mappings[task_name] = {
                            class_: idx + 1
                            for class_, idx in self.class_mappings[
                                task_name
                            ].items()
                        }
                        self.class_mappings[task_name]["background"] = 0

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

        if self.color_space == "BGR":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return img, labels

    def _load_data(self, idx: int) -> Tuple[np.ndarray, Labels]:
        """Loads image and its annotations based on index.

        @type idx: int
        @param idx: Index of the image
        @rtype: Tuple[np.ndarray, Labels]
        @return: Image as C{np.ndarray} in RGB format and a dictionary
            with all the present annotations
        """

        if not self.idx_to_df_row:
            raise ValueError(
                f"No data found in dataset '{self.dataset.identifier}' for {self.view} views"
            )

        ann_indices = self.idx_to_df_row[idx]

        ann_rows = [self.df.row(row) for row in ann_indices]
        if not self.dataset.is_remote:
            img_path = ann_rows[0][-1]
        else:
            uuid = ann_rows[0][7]
            file_extension = ann_rows[0][0].rsplit(".", 1)[-1]
            img_path = self.dataset.media_path / f"{uuid}.{file_extension}"

        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)

        labels_by_task: Dict[str, List[Annotation]] = defaultdict(list)
        class_ids_by_task: Dict[str, List[int]] = defaultdict(list)
        instance_ids_by_task: Dict[str, List[int]] = defaultdict(list)
        metadata_by_task: Dict[str, List[Union[str, int, float]]] = (
            defaultdict(list)
        )

        for annotation_data in ann_rows:
            task_name: str = annotation_data[2]
            class_name: Optional[str] = annotation_data[3]
            instance_id: int = annotation_data[4]
            task_type: str = annotation_data[5]
            ann_str: Optional[str] = annotation_data[6]

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
                        self.class_mappings[task_name][class_name]
                    )
                else:
                    class_ids_by_task[full_task_name].append(0)
                instance_ids_by_task[full_task_name].append(instance_id)

        labels: Labels = {}
        for task, metadata in metadata_by_task.items():
            labels[task] = np.array(metadata)

        for task, anns in labels_by_task.items():
            assert anns, f"No annotations found for task {task_name}"
            instance_ids = instance_ids_by_task[task]

            anns = [
                ann
                for _, ann in sorted(
                    zip(instance_ids, anns), key=lambda x: x[0]
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
                background_idx = self.class_mappings[task_name]["background"]
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
                    "Samples will include repetitions."
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
        augmentation_engine: Union[Literal["albumentations"], str],
        augmentation_config: Union[List[Dict[str, Any]], PathType],
        height: Optional[int],
        width: Optional[int],
        keep_aspect_ratio: bool,
    ) -> Optional[AugmentationEngine]:
        if isinstance(augmentation_config, (Path, str)):
            with open(augmentation_config) as file:
                augmentation_config = cast(
                    List[Dict[str, Any]], yaml.safe_load(file) or []
                )
        if augmentation_config and (width is None or height is None):
            raise ValueError(
                "Height and width must be provided when using augmentations"
            )

        if height is None or width is None:
            return None

        targets = {
            f"{task_name}/{task_type}": task_type
            for task_name, task_types in self.dataset.get_tasks().items()
            for task_type in task_types
        }

        return AUGMENTATION_ENGINES.get(augmentation_engine)(
            height=height,
            width=width,
            config=augmentation_config,
            targets=targets,
            keep_aspect_ratio=keep_aspect_ratio,
            is_validation_pipeline="train" not in self.view,
        )
