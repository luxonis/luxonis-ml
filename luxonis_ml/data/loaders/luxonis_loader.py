import json
import logging
import random
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from typing_extensions import override

from luxonis_ml.data.augmentations import Augmentations
from luxonis_ml.data.datasets import (
    Annotation,
    LuxonisDataset,
    load_annotation,
)
from luxonis_ml.data.loaders.base_loader import (
    BaseLoader,
    LuxonisLoaderOutput,
)
from luxonis_ml.data.utils import (
    Labels,
    get_task_name,
    get_task_type,
    split_task,
    task_type_iterator,
)

logger = logging.getLogger(__name__)


class LuxonisLoader(BaseLoader):
    def __init__(
        self,
        dataset: LuxonisDataset,
        view: Union[str, List[str]] = "train",
        stream: bool = False,
        augmentations: Optional[Augmentations] = None,
        *,
        force_resync: bool = False,
    ) -> None:
        """A loader class used for loading data from L{LuxonisDataset}.

        @type dataset: LuxonisDataset
        @param dataset: Instance of C{LuxonisDataset} to use.
        @type view: Union[str, List[str]]
        @param view: What splits to use. Can be either a single split or
            a list of splits. Defaults to C{"train"}.
        @type stream: bool
        @param stream: Flag for data streaming. Defaults to C{False}.
        @type augmentations: Optional[luxonis_ml.loader.Augmentations]
        @param augmentations: Augmentation class that performs
            augmentations. Defaults to C{None}.
        @type force_resync: bool
        @param force_resync: Flag to force resync from cloud. Defaults
            to C{False}.
        """

        self.logger = logging.getLogger(__name__)

        self.dataset = dataset
        self.stream = stream
        self.sync_mode = self.dataset.is_remote and not self.stream

        if self.sync_mode:
            self.dataset.sync_from_cloud(force=force_resync)

        if isinstance(view, str):
            view = [view]
        self.view = view

        df = self.dataset._load_df_offline()
        if df is None:
            raise FileNotFoundError("Cannot find dataframe")
        self.df = df

        if not self.dataset.is_remote or not self.stream:
            file_index = self.dataset._get_file_index()
            if file_index is None:
                raise FileNotFoundError("Cannot find file index")
            self.df = self.df.join(file_index, on="uuid").drop("file_right")
        else:
            raise NotImplementedError(
                "Streaming for remote bucket storage not implemented yet"
            )

        self.classes, self.classes_by_task = self.dataset.get_classes()
        self.augmentations = augmentations
        self.instances = []
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
            boolean_mask = df["uuid"] == uuid
            row_indexes = boolean_mask.arg_true().to_list()
            self.idx_to_df_row.append(row_indexes)

        self.class_mappings: Dict[str, Dict[str, int]] = {}
        for task in df["task_name"].unique():
            if not task:
                continue
            class_mapping = {
                class_: i
                for i, class_ in enumerate(
                    sorted(
                        self.classes_by_task.get(task, []),
                        key=lambda x: {"background": -1}.get(x, 0),
                    )
                )
            }
            self.class_mappings[task] = class_mapping

        self.add_background = False
        _, test_labels = self._load_image_with_annotations(0)
        for task, seg_masks in task_type_iterator(test_labels, "segmentation"):
            task = get_task_name(task)
            if seg_masks.shape[0] > 1:
                unassigned_pixels = np.sum(seg_masks, axis=0) == 0

                if np.any(unassigned_pixels):
                    logger.warning(
                        "Found unassigned pixels in segmentation masks. Assigning them to `background` class (class index 0). If this is not desired then make sure all pixels are assigned to one class or rename your background class."
                    )
                    self.add_background = True
                    if "background" not in self.classes_by_task[task]:
                        self.classes_by_task[task].append("background")
                        self.class_mappings[task] = {
                            class_: idx + 1
                            for class_, idx in self.class_mappings[
                                task
                            ].items()
                        }
                        self.class_mappings[task]["background"] = 0

    @override
    def __len__(self) -> int:
        """Returns length of the dataset.

        @rtype: int
        @return: Length of the loader.
        """
        return len(self.instances)

    @override
    def __getitem__(self, idx: int) -> LuxonisLoaderOutput:
        """Function to load a sample consisting of an image and its
        annotations.

        @type idx: int
        @param idx: The integer index of the sample to retrieve.
        @rtype: L{LuxonisLoaderOutput}
        @return: The loader ouput consisting of the image and a
            dictionary defining its annotations.
        """

        if self.augmentations is None:
            return self._load_image_with_annotations(idx)

        indices = [idx]
        if self.augmentations.is_batched:
            if self.augmentations.batch_size > len(self):
                warnings.warn(
                    f"Augmentations batch_size ({self.augmentations.batch_size}) is larger than dataset size ({len(self)}), samples will include repetitions."
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

        loaded_anns = [self._load_image_with_annotations(i) for i in indices]
        return self.augmentations.apply(loaded_anns)

    def _load_image_with_annotations(
        self, idx: int
    ) -> Tuple[np.ndarray, Labels]:
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
        elif not self.stream:
            uuid = ann_rows[0][8]
            file_extension = ann_rows[0][0].rsplit(".", 1)[-1]
            img_path = self.dataset.media_path / f"{uuid}.{file_extension}"
        else:
            # TODO: add support for streaming remote storage
            raise NotImplementedError(
                "Streaming for remote bucket storage not implemented yet"
            )

        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)

        labels_by_task: Dict[str, List[Annotation]] = defaultdict(list)
        class_ids_by_task: Dict[str, List[int]] = defaultdict(list)
        instance_ids_by_task: Dict[str, List[int]] = defaultdict(list)
        metadata_by_task: Dict[str, List[Union[str, int, float]]] = (
            defaultdict(list)
        )

        for annotation_data in ann_rows:
            task_name: str = annotation_data[2]
            class_name: Optional[str] = annotation_data[4]
            instance_id: int = annotation_data[5]
            task_type: str = annotation_data[6]
            ann_str: Optional[str] = annotation_data[7]

            if ann_str is None:
                continue

            data = json.loads(ann_str)
            full_task_name = f"{task_name}/{task_type}"
            task_type = get_task_type(full_task_name)
            if task_type == "array" and self.dataset.is_remote:
                data["path"] = self.dataset.arrays_path / data["path"]

            if task_type.startswith("metadata/"):
                metadata_by_task[full_task_name].append(data)
            else:
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
                len(self.classes_by_task[task_name]),
            )
            if (
                self.add_background
                and task_type == "segmentation"
                and len(self.class_mappings[task_name]) > 1
            ):
                unassigned_pixels = ~np.any(array, axis=0)
                background_idx = self.class_mappings[task_name]["background"]
                array[background_idx, unassigned_pixels] = 1

            labels[task] = array

        return img, labels
