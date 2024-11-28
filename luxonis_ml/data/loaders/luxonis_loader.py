import json
import logging
import random
import warnings
from collections import defaultdict
from operator import itemgetter
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from ..augmentations import Augmentations
from ..datasets import LuxonisDataset, load_annotation
from ..utils.enums import LabelType
from .base_loader import BaseLoader, Labels, LuxonisLoaderOutput

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

        self.class_mappings = {}
        for task in df["task"].unique():
            if not task:
                continue
            class_mapping = {
                class_: i
                for i, class_ in enumerate(
                    sorted(
                        self.classes_by_task[task],
                        key=lambda x: {"background": -1}.get(x, 0),
                    )
                )
            }
            self.class_mappings[task] = class_mapping

        self.add_background = False
        # TODO: possibly check more labels
        test_image, test_labels = self._load_image_with_annotations(0)
        if LabelType.SEGMENTATION in test_labels:
            seg_masks = test_labels[LabelType.SEGMENTATION][0]
            if seg_masks.shape[0] > 1:
                unassigned_pixels = np.sum(seg_masks, axis=0) == 0

                if np.any(unassigned_pixels):
                    logger.warning(
                        "Found unassigned pixels in segmentation masks. Assigning them to `background` class (class index 0). If this is not desired then make sure all pixels are assigned to one class or rename your background class."
                    )
                    self.add_background = True
                    if (
                        "background"
                        not in self.classes_by_task[LabelType.SEGMENTATION]
                    ):
                        self.classes_by_task[LabelType.SEGMENTATION].append(
                            "background"
                        )
                        self.class_mappings[LabelType.SEGMENTATION] = {
                            class_: idx + 1
                            for class_, idx in self.class_mappings[
                                task
                            ].items()
                        }
                        self.class_mappings[LabelType.SEGMENTATION][
                            "background"
                        ] = 0

    def __len__(self) -> int:
        """Returns length of the dataset.

        @rtype: int
        @return: Length of the loader.
        """
        return len(self.instances)

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
            if self.augmentations.aug_batch_size > len(self):
                warnings.warn(
                    f"Augmentations batch_size ({self.augmentations.aug_batch_size}) is larger than dataset size ({len(self)}), samples will include repetitions."
                )
                other_indices = [i for i in range(len(self)) if i != idx]
                picked_indices = random.choices(
                    other_indices, k=self.augmentations.aug_batch_size - 1
                )
            else:
                picked_indices = set()
                max_val = len(self)
                while (
                    len(picked_indices) < self.augmentations.aug_batch_size - 1
                ):
                    rand_idx = random.randint(0, max_val - 1)
                    if rand_idx != idx and rand_idx not in picked_indices:
                        picked_indices.add(rand_idx)
                picked_indices = list(picked_indices)

            indices.extend(picked_indices)

        out_dict: Dict[str, Tuple[np.ndarray, LabelType]] = {}
        loaded_anns = [self._load_image_with_annotations(i) for i in indices]
        random_state = random.getstate()
        np_random_state = np.random.get_state()
        if not loaded_anns[0][1]:
            img, aug_annotations = self.augmentations(
                [(loaded_anns[i][0], {}) for i in range(len(loaded_anns))],
            )

        while loaded_anns[0][1]:
            aug_input_data = []
            label_to_task = {}
            nk = 0
            ns = 0
            for img, annotations in loaded_anns:
                label_dict: Dict[LabelType, np.ndarray] = {}
                task_dict: Dict[LabelType, str] = {}
                for task in sorted(list(annotations.keys())):
                    array, label_type = annotations[task]
                    if label_type not in label_dict:
                        # ensure that bounding box annotations are added to the
                        # `label_dict` before keypoints
                        if label_type == LabelType.KEYPOINTS:
                            if (
                                LabelType.BOUNDINGBOX
                                in map(
                                    itemgetter(1), list(annotations.values())
                                )
                                and LabelType.BOUNDINGBOX not in label_dict  # type: ignore
                            ):
                                continue

                            if (
                                LabelType.BOUNDINGBOX in label_dict  # type: ignore
                                and LabelType.BOUNDINGBOX
                                in map(
                                    itemgetter(1), list(annotations.values())
                                )
                            ):
                                bbox_task = task_dict[LabelType.BOUNDINGBOX]
                                *_, bbox_suffix = bbox_task.split("-", 1)
                                *_, kp_suffix = task.split("-", 1)
                                if bbox_suffix != kp_suffix:
                                    continue

                        label_dict[label_type] = array
                        label_to_task[label_type] = task
                        task_dict[label_type] = task
                        annotations.pop(task)
                        if label_type == LabelType.KEYPOINTS:
                            nk = (array.shape[1] - 1) // 3
                        if label_type == LabelType.SEGMENTATION:
                            ns = array.shape[0]

                aug_input_data.append((img, label_dict))

            # NOTE: To ensure the same augmentation is applied to all samples
            # in case of multiple tasks per LabelType
            random.setstate(random_state)
            np.random.set_state(np_random_state)

            img, aug_annotations = self.augmentations(
                aug_input_data, nk=nk, ns=ns
            )
            for label_type, array in aug_annotations.items():
                out_dict[label_to_task[label_type]] = (array, label_type)

        return img, out_dict

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
            img_path = ann_rows[0][8]
        elif not self.stream:
            uuid = ann_rows[0][0]
            file_extension = ann_rows[0][8].rsplit(".", 1)[-1]
            img_path = self.dataset.media_path / f"{uuid}.{file_extension}"
        else:
            # TODO: add support for streaming remote storage
            raise NotImplementedError(
                "Streaming for remote bucket storage not implemented yet"
            )

        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        height, width, _ = img.shape

        labels_by_task = defaultdict(list)
        instance_counters = defaultdict(int)
        for annotation_data in ann_rows:
            _, _, type_, _, class_, instance_id, task, ann_str, _ = (
                annotation_data
            )
            if instance_id < 0:
                instance_counters[task] += 1
                instance_id = instance_counters[task]
            data = json.loads(ann_str)
            if type_ == "ArrayAnnotation" and self.dataset.is_remote:
                data["path"] = self.dataset.arrays_path / data["path"]
            data.update(
                {
                    "class": class_,
                    "task": task,
                    "instance_id": instance_id,
                }
            )
            if type_ != "NoneType":
                annotation = load_annotation(type_, data)
                labels_by_task[task].append(annotation)

        labels: Labels = {}
        for task, anns in labels_by_task.items():
            assert anns, f"No annotations found for task {task}"
            anns.sort(key=lambda x: x.instance_id)
            array = anns[0].combine_to_numpy(
                anns, self.class_mappings[task], width=width, height=height
            )
            if self.add_background and task == LabelType.SEGMENTATION:
                unassigned_pixels = ~np.any(array, axis=0)
                background_idx = self.class_mappings[task]["background"]
                array[background_idx, unassigned_pixels] = 1

            labels[task] = (array, anns[0]._label_type)

        missing_tasks = set(self.classes_by_task) - set(labels_by_task)
        if missing_tasks:
            for task in missing_tasks:
                class_mapping_len = len(self.class_mappings[task])
                if task == LabelType.SEGMENTATION:
                    empty_array = np.zeros(
                        (class_mapping_len, height, width),
                        dtype=np.uint8,
                    )
                elif task == LabelType.BOUNDINGBOX:
                    empty_array = np.zeros((0, 6), dtype=np.float32)
                elif task == LabelType.KEYPOINTS:
                    empty_array = np.zeros((0, 3), dtype=np.float32)
                elif task == LabelType.CLASSIFICATION:
                    empty_array = np.zeros(
                        (0, class_mapping_len), dtype=np.float32
                    )
                labels[task] = (empty_array, task)

        return img, labels
