import json
import logging
import random
import warnings
from collections import defaultdict
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from ..augmentations import Augmentations
from ..datasets import LuxonisDataset, load_annotation
from ..utils.enums import LabelType
from .base_loader import BaseLoader, Labels, LuxonisLoaderOutput


class LuxonisLoader(BaseLoader):
    def __init__(
        self,
        dataset: LuxonisDataset,
        view: str = "train",
        stream: bool = False,
        augmentations: Optional[Augmentations] = None,
        *,
        force_resync: bool = False,
    ) -> None:
        """A loader class used for loading data from L{LuxonisDataset}.

        @type dataset: LuxonisDataset
        @param dataset: LuxonisDataset to use
        @type view: str
        @param view: View of the dataset. Defaults to "train".
        @type stream: bool
        @param stream: Flag for data streaming. Defaults to C{False}.
        @type augmentations: Optional[luxonis_ml.loader.Augmentations]
        @param augmentations: Augmentation class that performs augmentations. Defaults
            to C{None}.
        @type force_resync: bool
        @param force_resync: Flag to force resync from cloud. Defaults to C{False}.
        """

        self.logger = logging.getLogger(__name__)

        self.dataset = dataset
        self.stream = stream
        self.sync_mode = self.dataset.is_remote and not self.stream

        if self.sync_mode:
            self.dataset.sync_from_cloud(force=force_resync)

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
        if self.view in ["train", "val", "test"]:
            splits_path = self.dataset.metadata_path / "splits.json"
            if not splits_path.exists():
                raise RuntimeError(
                    "Cannot find splits! Ensure you call dataset.make_splits()"
                )
            with open(splits_path, "r") as file:
                splits = json.load(file)
            self.instances = splits[self.view]
        else:
            raise NotImplementedError

        self.idx_to_df_row = []
        for uuid in self.instances:
            boolean_mask = df["uuid"] == uuid
            row_indexes = boolean_mask.arg_true().to_list()
            self.idx_to_df_row.append(row_indexes)

        self.class_mappings = {}
        for task in df["task"].unique():
            class_mapping = {
                class_: i for i, class_ in enumerate(self.classes_by_task[task])
            }
            self.class_mappings[task] = class_mapping

    def __len__(self) -> int:
        """Returns length of the dataset.

        @rtype: int
        @return: Length of dataset.
        """
        return len(self.instances)

    def __getitem__(self, idx: int) -> LuxonisLoaderOutput:
        """Function to load a sample consisting of an image and its annotations.

        @type idx: int
        @param idx: The (often random) integer index to retrieve a sample from the
            dataset.
        @rtype: LuxonisLoaderOutput
        @return: The loader ouput consisting of the image and a dictionary defining its
            annotations.
        """

        if self.augmentations is None:
            return self._load_image_with_annotations(idx)

        indices = [idx]
        if self.augmentations.is_batched:
            other_indices = [i for i in range(len(self)) if i != idx]
            if self.augmentations.aug_batch_size > len(self):
                warnings.warn(
                    f"Augmentations batch_size ({self.augmentations.aug_batch_size}) is larger than dataset size ({len(self)}), samples will include repetitions."
                )
                random_fun = random.choices
            else:
                random_fun = random.sample
            picked_indices = random_fun(
                other_indices, k=self.augmentations.aug_batch_size - 1
            )
            indices.extend(picked_indices)

        out_dict: Dict[str, Tuple[np.ndarray, LabelType]] = {}
        loaded_anns = [self._load_image_with_annotations(i) for i in indices]
        random_state = random.getstate()
        np_random_state = np.random.get_state()
        while loaded_anns[0][1]:
            aug_input_data = []
            label_to_task = {}
            nk = 0
            ns = 0
            for img, annotations in loaded_anns:
                label_dict: Dict[LabelType, np.ndarray] = {}
                for task in sorted(list(annotations.keys())):
                    array, label_type = annotations[task]
                    if label_type not in label_dict:
                        label_dict[label_type] = array
                        label_to_task[label_type] = task
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

            img, aug_annotations = self.augmentations(aug_input_data, nk=nk, ns=ns)
            for label_type, array in aug_annotations.items():
                out_dict[label_to_task[label_type]] = (array, label_type)

        return img, out_dict  # type: ignore

    def _load_image_with_annotations(self, idx: int) -> Tuple[np.ndarray, Labels]:
        """Loads image and its annotations based on index.

        @type idx: int
        @param idx: Index of the image
        @rtype: Tuple[L{np.ndarray}, dict]
        @return: Image as L{np.ndarray} in RGB format and a dictionary with all the
            present annotations
        """

        ann_indices = self.idx_to_df_row[idx]
        ann_rows = [self.df.row(row) for row in ann_indices]
        if not self.dataset.is_remote:
            img_path = ann_rows[0][8]
        elif not self.stream:
            uuid = ann_rows[0][0]
            img_path = next(self.dataset.media_path.glob(f"{uuid}.*"))
        else:
            # TODO: add support for streaming remote storage
            raise NotImplementedError(
                "Streaming for remote bucket storage not implemented yet"
            )

        img = np.array(Image.open(img_path).convert("RGB"))
        height, width, _ = img.shape

        labels_by_task = defaultdict(list)
        instance_counters = defaultdict(int)
        for annotation_data in ann_rows:
            _, _, type_, _, class_, instance_id, task, ann_str, _ = annotation_data
            if instance_id < 0:
                instance_counters[task] += 1
                instance_id = instance_counters[task]
            annotation = load_annotation(
                type_,
                ann_str,
                {"class": class_, "task": task, "instance_id": instance_id},
            )
            labels_by_task[task].append(annotation)

        labels: Labels = {}
        for task, anns in labels_by_task.items():
            assert anns, f"No annotations found for task {task}"
            anns.sort(key=lambda x: x.instance_id)
            array = anns[0].combine_to_numpy(
                anns, self.class_mappings[task], width=width, height=height
            )
            labels[task] = (array, anns[0]._label_type)

        return img, labels
