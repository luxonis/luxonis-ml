import glob
import json
import logging
import os
import random
import warnings
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from typing_extensions import TypeAlias

from .augmentations import Augmentations
from .datasets import Annotation, LuxonisDataset, load_annotation
from .utils.enums import BucketStorage, LabelType

Labels: TypeAlias = Dict[str, Tuple[np.ndarray, LabelType]]
"""C{Labels} is a dictionary of a label type and its annotations as L{numpy
arrays<np.ndarray>}."""


LuxonisLoaderOutput: TypeAlias = Tuple[np.ndarray, Labels]
"""C{LuxonisLoaderOutput} is a tuple of an image as a L{numpy array<np.ndarray>} and a
dictionary of task group names and their annotations as L{Annotations}."""


class BaseLoader(ABC):
    """Base abstract loader class.

    Enforces the L{LuxonisLoaderOutput} output label structure.
    """

    @abstractmethod
    def __len__(self) -> int:
        """Returns the length of the dataset.

        @rtype: int
        @return: Length of the dataset.
        """
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> LuxonisLoaderOutput:
        """Loads sample from dataset.

        @type idx: int
        @param idx: Index of the sample to load.
        @rtype: LuxonisLoaderOutput
        @return: Sample's data in L{LuxonisLoaderOutput} format.
        """
        pass


class LuxonisLoader(BaseLoader):
    def __init__(
        self,
        dataset: LuxonisDataset,
        view: str = "train",
        stream: bool = False,
        augmentations: Optional[Augmentations] = None,
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
        """

        self.logger = logging.getLogger(__name__)

        self.dataset = dataset
        self.stream = stream
        self.sync_mode = (
            self.dataset.bucket_storage != BucketStorage.LOCAL and not self.stream
        )

        if self.sync_mode:
            self.logger.info("Syncing from cloud...")
            self.dataset.sync_from_cloud()
            classes_file = os.path.join(self.dataset.metadata_path, "classes.json")
            with open(classes_file) as file:
                synced_classes = json.load(file)
            for task in synced_classes:
                self.dataset.set_classes(classes=synced_classes[task], task=task)
            skeletons_file = os.path.join(self.dataset.metadata_path, "skeletons.json")
            try:
                with open(skeletons_file, "r") as file:
                    synced_skeletons = json.load(file)
                self.dataset.set_skeletons(synced_skeletons)
            except FileNotFoundError:
                self.logger.warning("Skeletons file not found at %s", skeletons_file)

        if self.dataset.bucket_storage == BucketStorage.LOCAL or not self.stream:
            file_index = self.dataset._get_file_index()
            if file_index is None:
                raise Exception("Cannot find file index")
            self.file_index = file_index
        else:
            raise NotImplementedError(
                "Streaming for remote bucket storage not implemented yet"
            )

        self.view = view

        self.classes, self.classes_by_task = self.dataset.get_classes(
            sync_mode=self.sync_mode
        )
        self.nc = len(self.classes)
        self.ns = len(self.classes_by_task.get("segmentation", []))
        self.nk = {
            cls: len(skeleton["labels"])
            for cls, skeleton in self.dataset.get_skeletons(
                sync_mode=self.sync_mode
            ).items()
        }
        if len(list(self.nk.values())):
            self.max_nk = max(list(self.nk.values()))
        else:
            self.max_nk = 0
        self.augmentations = augmentations

        if self.view in ["train", "val", "test"]:
            splits_path = os.path.join(dataset.metadata_path, "splits.json")
            if not os.path.exists(splits_path):
                raise Exception(
                    "Cannot find splits! Ensure you call dataset.make_splits()"
                )
            with open(splits_path, "r") as file:
                splits = json.load(file)
            self.instances = splits[self.view]
        else:
            raise NotImplementedError

        df = dataset._load_df_offline(sync_mode=self.sync_mode)
        if df is None:
            raise Exception("Cannot find dataframe")
        self.df = df
        self.df.set_index(["instance_id"], inplace=True)

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

        img, group_annotations = self._load_image_with_annotations(idx)

        for task_group in list(group_annotations.keys()):
            annotations = group_annotations[task_group]
            if self.augmentations is not None:
                aug_input_data = [(img, annotations)]
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
                    for i in picked_indices:
                        _img, _group_annotations = self._load_image_with_annotations(i)
                        aug_input_data.append((_img, _group_annotations[task_group]))

                img, annotations = self.augmentations(
                    aug_input_data, nc=self.nc, ns=self.ns, nk=self.max_nk
                )
                group_annotations[task_group] = annotations

        return img, group_annotations

    def _load_image_with_annotations(self, idx: int) -> Tuple[np.ndarray, Labels]:
        """Loads image and its annotations based on index.

        @type idx: int
        @param idx: Index of the image
        @rtype: Tuple[L{np.ndarray}, dict]
        @return: Image as L{np.ndarray} in RGB format and a dictionary with all the
            present annotations
        """

        instance_id = self.instances[idx]
        df = self.df.loc[instance_id]
        if self.dataset.bucket_storage.value == "local":
            matched = self.file_index[self.file_index["instance_id"] == instance_id]
            img_path = list(matched["original_filepath"])[0]
        else:
            if self.dataset.bucket_storage == BucketStorage.LOCAL or not self.stream:
                img_path = os.path.join(self.dataset.media_path, f"{instance_id}.*")
                img_path = glob.glob(img_path)[0]
            else:
                # TODO: add support for streaming remote storage
                raise NotImplementedError

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        height, width, _ = img.shape
        labels: Labels = {}

        if df.ndim == 1:
            df = pd.DataFrame([df])

        for task in df["task"].unique():
            sub_df = df[df["task"] == task]
            annotations: List[Annotation] = []
            class_mapping = {
                class_: i for i, class_ in enumerate(self.classes_by_task[task])
            }
            for _, row in sub_df.iterrows():
                type_ = row["type"]
                class_ = row["class"]
                annotation = load_annotation(
                    type_, row["annotation"], {"class": class_, "task": task}
                )
                annotations.append(annotation)
            array = annotations[0].combine_to_numpy(
                annotations, class_mapping, width=width, height=height
            )
            labels[task] = (array, annotations[0]._label_type)

        return img, labels
