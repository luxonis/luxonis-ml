import json
import logging
import random
import warnings
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import polars as pl

from ..augmentations import Augmentations
from ..datasets import Annotation, LuxonisDataset, load_annotation
from ..utils.enums import LabelType
from .base_loader import BaseLoader, Labels, LuxonisLoaderOutput


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
        self.sync_mode = self.dataset.is_remote and not self.stream

        if self.sync_mode:
            self.logger.info("Syncing from cloud...")
            self.dataset.sync_from_cloud()
            classes_file = self.dataset.metadata_path / "classes.json"
            with open(classes_file) as file:
                synced_classes = json.load(file)
            for task in synced_classes:
                self.dataset.set_classes(classes=synced_classes[task], task=task)
            skeletons_file = self.dataset.metadata_path / "skeletons.json"
            try:
                with open(skeletons_file, "r") as file:
                    synced_skeletons = json.load(file)
                self.dataset.set_skeletons(synced_skeletons)
            except FileNotFoundError:
                self.logger.warning("Skeletons file not found at %s", skeletons_file)

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

                aug_input_data.append((img, label_dict))

            # NOTE: To ensure the same augmentation is applied to all samples
            # in case of multiple tasks per LabelType
            random.setstate(random_state)
            np.random.set_state(np_random_state)

            img, aug_annotations = self.augmentations(aug_input_data, nk=nk)
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

        uuid = self.instances[idx]
        df = self.df.filter(pl.col("uuid") == uuid)
        if not self.dataset.is_remote:
            img_path = list(df.select("original_filepath"))[0][0]
        elif not self.stream:
            img_path = next(self.dataset.media_path.glob(f"{uuid}.*"))
        else:
            # TODO: add support for streaming remote storage
            raise NotImplementedError(
                "Streaming for remote bucket storage not implemented yet"
            )

        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)

        height, width, _ = img.shape
        labels: Labels = {}

        for task in df["task"].unique():
            if not task:
                continue
            sub_df = df.filter(pl.col("task") == task)
            annotations: List[Annotation] = []
            class_mapping = {
                class_: i for i, class_ in enumerate(self.classes_by_task[task])
            }
            for i, (*_, type_, _, class_, instance_id, _, ann_str, _) in enumerate(
                sub_df.rows(named=False)
            ):
                instance_id = instance_id if instance_id > 0 else i
                annotation = load_annotation(
                    type_,
                    ann_str,
                    {"class": class_, "task": task, "instance_id": instance_id},
                )
                annotations.append(annotation)
            assert annotations, f"No annotations found for task {task}"
            annotations.sort(key=lambda x: x.instance_id)
            array = annotations[0].combine_to_numpy(
                annotations, class_mapping, width=width, height=height
            )
            labels[task] = (array, annotations[0]._label_type)

        return img, labels
