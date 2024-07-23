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
from ..utils.enums import BucketStorage, LabelType
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
        self.sync_mode = (
            self.dataset.bucket_storage != BucketStorage.LOCAL and not self.stream
        )

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

        self.classes, self.classes_by_task = self.dataset.get_classes(
            sync_mode=self.sync_mode
        )
        self.augmentations = augmentations
        if self.view in ["train", "val", "test"]:
            splits_path = dataset.metadata_path / "splits.json"
            if not splits_path.exists():
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
            raise FileNotFoundError("Cannot find dataframe")
        self.df = df

        if self.dataset.bucket_storage == BucketStorage.LOCAL or not self.stream:
            file_index = self.dataset._get_file_index()
            if file_index is None:
                raise FileNotFoundError("Cannot find file index")
            self.df = self.df.join(file_index, on="uuid").drop("file_right")
        else:
            raise NotImplementedError(
                "Streaming for remote bucket storage not implemented yet"
            )

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

        ############################################## DUMMY ANOTATIONS
        
        image_shape = (416, 416, 3)
        dummy_image = np.random.rand(*image_shape).astype(np.float32)
        fixed_annotation = {
            'boundingbox': (
                np.array([
                    [46, 0.2104375, 0.212889355, 0.2089375, 0.615866739],
                    [46, 0.0, 0.24195084, 0.2891875, 0.58977993],
                    [46, 0.07146875, 0.192160076, 0.254640625, 0.41410279],
                    [46, 0.17371875, 0.290931744, 0.053921875, 0.136741603],
                    [46, 0.595453125, 0.204821912, 0.2746875, 0.616692172],
                    [20, 0.56953125, 0.517022616, 0.038203125, 0.0931026544],
                    [20, 0.22865625, 0.411304848, 0.07759375, 0.154667524],
                    [20, 0.3678125, 0.401072589, 0.038828125, 0.0912337486],
                    [46, 0.0, 0.536521533, 0.18696875, 0.287749187],
                    [20, 0.000046875, 0.433747291, 0.105546875, 0.19807286],
                    [46, 0.5403125, 0.282116739, 0.077953125, 0.274542254],
                    [21, 0.681625, 0.538733072, 0.020453125, 0.0203866468],
                    [11, 0.431625, 0.266604821, 0.136890625, 0.116121343]
                ]),
                LabelType.BOUNDINGBOX
            )
        }
        return dummy_image, fixed_annotation

        ##############################################
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

        uuid = self.instances[idx]
        df = self.df.filter(pl.col("uuid") == uuid)
        if self.dataset.bucket_storage == BucketStorage.LOCAL:
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
