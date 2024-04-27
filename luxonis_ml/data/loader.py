import glob
import json
import logging
import os
import random
import warnings
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import pycocotools.mask as mask_util
from PIL import Image, ImageDraw
from typing_extensions import TypeAlias

from luxonis_ml.enums import LabelType

from .augmentations import Augmentations
from .datasets import LuxonisDataset
from .utils.enums import BucketStorage

Labels: TypeAlias = Dict[LabelType, np.ndarray]
"""C{Labels} is a dictionary of a label type and its annotations as L{numpy
arrays<np.ndarray>}."""


class Annotations(dict):
    """A dictionary of task group names and their annotations.

    Inherits from the built-in Python dictionary class.

    Acts like simple dictionary when only one group is present, otherwise acts like a
    nested dictionary of task group names and their annotations.
    """

    @property
    def groups(self) -> List[str]:
        return list(self.keys())

    def __getitem__(self, key: Union[LabelType, str]) -> Union[np.ndarray, Labels]:
        if isinstance(key, LabelType):
            if len(self.groups) == 1:
                return super().__getitem__(self.groups[0])[key]
            raise ValueError(
                "Multiple groups present, please access the specific group first."
            )
        elif isinstance(key, str):
            if key in self.groups:
                return super().__getitem__(key)
            raise KeyError(f"Group '{key}' not found in annotations.")
        else:
            raise TypeError("Key must be of type LabelType or str.")

    def __contains__(self, key: Union[LabelType, str]) -> bool:
        if isinstance(key, LabelType):
            if len(self.groups) == 1:
                return key in super().__getitem__(self.groups[0])
            raise ValueError(
                "Multiple groups present, please access the specific group first."
            )
        elif isinstance(key, str):
            return key in self.groups
        else:
            raise TypeError("Key must be of type LabelType or str.")


LuxonisLoaderOutput: TypeAlias = Tuple[np.ndarray, Annotations]
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
        self.ns = len(self.classes_by_task[LabelType.SEGMENTATION])
        self.nk = {
            cls: len(skeleton["labels"])
            for cls, skeleton in self.dataset.get_skeletons().items()
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

        return img, Annotations(group_annotations)

    def _load_classification(self, rows: pd.DataFrame) -> np.ndarray:
        classes = [row[1]["class"] for row in rows.iterrows() if bool(row[1]["value"])]
        classify = np.zeros(self.nc)
        for class_ in classes:
            class_ = self.classes_by_task[LabelType.CLASSIFICATION].index(class_)
            classify[class_] = classify[class_] + 1
        classify[classify > 0] = 1
        return classify

    def _load_box(self, rows: pd.DataFrame) -> np.ndarray:
        boxes = np.zeros((0, 5))
        for row in rows.iterrows():
            row = row[1]
            class_ = self.classes_by_task[LabelType.BOUNDINGBOX].index(row["class"])
            det = json.loads(row["value"])
            box = np.array([class_, det[0], det[1], det[2], det[3]]).reshape(1, 5)
            boxes = np.append(boxes, box, axis=0)
        return boxes

    def _load_polyline(self, rows: pd.DataFrame, height: int, width: int) -> np.ndarray:
        seg = np.zeros((self.ns, height, width))
        for row in rows.iterrows():
            row = row[1]
            class_ = self.classes_by_task[LabelType.SEGMENTATION].index(row["class"])
            polyline = json.loads(row["value"])
            polyline = [
                (round(coord[0] * width), round(coord[1] * height))
                for coord in polyline
            ]
            mask = Image.new("L", (width, height), 0)
            draw = ImageDraw.Draw(mask)
            draw.polygon(polyline, fill=1, outline=1)
            mask = np.array(mask)
            seg[class_, ...] = seg[class_, ...] + mask
        seg[seg > 0] = 1
        return seg

    def _load_segmentation(
        self, rows: pd.DataFrame, height: int, width: int
    ) -> np.ndarray:
        seg = np.zeros((self.ns, height, width))
        for row in rows.iterrows():
            row = row[1]
            class_ = self.classes_by_task[LabelType.SEGMENTATION].index(row["class"])
            height, width, counts_str = json.loads(row["value"])
            mask = mask_util.decode(
                {"counts": counts_str.encode("utf-8"), "size": [height, width]}
            )
            seg[class_, ...] = seg[class_, ...] + mask
        seg[seg > 0] = 1
        return seg

    def _load_keypoints(self, rows: pd.DataFrame) -> np.ndarray:
        keypoints = np.zeros((0, self.max_nk * 3 + 1))
        for row in rows.iterrows():
            row = row[1]
            class_ = self.classes_by_task[LabelType.KEYPOINT].index(row["class"])
            kps = np.array(json.loads(row["value"])).reshape((-1, 3)).astype(np.float32)
            kps = kps.flatten()
            nk = len(kps)
            kps = np.concatenate([[class_], kps])
            points = np.zeros((1, self.max_nk * 3 + 1))
            points[0, : nk + 1] = kps
            keypoints = np.append(keypoints, points, axis=0)
        return keypoints

    def _load_image_with_annotations(
        self, idx: int
    ) -> Tuple[np.ndarray, Dict[str, dict]]:
        """Loads image and its annotations based on index.

        @type idx: int
        @param idx: Index of the image
        @rtype: Tuple[L{np.ndarray}, dict]
        @return: Image as L{np.ndarray} in RGB format and a dictionary with all the
            present annotations
        """

        instance_id = self.instances[idx]
        sub_df = self.df.loc[instance_id]
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
        annotations = {}

        if sub_df.ndim == 1:
            sub_df = pd.DataFrame([sub_df])

        task_groups = sub_df["task_group"].unique()

        for task_group in task_groups:
            group_df = sub_df[sub_df["task_group"] == task_group]
            _annotations = {}
            classification_rows = group_df[sub_df["type"] == "classification"]
            box_rows = group_df[sub_df["type"] == "box"]
            segmentation_rows = group_df[sub_df["type"] == "segmentation"]
            polyline_rows = group_df[sub_df["type"] == "polyline"]
            keypoints_rows = group_df[sub_df["type"] == "keypoints"]

            if not classification_rows.empty:
                _annotations[LabelType.CLASSIFICATION] = self._load_classification(
                    classification_rows
                )

            if not box_rows.empty:
                _annotations[LabelType.BOUNDINGBOX] = self._load_box(box_rows)

            if not polyline_rows.empty:
                _annotations[LabelType.SEGMENTATION] = self._load_polyline(
                    polyline_rows, height, width
                )

            if not segmentation_rows.empty:
                _annotations[LabelType.SEGMENTATION] = self._load_segmentation(
                    segmentation_rows, height, width
                )

            if not keypoints_rows.empty:
                _annotations[LabelType.KEYPOINT] = self._load_keypoints(keypoints_rows)

            annotations[task_group] = _annotations

        return img, annotations
