import glob
import json
import logging
import os
import random
import warnings
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import pycocotools.mask as mask_util
from PIL import Image, ImageDraw

from luxonis_ml.enums import LabelType

from .augmentations import Augmentations
from .dataset import LuxonisDataset
from .utils.enums import BucketStorage

Labels = Dict[LabelType, np.ndarray]
"""C{Labels} is a dictionary of a label type and its annotations as L{numpy
arrays<np.ndarray>}."""

LuxonisLoaderOutput = Tuple[np.ndarray, Labels]
"""C{LuxonisLoaderOutput} is a tuple of image and its annotations."""


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
        @param stream: Flag for data streaming. Defaults to False.
        @type augmentations: Optional[luxonis_ml.loader.Augmentations]
        @param augmentations: Augmentation class that performs augmentations. Defaults
            to None.
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

        if self.dataset.bucket_storage == BucketStorage.LOCAL or not self.stream:
            self.file_index = self.dataset._get_file_index()
            if self.file_index is None:
                raise Exception("Cannot find file index")
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

        if self.dataset.online:
            raise NotImplementedError

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

        self.df = dataset._load_df_offline(sync_mode=self.sync_mode)
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

        img, annotations = self._load_image_with_annotations(idx)

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
                aug_input_data.extend(
                    [self._load_image_with_annotations(i) for i in picked_indices]
                )

            img, annotations = self.augmentations(
                aug_input_data, nc=self.nc, ns=self.ns, nk=self.max_nk
            )

        return img, annotations

    def _load_image_with_annotations(self, idx: int) -> Tuple[np.ndarray, Dict]:
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

        ih, iw, _ = img.shape
        annotations = {}

        if sub_df.ndim == 1:
            sub_df = pd.DataFrame([sub_df])

        classification_rows = sub_df[sub_df["type"] == "classification"]
        box_rows = sub_df[sub_df["type"] == "box"]
        segmentation_rows = sub_df[sub_df["type"] == "segmentation"]
        polyline_rows = sub_df[sub_df["type"] == "polyline"]
        keypoints_rows = sub_df[sub_df["type"] == "keypoints"]

        seg = np.zeros((self.ns, ih, iw))

        if len(classification_rows):
            classes = [
                row[1]["class"]
                for row in classification_rows.iterrows()
                if bool(row[1]["value"])
            ]
            classify = np.zeros(self.nc)
            for cls in classes:
                cls = self.classes_by_task[LabelType.CLASSIFICATION].index(cls)
                classify[cls] = classify[cls] + 1
            classify[classify > 0] = 1
            annotations[LabelType.CLASSIFICATION] = classify

        if len(box_rows):
            boxes = np.zeros((0, 5))
            for row in box_rows.iterrows():
                row = row[1]
                cls = self.classes_by_task[LabelType.BOUNDINGBOX].index(row["class"])
                det = json.loads(row["value"])
                box = np.array([cls, det[0], det[1], det[2], det[3]]).reshape(1, 5)
                boxes = np.append(boxes, box, axis=0)
            annotations[LabelType.BOUNDINGBOX] = boxes

        if len(polyline_rows):
            for row in polyline_rows.iterrows():
                row = row[1]
                cls = self.classes_by_task[LabelType.SEGMENTATION].index(row["class"])
                polyline = json.loads(row["value"])
                polyline = [
                    (round(coord[0] * iw), round(coord[1] * ih)) for coord in polyline
                ]
                mask = Image.new("L", (iw, ih), 0)
                draw = ImageDraw.Draw(mask)
                draw.polygon(polyline, fill=1, outline=1)
                mask = np.array(mask)
                seg[cls, ...] = seg[cls, ...] + mask
            seg[seg > 0] = 1
            annotations[LabelType.SEGMENTATION] = seg

        if len(segmentation_rows):
            for row in segmentation_rows.iterrows():
                row = row[1]
                cls = self.classes_by_task[LabelType.SEGMENTATION].index(row["class"])
                height, width, counts_str = json.loads(row["value"])
                mask = mask_util.decode(
                    {"counts": counts_str.encode("utf-8"), "size": [height, width]}
                )
                seg[cls, ...] = seg[cls, ...] + mask
            seg[seg > 0] = 1
            annotations[LabelType.SEGMENTATION] = seg

        if len(keypoints_rows):
            # TODO: test with multi-class keypoint instances where nk's are not equal
            keypoints = np.zeros((0, self.max_nk * 3 + 1))
            for row in keypoints_rows.iterrows():
                row = row[1]
                cls = self.classes_by_task[LabelType.KEYPOINT].index(row["class"])
                kps = (
                    np.array(json.loads(row["value"]))
                    .reshape((-1, 3))
                    .astype(np.float32)
                )
                kps = kps.flatten()
                nk = len(kps)
                kps = np.concatenate([[cls], kps])
                points = np.zeros((1, self.max_nk * 3 + 1))
                points[0, : nk + 1] = kps
                keypoints = np.append(keypoints, points, axis=0)
            annotations[LabelType.KEYPOINT] = keypoints

        return img, annotations
