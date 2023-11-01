import cv2
from PIL import Image, ImageDraw
import numpy as np
import torch
import random
import warnings
import os, glob
import json
from typing import Optional, Tuple, Dict
from luxonis_ml.enums import LabelType


class LuxonisLoader(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset: "luxonis_ml.data.LuxonisDataset",
        view: str = "train",
        stream: bool = False,
        augmentations: Optional["luxonis_ml.loader.Augmentations"] = None,
    ) -> None:
        """LuxonisLoader for loading data from LuxonisDataset

        Args:
            dataset (luxonis_ml.data.LuxonisDataset): LuxonisDataset to use
            view (str, optional): View of the dataset. Defaults to "train".
            stream (bool, optional): Flag for data streaming. Defaults to False.
            augmentations (Optional[luxonis_ml.loader.Augmentations], optional): Augmentation class that performs augmentations. Defaults to None.
        """

        self.dataset = dataset
        self.stream = stream
        self.view = view
        self.classes, self.classes_by_task = self.dataset.get_classes()
        self.nc = len(self.classes)
        self.ns = len(self.classes_by_task[LabelType.SEGMENTATION])
        self.nk = {
            cls: len(skeleton["labels"])
            for cls, skeleton in self.dataset.get_skeletons().items()
        }
        self.max_nk = max(list(self.nk.values()))
        self.augmentations = augmentations

        if not self.stream and self.dataset.bucket_storage.value != "local":
            self.dataset.sync_from_cloud()

        if self.view in ["train", "val", "test"]:
            splits_path = os.path.join(dataset.path, "splits.json")
            if not os.path.exists(splits_path):
                raise Exception(
                    "Cannot find splits! Ensure you call dataset.make_splits()"
                )
            with open(splits_path, "r") as file:
                splits = json.load(file)
            self.instances = splits[self.view]
        else:
            raise NotImplementedError

        self.df = dataset._load_df_offline()
        self.df.set_index(["instance_id"], inplace=True)

    def __len__(self) -> int:
        """Returns length of the pytorch dataset"""
        return len(self.instances)

    def __getitem__(self, idx: int) -> Tuple[torch.tensor, Dict]:
        """Function to load a sample"""

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

        img = np.transpose(img, (2, 0, 1))  # HWC to CHW
        img = torch.tensor(img)
        for key in annotations:
            annotations[key] = torch.tensor(annotations[key])

        return img, annotations

    def _load_image_with_annotations(self, idx: int) -> Tuple[np.ndarray, Dict]:
        """Loads image and its annotations based on index.

        Args:
            idx (int): Index of the image

        Returns:
            Tuple[np.ndarray, dict]: Image as np.ndarray in RGB format and dict with all present annotations
        """

        instance_id = self.instances[idx]
        sub_df = self.df.loc[instance_id]
        img_path = os.path.join(self.dataset.media_path, f"{instance_id}.*")
        img_path = glob.glob(img_path)[0]

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        ih, iw, _ = img.shape
        annotations = {}

        classification_rows = sub_df[sub_df["type"] == "classification"]
        box_rows = sub_df[sub_df["type"] == "box"]
        segmentation_rows = sub_df[sub_df["type"] == "polyline"]
        keypoints_rows = sub_df[sub_df["type"] == "keypoints"]

        if len(classification_rows):
            classes = [
                row[1]["class"]
                for row in classification_rows.iterrows()
                if bool(row[1]["value"])
            ]
            classify = np.zeros(self.nc)
            for cls in classes:
                cls = self.classes.index(cls)
                classify[cls] = classify[cls] + 1
            classify[classify > 0] = 1
            annotations[LabelType.CLASSIFICATION] = classify

        if len(box_rows):
            boxes = np.zeros((0, 5))
            for row in box_rows.iterrows():
                row = row[1]
                cls = self.classes.index(row["class"])
                det = json.loads(row["value"])
                box = np.array([cls, det[0], det[1], det[2], det[3]]).reshape(1, 5)
                boxes = np.append(boxes, box, axis=0)
            annotations[LabelType.BOUNDINGBOX] = boxes

        if len(segmentation_rows):
            seg = np.zeros((self.ns, ih, iw))
            for row in segmentation_rows.iterrows():
                row = row[1]
                cls = self.classes.index(row["class"])
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

        if len(keypoints_rows):
            # TODO: test with multi-class keypoint instances where nk's are not equal
            keypoints = np.zeros((0, self.max_nk * 3 + 1))
            for row in keypoints_rows.iterrows():
                row = row[1]
                cls = self.classes.index(row["class"])
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

    @staticmethod
    def collate_fn(batch: list) -> Tuple[torch.tensor, Dict]:
        """Default collate function used for training

        Args:
            batch (list): List of images and their annotations

        Returns:
            Tuple[torch.FloatTensor, Dict]:
                imgs: Tensor of images (torch.float32) of shape [N, 3, H, W]
                out_annotations: Dictionary with annotations
                    {
                        LabelType.CLASSIFICATION: Tensor of shape [N, classes] with value 1 for present class
                        LabelType.SEGMENTATION: Tensor of shape [N, classes, H, W] with value 1 for pixels that are part of the class
                        LabelType.BOUNDINGBOX: Tensor of shape [instances, 6] with [image_id, class, x_min_norm, y_min_norm, w_norm, h_norm]
                        LabelType.KEYPOINT: Tensor of shape [instances, n_keypoints*3] with [image_id, x1_norm, y1_norm, vis1, x2_norm, y2_norm, vis2, ...]
                    }
        """

        zipped = zip(*batch)
        img, anno_dicts = zipped
        imgs = torch.stack(img, 0)

        present_annotations = anno_dicts[0].keys()
        out_annotations = {anno: None for anno in present_annotations}

        if LabelType.CLASSIFICATION in present_annotations:
            class_annos = [anno[LabelType.CLASSIFICATION] for anno in anno_dicts]
            out_annotations[LabelType.CLASSIFICATION] = torch.stack(class_annos, 0)

        if LabelType.SEGMENTATION in present_annotations:
            seg_annos = [anno[LabelType.SEGMENTATION] for anno in anno_dicts]
            out_annotations[LabelType.SEGMENTATION] = torch.stack(seg_annos, 0)

        if LabelType.BOUNDINGBOX in present_annotations:
            bbox_annos = [anno[LabelType.BOUNDINGBOX] for anno in anno_dicts]
            label_box = []
            for i, box in enumerate(bbox_annos):
                l_box = torch.zeros((box.shape[0], 6))
                l_box[:, 0] = i  # add target image index for build_targets()
                l_box[:, 1:] = box
                label_box.append(l_box)
            out_annotations[LabelType.BOUNDINGBOX] = torch.cat(label_box, 0)

        if LabelType.KEYPOINT in present_annotations:
            keypoint_annos = [anno[LabelType.KEYPOINT] for anno in anno_dicts]
            label_keypoints = []
            for i, points in enumerate(keypoint_annos):
                l_kps = torch.zeros((points.shape[0], points.shape[1] + 1))
                l_kps[:, 0] = i  # add target image index for build_targets()
                l_kps[:, 1:] = points
                label_keypoints.append(l_kps)
            out_annotations[LabelType.KEYPOINT] = torch.cat(label_keypoints, 0)

        return imgs, out_annotations
