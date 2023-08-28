import cv2
import numpy as np
import torch
import random
import warnings
from pathlib import Path
from fiftyone import ViewField as F
from enum import Enum
from typing import Optional


class LabelType(str, Enum):
    CLASSIFICATION = "class"
    SEGMENTATION = "segmentation"
    BOUNDINGBOX = "boxes"
    KEYPOINT = "keypoints"


class LuxonisLoader(torch.utils.data.Dataset):
    def __init__(self, dataset: "luxonis_ml.data.LuxonisDataset", view: str = "train", stream: bool = False, augmentations: Optional["luxonis_ml.loader.Augmentations"]=None):
        """LuxonisLoader used for loading LuxonisDataset

        Args:
            dataset (luxonis_ml.data.LuxonisDataset): LuxonisDataset to use
            view (str, optional): View of the dataset. Defaults to "train".
            stream (bool, optional): Flag for data streaming. Defaults to False.
            augmentations (Optional[luxonis_ml.loader.Augmentations], optional): Augmentation class that performs augmentations. Defaults to None.
        """

        self.dataset = dataset
        if view in ["train", "val", "test"]:
            version_view = self.dataset.fo_dataset.load_saved_view(
                f"version_{dataset.version}"
            )
            self.samples = version_view.match(
                (F("latest") == True) & (F("split") == view)
            )
        else:
            self.samples = dataset.fo_dataset.load_saved_view(view)

        # this will automatically use the main component in a group
        # if another component is desired, you must set dataset.fo_dataset.group_slice = 'other component'
        self.ids = self.samples.values("id")
        self.paths = self.samples.values("filepath")
        self.stream = stream
        self.classes, self.classes_by_task = self.dataset.get_classes()
        self.nc = len(self.classes)
        self.ns = len(
            self.dataset.fo_dataset.mask_targets.get(LabelType.SEGMENTATION, {}).keys()
        )
        if LabelType.KEYPOINT in self.dataset.fo_dataset.skeletons.keys():
            self.nk = len(self.dataset.fo_dataset.skeletons[LabelType.KEYPOINT]["labels"])
        else:
            self.nk = 0
        self.augmentations = augmentations

        if not self.stream and self.dataset.bucket_storage.value != "local":
            self.dataset.sync_from_cloud()

        self.dataset.fo_dataset.group_slice = self.dataset.source.main_component

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int):
        img, annotations = self.load_image_with_annotations(idx)

        if self.augmentations is not None:
            aug_input_data = [(img, annotations)]
            if self.augmentations.is_batched:
                other_indices = [i for i in range(len(self)) if i != idx]
                if self.augmentations.aug_batch_size > len(self):
                    warnings.warn(f"Augmentations batch_size ({self.augmentations.aug_batch_size}) is larger than dataset size ({len(self)}), samples will include repetitions.")
                    random_fun = random.choices
                else:
                    random_fun = random.sample
                picked_indices = random_fun(other_indices, k=self.augmentations.aug_batch_size-1)
                aug_input_data.extend([self.load_image_with_annotations(i) for i in picked_indices])

            img, annotations = self.augmentations(aug_input_data, nc=self.nc, ns=self.ns, nk=self.nk)

        img = np.transpose(img, (2, 0, 1)) # HWC to CHW
        img = torch.tensor(img)
        for key in annotations:
            annotations[key] = torch.tensor(annotations[key])

        return img, annotations

    def load_image_with_annotations(self, idx: int):
        """Loads image and its annotations based on index.
        
        Args:
            idx (int): Index of the image

        Returns:
            Tuple[np.ndarray, dict]: Image as np.ndarray in RGB format and dict with all present annotations
        """

        sample_id = self.ids[idx]
        path = self.paths[idx]
        sample = self.dataset.fo_dataset[sample_id]
        if self.stream and self.dataset.bucket_storage.value != "local":
            img_path = str(Path.home() / ".luxonis_mount" / path[1:])
        else:
            img_path = str(Path.home() / ".cache" / "luxonis_ml" / "data" / path[1:])

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        ih, iw, _ = img.shape
        annotations = {}

        if LabelType.CLASSIFICATION in sample and sample[LabelType.CLASSIFICATION] is not None:
            classes = sample[LabelType.CLASSIFICATION]["classifications"]
            classify = np.zeros(self.nc)
            for cls in classes:
                cls = self.classes.index(cls.label)
                classify[cls] = classify[cls] + 1
            classify[classify > 0] = 1
            annotations[LabelType.CLASSIFICATION] = classify

        if LabelType.SEGMENTATION in sample and sample[LabelType.SEGMENTATION] is not None:
            mask = sample.segmentation.mask
            seg = np.zeros((self.ns, ih, iw))
            for key in np.unique(mask):
                if key != 0:
                    seg[int(key) - 1, ...] = mask == key
            seg[seg > 0] = 1
            annotations[LabelType.SEGMENTATION] = seg

        if LabelType.BOUNDINGBOX in sample and sample[LabelType.BOUNDINGBOX] is not None:
            detections = sample.boxes.detections
            boxes = np.zeros((0, 5))
            for det in detections:
                box = np.array(
                    [
                        self.classes.index(det.label),
                        det.bounding_box[0],
                        det.bounding_box[1],
                        det.bounding_box[2],
                        det.bounding_box[3],
                    ]
                ).reshape(1, 5)
                boxes = np.append(boxes, box, axis=0)
            annotations[LabelType.BOUNDINGBOX] = boxes
    
        if LabelType.KEYPOINT in sample and sample[LabelType.KEYPOINT] is not None:
            keypoints = np.zeros((0, self.nk * 3 + 1))
            for kps in sample.keypoints.keypoints:
                cls = self.classes.index(kps.label)
                pnts = np.array(kps.points).reshape((-1, 2)).astype(np.float32)
                kps = np.zeros((len(pnts), 3))
                nan_key = np.isnan(pnts[:, 0])
                kps[~nan_key, 2] = 2
                kps[:, :2] = pnts
                kps[nan_key, :2] = 0  # use 0 instead of NaN
                kps = kps.flatten()
                nk = len(kps)
                kps = np.concatenate([[cls], kps])
                points = np.zeros((1, self.nk * 3 + 1))
                points[0, : nk + 1] = kps
                keypoints = np.append(keypoints, points, axis=0)

            annotations[LabelType.KEYPOINT] = keypoints


        return img, annotations

    @staticmethod
    def collate_fn(batch: list):
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
