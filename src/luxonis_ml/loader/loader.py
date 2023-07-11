from pathlib import Path
import cv2
import numpy as np
import torch
from fiftyone import ViewField as F


class LuxonisLoader(torch.utils.data.Dataset):
    def __init__(self, dataset, view="train", stream=False, augmentations=None):
        """
        LuxonisDataset dataset: LuxonisDataset to use
        str view: either a saved view from a query or the name of a split
        bool stream: if False, data is downloaded locally before training else data is streamed from the cloud
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
            self.dataset.fo_dataset.mask_targets.get("segmentation", {}).keys()
        )
        if "keypoints" in self.dataset.fo_dataset.skeletons.keys():
            self.nk = len(self.dataset.fo_dataset.skeletons["keypoints"]["labels"])
        else:
            self.nk = 0
        self.augmentations = augmentations

        if not self.stream and self.dataset.bucket_storage.value != "local":
            self.dataset.sync_from_cloud()

        self.dataset.fo_dataset.group_slice = self.dataset.source.main_component

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample_id = self.ids[idx]
        path = self.paths[idx]
        sample = self.dataset.fo_dataset[sample_id]
        if self.stream and self.dataset.bucket_storage.value != "local":
            img_path = str(Path.home() / ".luxonis_mount" / path[1:])
        else:
            img_path = str(Path.home() / ".cache" / "luxonis_ml" / "data" / path[1:])
        img = np.transpose(cv2.imread(img_path), (2, 0, 1))
        _, ih, iw = img.shape

        classify = np.zeros(self.nc)
        bboxes = np.zeros((0, 5))
        seg = np.zeros((self.ns, ih, iw))
        keypoints = np.zeros((0, self.nk * 3 + 1))
        present_annotations = set()

        anno_dict = {}
        if "class" in sample and sample["class"] is not None:
            classes = sample["class"]["classifications"]
            for cls in classes:
                cls = self.classes.index(cls.label)
                classify[cls] = classify[cls] + 1
            present_annotations.add("class")
        if "boxes" in sample and sample["boxes"] is not None:
            detections = sample.boxes.detections
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
                bboxes = np.append(bboxes, box, axis=0)
            present_annotations.add("bbox")
        if "segmentation" in sample and sample["segmentation"] is not None:
            mask = sample.segmentation.mask
            for key in np.unique(mask):
                if key != 0:
                    seg[int(key) - 1, ...] = mask == key
            present_annotations.add("segmentation")
        if "keypoints" in sample and sample["keypoints"] is not None:
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

            present_annotations.add("keypoints")

        classify[classify > 0] = 1
        seg[seg > 0] = 1

        anno_dict = {}
        if "class" in present_annotations:
            anno_dict["class"] = classify
        if "bbox" in present_annotations:
            anno_dict["bbox"] = bboxes
        if "segmentation" in present_annotations:
            anno_dict["segmentation"] = seg
        if "keypoints" in present_annotations:
            anno_dict["keypoints"] = keypoints

        img = torch.tensor(img)
        for key in anno_dict:
            anno_dict[key] = torch.tensor(anno_dict[key])
        if self.augmentations is not None:
            img, anno_dict = self.augmentations((img, anno_dict))

        return img, anno_dict

    @staticmethod
    def collate_fn(batch):
        zipped = zip(*batch)
        img, anno_dicts = zipped
        imgs = torch.stack(img, 0)

        present_annotations = anno_dicts[0].keys()
        out_annotations = {anno: None for anno in present_annotations}

        if "class" in present_annotations:
            class_annos = [anno["class"] for anno in anno_dicts]
            out_annotations["class"] = torch.stack(class_annos, 0)

        if "bbox" in present_annotations:
            bbox_annos = [anno["bbox"] for anno in anno_dicts]
            label_box = []
            for i, box in enumerate(bbox_annos):
                l_box = torch.zeros((box.shape[0], 6))
                l_box[:, 0] = i  # add target image index for build_targets()
                l_box[:, 1:] = box
                label_box.append(l_box)
            out_annotations["bbox"] = torch.cat(label_box, 0)

        if "segmentation" in present_annotations:
            seg_annos = [anno["segmentation"] for anno in anno_dicts]
            out_annotations["segmentation"] = torch.stack(seg_annos, 0)

        if "keypoints" in present_annotations:
            keypoint_annos = [anno["keypoints"] for anno in anno_dicts]
            label_keypoints = []
            for i, points in enumerate(keypoint_annos):
                l_kps = torch.zeros((points.shape[0], points.shape[1] + 1))
                l_kps[:, 0] = i  # add target image index for build_targets()
                l_kps[:, 1:] = points
                label_keypoints.append(l_kps)
            out_annotations["keypoints"] = torch.cat(label_keypoints, 0)

        return imgs, out_annotations
