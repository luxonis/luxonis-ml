#!/usr/bin/env python3

import webdataset as wds
import cv2
import numpy as np
import glob
import torch
from .dataset import HType, IType, Bough
from pycocotools import mask as maskUtils

class LuxonisLoader:

    def __init__(self, dataset, view='train', stream=False):
        """
        dataset: a LuxonisDataset instance
        """
        self.dataset = dataset
        self.view = view
        self.stream = stream

        tar_numbers = self.get_tar_numbers()

        if not self.dataset.s3 and not self.stream:
            self.url = f"file://{self.dataset.path}/{Bough.WEBDATASET.value}/{self.view}_{tar_numbers}.tar"
        else:
            self.url = f"pipe:s3cmd -q get s3://{self.dataset.bucket}/{self.dataset.bucket_path}/{Bough.WEBDATASET.value}/{self.view}_{tar_numbers}.tar -"

        handlers = self.get_source_handlers()
        self.webdataset = wds.WebDataset(self.url).shuffle(1000).decode(*handlers)

        self.nc = len(self.dataset.classes)
        self.max_nk = max([len(definition['keypoints']) for definition in self.dataset.keypoint_definitions.values()])

    def get_tar_numbers(self):
        if not self.dataset.s3 and not self.stream:
            tar_paths = sorted(glob.glob(f"{self.dataset.path}/{Bough.WEBDATASET.value}/{self.view}_*.tar"))
        else:
            resp = self.dataset.client.list_objects(
                Bucket=self.dataset.bucket,
                Prefix=f"{self.dataset.bucket_path}/{Bough.WEBDATASET.value}/",
                Delimiter='/'
            )
            if 'Contents' not in resp:
                raise Exception("Cannot find tar files in S3 bucket")
            tar_paths = [content['Key'] for content in resp['Contents']]

        tar_numbers = [int(path.split('_')[-1].split('.tar')[0]) for path in tar_paths]
        if len(tar_numbers) == 0:
            raise Exception("No tar files found in webdataset!")
        elif len(tar_numbers) == 1:
            return "000000"
        else:
            max_number = str(max(tar_numbers)).zfill(6)
            return "{000000.."+max_number+"}"


    def get_source_handlers(self):
        handlers = []
        sources = self.dataset.sources
        for source_name in sources:
            components = sources[source_name].components
            for component_name in components:
                component = components[component_name]
                if component.htype == HType.IMAGE:
                    ext = f".{source_name}.{component_name}.{component.compression}"
                    if component.itype == IType.BGR:
                        handlers.append(wds.handle_extension(ext, self.bgr_decoder))
                    else:
                        handlers.append(wds.handle_extension(ext, self.mono_decoder))

        return handlers

    def bgr_decoder(self, data):
        image = np.asarray(bytearray(data), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return image

    def mono_decoder(self, data):
        # should handle uint8 or uint16
        image = np.asarray(bytearray(data))
        image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
        return image

    def map(self, map_function):
        self.webdataset.map(map_function)

    def to_pytorch(self, batch_size, num_workers=0, collate_fn='default'):
        if collate_fn == 'default':
            collate_fn = self.collate_fn
        return wds.WebLoader(self.webdataset, num_workers=num_workers, batch_size=batch_size, collate_fn=collate_fn)

    def _auto_preprocess(self, data):
        """
        Loads main component as image
        and automatically processes annotations -> numpy
        """

        key = data['__key__']
        source_name = [key for key in data if not key.startswith('__')][0].split('.')[0]
        source = self.dataset.sources[source_name]
        image_component = source.components[source.main_component]
        image_key = f"{source_name}.{source.main_component}.{image_component.compression}"
        img = np.transpose(data[image_key], (2, 0, 1)) # convert HWC to CHW
        _, ih, iw = img.shape

        json_key = f"{source_name}.json"
        json_data = data[json_key]
        annotations = json_data[source.main_component]["annotations"]

        bboxes = np.zeros((0,5))
        seg = np.zeros((self.nc, ih, iw))
        classify = np.zeros(self.nc)
        keypoints = np.zeros((0, self.max_nk*3+1))

        for ann in annotations:
            cls = ann['class']
            classify[cls] = classify[cls] + 1

            if 'bbox' in ann.keys():
                x, y, w, h = ann['bbox']
                box = np.array([cls, x/iw, y/ih, w/iw, h/ih]).reshape((1,5))
                bboxes = np.append(bboxes, box, axis=0)

            if 'segmentation' in ann.keys():
                segmentation = ann['segmentation']['data']
                if isinstance(segmentation, list): # polygon format
                    rles = maskUtils.frPyObjects(segmentation, ih, iw)
                    rle = maskUtils.merge(rles)
                elif isinstance(segmentation, dict): # rle format
                    rle = maskUtils.frPyObjects(segmentation, ih, iw)

                mask = maskUtils.decode(rle)
                seg[cls] = seg[cls] + mask

            if 'keypoints' in ann.keys():
                kps = np.array(ann['keypoints']).reshape((-1,3)).flatten()
                nk = len(kps)
                kps = np.concatenate([[cls], kps])
                points = np.zeros((1, self.max_nk*3+1))
                points[0,:nk+1] = kps
                keypoints = np.append(keypoints, points, axis=0)

        classify[classify > 0] = 1
        seg[seg > 0] = 1

        return img, classify, bboxes, seg, keypoints

    def auto_preprocess_numpy(self, data):
        return self._auto_preprocess(data)

    def auto_preprocess(self, data):
        img, classify, bboxes, seg, keypoints = self._auto_preprocess(data)
        img = torch.tensor(img)
        classify = torch.tensor(classify)
        bboxes = torch.tensor(bboxes)
        seg = torch.tensor(seg)
        keypoints = torch.tensor(keypoints)
        return img, classify, bboxes, seg, keypoints

    @staticmethod
    def collate_fn(batch):
        # TODO: will also want to implement this for keypoints
        zipped = zip(*batch)
        img, label_cls, label_bboxes, label_seg, label_kps = zipped

        label_box = []
        for i, box in enumerate(label_bboxes):
            l_box = torch.zeros((box.shape[0], 6))
            l_box[:, 0] = i  # add target image index for build_targets()
            l_box[:, 1:] = box
            label_box.append(l_box)

        label_keypoints = []
        for i, points in enumerate(label_kps):
            l_kps = torch.zeros((points.shape[0], points.shape[1]+1))
            l_kps[:, 0] = i  # add target image index for build_targets()
            l_kps[:, 1:] = points
            label_keypoints.append(l_kps)

        imgs = torch.stack(img, 0)
        cls = torch.stack(label_cls, 0)
        boxs = torch.cat(label_box, 0)
        segs = torch.stack(label_seg, 0)
        kps = torch.cat(label_keypoints, 0)

        return imgs, cls, boxs, segs, kps
