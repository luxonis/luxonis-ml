import numpy as np

from luxonis_ml.data.augmentations import BatchTransform


class KeepFirstSample(BatchTransform):
    """Passes the first sample's targets through untouched.

    Lets a test see exactly what `BatchCompose` did to the data, without a
    real transform rewriting the targets on the way through.
    """

    def __init__(self):
        super().__init__(batch_size=2, p=1.0)

    def apply(self, image_batch: list[np.ndarray], **_) -> np.ndarray:
        return image_batch[0]

    def apply_to_mask(self, masks_batch: list[np.ndarray], **_) -> np.ndarray:
        return masks_batch[0]

    def apply_to_instance_mask(
        self, masks_batch: list[np.ndarray], **_
    ) -> np.ndarray:
        return masks_batch[0]

    def apply_to_bboxes(
        self, bboxes_batch: list[np.ndarray], **_
    ) -> np.ndarray:
        return bboxes_batch[0]

    def apply_to_keypoints(
        self, keypoints_batch: list[np.ndarray], **_
    ) -> np.ndarray:
        return keypoints_batch[0]

    def apply_to_metadata(
        self, metadata_batch: list[np.ndarray], **_
    ) -> np.ndarray:
        return metadata_batch[0]

    def apply_to_array(self, array_batch: list[np.ndarray], **_) -> np.ndarray:
        return array_batch[0]
