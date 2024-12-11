from typing import Any, Dict

import albumentations as A
from typing_extensions import override


class BatchBasedTransform(A.DualTransform):
    def __init__(self, batch_size: int, **kwargs):
        """Transform for multi-image.

        @param batch_size: Batch size needed for augmentation to work
        @type batch_size: int
        @param kwargs: Additional BasicTransform parameters
        @type kwargs: Any
        """
        super().__init__(**kwargs)

        self.batch_size = batch_size

    @override
    def update_params(self, params: Dict[str, Any], **_) -> Dict[str, Any]:
        return params

    @override
    def update_params_shape(
        self, params: Dict[str, Any], data: Dict[str, Any]
    ) -> Dict[str, Any]:
        shape = data["image"][0].shape
        params["shape"] = shape
        params.update({"cols": shape[1], "rows": shape[0]})
        return params
