from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping

from luxonis_ml.typing import LoaderMultiOutput, Params
from luxonis_ml.utils import AutoRegisterMeta, Registry

AUGMENTATION_ENGINES: Registry[type["AugmentationEngine"]] = Registry(
    name="augmentation_engines"
)


# TODO: The engine should probably also handle normalization
# so it doesn't have to be done by injecting a normalization
# transformation to the config in LuxonisTrain.
class AugmentationEngine(
    ABC,
    metaclass=AutoRegisterMeta,
    registry=AUGMENTATION_ENGINES,
    register=False,
):
    @abstractmethod
    def __init__(
        self,
        height: int,
        width: int,
        targets: Mapping[str, str],
        n_classes: Mapping[str, int],
        source_names: list[str],
        config: Iterable[Params],
        keep_aspect_ratio: bool,
        is_validation_pipeline: bool,
        min_bbox_visibility: float = 0.0,
        seed: int | None = None,
        bbox_area_threshold: float = 0.0004,
    ):
        """Initialize augmentation pipeline from configuration.

        @type height: int
        @param height: Target image height
        @type width: int
        @param width: Target image width

        @type targets: Dict[str, str]
        @param targets: Dictionary mapping task names to task types.
            Example::
                {
                    "detection/boundingbox": "bbox",
                    "detection/segmentation": "mask",
                }

        @type n_classes: Dict[str, int]
        @param n_classes: Dictionary mapping task names to the number
            of associated classes. Example::
                {
                    "cars/boundingbox": 2,
                    "cars/segmentation": 2,
                    "motorbikes/boundingbox": 1,
                }

        @type config: List[Params]
        @param config: List of dictionaries with configuration for each
            augmentation. It is up to the augmentation engine to parse
            and interpret this configuration.

        @type keep_aspect_ratio: bool
        @param keep_aspect_ratio: Whether to keep aspect ratio
        @type is_validation_pipeline: bool
        @param is_validation_pipeline: Whether this is a validation
            pipeline (in which case some augmentations are skipped)
        @type min_bbox_visibility: float
        @param min_bbox_visibility: Minimum fraction of the original bounding box that must remain visible after augmentation.
        @type bbox_area_threshold: float
        @param bbox_area_threshold: Minimum area threshold for bounding boxes to be considered valid. In the range [0, 1].
            Default is 0.0004, which corresponds to a small area threshold to remove invalid bboxes and respective keypoints.
        @type seed: Optional[int]
        @param seed: Random seed for reproducibility. If None, a random seed will be used.
            If provided, it will be used to initialize the random number generator.
        """
        ...

    @abstractmethod
    def apply(self, data: list[LoaderMultiOutput]) -> LoaderMultiOutput:
        """Apply the augmentation pipeline to the data.

        @type data: List[LoaderMultiOutput]
        @param data: List of data to augment. The length of the list
            must be equal to the batch size.
        @rtype: LoaderMultiOutput
        @return: Augmented data
        """
        ...

    @property
    @abstractmethod
    def batch_size(self) -> int:
        """Getter for the batch size.

        The batch size is the number of images necessary for the
        augmentation pipeline to work in case of batch-based
        augmentations.

        For example, if the augmentation pipeline contains the MixUp
        augmentation, the batch size should be 2.

        If the pipeline requires MixUp and also Mosaic4 augmentations,
        the batch size should be 8 (2 * 4).
        """
        ...
