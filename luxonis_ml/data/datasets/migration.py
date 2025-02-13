from typing import Any, Dict, Final, List, Literal, Set

from typing_extensions import TypedDict

from .metadata import Skeletons

LDF_1_0_0_TASKS: Final[Set[str]] = {
    "classification",
    "segmentation",
    "boundingbox",
    "keypoints",
    "array",
}

LDF_1_0_0_TASK_TYPES: Final[Dict[str, str]] = {
    "BBoxAnnotation": "boundingbox",
    "ClassificationAnnotation": "classification",
    "PolylineSegmentationAnnotation": "segmentation",
    "RLESegmentationAnnotation": "segmentation",
    "MaskSegmentationAnnotation": "segmentation",
    "KeypointAnnotation": "keypoints",
    "ArrayAnnotation": "array",
}


class LDF_1_0_0_MetadataDict(TypedDict):
    source: Dict[str, Any]
    ldf_version: str
    classes: Dict[str, List[str]]
    tasks: Dict[str, List[str]]
    skeletons: Dict[str, Skeletons]
    categorical_encodings: Dict[str, Dict[str, int]]
    metadata_types: Dict[str, Literal["float", "int", "str", "Category"]]
