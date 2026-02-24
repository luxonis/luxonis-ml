from collections.abc import Callable
from pathlib import Path

import pytest

from luxonis_ml.data import LuxonisDataset
from luxonis_ml.data.parsers import LuxonisParser
from luxonis_ml.enums import DatasetType

from .utils import LDFEquivalence


def _export_and_reimport(
    url: str,
    dataset_type: DatasetType,
    dataset_name: str,
    storage_url: str,
    tempdir: Path,
    initial_parse_kwargs: dict | None = None,
) -> tuple[LuxonisDataset, LuxonisDataset]:
    """Parse -> export -> re-import and return (original_dataset,
    reimported_dataset) to compare the two."""
    url = f"{storage_url}/{url}"
    dataset = LuxonisParser(
        url,
        dataset_name=dataset_name,
        delete_local=True,
        save_dir=tempdir,
    ).parse(**(initial_parse_kwargs or {}))

    export_dir = tempdir / "exported"
    dataset.export(
        output_path=export_dir,
        zip_output=True,
        dataset_type=dataset_type,
    )

    zip_files = sorted(export_dir.glob("*.zip"))
    exported_zip = zip_files[0]

    parser = LuxonisParser(
        str(exported_zip),
        dataset_name=f"{dataset_name}_reimported",
        delete_local=True,
        save_dir=tempdir,
    )
    parse_kwargs = {}
    if dataset_type == DatasetType.COCO:
        parse_kwargs["split_val_to_test"] = False

    new_dataset = parser.parse(**parse_kwargs)
    return dataset, new_dataset


# Which (DatasetType, collector) pairs to run for each logical annotation type
ANNOTATION_REGISTRY: dict[str, list[tuple[DatasetType, Callable]]] = {
    "boundingbox": [
        (DatasetType.YOLOV4, LDFEquivalence.collect_bbox_multiset),
        (DatasetType.YOLOV6, LDFEquivalence.collect_bbox_multiset),
        (DatasetType.YOLOV8BOUNDINGBOX, LDFEquivalence.collect_bbox_multiset),
        (DatasetType.COCO, LDFEquivalence.collect_bbox_multiset),
        (DatasetType.DARKNET, LDFEquivalence.collect_bbox_multiset),
        (DatasetType.VOC, LDFEquivalence.collect_bbox_multiset),
        (DatasetType.NATIVE, LDFEquivalence.collect_bbox_multiset),
        (DatasetType.CREATEML, LDFEquivalence.collect_bbox_multiset),
        (DatasetType.TFCSV, LDFEquivalence.collect_bbox_multiset),
    ],
    "instance_segmentation": [
        (
            DatasetType.NATIVE,
            LDFEquivalence.collect_instance_segmentation_multiset,
        ),
        (
            DatasetType.COCO,
            LDFEquivalence.collect_instance_segmentation_multiset,
        ),
        (
            DatasetType.YOLOV8INSTANCESEGMENTATION,
            LDFEquivalence.collect_instance_segmentation_mask_overlap_multiset,
        ),
    ],
    "keypoints": [
        (DatasetType.COCO, LDFEquivalence.collect_keypoint_multiset),
        (DatasetType.NATIVE, LDFEquivalence.collect_keypoint_multiset),
        (
            DatasetType.YOLOV8KEYPOINTS,
            LDFEquivalence.collect_keypoint_multiset,
        ),
    ],
    "classification": [
        (DatasetType.NATIVE, LDFEquivalence.collect_classification_multiset),
        (DatasetType.CLSDIR, LDFEquivalence.collect_classification_multiset),
        (
            DatasetType.FIFTYONECLS,
            LDFEquivalence.collect_classification_multiset,
        ),
    ],
    "segmentation": [
        (DatasetType.SEGMASK, LDFEquivalence.collect_segmentation_multiset),
        (DatasetType.NATIVE, LDFEquivalence.collect_segmentation_multiset),
        (
            DatasetType.SEGMASK,
            LDFEquivalence.collect_segmentation_mask_overlap_multiset,
        ),
    ],
}

DATASETS = [
    {
        "url": "Thermal_Dogs_and_People.v1-resize-416x416.coco.zip",
        "types": ["boundingbox"],
    },
    {"url": "D2_Tile.png-mask-semantic.zip", "types": ["segmentation"]},
    {
        "url": "COCO_people_subset.zip",
        "types": ["instance_segmentation", "boundingbox", "keypoints"],
    },
    {
        "url": "Flowers_Classification.v2-raw.folder.zip",
        "types": ["classification"],
    },
    {"url": "D2_Tile.png-mask-semantic.zip", "types": ["segmentation"]},
    {
        "url": "coco-2017.zip",
        "types": ["instance_segmentation", "boundingbox"],
    },
    {
        "url": "coco-2017.zip",
        "types": ["keypoints"],
        "initial_parse_kwargs": {"use_keypoint_ann": True},
    },
]


def build_params():
    """Expand DATASETS x supported annotation types x registry combos
    into pytest params."""
    params = []
    for ds in DATASETS:
        url = ds["url"]
        initial_parse_kwargs = ds.get("initial_parse_kwargs")
        for anno_type in ds["types"]:
            combos = ANNOTATION_REGISTRY.get(anno_type, [])
            for dataset_type, collector in combos:
                params.append(
                    pytest.param(
                        url,
                        dataset_type,
                        collector,
                        initial_parse_kwargs,
                        id=f"{url}::{anno_type}::{dataset_type.name}",
                    )
                )
    return params


@pytest.mark.parametrize(
    ("url", "dataset_type", "collector", "initial_parse_kwargs"),
    build_params(),
)
def test_export_import_equivalence(
    dataset_name: str,
    storage_url: str,
    tempdir: Path,
    url: str,
    dataset_type: DatasetType,
    collector: Callable,
    initial_parse_kwargs: dict | None,
):
    original, reimported = _export_and_reimport(
        url=url,
        dataset_type=dataset_type,
        dataset_name=dataset_name,
        storage_url=storage_url,
        tempdir=tempdir,
        initial_parse_kwargs=initial_parse_kwargs,
    )
    LDFEquivalence.assert_equivalence(original, reimported, collector)
