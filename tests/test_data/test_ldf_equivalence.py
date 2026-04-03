import json
import zipfile
from pathlib import Path

import pytest

from luxonis_ml.data import LuxonisDataset, LuxonisParser, ldf_equivalent
from luxonis_ml.data.parsers.coco_parser import COCOParser
from luxonis_ml.utils import LuxonisFileSystem


def _parse_dataset(
    source: str,
    dataset_name: str,
    tempdir: Path,
    **parse_kwargs,
) -> LuxonisDataset:
    return LuxonisParser(
        source,
        dataset_name=dataset_name,
        delete_local=True,
        save_dir=tempdir,
    ).parse(**parse_kwargs)


def _load_coco_annotation_payload(
    dataset_root: Path,
) -> tuple[Path, Path, dict]:
    split_dirs = [
        dataset_root,
        *sorted(path for path in dataset_root.rglob("*") if path.is_dir()),
    ]
    for split_dir in split_dirs:
        split_info = COCOParser.validate_split(split_dir)
        if split_info is None:
            continue

        json_path = split_info["annotation_path"]
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        annotations = (
            payload.get("annotations") if isinstance(payload, dict) else None
        )
        images = payload.get("images") if isinstance(payload, dict) else None
        if isinstance(annotations, list) and isinstance(images, list):
            return json_path, split_info["image_dir"], payload

    raise AssertionError(
        "Unable to find a split annotation json used by the COCO parser."
    )


def _tweak_first_coco_keypoint(dataset_root: Path) -> None:
    json_path, image_dir, payload = _load_coco_annotation_payload(dataset_root)

    annotations = payload["annotations"]
    images = payload["images"]

    image_info = {
        image["id"]: image
        for image in images
        if isinstance(image, dict)
        and "id" in image
        and "width" in image
        and "file_name" in image
        and (image_dir / image["file_name"]).exists()
    }

    for annotation in annotations:
        keypoints = annotation.get("keypoints")
        image = image_info.get(annotation.get("image_id"))
        if (
            isinstance(keypoints, list)
            and len(keypoints) >= 3
            and isinstance(image, dict)
            and isinstance(image["width"], int)
            and image["width"] > 1
        ):
            annotation["keypoints"][0] = min(
                image["width"] - 1,
                float(keypoints[0]) + 50,
            )
            json_path.write_text(json.dumps(payload), encoding="utf-8")
            return


def _zip_directory(dataset_root: Path, output_zip: Path) -> Path:
    with zipfile.ZipFile(
        output_zip, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for path in dataset_root.rglob("*"):
            archive.write(path, path.relative_to(dataset_root))
    return output_zip


def _create_modified_dataset_zip(source_url: str, tempdir: Path) -> Path:
    downloaded_zip = LuxonisFileSystem.download(
        source_url, tempdir / Path(source_url).name
    )
    extracted_dir = tempdir / f"{downloaded_zip.stem}_modified"
    with zipfile.ZipFile(downloaded_zip, "r") as archive:
        archive.extractall(extracted_dir)

    _tweak_first_coco_keypoint(extracted_dir)
    return _zip_directory(extracted_dir, tempdir / "modified_dataset.zip")


@pytest.mark.parametrize(
    "dataset_url",
    [
        pytest.param(
            "Thermal_Dogs_and_People.v1-resize-416x416.coco.zip",
            id="thermal-dogs-and-people",
        ),
        pytest.param(
            "Flowers_Classification.v2-raw.folder.zip",
            id="flowers-classification",
        ),
        pytest.param(
            "COCO_people_subset.zip",
            id="coco-people-subset",
        ),
        pytest.param(
            "D2_Tile.png-mask-semantic.zip",
            id="d2-tile-segmentation",
        ),
    ],
)
def test_ldf_equivalent_returns_true_for_same_dataset_parse(
    dataset_name: str,
    storage_url: str,
    tempdir: Path,
    dataset_url: str,
):
    first_dataset: LuxonisDataset | None = None
    second_dataset: LuxonisDataset | None = None
    try:
        first_dataset = _parse_dataset(
            f"{storage_url}/{dataset_url}",
            f"{dataset_name}_first",
            tempdir,
        )
        second_dataset = _parse_dataset(
            f"{storage_url}/{dataset_url}",
            f"{dataset_name}_second",
            tempdir,
        )

        assert ldf_equivalent(
            first_dataset.dataset_name,
            second_dataset.dataset_name,
        )
    finally:
        if first_dataset is not None:
            first_dataset.delete_dataset(delete_local=True)
        if second_dataset is not None:
            second_dataset.delete_dataset(delete_local=True)


@pytest.mark.parametrize(
    ("first_dataset_url", "second_dataset_url"),
    [
        pytest.param(
            "Thermal_Dogs_and_People.v1-resize-416x416.coco.zip",
            "Flowers_Classification.v2-raw.folder.zip",
            id="bbox-vs-classification",
        ),
        pytest.param(
            "Thermal_Dogs_and_People.v1-resize-416x416.coco.zip",
            "COCO_people_subset.zip",
            id="bbox-vs-coco-people",
        ),
        pytest.param(
            "Flowers_Classification.v2-raw.folder.zip",
            "D2_Tile.png-mask-semantic.zip",
            id="classification-vs-segmentation",
        ),
        pytest.param(
            "D2_Tile.png-mask-semantic.zip",
            "COCO_people_subset.zip",
            id="segmentation-vs-coco-people",
        ),
    ],
)
def test_ldf_equivalent_returns_false_for_different_datasets(
    dataset_name: str,
    storage_url: str,
    tempdir: Path,
    first_dataset_url: str,
    second_dataset_url: str,
):
    first_dataset: LuxonisDataset | None = None
    second_dataset: LuxonisDataset | None = None
    try:
        first_dataset = _parse_dataset(
            f"{storage_url}/{first_dataset_url}",
            f"{dataset_name}_bbox",
            tempdir,
        )
        second_dataset = _parse_dataset(
            f"{storage_url}/{second_dataset_url}",
            f"{dataset_name}_other",
            tempdir,
        )

        assert not ldf_equivalent(
            first_dataset.dataset_name,
            second_dataset.dataset_name,
        )
    finally:
        if first_dataset is not None:
            first_dataset.delete_dataset(delete_local=True)
        if second_dataset is not None:
            second_dataset.delete_dataset(delete_local=True)


def test_ldf_equivalent_returns_false_when_annotations_change(
    dataset_name: str,
    storage_url: str,
    tempdir: Path,
):
    original_dataset: LuxonisDataset | None = None
    modified_dataset: LuxonisDataset | None = None
    try:
        original_dataset = _parse_dataset(
            f"{storage_url}/COCO_people_subset.zip",
            f"{dataset_name}_original",
            tempdir,
        )
        modified_zip = _create_modified_dataset_zip(
            f"{storage_url}/COCO_people_subset.zip", tempdir
        )
        modified_dataset = _parse_dataset(
            str(modified_zip),
            f"{dataset_name}_modified",
            tempdir,
        )

        assert not ldf_equivalent(
            original_dataset.dataset_name,
            modified_dataset.dataset_name,
        )
    finally:
        if original_dataset is not None:
            original_dataset.delete_dataset(delete_local=True)
        if modified_dataset is not None:
            modified_dataset.delete_dataset(delete_local=True)


def test_luxonis_dataset_eq_uses_ldf_equivalence(
    dataset_name: str,
    storage_url: str,
    tempdir: Path,
):
    first_dataset: LuxonisDataset | None = None
    second_dataset: LuxonisDataset | None = None
    different_dataset: LuxonisDataset | None = None
    try:
        first_dataset = _parse_dataset(
            f"{storage_url}/COCO_people_subset.zip",
            f"{dataset_name}_eq_first",
            tempdir,
        )
        second_dataset = _parse_dataset(
            f"{storage_url}/COCO_people_subset.zip",
            f"{dataset_name}_eq_second",
            tempdir,
        )
        different_dataset = _parse_dataset(
            f"{storage_url}/Flowers_Classification.v2-raw.folder.zip",
            f"{dataset_name}_eq_different",
            tempdir,
        )

        assert first_dataset == second_dataset
        assert first_dataset == second_dataset.dataset_name
        assert first_dataset != different_dataset
        assert first_dataset != different_dataset.dataset_name
        assert first_dataset != object()
    finally:
        if first_dataset is not None:
            first_dataset.delete_dataset(delete_local=True)
        if second_dataset is not None:
            second_dataset.delete_dataset(delete_local=True)
        if different_dataset is not None:
            different_dataset.delete_dataset(delete_local=True)
