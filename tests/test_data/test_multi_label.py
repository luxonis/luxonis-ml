"""Multi-label classification through an export and a parse.

An image with more than one class has one classification row for each
class. NATIVE, CLSDIR and FIFTYONECLS can all store such an image, so an
export and a parse must give the image all of its classes back.
"""

import json
import shutil
from collections import Counter
from pathlib import Path

import pytest

from luxonis_ml.data import (
    LDFEquivalence,
    LuxonisDataset,
    LuxonisLoader,
    LuxonisParser,
)
from luxonis_ml.data.datasets.base_dataset import DatasetIterator
from luxonis_ml.data.exporters import PreparedLDF
from luxonis_ml.enums import DatasetType

from .utils import create_dataset, create_image

IMAGE_CLASSES = [{"cat", "dog"}, {"cat"}, {"bird", "cat", "dog"}]
EXPECTED_CLASSES = sorted(sorted(classes) for classes in IMAGE_CLASSES)


def _multi_label_images(tempdir: Path, task_name: str = "") -> DatasetIterator:
    for i, classes in enumerate(IMAGE_CLASSES):
        for class_name in sorted(classes):
            yield {
                "file": create_image(i, tempdir),
                "task_name": task_name,
                "annotation": {"class": class_name},
            }


def _classes_and_boxes(tempdir: Path) -> DatasetIterator:
    """Image-level classes next to boxes of the same class.

    The box without an ``instance_id`` shares the ID ``-1`` with the
    image-level classes, so only the row order tells them apart.
    """
    image = create_image(0, tempdir)
    box = {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}
    for class_name in ("car", "cat"):
        yield {"file": image, "annotation": {"class": class_name}}
    yield {
        "file": image,
        "annotation": {
            "class": "car",
            "instance_id": 0,
            "boundingbox": box,
            "metadata": {"text": "red"},
        },
    }
    yield {"file": image, "annotation": {"class": "car", "boundingbox": box}}
    yield {"file": create_image(1, tempdir)}


def _round_trip(
    dataset: LuxonisDataset, dataset_type: DatasetType, tempdir: Path
) -> LuxonisDataset:
    """Export the dataset and parse the export into a new dataset."""
    name = f"{dataset.identifier}_{dataset_type.name.lower()}"
    export_path = dataset.export(tempdir / name, dataset_type=dataset_type)
    assert isinstance(export_path, Path)
    return LuxonisParser(
        str(export_path / dataset.identifier),
        dataset_type=dataset_type,
        dataset_name=name,
        delete_local=True,
        save_dir=tempdir,
    ).parse()


def _loaded_classes(
    dataset: LuxonisDataset, task_name: str = ""
) -> list[list[str]]:
    """Return the class names of each image from its multi-hot vector."""
    class_ids = dataset.get_classes()[task_name]
    return sorted(
        sorted(
            name
            for name, i in class_ids.items()
            if labels[f"{task_name}/classification"][i]
        )
        for _, labels in LuxonisLoader(dataset)
    )


def _rows(
    dataset: LuxonisDataset, task_type: str
) -> dict[tuple[str], Counter]:
    """Count the rows of a task type for each image, all instances
    included.
    """
    return LDFEquivalence.collect_annotation_multiset(
        PreparedLDF.from_dataset(dataset), task_type
    )


@pytest.mark.parametrize(
    "dataset_type",
    [DatasetType.NATIVE, DatasetType.CLSDIR, DatasetType.FIFTYONECLS],
)
def test_round_trip_keeps_the_classes_of_each_image(
    dataset_name: str, tempdir: Path, dataset_type: DatasetType
):
    dataset = create_dataset(
        dataset_name, _multi_label_images(tempdir), splits=(1, 0, 0)
    )

    imported = _round_trip(dataset, dataset_type, tempdir)

    LDFEquivalence.assert_equivalence(
        dataset, imported, LDFEquivalence.collect_classification_multiset
    )
    assert _loaded_classes(imported) == EXPECTED_CLASSES


def test_native_round_trip_keeps_the_classes_of_each_task(
    dataset_name: str, tempdir: Path
):
    def generator() -> DatasetIterator:
        yield from _multi_label_images(tempdir, "animals")
        for i in range(len(IMAGE_CLASSES)):
            yield {
                "file": create_image(i, tempdir),
                "task_name": "pets",
                "annotation": {"class": "dog"},
            }

    dataset = create_dataset(dataset_name, generator(), splits=(1, 0, 0))

    imported = _round_trip(dataset, DatasetType.NATIVE, tempdir)

    assert _loaded_classes(imported, "animals") == EXPECTED_CLASSES
    assert _loaded_classes(imported, "pets") == [["dog"]] * len(IMAGE_CLASSES)


def test_native_round_trip_keeps_every_row(dataset_name: str, tempdir: Path):
    dataset = create_dataset(
        dataset_name, _classes_and_boxes(tempdir), splits=(1, 0, 0)
    )

    imported = _round_trip(dataset, DatasetType.NATIVE, tempdir)

    assert LDFEquivalence.collect_image_multiset(
        PreparedLDF.from_dataset(dataset)
    ) == LDFEquivalence.collect_image_multiset(
        PreparedLDF.from_dataset(imported)
    )
    for task_type in ("boundingbox", "classification", "metadata/text"):
        assert _rows(dataset, task_type) == _rows(imported, task_type)


def test_native_export_writes_one_entry_per_detection(
    dataset_name: str, tempdir: Path
):
    records = list(_classes_and_boxes(tempdir))
    dataset = create_dataset(dataset_name, iter(records), splits=(1, 0, 0))

    export_path = dataset.export(
        tempdir / "exported", dataset_type=DatasetType.NATIVE
    )

    assert isinstance(export_path, Path)
    entries = json.loads(
        (
            export_path / dataset.identifier / "train" / "annotations.json"
        ).read_text()
    )
    assert len(entries) == len(records)


def test_clsdir_parser_keeps_different_images_with_one_name_apart(
    dataset_name: str, tempdir: Path
):
    for i, class_name in enumerate(("cat", "dog")):
        class_dir = tempdir / "clsdir" / "train" / class_name
        class_dir.mkdir(parents=True)
        shutil.copy(create_image(i, tempdir), class_dir / "0.jpg")

    dataset = LuxonisParser(
        str(tempdir / "clsdir"),
        dataset_type=DatasetType.CLSDIR,
        dataset_name=dataset_name,
        delete_local=True,
        save_dir=tempdir,
    ).parse()

    assert _loaded_classes(dataset) == [["cat"], ["dog"]]
