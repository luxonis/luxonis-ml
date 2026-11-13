"""Multi-label classification in the NATIVE, CLSDIR and FIFTYONECLS formats.

An image with more than one class has one classification row for each
class. The tests check the files that an export writes, the classes that
a parse reads, and that an export and a parse keep all labels.

In the test data, an integer is the number of a test image. In a record
it is the image file, and in a file tree it is the content of a file.
"""

import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from luxonis_ml.data import LuxonisDataset, LuxonisParser
from luxonis_ml.data.exporters import PreparedLDF
from luxonis_ml.data.utils.constants import LDF_VERSION
from luxonis_ml.enums import DatasetType

from .utils import create_dataset, create_image

BOX = {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}
OTHER_BOX = {"x": 0.5, "y": 0.5, "w": 0.2, "h": 0.2}

MULTI_LABEL = [
    {"file": 0, "annotation": {"class": "cat"}},
    {"file": 0, "annotation": {"class": "dog"}},
    {"file": 1, "annotation": {"class": "cat"}},
    {"file": 2, "annotation": {"class": "bird"}},
    {"file": 2, "annotation": {"class": "cat"}},
    {"file": 2, "annotation": {"class": "dog"}},
]

# The "dog" of the second task must stay in that task.
TWO_TASKS = [
    *MULTI_LABEL,
    {"file": 0, "task_name": "pets", "annotation": {"class": "dog"}},
    {"file": 1, "task_name": "pets", "annotation": {"class": "dog"}},
]

# The box without an instance ID gets the ID -1, as the image classes do.
CLASSES_AND_BOXES = [
    {"file": 0, "annotation": {"class": "car"}},
    {"file": 0, "annotation": {"class": "cat"}},
    {
        "file": 0,
        "annotation": {
            "class": "car",
            "instance_id": 0,
            "boundingbox": BOX,
            "metadata": {"text": "red"},
        },
    },
    {"file": 0, "annotation": {"class": "car", "boundingbox": BOX}},
    {"file": 1},
]

# The export must keep each record apart: the first two have no key that
# separates them, the last two differ only in the sample metadata.
CLASSLESS_RECORDS = [
    {"file": 0, "annotation": {"boundingbox": BOX}},
    {"file": 0, "annotation": {"metadata": {"text": "red"}}},
    {
        "file": 0,
        "sample_metadata": {"frame": 1},
        "annotation": {"instance_id": 0, "boundingbox": OTHER_BOX},
    },
    {
        "file": 0,
        "sample_metadata": {"frame": 2},
        "annotation": {"instance_id": 0, "metadata": {"text": "red"}},
    },
]


def _entry(
    file: str,
    annotation: dict[str, Any] | None = None,
    sample_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return an entry of a native ``annotations.json``."""
    entry: dict[str, Any] = {
        "file": file,
        "task_name": "",
        "sample_metadata": sample_metadata or {},
    }
    if annotation is not None:
        entry["annotation"] = annotation
    return entry


def _images(tempdir: Path) -> list[Path]:
    """Create the test images. Each one has different content."""
    return [create_image(i, tempdir) for i in range(3)]


def _dataset(
    name: str, records: list[dict[str, Any]], images: list[Path]
) -> LuxonisDataset:
    """Add records in which ``file`` is the number of a test image."""
    return create_dataset(
        name,
        ({**record, "file": images[record["file"]]} for record in records),
        splits=(1, 0, 0),
    )


def _read_tree(root: Path, images: list[Path]) -> dict[str, Any]:
    """Read the files under root: JSON as data, an image as its number."""
    numbers = {image.read_bytes(): i for i, image in enumerate(images)}
    return {
        path.relative_to(root).as_posix(): (
            json.loads(path.read_text())
            if path.suffix == ".json"
            else numbers[path.read_bytes()]
        )
        for path in root.rglob("*")
        if path.is_file()
    }


def _write_tree(root: Path, tree: dict[str, Any], images: list[Path]) -> None:
    """Write files under root: data as JSON, a number as that image."""
    for name, content in tree.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, int):
            shutil.copy(images[content], path)
        else:
            path.write_text(json.dumps(content))


def _parse(
    source: Path, dataset_type: DatasetType, dataset_name: str
) -> LuxonisDataset:
    """Parse the dataset at source in place."""
    return LuxonisParser(
        str(source),
        dataset_type=dataset_type,
        dataset_name=dataset_name,
        delete_local=True,
        save_dir=source.parent,
    ).parse()


def _rows_by_image(
    dataset: LuxonisDataset, images: list[Path]
) -> dict[int, Counter]:
    """Count the label rows of each image, by image number."""
    numbers = {image.read_bytes(): i for i, image in enumerate(images)}
    rows_by_image: dict[int, Counter] = {}
    df = PreparedLDF.from_dataset(dataset).processed_df
    for row in df.iter_rows(named=True):
        image = numbers[Path(row["file"]).read_bytes()]
        rows = rows_by_image.setdefault(image, Counter())
        if row["task_type"] is not None:
            rows[
                row["task_name"],
                row["class_name"],
                row["instance_id"],
                row["task_type"],
                row["annotation"],
            ] += 1
    return rows_by_image


def _classes_by_image(
    dataset: LuxonisDataset, images: list[Path]
) -> dict[int, set[str]]:
    """Return the image-level classes of each image, by image number."""
    return {
        image: {
            class_name
            for _, class_name, instance_id, task_type, _ in rows
            if task_type == "classification" and instance_id == -1
        }
        for image, rows in _rows_by_image(dataset, images).items()
    }


@pytest.mark.parametrize(
    ("dataset_type", "records", "expected"),
    [
        pytest.param(
            DatasetType.NATIVE,
            CLASSES_AND_BOXES,
            {
                "metadata.json": {"ldf_version": str(LDF_VERSION)},
                "train/annotations.json": [
                    _entry(
                        "images/0.jpg", {"instance_id": -1, "class": "car"}
                    ),
                    _entry(
                        "images/0.jpg", {"instance_id": -1, "class": "cat"}
                    ),
                    _entry(
                        "images/0.jpg",
                        {
                            "instance_id": 0,
                            "class": "car",
                            "boundingbox": BOX,
                            "metadata": {"text": "red"},
                        },
                    ),
                    _entry(
                        "images/0.jpg",
                        {
                            "instance_id": -1,
                            "class": "car",
                            "boundingbox": BOX,
                        },
                    ),
                    _entry("images/1.jpg"),
                ],
                "train/images/0.jpg": 0,
                "train/images/1.jpg": 1,
                "val/annotations.json": [],
                "test/annotations.json": [],
            },
            id="native",
        ),
        pytest.param(
            DatasetType.NATIVE,
            CLASSLESS_RECORDS,
            {
                "metadata.json": {"ldf_version": str(LDF_VERSION)},
                "train/annotations.json": [
                    _entry(
                        "images/0.jpg",
                        {
                            "instance_id": -1,
                            "class": None,
                            "boundingbox": BOX,
                        },
                    ),
                    _entry(
                        "images/0.jpg",
                        {
                            "instance_id": -1,
                            "class": None,
                            "metadata": {"text": "red"},
                        },
                    ),
                    _entry(
                        "images/0.jpg",
                        {
                            "instance_id": 0,
                            "class": None,
                            "boundingbox": OTHER_BOX,
                        },
                        {"frame": 1},
                    ),
                    _entry(
                        "images/0.jpg",
                        {
                            "instance_id": 0,
                            "class": None,
                            "metadata": {"text": "red"},
                        },
                        {"frame": 2},
                    ),
                ],
                "train/images/0.jpg": 0,
                "val/annotations.json": [],
                "test/annotations.json": [],
            },
            id="native-classless-records",
        ),
        pytest.param(
            DatasetType.CLSDIR,
            MULTI_LABEL,
            {
                "train/bird/2.jpg": 2,
                "train/cat/0.jpg": 0,
                "train/cat/1.jpg": 1,
                "train/cat/2.jpg": 2,
                "train/dog/0.jpg": 0,
                "train/dog/2.jpg": 2,
            },
            id="clsdir",
        ),
        pytest.param(
            DatasetType.FIFTYONECLS,
            MULTI_LABEL,
            {
                "train/data/000001.jpg": 0,
                "train/data/000002.jpg": 1,
                "train/data/000003.jpg": 2,
                "train/labels.json": {
                    "classes": ["bird", "cat", "dog"],
                    "labels": {
                        "000001": [1, 2],
                        "000002": 1,
                        "000003": [0, 1, 2],
                    },
                },
            },
            id="fiftyone",
        ),
    ],
)
def test_export_writes_the_format(
    dataset_name: str,
    tempdir: Path,
    dataset_type: DatasetType,
    records: list[dict[str, Any]],
    expected: dict[str, Any],
):
    images = _images(tempdir)
    dataset = _dataset(dataset_name, records, images)

    dataset.export(tempdir / "export", dataset_type=dataset_type)

    root = tempdir / "export" / dataset.identifier
    assert _read_tree(root, images) == expected


@pytest.mark.parametrize(
    ("dataset_type", "tree", "expected"),
    [
        pytest.param(
            DatasetType.NATIVE,
            {
                "train/annotations.json": [
                    {"file": "images/a.jpg", "annotation": {"class": "cat"}},
                    {"file": "images/a.jpg", "annotation": {"class": "dog"}},
                    {"file": "images/b.jpg", "annotation": {"class": "cat"}},
                ],
                "train/images/a.jpg": 0,
                "train/images/b.jpg": 1,
            },
            {0: {"cat", "dog"}, 1: {"cat"}},
            id="native",
        ),
        pytest.param(
            DatasetType.CLSDIR,
            {
                "train/cat/a.jpg": 0,
                "train/cat/b.jpg": 1,
                "train/dog/a.jpg": 0,
            },
            {0: {"cat", "dog"}, 1: {"cat"}},
            id="clsdir",
        ),
        pytest.param(
            DatasetType.CLSDIR,
            {"train/cat/a.jpg": 0, "train/dog/a.jpg": 1},
            {0: {"cat"}, 1: {"dog"}},
            id="clsdir-other-image-same-name",
        ),
        pytest.param(
            DatasetType.FIFTYONECLS,
            {
                "train/data/a.jpg": 0,
                "train/data/b.jpg": 1,
                "train/labels.json": {
                    "classes": ["cat", "dog"],
                    "labels": {"a": [0, 1], "b": 0},
                },
            },
            {0: {"cat", "dog"}, 1: {"cat"}},
            id="fiftyone",
        ),
        pytest.param(
            DatasetType.FIFTYONECLS,
            {
                "train/data/a.jpg": 0,
                "train/data/b.jpg": 1,
                "train/labels.json": {
                    "classes": ["cat", "dog"],
                    "labels": {"a": [0, 1], "b": []},
                },
            },
            {0: {"cat", "dog"}, 1: set()},
            id="fiftyone-empty-list",
        ),
    ],
)
def test_parse_reads_the_format(
    dataset_name: str,
    tempdir: Path,
    dataset_type: DatasetType,
    tree: dict[str, Any],
    expected: dict[int, set[str]],
):
    images = _images(tempdir)
    _write_tree(tempdir / "source", tree, images)

    dataset = _parse(tempdir / "source", dataset_type, dataset_name)

    assert _classes_by_image(dataset, images) == expected


def test_clsdir_parse_groups_links_by_their_name(
    dataset_name: str, tempdir: Path
):
    images = _images(tempdir)
    _write_tree(tempdir / "pool", {"a.jpg": 0, "b.jpg": 0}, images)
    for link, target in (("cat/x.jpg", "a.jpg"), ("dog/x.jpg", "b.jpg")):
        path = tempdir / "source" / "train" / link
        path.parent.mkdir(parents=True)
        path.symlink_to((tempdir / "pool" / target).absolute())

    dataset = _parse(tempdir / "source", DatasetType.CLSDIR, dataset_name)

    assert _classes_by_image(dataset, images) == {0: {"cat", "dog"}}


@pytest.mark.parametrize(
    ("dataset_type", "records"),
    [
        pytest.param(DatasetType.NATIVE, MULTI_LABEL, id="native"),
        pytest.param(DatasetType.CLSDIR, MULTI_LABEL, id="clsdir"),
        pytest.param(DatasetType.FIFTYONECLS, MULTI_LABEL, id="fiftyone"),
        pytest.param(DatasetType.NATIVE, TWO_TASKS, id="native-two-tasks"),
        pytest.param(DatasetType.NATIVE, CLASSES_AND_BOXES, id="native-boxes"),
    ],
)
def test_round_trip_keeps_the_labels(
    dataset_name: str,
    tempdir: Path,
    dataset_type: DatasetType,
    records: list[dict[str, Any]],
):
    images = _images(tempdir)
    dataset = _dataset(dataset_name, records, images)

    dataset.export(tempdir / "export", dataset_type=dataset_type)
    imported = _parse(
        tempdir / "export" / dataset.identifier,
        dataset_type,
        f"{dataset_name}_imported",
    )

    assert _rows_by_image(imported, images) == _rows_by_image(dataset, images)
