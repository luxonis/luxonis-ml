"""`LuxonisDataset` keeps what class names say, and the loader uses it."""

from pathlib import Path

import numpy as np
import pytest
from semver.version import Version

from luxonis_ml.data import LuxonisLoader
from luxonis_ml.ldf import ClassMeaning, ClassMetadata

from .utils import create_dataset, create_image, export_and_import

HFLIP = [
    {
        "name": "HorizontalFlip",
        "params": {
            "p": 1.0,
        },
    },
]


def hand_records(tempdir: Path, x: float = 0.1):
    yield {
        "media": str(create_image(0, tempdir)),
        "annotation": {
            "hands": {
                "class": {
                    "name": "left_hand",
                    "mirror": "right_hand",
                },
                "boundingbox": {
                    "x": x,
                    "y": 0.4,
                    "w": 0.2,
                    "h": 0.2,
                },
            },
        },
    }


def test_add_keeps_the_meaning_of_each_class(dataset_name: str, tempdir: Path):
    dataset = create_dataset(dataset_name, hand_records(tempdir))

    assert dataset.get_class_metadata() == {
        "hands": ClassMetadata(mirror_pairs=[("left_hand", "right_hand")]),
    }


def test_a_partner_without_rows_gets_the_next_free_id(
    dataset_name: str, tempdir: Path
):
    dataset = create_dataset(dataset_name, hand_records(tempdir))

    assert dataset.get_classes()["hands"] == {
        "left_hand": 0,
        "right_hand": 1,
    }


def test_two_meanings_of_one_class_are_rejected(
    dataset_name: str, tempdir: Path
):
    records = [
        *hand_records(tempdir),
        {
            "media": str(create_image(1, tempdir)),
            "annotation": {
                "hands": {
                    "class": {
                        "name": "left_hand",
                        "mirror": "other_hand",
                    },
                },
            },
        },
    ]

    with pytest.raises(ValueError, match="different meanings"):
        create_dataset(dataset_name, iter(records))


def test_set_class_metadata_adds_and_removes_meanings(
    dataset_name: str, tempdir: Path
):
    dataset = create_dataset(dataset_name, hand_records(tempdir))

    dataset.set_class_metadata(
        task="hands",
        classes={
            "fist": {
                "upright": True,
            },
            "left_hand": None,
        },
    )

    assert dataset.get_class_metadata() == {
        "hands": ClassMetadata(upright=["fist"]),
    }
    assert dataset.get_classes()["hands"]["fist"] == 2


def test_set_class_metadata_rejects_an_unknown_task(
    dataset_name: str, tempdir: Path
):
    dataset = create_dataset(dataset_name, hand_records(tempdir))

    with pytest.raises(ValueError, match="no task 'digits'"):
        dataset.set_class_metadata(
            task="digits",
            classes={
                "digit_6": {
                    "upright": True,
                },
            },
        )


def test_merge_keeps_the_meanings_of_both_datasets(
    dataset_name: str, tempdir: Path
):
    hands = create_dataset(dataset_name, hand_records(tempdir))
    fists = create_dataset(f"{dataset_name}_fists", hand_records(tempdir))
    fists.set_class_metadata(
        task="hands",
        classes={
            "fist": ClassMeaning(upright=True),
        },
    )

    merged = hands.merge_with(fists, new_dataset_name=f"{dataset_name}_m")

    assert merged.get_class_metadata() == {
        "hands": ClassMetadata(
            mirror_pairs=[("left_hand", "right_hand")],
            upright=["fist"],
        ),
    }


def test_native_export_keeps_the_meanings(dataset_name: str, tempdir: Path):
    dataset = create_dataset(dataset_name, hand_records(tempdir))
    dataset.set_class_metadata(
        task="hands",
        classes={
            "fist": {
                "upright": True,
            },
        },
    )

    imported = export_and_import(dataset, tempdir)

    assert imported.get_class_metadata() == dataset.get_class_metadata()


def test_an_export_for_ldf_2_drops_the_meanings(
    dataset_name: str, tempdir: Path
):
    dataset = create_dataset(dataset_name, hand_records(tempdir))

    imported = export_and_import(
        dataset, tempdir, ldf_version=Version.parse("2.2.0")
    )

    assert imported.get_class_metadata() == {}
    assert set(imported.get_classes()["hands"]) == {"left_hand"}


def test_the_loader_swaps_a_mirrored_box_class(
    dataset_name: str, tempdir: Path
):
    dataset = create_dataset(
        dataset_name, hand_records(tempdir), splits={"train": 1.0}
    )

    labels = LuxonisLoader(
        dataset, height=512, width=512, augmentation_config=HFLIP
    )[0].labels

    box = labels["hands/boundingbox"][0]
    assert box[0] == 1
    assert box[1] == pytest.approx(0.7, abs=1e-4)
    assert labels["hands/classification"].tolist() == [0, 1]


def test_the_loader_moves_a_mirrored_segmentation_channel(
    dataset_name: str, tempdir: Path
):
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[:, :100] = 1
    record = {
        "media": str(create_image(0, tempdir)),
        "annotation": {
            "hands": {
                "class": {
                    "name": "left_hand",
                    "mirror": "right_hand",
                },
                "segmentation": {
                    "mask": mask,
                },
            },
        },
    }
    dataset = create_dataset(
        dataset_name, iter([record]), splits={"train": 1.0}
    )

    labels = LuxonisLoader(
        dataset, height=512, width=512, augmentation_config=HFLIP
    )[0].labels

    # The loader adds the background as class 0.
    background, left_hand, right_hand = labels["hands/segmentation"]
    assert left_hand.sum() == 0
    assert right_hand[:, -100:].all()
    assert background[:, :100].all()


def test_the_loader_keeps_an_upright_class_upright(
    dataset_name: str, tempdir: Path
):
    dataset = create_dataset(
        dataset_name, hand_records(tempdir), splits={"train": 1.0}
    )
    dataset.set_class_metadata(
        task="hands",
        classes={
            "left_hand": {
                "upright": True,
            },
        },
    )

    labels = LuxonisLoader(
        dataset, height=512, width=512, augmentation_config=HFLIP
    )[0].labels

    box = labels["hands/boundingbox"][0]
    assert box[0] == 0
    assert box[1] == pytest.approx(0.1, abs=1e-4)
