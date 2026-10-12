"""`LuxonisDataset` keeps what each keypoint of a record means."""

from pathlib import Path

from semver.version import Version

from luxonis_ml.data.exporters.ldf_downgrade import LDFDowngrader
from luxonis_ml.ldf import KeypointMetadata

from .utils import create_dataset, create_image, export_and_import

CORNERS = ["top_left", "top_right", "bottom_right", "bottom_left"]


def tile_records(tempdir: Path, n: int = 2):
    for i in range(n):
        yield {
            "media": str(create_image(i, tempdir)),
            "annotation": {
                "tiles": {
                    "class": "tile",
                    "keypoints": {
                        "keypoints": {
                            name: {"x": x, "y": y, "picture": name}
                            for name, (x, y) in zip(
                                CORNERS,
                                [
                                    (0.2, 0.2),
                                    (0.8, 0.2),
                                    (0.8, 0.8),
                                    (0.2, 0.8),
                                ],
                                strict=True,
                            )
                        }
                    },
                }
            },
        }


def eye_records(tempdir: Path):
    yield {
        "media": str(create_image(0, tempdir)),
        "annotation": {
            "pose": {
                "class": "person",
                "keypoints": {
                    "keypoints": {
                        "nose": (0.5, 0.3, 2),
                        "left_eye": {
                            "x": 0.55,
                            "y": 0.25,
                            "mirror": "right_eye",
                            "connected_to": "nose",
                        },
                        "right_eye": (0.45, 0.25, 2),
                    }
                },
            }
        },
    }


def test_add_keeps_the_meaning_of_each_keypoint(
    dataset_name: str, tempdir: Path
):
    dataset = create_dataset(dataset_name, eye_records(tempdir))

    assert dataset.get_keypoint_metadata()["pose"] == KeypointMetadata(
        labels=["nose", "left_eye", "right_eye"],
        edges=[(0, 1)],
        flip_pairs=[(1, 2)],
    )


def test_names_of_places_get_no_inferred_partner(
    dataset_name: str, tempdir: Path
):
    """``top_left`` and ``top_right`` name places, not sides of the tile."""
    dataset = create_dataset(dataset_name, tile_records(tempdir))

    tiles = dataset.get_keypoint_metadata()["tiles"]
    assert tiles.picture == CORNERS
    assert tiles.flip_pairs == []


def test_set_keypoint_metadata_gives_places(dataset_name: str, tempdir: Path):
    dataset = create_dataset(dataset_name, tile_records(tempdir))

    dataset.set_keypoint_metadata(task="tiles", picture=[None] * 4)

    assert dataset.get_keypoint_metadata()["tiles"].picture == [None] * 4


def test_native_export_keeps_the_places(dataset_name: str, tempdir: Path):
    dataset = create_dataset(dataset_name, tile_records(tempdir))

    imported = export_and_import(dataset, tempdir)

    assert imported.get_keypoint_metadata()["tiles"].picture == CORNERS


def test_an_export_for_ldf_2_drops_the_places():
    keypoints = {"keypoints": {"a": [0.2, 0.2, 2]}, "picture": ["center"]}

    downgraded = LDFDowngrader(Version.parse("2.2.0"))(
        {"media": "a.jpg", "annotation": {"tiles": [{"keypoints": keypoints}]}}
    )

    assert "picture" not in downgraded["annotation"]["keypoints"]
