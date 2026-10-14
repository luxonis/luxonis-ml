import numpy as np
import pytest

import luxonis_ml.data.__main__ as data_cli
from luxonis_ml.data import LuxonisDataset, LuxonisLoader
from luxonis_ml.testing import FeatureDatasets, build_feature_datasets
from luxonis_ml.testing.__main__ import viz
from luxonis_ml.testing.feature_datasets import ARRAY_TASKS, IMAGE_TASKS
from luxonis_ml.testing.street import (
    POSE_NAMES,
    POSE_SIGMAS,
    Camera,
    Street,
    Thing,
    random_street,
)

SMALL = Camera(width=160, height=100)


@pytest.fixture(scope="module")
def names(worker_id: str) -> FeatureDatasets:
    return build_feature_datasets(
        f"feature_datasets_{worker_id}", camera=SMALL, sequences=1, frames=2
    )


def test_built_datasets_are_found_again(names: FeatureDatasets):
    assert names.exist()


def test_main_dataset_declares_every_task(names: FeatureDatasets):
    tasks = set(LuxonisDataset(names.main).get_tasks())
    nested = {"vehicles/plate", "people/face"}
    assert {*IMAGE_TASKS, *nested, *ARRAY_TASKS} <= tasks


def test_keypoint_tasks_keep_their_own_skeletons(names: FeatureDatasets):
    metadata = LuxonisDataset(names.main).get_keypoint_metadata()
    assert metadata["people"].labels == list(POSE_NAMES)
    assert metadata["people"].sigmas == POSE_SIGMAS


def test_first_case_shows_every_class(names: FeatureDatasets):
    loader = LuxonisLoader(LuxonisDataset(names.main), view="train")
    record = loader[0].to_ldf()
    classes = {
        detection.class_name
        for detections in record.annotation.values()
        for detection in detections
    }
    expected = {"car", "truck", "bus", "person", "child"}
    assert expected | {"stop", "yield", "speed_limit", "info"} <= classes


def test_true_negative_carries_no_detection(names: FeatureDatasets):
    loader = LuxonisLoader(LuxonisDataset(names.main), view="train")
    record = loader[1].to_ldf()
    assert not any(record.annotation.values())


def test_predictions_leave_out_two_images_and_add_one(
    names: FeatureDatasets,
):
    main = LuxonisDataset(names.main)
    predictions = LuxonisDataset(names.predictions)
    assert len(predictions) == len(main) - 2 + 1


def test_stereo_samples_have_two_sources(names: FeatureDatasets):
    sources = LuxonisDataset(names.stereo).get_source_names()
    assert sorted(sources) == ["left", "right"]


def test_augmentation_config_runs_on_the_main_dataset(
    names: FeatureDatasets,
):
    loader = LuxonisLoader(
        LuxonisDataset(names.main),
        view="train",
        augmentation_config=names.augmentations,
        height=SMALL.height,
        width=SMALL.width,
    )
    assert loader[0].images["image"].shape[:2] == (SMALL.height, SMALL.width)


def test_joint_behind_a_nearer_object_is_occluded():
    street = Street(
        things=[
            Thing("vehicle", "car", 0.0, 8.0, track=1),
            Thing("person", "person", 0.2, 10.0, track=2),
        ],
        buildings=[],
    )
    frame = street.render(Camera())
    person = next(i for i in frame.instances if i.thing.kind == "person")
    # The car roof is lower than the head, so the head stays visible.
    assert person.keypoints["left_knee"][2] == 1
    assert person.keypoints["nose"][2] == 2


def test_ground_flows_away_from_the_horizon():
    rng = np.random.default_rng(0)
    street = random_street(rng, vehicles=0, people=0, signs=0)
    bottom_row = street.render(SMALL).flow[-1]
    assert bottom_row[0, 0] < 0 < bottom_row[-1, 0]
    assert (bottom_row[:, 1] > 0).all()


@pytest.mark.parametrize(
    ("flag", "command", "dataset"),
    [
        ("", "inspect", "main"),
        ("compare", "compare", "main"),
        ("stereo", "inspect", "stereo"),
        ("augment", "inspect", "main"),
    ],
)
def test_viz_opens_the_chosen_view(
    names: FeatureDatasets,
    monkeypatch: pytest.MonkeyPatch,
    flag: str,
    command: str,
    dataset: str,
):
    calls = []
    for name in ("inspect", "compare"):
        monkeypatch.setattr(
            data_cli,
            name,
            lambda *args, name=name, **kwargs: calls.append((name, args)),
        )
    viz(prefix=names.main, **({flag: True} if flag else {}))
    assert calls[0][0] == command
    assert calls[0][1][0] == getattr(names, dataset)


def test_viz_refuses_two_views():
    with pytest.raises(ValueError, match="at most one"):
        viz(compare=True, stereo=True)
