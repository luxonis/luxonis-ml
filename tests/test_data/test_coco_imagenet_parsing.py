import os
import subprocess
import sys
from pathlib import Path

import pytest

from luxonis_ml.data import LuxonisDataset, LuxonisLoader, LuxonisParser
from luxonis_ml.data.utils import get_task_type
from luxonis_ml.utils.environ import environ

from tests.conftest import CocoSplitConfig

COCO_TRAIN_IMAGES = 100
COCO_VAL_IMAGES = 100
COCO_TEST_IMAGES = 100  # test split has images but 0 annotations
IMAGENET_IMAGES = 1000


def _split_counts(dataset: LuxonisDataset) -> dict[str, int]:
    """Return {split_name: n_items}."""
    splits = dataset.get_splits()
    assert splits is not None, "Dataset has no splits"
    return {name: len(ids) for name, ids in splits.items()}


def _task_types(dataset: LuxonisDataset) -> set[str]:
    loader = LuxonisLoader(dataset)
    _, ann = next(iter(loader))
    return {get_task_type(task) for task in ann}


def _run_cli_parse(
        dataset_dir: str | Path,
        dataset_name: str,
        split_ratio: str,
        *,
        extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        "-m",
        "luxonis_ml.data",
        "parse",
        str(dataset_dir),
        "--name",
        dataset_name,
        "--delete",
        "--split-ratio",
        split_ratio,
    ]
    if extra_args:
        cmd.extend(extra_args)
    env = {
        **os.environ,
        "PYTHONIOENCODING": "utf-8",
        "LUXONISML_BASE_PATH": str(environ.LUXONISML_BASE_PATH),
    }
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=600, env=env
    )


class TestCLIParseCoco:
    """luxonis-ml data parse CLI command with coco-2017"""

    def test_all_splits_ratio(
            self, coco_2017_all_splits: Path, dataset_name: str
    ):
        result = _run_cli_parse(
            coco_2017_all_splits, dataset_name, "[0.8, 0.1, 0.1]"
        )
        assert result.returncode == 0, result.stderr

        dataset = LuxonisDataset(dataset_name)
        counts = _split_counts(dataset)
        total = sum(counts.values())

        assert total > 0
        assert set(counts.keys()) == {"train", "val", "test"}
        assert counts["train"] / total == pytest.approx(0.8, abs=0.05)
        assert counts["val"] / total == pytest.approx(0.1, abs=0.05)
        assert counts["test"] / total == pytest.approx(0.1, abs=0.05)

    def test_all_splits_counts(
            self, coco_2017_all_splits: Path, dataset_name: str
    ):
        result = _run_cli_parse(
            coco_2017_all_splits, dataset_name, "[50, 30, 20]"
        )
        assert result.returncode == 0, result.stderr

        dataset = LuxonisDataset(dataset_name)
        counts = _split_counts(dataset)

        assert counts["train"] == 50
        assert counts["val"] == 30
        assert counts["test"] == 20

    def test_train_val_ratio(
            self, coco_2017_train_val: Path, dataset_name: str
    ):
        """Two-split source redistributed into three splits."""
        result = _run_cli_parse(
            coco_2017_train_val, dataset_name, "[0.7, 0.2, 0.1]"
        )
        assert result.returncode == 0, result.stderr

        dataset = LuxonisDataset(dataset_name)
        counts = _split_counts(dataset)
        total = sum(counts.values())

        assert total > 0
        assert set(counts.keys()) == {"train", "val", "test"}
        assert counts["train"] / total == pytest.approx(0.7, abs=0.05)

    def test_train_only_as_single_split(
            self, coco_2017_train_only: Path, dataset_name: str
    ):
        """Train-only detected as single split and redistributed.

        The CLI is pointed at the train subfolder so the parser enters
        parse_split mode (single-split COCO).
        """
        train_dir = coco_2017_train_only / "train"
        result = _run_cli_parse(
            train_dir, dataset_name, "[0.8, 0.1, 0.1]"
        )
        assert result.returncode == 0, result.stderr

        dataset = LuxonisDataset(dataset_name)
        counts = _split_counts(dataset)
        total = sum(counts.values())

        assert total > 0
        assert set(counts.keys()) == {"train", "val", "test"}


class TestCLIParseImagenet:
    """luxonis-ml data parse with imagenet-sample."""

    def test_ratio(
            self, imagenet_sample_dir: Path, dataset_name: str
    ):
        result = _run_cli_parse(
            imagenet_sample_dir, dataset_name, "[0.8, 0.1, 0.1]"
        )
        assert result.returncode == 0, result.stderr

        dataset = LuxonisDataset(dataset_name)
        counts = _split_counts(dataset)
        total = sum(counts.values())

        assert total == IMAGENET_IMAGES
        assert counts["train"] / total == pytest.approx(0.8, abs=0.05)
        assert counts["val"] / total == pytest.approx(0.1, abs=0.05)
        assert counts["test"] / total == pytest.approx(0.1, abs=0.05)

    def test_counts(
            self, imagenet_sample_dir: Path, dataset_name: str
    ):
        result = _run_cli_parse(
            imagenet_sample_dir, dataset_name, "[500, 100, 50]"
        )
        assert result.returncode == 0, result.stderr

        dataset = LuxonisDataset(dataset_name)
        counts = _split_counts(dataset)

        assert counts["train"] == 500
        assert counts["val"] == 100
        assert counts["test"] == 50


class TestCoco2017Keypoints:
    """``LuxonisParser(...).parse(use_keypoint_ann=True, ...)``"""

    def test_keypoints_all_splits_ratio(
            self, coco_2017_all_splits: Path, dataset_name: str
    ):
        """All three split folders present, but only train and val have
        keypoint JSONs.

        The test split keypoint JSON does not exist, so the parser
        skips it entirely. With ``split_ratios`` as percentages, all
        keypoint data from train+val is redistributed across the
        requested ratios.
        """
        parser = LuxonisParser(
            str(coco_2017_all_splits),
            dataset_name=dataset_name,
            delete_local=True,
        )
        dataset = parser.parse(
            split_ratios={"train": 0.8, "val": 0.1, "test": 0.1},
            use_keypoint_ann=True,
        )

        counts = _split_counts(dataset)
        total = sum(counts.values())

        assert total > 0
        assert counts["train"] / total == pytest.approx(0.8, abs=0.05)
        assert counts["val"] / total == pytest.approx(0.1, abs=0.05)
        assert counts["test"] / total == pytest.approx(0.1, abs=0.05)

        types = _task_types(dataset)
        assert "keypoints" in types
        assert "boundingbox" in types
        assert "classification" in types

    def test_keypoints_train_val_ratio(
            self, coco_2017_train_val: Path, dataset_name: str
    ):
        parser = LuxonisParser(
            str(coco_2017_train_val),
            dataset_name=dataset_name,
            delete_local=True,
        )
        dataset = parser.parse(
            split_ratios={"train": 0.7, "val": 0.2, "test": 0.1},
            use_keypoint_ann=True,
        )

        counts = _split_counts(dataset)
        total = sum(counts.values())

        assert total > 0
        assert set(counts.keys()) == {"train", "val", "test"}
        assert "keypoints" in _task_types(dataset)

    def test_keypoints_counts(
            self, coco_2017_all_splits: Path, dataset_name: str
    ):
        parser = LuxonisParser(
            str(coco_2017_all_splits),
            dataset_name=dataset_name,
            delete_local=True,
        )
        dataset = parser.parse(
            split_ratios={"train": 50, "val": 30, "test": 10},
            use_keypoint_ann=True,
        )

        counts = _split_counts(dataset)
        assert counts["train"] == 50
        assert counts["val"] == 30
        # test split has images but zero keypoint annotations
        assert counts["test"] == 0
        assert "keypoints" in _task_types(dataset)

    def test_keypoints_default_splits(
            self, coco_2017_all_splits: Path, dataset_name: str
    ):
        """No explicit split_ratios – uses original splits.

        The test split has no keypoint annotations, so
        ``split_val_to_test`` (default ``True``) splits the validation
        set 50/50 into val and test.
        """
        parser = LuxonisParser(
            str(coco_2017_all_splits),
            dataset_name=dataset_name,
            delete_local=True,
        )
        dataset = parser.parse(use_keypoint_ann=True)

        counts = _split_counts(dataset)
        assert counts.get("train", 0) > 0
        assert counts.get("val", 0) > 0
        # test should be populated from val-to-test auto-split
        assert counts.get("test", 0) > 0

    def test_keypoints_train_test_only(
            self, coco_2017_train_test: Path, dataset_name: str
    ):
        """Only train and test directories, no validation.

        Only the train keypoint JSON exists. The parser should parse
        train keypoints and redistribute across all three splits.
        """
        parser = LuxonisParser(
            str(coco_2017_train_test),
            dataset_name=dataset_name,
            delete_local=True,
        )
        dataset = parser.parse(
            split_ratios={"train": 0.5, "val": 0.4, "test": 0.1},
            use_keypoint_ann=True,
        )

        counts = _split_counts(dataset)
        total = sum(counts.values())

        assert total > 0
        assert set(counts.keys()) == {"train", "val", "test"}
        assert "keypoints" in _task_types(dataset)

    @pytest.mark.parametrize(
        "coco_2017",
        [
            CocoSplitConfig.ALL_SPLITS,
            CocoSplitConfig.TRAIN_VAL,
        ],
        indirect=True,
    )
    def test_keypoints_across_layouts(
            self, coco_2017: Path, dataset_name: str
    ):
        """Keypoint parsing works regardless of test being present."""
        parser = LuxonisParser(
            str(coco_2017),
            dataset_name=dataset_name,
            delete_local=True,
        )
        dataset = parser.parse(
            split_ratios={"train": 0.6, "val": 0.3, "test": 0.1},
            use_keypoint_ann=True,
        )

        assert len(dataset) > 0
        assert "keypoints" in _task_types(dataset)


class TestCoco2017NoKeypoints:
    """``LuxonisParser(...).parse(...)`` using default labels.json."""

    def test_no_keypoints_ratio(
            self, coco_2017_all_splits: Path, dataset_name: str
    ):
        parser = LuxonisParser(
            str(coco_2017_all_splits),
            dataset_name=dataset_name,
            delete_local=True,
        )
        dataset = parser.parse(
            split_ratios={"train": 0.5, "val": 0.4, "test": 0.1},
        )

        counts = _split_counts(dataset)
        total = sum(counts.values())

        assert total > 0
        types = _task_types(dataset)
        assert "boundingbox" in types
        assert "classification" in types

    def test_no_keypoints_default_splits(
            self, coco_2017_all_splits: Path, dataset_name: str
    ):
        parser = LuxonisParser(
            str(coco_2017_all_splits),
            dataset_name=dataset_name,
            delete_local=True,
        )
        dataset = parser.parse()

        counts = _split_counts(dataset)
        assert sum(counts.values()) > 0
        types = _task_types(dataset)
        assert "boundingbox" in types
        assert "classification" in types


class TestImagenetSample:

    def test_ratio(
            self, imagenet_sample_dir: Path, dataset_name: str
    ):
        parser = LuxonisParser(
            str(imagenet_sample_dir),
            dataset_name=dataset_name,
            delete_local=True,
        )
        dataset = parser.parse(
            split_ratios={"train": 0.8, "val": 0.1, "test": 0.1},
        )

        counts = _split_counts(dataset)
        total = sum(counts.values())

        assert total == IMAGENET_IMAGES
        assert counts["train"] / total == pytest.approx(0.8, abs=0.05)
        assert _task_types(dataset) == {"classification"}

    def test_counts(
            self, imagenet_sample_dir: Path, dataset_name: str
    ):
        parser = LuxonisParser(
            str(imagenet_sample_dir),
            dataset_name=dataset_name,
            delete_local=True,
        )
        dataset = parser.parse(
            split_ratios={"train": 500, "val": 100, "test": 50},
        )

        counts = _split_counts(dataset)
        assert counts["train"] == 500
        assert counts["val"] == 100
        assert counts["test"] == 50
