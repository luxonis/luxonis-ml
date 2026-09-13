import math
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import NoReturn

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from loguru import logger

from luxonis_ml.data import BucketStorage, LuxonisDataset
from luxonis_ml.data.datasets.base_dataset import DatasetIterator
from luxonis_ml.data.datasets.luxonis_dataset import (
    _resolve_splits,
    _split_sizes,
)

from .utils import create_dataset, create_image


def test_split_sizes():
    """A ``ceil`` of each share starved the last split.

    Every split rounded up in turn, so the last one absorbed all the
    error. With the default 80/10/10 ratios, the test split was empty
    for 9 groups or fewer, and again for 11 to 14 groups.
    """
    default = {"train": 0.8, "val": 0.1, "test": 0.1}

    # These counts gave an empty test split before the fix.
    assert _split_sizes(7, default) == {"train": 5, "val": 1, "test": 1}
    assert _split_sizes(9, default) == {"train": 7, "val": 1, "test": 1}
    assert _split_sizes(14, default) == {"train": 11, "val": 2, "test": 1}

    # Below 7 groups the test split cannot get a whole group.
    assert _split_sizes(6, default) == {"train": 5, "val": 1, "test": 0}

    for n_groups in range(1, 200):
        sizes = _split_sizes(n_groups, default)
        # No group is lost and no group is counted two times.
        assert sum(sizes.values()) == n_groups, (n_groups, sizes)
        if n_groups >= 7:
            assert all(size > 0 for size in sizes.values()), (n_groups, sizes)

    # A zero ratio still gets nothing.
    assert _split_sizes(10, {"train": 1.0, "val": 0.0, "test": 0.0}) == {
        "train": 10,
        "val": 0,
        "test": 0,
    }

    # Three equal splits divide three groups evenly.
    thirds = {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}
    assert _split_sizes(3, thirds) == {"a": 1, "b": 1, "c": 1}


@given(
    n_groups=st.integers(min_value=0, max_value=500),
    weights=st.lists(
        st.floats(
            min_value=0, max_value=1000, allow_nan=False, allow_infinity=False
        ),
        min_size=1,
        max_size=6,
    ),
)
def test_split_sizes_gives_every_group_to_exactly_one_split(
    n_groups: int, weights: list[float]
):
    total = sum(weights)
    assume(total > 0)
    ratios = {f"s{i}": weight / total for i, weight in enumerate(weights)}
    assume(math.isclose(sum(ratios.values()), 1.0))

    sizes = _split_sizes(n_groups, ratios)

    assert sizes.keys() == ratios.keys()
    assert sum(sizes.values()) == n_groups
    assert all(size >= 0 for size in sizes.values())
    # A split asked for nothing never takes a group from another split.
    for split, ratio in ratios.items():
        if ratio == 0:
            assert sizes[split] == 0


@given(
    weights=st.dictionaries(
        st.text(min_size=1),
        st.floats(
            min_value=0, max_value=1000, allow_nan=False, allow_infinity=False
        ),
        min_size=1,
        max_size=5,
    )
)
def test_resolve_splits_reads_numbers_as_ratios(weights: dict[str, float]):
    total = sum(weights.values())
    assume(total > 0)
    ratios = {split: weight / total for split, weight in weights.items()}
    assume(math.isclose(sum(ratios.values()), 1.0))

    resolved, definitions = _resolve_splits(ratios)

    assert definitions is None
    assert resolved == ratios


@given(
    definitions=st.dictionaries(
        st.text(min_size=1),
        st.lists(st.text(min_size=1), max_size=4),
        min_size=1,
        max_size=4,
    )
)
def test_resolve_splits_reads_lists_as_filepaths(
    definitions: dict[str, list[str]],
):
    ratios, resolved = _resolve_splits(definitions)

    assert ratios is None
    assert resolved == definitions


def test_resolve_splits_reads_every_spelling_of_one_the_same_way():
    """`1`, `1.0`, and `True` are the same ratio."""
    assert (
        _resolve_splits({"train": 1})
        == _resolve_splits({"train": 1.0})
        == _resolve_splits({"train": True})
        == ({"train": 1.0}, None)
    )
    assert (
        _resolve_splits((1, 0, 0))
        == _resolve_splits((1.0, 0.0, 0.0))
        == _resolve_splits((True, False, False))
    )


def test_resolve_splits_rejects_a_mapping_that_is_neither():
    with pytest.raises(TypeError, match="ratios or filepath lists"):
        _resolve_splits({"train": 0.5, "val": ["a.jpg"]})  # type: ignore


@pytest.mark.parametrize(
    ("splits", "expected"),
    [
        pytest.param(None, {"train": 8, "val": 1, "test": 1}, id="none"),
        pytest.param(
            (0.8, 0.1, 0.1), {"train": 8, "val": 1, "test": 1}, id="tuple"
        ),
        pytest.param(
            {"train": 0.8, "val": 0.1, "test": 0.1},
            {"train": 8, "val": 1, "test": 1},
            id="mapping",
        ),
        pytest.param(
            {"train": 0.8, "val": 0.2, "test": 0},
            {"train": 8, "val": 2, "test": 0},
            id="zero_ratio",
        ),
        pytest.param({"train": 1.0}, {"train": 10}, id="one_split"),
        pytest.param(
            {"real": 0.5, "synthetic": 0.5},
            {"real": 5, "synthetic": 5},
            id="custom_names",
        ),
    ],
)
def test_make_splits_divides_by_ratio(
    splits: Mapping[str, float] | tuple[float, float, float] | None,
    expected: dict[str, int],
    dataset_name: str,
    tempdir: Path,
):
    dataset, _ = _ten_image_dataset(dataset_name, tempdir)
    dataset.make_splits(splits)
    assert _sizes(dataset) == expected


@pytest.mark.parametrize(
    "splits",
    [
        (1, 0, 0),
        (1.0, 0.0, 0.0),
        (True, False, False),
        (1, 0.0, 0.0),
        {"train": 1},
        {"train": 1.0},
        {"train": True},
    ],
    ids=str,
)
def test_make_splits_reads_every_spelling_of_one_alike(
    splits: Mapping[str, float] | tuple[float, float, float],
    dataset_name: str,
    tempdir: Path,
):
    dataset, _ = _ten_image_dataset(dataset_name, tempdir)
    dataset.make_splits(splits)
    assert _sizes(dataset)["train"] == 10


@pytest.mark.parametrize("element", [str, Path], ids=["str", "path"])
@pytest.mark.parametrize("sequence", [list, tuple], ids=["list", "tuple"])
def test_make_splits_accepts_any_filepath_sequence(
    sequence: type[list] | type[tuple],
    element: type[str] | type[Path],
    dataset_name: str,
    tempdir: Path,
):
    dataset, paths = _ten_image_dataset(dataset_name, tempdir)
    dataset.make_splits(
        {
            "train": sequence(element(path) for path in paths[:6]),
            "val": sequence(element(path) for path in paths[6:8]),
            "test": sequence(element(path) for path in paths[8:]),
        }
    )
    assert _sizes(dataset) == {"train": 6, "val": 2, "test": 2}


def test_make_splits_accepts_custom_names_for_filepaths(
    dataset_name: str, tempdir: Path
):
    dataset, paths = _ten_image_dataset(dataset_name, tempdir)
    dataset.make_splits({"real": paths[:4], "synthetic": paths[4:]})
    assert _sizes(dataset) == {"real": 4, "synthetic": 6}


def test_make_splits_never_puts_one_group_in_two_splits(
    dataset_name: str, tempdir: Path
):
    dataset, paths = _ten_image_dataset(dataset_name, tempdir)
    dataset.make_splits({"train": paths[:6], "val": paths[4:]})

    splits = dataset.get_splits()
    assert splits is not None
    group_ids = [group for data in splits.values() for group in data]
    assert len(group_ids) == len(set(group_ids))


@pytest.mark.parametrize(
    ("splits", "expected"),
    [
        pytest.param(None, {"train": 8, "val": 1, "test": 1}, id="none"),
        pytest.param(
            (0.0, 1.0, 0.0), {"train": 0, "val": 10, "test": 0}, id="tuple"
        ),
        pytest.param({"val": 1.0}, {"val": 10}, id="mapping"),
    ],
)
def test_make_splits_replaces_the_old_splits(
    splits: Mapping[str, float] | tuple[float, float, float] | None,
    expected: dict[str, int],
    dataset_name: str,
    tempdir: Path,
):
    dataset, paths = _ten_image_dataset(dataset_name, tempdir)
    dataset.make_splits({"train": paths})
    assert _sizes(dataset)["train"] == 10

    dataset.make_splits(splits, replace_old_splits=True)
    assert _sizes(dataset) == expected


def test_make_splits_replaces_the_old_splits_for_filepaths(
    dataset_name: str, tempdir: Path
):
    dataset, paths = _ten_image_dataset(dataset_name, tempdir)
    dataset.make_splits({"train": paths})
    assert _sizes(dataset)["train"] == 10

    dataset.make_splits({"test": paths[:3]}, replace_old_splits=True)
    assert _sizes(dataset) == {"test": 3}


def test_small_dataset_keeps_every_split(
    bucket_storage: BucketStorage, dataset_name: str, tempdir: Path
):
    """A small dataset keeps a test split with the default ratios."""

    def generator() -> DatasetIterator:
        for i in range(7):
            yield {
                "file": str(create_image(i, tempdir)),
                "annotation": {"class": "dog"},
            }

    dataset = create_dataset(
        dataset_name, generator(), bucket_storage, splits=False
    )
    dataset.make_splits()
    splits = dataset.get_splits()
    assert splits is not None
    assert {split: len(data) for split, data in splits.items()} == {
        "train": 5,
        "val": 1,
        "test": 1,
    }


def test_make_splits_is_atomic(
    bucket_storage: BucketStorage,
    dataset_name: str,
    tempdir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """A failed write must not destroy the existing splits.

    The method wrote straight to ``splits.json``. A failure in the
    middle of the write left a truncated file that no longer parses,
    and the old splits were gone.
    """

    def generator() -> DatasetIterator:
        for i in range(10):
            yield {
                "file": str(create_image(i, tempdir)),
                "annotation": {"class": "dog"},
            }

    dataset = create_dataset(dataset_name, generator(), bucket_storage)
    old_splits = dataset.get_splits()
    assert old_splits is not None

    write_text = Path.write_text

    def truncating_write_text(self: Path, *_) -> NoReturn:
        write_text(self, "truncated")
        raise RuntimeError("the disk is full")

    monkeypatch.setattr(Path, "write_text", truncating_write_text)
    with pytest.raises(RuntimeError, match="the disk is full"):
        dataset.make_splits((0.5, 0.5, 0.0), replace_old_splits=True)
    monkeypatch.undo()

    assert dataset.get_splits() == old_splits
    assert not list(dataset._metadata_path.glob("*.tmp"))


def test_definitions_skip_a_non_filepath(
    bucket_storage: BucketStorage, dataset_name: str, tempdir: Path
):
    """The method does not check the elements of a file list.

    Before the fix, an ``int`` in a definition list reached ``Path()``
    and raised a TypeError far away from the caller.
    """
    paths = [str(create_image(i, tempdir)) for i in range(4)]

    def generator() -> DatasetIterator:
        for path in paths:
            yield {"file": path, "annotation": {"class": "dog"}}

    dataset = create_dataset(
        dataset_name, generator(), bucket_storage, splits=False
    )

    messages: list[str] = []
    sink_id = logger.add(
        lambda message: messages.append(str(message).strip()),
        format="{message}",
        level="WARNING",
    )
    try:
        dataset.make_splits(
            {
                "train": paths[:2],
                "val": [1, None],  # type: ignore
                "test": paths[2:],
            }
        )
    finally:
        logger.remove(sink_id)

    splits = dataset.get_splits()
    assert splits is not None
    assert len(splits["train"]) == 2
    assert len(splits["test"]) == 2
    assert not splits["val"]
    assert sum("not a filepath; skipping" in m for m in messages) == 2


@pytest.mark.dependency(name="test_dataset[BucketStorage.LOCAL]")
def test_make_splits(
    bucket_storage: BucketStorage, dataset_name: str, tempdir: Path
):
    definitions: dict[str, list[str]] = defaultdict(list)

    _start_index: int = 0

    def generator(step: int = 15) -> DatasetIterator:
        nonlocal _start_index
        definitions.clear()
        for i in range(_start_index, _start_index + step):
            path = create_image(i, tempdir)
            yield {
                "file": str(path),
                "annotation": {
                    "class": ["dog", "cat"][i % 2],
                },
            }
            definitions[["train", "val", "test"][i % 3]].append(str(path))
        _start_index += step

    dataset = create_dataset(
        dataset_name, generator(), bucket_storage, splits=False
    )

    assert len(dataset) == 15
    assert dataset.get_splits() is None
    dataset.make_splits(definitions)
    splits = dataset.get_splits()
    assert splits is not None
    assert set(splits.keys()) == {"train", "val", "test"}
    for split, split_data in splits.items():
        assert len(split_data) == 5, (
            f"Split {split} has {len(split_data)} samples"
        )

    dataset.make_splits(definitions)
    assert dataset.get_splits() == splits

    dataset.add(generator())
    splits = dataset.get_splits()
    assert splits is not None
    for split, split_data in splits.items():
        assert len(split_data) == 5, (
            f"Split {split} has {len(split_data)} samples"
        )
    dataset.make_splits(definitions)
    splits = dataset.get_splits()
    assert splits is not None
    for split, split_data in splits.items():
        assert len(split_data) == 10, (
            f"Split {split} has {len(split_data)} samples"
        )

    dataset.add(generator())
    dataset.make_splits((1, 0, 0))
    splits = dataset.get_splits()
    assert splits is not None
    for split, split_data in splits.items():
        expected_length = 25 if split == "train" else 10
        assert len(split_data) == expected_length, (
            f"Split {split} has {len(split_data)} samples"
        )

    with pytest.raises(ValueError, match="No new files"):
        dataset.make_splits()

    with pytest.raises(ValueError, match="Splits cannot be empty"):
        dataset.make_splits({})

    with pytest.raises(ValueError, match=r"Ratios must sum to 1.0"):
        dataset.make_splits((0.7, 0.1, 1))

    # An out-of-range ratio reports the range, not a misleading sum.
    with pytest.raises(ValueError, match=r"between 0\.0 and 1\.0"):
        dataset.make_splits({"train": 1.5})

    # A ratio that is in range but sums wrong still reports the sum.
    with pytest.raises(ValueError, match=r"Ratios must sum to 1.0"):
        dataset.make_splits({"train": 0.5, "val": 0.2})

    with pytest.raises(ValueError, match=r"between 0\.0 and 1\.0"):
        dataset.make_splits({"train": -0.1, "val": 1.1})

    with pytest.raises(TypeError, match="ratios or filepath lists"):
        dataset.make_splits({"train": "invalid"})

    # Counts are not ratios; no ratio can exceed 1.
    with pytest.raises(ValueError, match=r"between 0\.0 and 1\.0"):
        dataset.make_splits({"train": 8, "val": 1, "test": 1})

    dataset.add(generator(10))
    dataset.make_splits({"custom_split": 1.0})
    splits = dataset.get_splits()
    assert splits is not None
    assert set(splits.keys()) == {"train", "val", "test", "custom_split"}
    for split, split_data in splits.items():
        expected_length = 25 if split == "train" else 10
        assert len(split_data) == expected_length, (
            f"Split {split} has {len(split_data)} samples"
        )

    dataset.make_splits(replace_old_splits=True)
    splits = dataset.get_splits()
    assert splits is not None
    for split, split_data in splits.items():
        expected_length = {"train": 44, "val": 6, "test": 5}
        assert len(split_data) == expected_length[split], (
            f"Split {split} has {len(split_data)} samples"
        )

    # The definitions match no file, but `replace_old_splits=True` still
    # discards the old splits.
    dataset.make_splits({"train": ["missing.jpg"]}, replace_old_splits=True)
    assert dataset.get_splits() == {"train": []}


def _sizes(dataset: LuxonisDataset) -> dict[str, int]:
    splits = dataset.get_splits()
    assert splits is not None
    return {split: len(data) for split, data in splits.items()}


def _ten_image_dataset(
    dataset_name: str, tempdir: Path
) -> tuple[LuxonisDataset, list[str]]:
    paths = [str(create_image(i, tempdir)) for i in range(10)]

    def generator() -> DatasetIterator:
        for path in paths:
            yield {"file": path, "annotation": {"class": "dog"}}

    return create_dataset(dataset_name, generator(), splits=False), paths
