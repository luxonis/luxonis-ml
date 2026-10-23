from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import polars as pl
import pytest

from luxonis_ml.data import LuxonisDataset
from luxonis_ml.data.__main__ import delete
from luxonis_ml.data.utils.enums import BucketStorage
from luxonis_ml.utils import environ

from .utils import create_dataset, create_image


@pytest.fixture
def dataset(randint: int, tempdir: Path) -> Iterator[LuxonisDataset]:
    """Create a local dataset with one labeled image."""
    dataset = create_dataset(
        f"test_dataset_errors_{randint}",
        iter(
            [
                {
                    "file": create_image(0, tempdir),
                    "annotation": {"class": "cat"},
                }
            ]
        ),
    )
    yield dataset
    dataset.delete_dataset(delete_local=True)


def test_merge_into_a_new_dataset_needs_a_name(dataset: LuxonisDataset):
    with pytest.raises(ValueError, match="specify a name"):
        dataset.merge_with(dataset, inplace=False)


def test_merge_into_a_new_dataset_needs_one_storage(dataset: LuxonisDataset):
    remote = cast(
        LuxonisDataset, SimpleNamespace(_bucket_storage=BucketStorage.GCS)
    )

    with pytest.raises(ValueError, match="different bucket storage"):
        dataset.merge_with(remote, inplace=False, new_dataset_name="merged")


def test_export_keeps_an_existing_directory(
    dataset: LuxonisDataset, tempdir: Path
):
    output = tempdir / "export"
    output.mkdir()

    with pytest.raises(ValueError, match="already exists"):
        dataset.export(output)


def test_push_needs_each_media_file(dataset: LuxonisDataset, tempdir: Path):
    index = pl.DataFrame(
        {"original_filepath": [str(tempdir / "gone.jpg")], "uuid": ["0000"]}
    )

    with pytest.raises(FileNotFoundError, match="do not exist"):
        dataset._local_media_paths(index)


def test_remote_delete_of_a_local_dataset_deletes_nothing(
    dataset: LuxonisDataset,
):
    with pytest.raises(SystemExit) as exit_info:
        delete(dataset.identifier, remote=True, yes=True)

    assert exit_info.value.code == 2
    assert LuxonisDataset.exists(dataset.identifier)


@pytest.mark.parametrize(
    "credentials",
    [{}, {"LUXONISML_BUCKET": None}],
    ids=["environment", "credentials file"],
)
def test_unset_credential_is_an_error(
    dataset: LuxonisDataset,
    monkeypatch: pytest.MonkeyPatch,
    credentials: dict[str, None],
):
    monkeypatch.setattr(dataset, "_credentials", credentials)
    # On the proxy instance, undo would pin the old value for the session.
    monkeypatch.setattr(type(environ), "LUXONISML_BUCKET", None, raising=False)

    with pytest.raises(ValueError, match="LUXONISML_BUCKET"):
        dataset._get_credential("LUXONISML_BUCKET")
