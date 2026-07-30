"""The deprecated `LuxonisParser` wrapper."""

import zipfile
from pathlib import Path
from typing import (
    Any,
)

import pytest

from luxonis_ml.data import (
    Layout,
    LuxonisDataset,
    LuxonisParser,
)
from tests.test_data.parsers.helpers import _SyntheticSplitParser
from tests.test_data.utils import create_image


def test_luxonis_parser_rejects_unsupported_source_at_construction(
    tempdir: Path,
):
    """`LuxonisParser` must fail at construction, as it did before deprecation.

    Regression: the deprecated wrapper deferred source acquisition and format
    detection into ``parse``, so ``try: LuxonisParser(d) except ValueError``
    — the documented way to probe whether a directory is a recognized layout
    — always succeeded and the error surfaced much later.
    """
    unrecognized = tempdir / "not_a_dataset"
    unrecognized.mkdir()

    with (
        pytest.warns(DeprecationWarning, match="LuxonisParser.*deprecated"),
        pytest.raises(
            ValueError,
            match="not in expected format for any registered parser",
        ),
    ):
        LuxonisParser(str(unrecognized))


def test_luxonis_parser_prepares_source_once(
    dataset_name: str,
    tempdir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Repeated ``parse`` calls must not re-acquire the source.

    Regression: with acquisition deferred into ``parse``, every call
    re-downloaded the Roboflow or S3 source and re-extracted the ZIP. A local
    ZIP stands in for the remote case here, since extraction is the observable
    half of the same work.
    """
    source_dir = tempdir / "clsdir_zip_source"
    for index, class_name in enumerate(("budgie", "parrot")):
        class_dir = source_dir / class_name
        class_dir.mkdir(parents=True)
        create_image(index, class_dir)

    archive_path = tempdir / "clsdir_zip_source.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for image_path in source_dir.rglob("*.jpg"):
            archive.write(
                image_path, image_path.relative_to(source_dir.parent)
            )

    extractions: list[Path] = []
    original_extractall = zipfile.ZipFile.extractall

    def counting_extractall(  # noqa: ANN202
        self,  # noqa: ANN001
        path=None,  # noqa: ANN001
        *args: Any,
        **kwargs: Any,
    ):
        extractions.append(Path(str(path)))
        return original_extractall(self, path, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "extractall", counting_extractall)

    with pytest.warns(DeprecationWarning, match="LuxonisParser.*deprecated"):
        parser = LuxonisParser(
            str(archive_path),
            dataset_name=dataset_name,
            delete_local=True,
        )
    assert len(extractions) == 1

    first = parser.parse()
    second = parser.parse()
    try:
        assert len(extractions) == 1
        assert first.identifier == second.identifier
        assert len(first) == 2
    finally:
        first.delete_dataset(delete_local=True)


def test_luxonis_parser_forwards_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """The deprecated wrapper forwards arguments to ``import_dataset``.

    Source acquisition and format resolution happen in ``__init__`` — that is
    what makes an unsupported source fail at construction, as it did before the
    deprecation, and what stops each ``parse`` call from downloading the source
    again — so the forwarded ``source`` is the resolved local path and the
    forwarded ``dataset_type`` is the resolved type name.
    """
    calls: list[dict[str, Any]] = []

    class Result:
        @staticmethod
        def get_parser_issue_messages() -> list[str]:
            return ["issue"]

    def import_dataset(cls: type[Any], source: str, **kwargs: Any) -> Result:
        calls.append({"cls": cls, "source": source, **kwargs})
        return Result()

    monkeypatch.setattr(
        "luxonis_ml.data.parsers.luxonis_parser.LuxonisDataset.import_dataset",
        classmethod(import_dataset),
    )
    monkeypatch.setattr(
        "luxonis_ml.data.parsers.luxonis_parser.get_parser_plugin",
        lambda source, dataset_type: (
            _SyntheticSplitParser,
            dataset_type or "synthetic-split",
            Layout({None: {}}),
        ),
    )
    with pytest.warns(DeprecationWarning, match="LuxonisParser.*deprecated"):
        parser = LuxonisParser(
            str(tmp_path),
            dataset_name="dataset",
            save_dir=tmp_path,
            dataset_type="coco",
            task_name="detection",
            full_warnings=True,
            delete_local=True,
        )
    assert parser.get_parser_issue_messages() == []
    result = parser.parse(
        split="train",
        random_split=False,
        split_ratios={"train": 1, "val": 0, "test": 0},
        use_keypoint_ann=True,
    )
    assert result.get_parser_issue_messages() == ["issue"]
    assert parser._get_parser_issue_messages() == ["issue"]
    assert calls == [
        {
            "cls": LuxonisDataset,
            "source": tmp_path,
            "dataset_name": "dataset",
            "save_dir": tmp_path,
            "dataset_type": "coco",
            "task_name": "detection",
            "full_warnings": True,
            "split": "train",
            "random_split": False,
            "split_ratios": {"train": 1, "val": 0, "test": 0},
            "parser_kwargs": {"use_keypoint_ann": True},
            "delete_local": True,
        }
    ]

    class PluginDataset:
        @classmethod
        def import_dataset(cls, source: str, **kwargs: Any) -> Result:
            return import_dataset(cls, source, **kwargs)

    monkeypatch.setattr(
        "luxonis_ml.data.parsers.luxonis_parser.DATASETS_REGISTRY.get",
        lambda name: PluginDataset,
    )
    with pytest.warns(DeprecationWarning, match="LuxonisParser.*deprecated"):
        plugin_parser = LuxonisParser(str(tmp_path), dataset_plugin="plugin")
    plugin_parser.parse()
    assert calls[-1]["cls"] is PluginDataset
