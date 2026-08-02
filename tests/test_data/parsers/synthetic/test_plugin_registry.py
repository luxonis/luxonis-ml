"""Parser registration, plugin discovery and format resolution."""

from pathlib import Path
from typing import (
    Any,
    ClassVar,
)

import pytest
from loguru import logger

import luxonis_ml.data as data_module
from luxonis_ml.data import (
    PARSERS_REGISTRY,
    Layout,
    LuxonisDataset,
    ParseResult,
    ParserPlugin,
    register_parser_plugin,
)
from luxonis_ml.data.parsers import (
    COCOParser,
    YOLOv8Parser,
)
from luxonis_ml.data.parsers.parser_plugin import (
    get_parser_plugin,
)
from tests.test_data.parsers.helpers import _SyntheticSplitParser
from tests.test_data.utils import create_image


def test_parser_entry_point_loading(monkeypatch: pytest.MonkeyPatch):
    class EntryPointParser(ParserPlugin):
        dataset_types = ("test-entry-point",)

        @classmethod
        def detect(cls, source: Path) -> Layout | None:
            return None

        def parse(
            self,
            source: Path,
            layout: Layout,
            **kwargs: Any,
        ) -> ParseResult:
            raise AssertionError("Plugin should only be registered")

    class EntryPoint:
        @staticmethod
        def load() -> type[ParserPlugin]:
            return EntryPointParser

    monkeypatch.setattr(
        data_module,
        "_get_entry_points_subset",
        lambda group: [EntryPoint()] if group == "parser_plugins" else [],
    )
    data_module._load_parser_plugins()
    assert PARSERS_REGISTRY.get("test-entry-point") is EntryPointParser


def test_custom_parser_plugin_import(
    dataset_name: str,
    tempdir: Path,
):
    source = tempdir / "sample.plugin"
    source.write_text("custom")
    image = create_image(0, tempdir)

    class CustomParser(ParserPlugin):
        dataset_types = ("test-custom", "test-custom-alias")

        @classmethod
        def detect(cls, source: Path) -> Layout | None:
            if source.suffix != ".plugin":
                return None
            return Layout({None: {}})

        def parse(
            self,
            source: Path,
            layout: Layout,
            **kwargs: Any,
        ) -> ParseResult:
            assert source.suffix == ".plugin"
            assert layout.splits == {None: {}}
            assert kwargs == {"label": "budgie"}
            return ParseResult(
                iter(
                    [
                        (
                            None,
                            {
                                "file": image,
                                "annotation": {"class": "budgie"},
                            },
                        )
                    ]
                ),
                {},
            )

    register_parser_plugin(CustomParser, force=True)
    with pytest.raises(KeyError, match="already registered"):
        register_parser_plugin(CustomParser)
    dataset = LuxonisDataset.import_dataset(
        source,
        dataset_name=dataset_name,
        dataset_type="test-custom-alias",
        parser_kwargs={"label": "budgie"},
        delete_local=True,
        save_dir=tempdir,
    )
    try:
        assert len(dataset) == 1
        assert dataset.get_parser_issue_messages() == []
    finally:
        dataset.delete_dataset(delete_local=True)


def test_partial_ultralytics_layout_reports_yolov6_yolov8_ambiguity(
    dataset_name: str,
    tempdir: Path,
):
    dataset_dir = tempdir / "yolo_partial"
    image_dir = dataset_dir / "images" / "test"
    label_dir = dataset_dir / "labels" / "test"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    create_image(16, image_dir)
    (label_dir / "img_16.txt").write_text("0 0.5 0.5 0.4 0.4\n")
    (dataset_dir / "data.yaml").write_text("names:\n  0: budgie\n")

    with pytest.raises(
        ValueError,
        match=(
            r"compatible with multiple parsers: yolov6, yolov8\. "
            r"Please specify `dataset_type`\."
        ),
    ):
        LuxonisDataset.import_dataset(
            str(dataset_dir),
            dataset_name=dataset_name,
            delete_local=True,
            save_dir=tempdir,
        )


def test_ultralytics_layout_with_val_split_detects_yolov8(
    dataset_name: str,
    tempdir: Path,
):
    dataset_dir = tempdir / "yolo_ultralytics"
    for index, split_name in enumerate(("train", "val")):
        image_dir = dataset_dir / "images" / split_name
        label_dir = dataset_dir / "labels" / split_name
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        create_image(index, image_dir)
        (label_dir / f"img_{index}.txt").write_text("0 0.5 0.5 0.4 0.4\n")
    (dataset_dir / "data.yaml").write_text("names:\n  0: budgie\n")

    # `images/valid` is absent, so only the YOLOv8 parser recognizes both
    # splits even though the YOLOv6 parser also recognizes `images/train`.
    plugin, dataset_type, _layout = get_parser_plugin(dataset_dir, None)
    assert dataset_type == "yolov8"
    assert plugin is YOLOv8Parser

    dataset = LuxonisDataset.import_dataset(
        str(dataset_dir),
        dataset_name=dataset_name,
        delete_local=True,
        save_dir=tempdir,
    )
    try:
        assert len(dataset) == 2
    finally:
        dataset.delete_dataset(delete_local=True)


def test_parser_entry_point_name_collision_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
):
    """A colliding entry-point plugin must not break ``import``.

    Regression: ``_load_parser_plugins`` runs unguarded at module scope, so a
    third-party plugin declaring a built-in ``dataset_types`` name made plain
    ``import luxonis_ml.data`` raise ``KeyError``, taking down every downstream
    consumer as soon as that package was installed. Registration is also
    all-or-nothing now: the collision is checked before any name is claimed, so
    the type declared ahead of ``"coco"`` must not be left behind.
    """

    class CollidingParser(ParserPlugin):
        dataset_types = ("test-before-collision", "coco")

        @classmethod
        def detect(cls, source: Path) -> Layout | None:
            return None

        def parse(
            self,
            source: Path,
            layout: Layout,
            **kwargs: Any,
        ) -> ParseResult:
            raise AssertionError("Plugin should never be selected")

    class EntryPoint:
        name = "colliding-plugin"

        @staticmethod
        def load() -> type[ParserPlugin]:
            return CollidingParser

    monkeypatch.setattr(
        data_module,
        "_get_entry_points_subset",
        lambda group: [EntryPoint()] if group == "parser_plugins" else [],
    )
    messages: list[str] = []
    sink_id = logger.add(
        lambda message: messages.append(str(message).strip()),
        format="{message}",
        level="WARNING",
    )
    try:
        data_module._load_parser_plugins()
    finally:
        logger.remove(sink_id)

    assert any("colliding-plugin" in message for message in messages)
    assert PARSERS_REGISTRY.get("coco") is COCOParser
    assert "test-before-collision" not in PARSERS_REGISTRY


def test_parser_registration_forms_and_validation():
    class MissingTypes(ParserPlugin):
        @classmethod
        def detect(cls, source: Path) -> Layout | None:
            return None

        def parse(
            self, source: Path, layout: Layout, **kwargs: Any
        ) -> ParseResult:
            return ParseResult(iter(()), {})

    with pytest.raises(ValueError, match="declare `dataset_types`"):
        register_parser_plugin(MissingTypes)

    @register_parser_plugin(force=True)
    class DecoratedParser(MissingTypes):
        dataset_types = ("synthetic-decorated",)

    assert PARSERS_REGISTRY.get("synthetic-decorated") is DecoratedParser


def test_get_parser_plugin_resolution_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    class NeverParser(_SyntheticSplitParser):
        dataset_types = ("never",)

        @classmethod
        def detect(cls, source: Path) -> Layout | None:
            return None

    class FirstParser(NeverParser):
        dataset_types = ("first",)

        @classmethod
        def detect(cls, source: Path) -> Layout | None:
            return Layout({None: {}})

    class SecondParser(FirstParser):
        dataset_types = ("second",)

    monkeypatch.setattr(PARSERS_REGISTRY, "get", lambda name: NeverParser)
    with pytest.raises(ValueError, match="never parser"):
        get_parser_plugin(tmp_path, "never")

    monkeypatch.setattr(PARSERS_REGISTRY, "get", lambda name: FirstParser)
    plugin, type_name, layout = get_parser_plugin(tmp_path, "first")
    assert (plugin, type_name) == (FirstParser, "first")
    assert layout.splits == {None: {}}

    monkeypatch.setattr(PARSERS_REGISTRY, "values", lambda: [NeverParser])
    with pytest.raises(ValueError, match="any registered parser"):
        get_parser_plugin(tmp_path, None)

    monkeypatch.setattr(
        PARSERS_REGISTRY, "values", lambda: [FirstParser, SecondParser]
    )
    with pytest.raises(ValueError, match="multiple parsers: first, second"):
        get_parser_plugin(tmp_path, None)

    monkeypatch.setattr(PARSERS_REGISTRY, "values", lambda: [FirstParser])
    plugin, type_name, _layout = get_parser_plugin(tmp_path, None)
    assert (plugin, type_name) == (FirstParser, "first")


def test_get_parser_plugin_resolves_by_split_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Multiple matches are broken by how many splits each one recognizes.

    YOLOv6 and Ultralytics YOLOv8 layouts differ only in split naming
    (``images/valid`` against ``images/val``), so both parsers report support
    for either tree — but only the right one recognizes every split present.
    Equal coverage stays genuinely ambiguous and must be reported, since
    guessing would silently parse the source with the wrong parser.
    """

    class SyntheticYoloV6(_SyntheticSplitParser):
        dataset_types = ("yolov6",)
        # `ClassVar`: reassigned per case below, never per instance.
        splits: ClassVar[dict[str | None, dict[str, Any]]] = {}

        @classmethod
        def detect(cls, source: Path) -> Layout | None:
            # Always recognizes the tree; only how many splits it names
            # differs, which is exactly what the tie-break weighs.
            return Layout(dict(cls.splits) or {None: {}})

    class SyntheticYoloV8(SyntheticYoloV6):
        dataset_types = ("yolov8",)
        splits: ClassVar[dict[str | None, dict[str, Any]]] = {}

    monkeypatch.setattr(
        PARSERS_REGISTRY,
        "values",
        lambda: [SyntheticYoloV6, SyntheticYoloV8],
    )

    SyntheticYoloV6.splits = {"train": {}, "valid": {}, "test": {}}
    SyntheticYoloV8.splits = {"train": {}}
    assert get_parser_plugin(tmp_path, None)[:2] == (SyntheticYoloV6, "yolov6")

    SyntheticYoloV6.splits = {"train": {}}
    SyntheticYoloV8.splits = {"train": {}, "val": {}}
    assert get_parser_plugin(tmp_path, None)[:2] == (SyntheticYoloV8, "yolov8")

    SyntheticYoloV6.splits = {"test": {}}
    SyntheticYoloV8.splits = {"test": {}}
    with pytest.raises(ValueError, match="multiple parsers: yolov6, yolov8"):
        get_parser_plugin(tmp_path, None)

    # No split at all is no evidence either way, so it stays ambiguous too.
    SyntheticYoloV6.splits = {}
    SyntheticYoloV8.splits = {}
    with pytest.raises(ValueError, match="multiple parsers: yolov6, yolov8"):
        get_parser_plugin(tmp_path, None)


def test_parser_entry_point_that_fails_to_import_is_skipped(
    monkeypatch: pytest.MonkeyPatch,
):
    """A plugin that cannot even be loaded must not break ``import``.

    Regression: only ``KeyError`` and ``ValueError`` were caught, but
    ``EntryPoint.load()`` imports the plugin's module — a missing optional
    dependency raises ``ModuleNotFoundError`` and a renamed class raises
    ``AttributeError``. Neither was caught, so ``_load_parser_plugins``
    running at module scope made plain ``import luxonis_ml.data`` raise,
    which is the exact case its docstring promises to survive.
    """

    class BrokenEntryPoint:
        name = "broken-plugin"

        @staticmethod
        def load() -> type[ParserPlugin]:
            raise ModuleNotFoundError("No module named 'not_installed'")

    monkeypatch.setattr(
        data_module,
        "_get_entry_points_subset",
        lambda group: (
            [BrokenEntryPoint()] if group == "parser_plugins" else []
        ),
    )
    messages: list[str] = []
    sink_id = logger.add(
        lambda message: messages.append(str(message).strip()),
        format="{message}",
        level="WARNING",
    )
    try:
        data_module._load_parser_plugins()
    finally:
        logger.remove(sink_id)

    assert any("broken-plugin" in message for message in messages)
