from collections.abc import (
    Callable,
    Sequence,
)
from pathlib import Path
from typing import (
    Any,
    TypeVar,
    cast,
)

from PIL import Image

from luxonis_ml.data import (
    DatasetIterator,
    Layout,
    ParseIssueCollector,
    ParseResult,
    ParserPlugin,
)
from luxonis_ml.data.parsers.parser_plugin import (
    SplitParserPlugin,
)
from tests.test_data.utils import create_image


def _collect_raw_records(records: DatasetIterator) -> list[dict[str, Any]]:
    """Collect parser output, which plugins emit as plain dictionaries."""
    collected = []
    for record in records:
        assert isinstance(record, dict)
        collected.append(record)
    return collected


def _write_yolov8_split(
    split_path: Path,
    image_indices: Sequence[int],
    *,
    annotate: Callable[[int], str | None] = lambda _: "0 0.5 0.5 0.4 0.4\n",
) -> None:
    """Write a Roboflow-style YOLOv8 split.

    Args:
        split_path: Split directory to populate.
        image_indices: Indices passed to `create_image`.
        annotate: Label file content per index. ``None`` writes no label file
            at all, which is how parsers see a background image.

    """
    image_dir = split_path / "images"
    label_dir = split_path / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    for index in image_indices:
        create_image(index, image_dir)
        content = annotate(index)
        if content is not None:
            (label_dir / f"img_{index}.txt").write_text(content)


T = TypeVar("T", bound=ParserPlugin)


def _image(
    path: Path,
    *,
    size: tuple[int, int] = (20, 10),
    value: int = 0,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(value, value, value)).save(path)
    return path


def _plugin(parser_type: type[T]) -> T:
    return parser_type(ParseIssueCollector())


def _records(parsed: "ParseResult | DatasetIterator") -> list[dict[str, Any]]:
    """Collect the records of a parse result or a raw record stream.

    `ParseResult.records` pairs each record with its split, while a
    plugin's `_split_records` yields the records on their own; both are
    reduced to the records themselves so a test can assert on either.
    """
    records = parsed.records if isinstance(parsed, ParseResult) else parsed
    collected = [
        record[1] if isinstance(record, tuple) else record
        for record in records
    ]
    return cast(list[dict[str, Any]], collected)


def _split_records(parsed: ParseResult) -> list[tuple[str | None, Any]]:
    """Collect the split-tagged records of a parse result."""
    return list(parsed.records)


class _SyntheticSplitParser(SplitParserPlugin):
    dataset_types = ("synthetic-split",)
    recognized = False
    splits: dict[str | None, dict[str, Any]] = {}

    @staticmethod
    def validate_split(split_path: Path) -> dict[str, Any] | None:
        return (
            {"source": split_path}
            if _SyntheticSplitParser.recognized
            else None
        )

    @classmethod
    def detect(cls, source: Path) -> Layout | None:
        if cls.splits:
            return Layout(dict(cls.splits))
        split_kwargs = cls.validate_split(source)
        return None if split_kwargs is None else Layout({None: split_kwargs})

    def _split_records(self, **kwargs: Any) -> DatasetIterator:
        del kwargs
        return iter(())


COCO_TRAIN_IMAGES = 100


COCO_VAL_IMAGES = 100


COCO_TEST_IMAGES = 100  # test split has images but 0 annotations
