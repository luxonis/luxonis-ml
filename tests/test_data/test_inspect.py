"""End-to-end coverage for the ``data inspect`` command (thin viewer adapter)."""

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import luxonis_ml.data.__main__ as data_main
import luxonis_ml.vizlab.viewer as viewer_module
from luxonis_ml.ldf import BBoxAnnotation, DatasetRecord, Detection


class _FakeBackend:
    """A headless `WindowBackend` recording shown windows, replaying keys."""

    def __init__(self, keys: list[int]) -> None:
        self._keys = list(keys)
        self.shown: list[str] = []

    def screen_size(self) -> tuple[int, int] | None:
        return None

    def create_window(self, name: str) -> None:
        pass

    def destroy_window(self, name: str) -> None:
        pass

    def show(self, name: str, frame: np.ndarray) -> None:
        self.shown.append(name)

    def resize(self, name: str, width: int, height: int) -> None:
        pass

    def center(
        self, name: str, width: int, height: int, screen: tuple[int, int]
    ) -> None:
        pass

    def set_mouse_handler(self, name: str, handler: object) -> None:
        pass

    def poll_key(self, timeout_ms: int) -> int:
        return self._keys.pop(0) if self._keys else ord("q")

    def close(self) -> None:
        pass


def test_per_instance_inspect_attaches_augmentation_panel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    image = np.zeros((32, 48, 3), dtype=np.uint8)
    record = DatasetRecord.model_construct(
        files={},
        annotation=[
            Detection(
                class_name="car",
                boundingbox=BBoxAnnotation(x=0.1, y=0.1, w=0.4, h=0.4),
            )
        ],
        task_name="objects",
    )

    class _Dataset:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def __len__(self) -> int:
            return 1

        def get_classes(self) -> dict[str, dict[str, int]]:
            return {"objects": {"car": 0}}

        def get_class_names(self) -> dict[str, list[str]]:
            return {"objects": ["car"]}

        def get_categorical_encodings(self) -> dict[str, object]:
            return {}

        def get_skeletons(self) -> dict[str, object]:
            return {}

    class _Loader:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self._augmentations = None

        def __getitem__(
            self, _index: int
        ) -> tuple[dict[str, np.ndarray], dict[str, object]]:
            return {"image": image}, {}

        def __iter__(self) -> Iterator[SimpleNamespace]:
            yield SimpleNamespace(
                images={"image": image},
                labels={},
                metadata={},
            )

        def _init_augmentations(self, **_kwargs: object) -> object:
            return object()

    class _Collector:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def get_applied_augmentations(self) -> list[str]:
            return ["HorizontalFlip"]

    from luxonis_ml.data.loaders import label_converter
    from luxonis_ml.vizlab import Image
    from luxonis_ml.vizlab.viewer import Viewer as RealViewer

    panels: list[object] = []

    def capture_panel(self: Image, data: object, **_kwargs: object) -> Image:
        panels.append(data)
        return self

    backend = _FakeBackend(keys=[ord("q")])
    monkeypatch.setattr(data_main, "check_exists", lambda *_args: None)
    monkeypatch.setattr(data_main, "LuxonisDataset", _Dataset)
    monkeypatch.setattr(data_main, "LuxonisLoader", _Loader)
    monkeypatch.setattr(data_main, "AugmentationsCollector", _Collector)
    # inspect imports Viewer from the viewer package at call time, so patch it
    # there; the real viewer drives a headless fake backend.
    monkeypatch.setattr(viewer_module, "Viewer", lambda: RealViewer(backend))
    monkeypatch.setattr(
        label_converter,
        "loader_output_to_records",
        lambda *_args, **_kwargs: {"objects": record},
    )
    monkeypatch.setattr(Image, "with_panel", capture_panel)

    from luxonis_ml.vizlab import (
        LIGHT_THEME,
        RenderOptions,
        get_default_theme,
        set_default_options,
    )

    aug_config = tmp_path / "augmentations.json"
    aug_config.write_text("[]")
    try:
        data_main.inspect(
            "dataset",
            aug_config=aug_config,
            per_instance=True,
            list_augmentations=True,
            theme="light",
        )
        # The chosen theme becomes the scope default (with the dataset palette
        # pinned onto it, so it is a light-background theme, not LIGHT_THEME itself).
        assert get_default_theme().background == LIGHT_THEME.background
    finally:
        set_default_options(RenderOptions())

    assert panels == [{"augmentations": ["HorizontalFlip"]}]


def test_inspect_grid_renders_real_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Two tasks -> the grid path: fit_grid + visualize_record + render_hits run
    # for real (only the window backend is faked), quitting after one sample.
    image = np.zeros((40, 60, 3), dtype=np.uint8)

    def _record(task: str) -> DatasetRecord:
        return DatasetRecord.model_construct(
            files={},
            sample_metadata={},
            annotation=[
                Detection(
                    class_name="car",
                    instance_id=1,
                    boundingbox=BBoxAnnotation(x=0.1, y=0.1, w=0.3, h=0.3),
                    metadata={"track_id": 7},
                )
            ],
            task_name=task,
        )

    records = {"a": _record("a"), "b": _record("b")}

    class _Dataset:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def __len__(self) -> int:
            return 1

        def get_classes(self) -> dict[str, dict[str, int]]:
            return {"a": {"car": 0}, "b": {"car": 0}}

        def get_class_names(self) -> dict[str, list[str]]:
            return {"a": ["car"], "b": ["car"]}

        def get_categorical_encodings(self) -> dict[str, object]:
            return {}

        def get_skeletons(self) -> dict[str, object]:
            return {}

    class _Loader:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self._augmentations = None

        def __iter__(self) -> Iterator[SimpleNamespace]:
            yield SimpleNamespace(
                images={"image": image}, labels={}, metadata={}
            )

    from luxonis_ml.data.loaders import label_converter
    from luxonis_ml.vizlab.viewer import Viewer as RealViewer

    backend = _FakeBackend(keys=[ord("q")])
    monkeypatch.setattr(data_main, "check_exists", lambda *_args: None)
    monkeypatch.setattr(data_main, "LuxonisDataset", _Dataset)
    monkeypatch.setattr(data_main, "LuxonisLoader", _Loader)
    monkeypatch.setattr(viewer_module, "Viewer", lambda: RealViewer(backend))
    monkeypatch.setattr(
        label_converter,
        "loader_output_to_records",
        lambda *_args, **_kwargs: records,
    )

    from luxonis_ml.vizlab import RenderOptions, set_default_options

    try:
        data_main.inspect("dataset", legend=True)
    finally:
        set_default_options(RenderOptions())

    # One composited window was presented for the sole source image.
    assert backend.shown == ["image"]
