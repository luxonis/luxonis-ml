"""End-to-end coverage for the ``data inspect`` command (thin viewer adapter)."""

from collections.abc import Callable, Iterator
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

    def set_key_handler(self, handler: object) -> None:
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


def test_inspect_layer_key_rerenders_and_toggles_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A layer-control key ('m') is intercepted by the viewer: it toggles the
    # shared state and re-renders the window in place instead of advancing.
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    record = DatasetRecord.model_construct(
        files={},
        sample_metadata={},
        annotation=[
            Detection(
                class_name="car",
                boundingbox=BBoxAnnotation(x=0.1, y=0.1, w=0.3, h=0.3),
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

        def __iter__(self) -> Iterator[SimpleNamespace]:
            yield SimpleNamespace(
                images={"image": image}, labels={}, metadata={}
            )

    from luxonis_ml.data.loaders import label_converter
    from luxonis_ml.vizlab.viewer import Viewer as RealViewer

    backend = _FakeBackend(keys=[ord("m"), ord("q")])
    created: list[RealViewer] = []

    def make_viewer() -> RealViewer:
        viewer = RealViewer(backend)
        created.append(viewer)
        return viewer

    monkeypatch.setattr(data_main, "check_exists", lambda *_args: None)
    monkeypatch.setattr(data_main, "LuxonisDataset", _Dataset)
    monkeypatch.setattr(data_main, "LuxonisLoader", _Loader)
    monkeypatch.setattr(viewer_module, "Viewer", make_viewer)
    monkeypatch.setattr(
        label_converter,
        "loader_output_to_records",
        lambda *_args, **_kwargs: {"objects": record},
    )

    from luxonis_ml.vizlab import RenderOptions, set_default_options

    try:
        data_main.inspect("dataset")
    finally:
        set_default_options(RenderOptions())

    viewer = created[0]
    assert viewer.layers.masks is False  # 'm' toggled masks off
    # The window was painted twice: the initial show plus the 'm' re-render.
    assert backend.shown == ["image", "image"]


def test_inspect_fast_lightens_the_render_style(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # `--fast` installs a theme whose default style skips mask contours and drop
    # shadows, while the default run keeps the crisp, shadowed look.
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    record = DatasetRecord.model_construct(
        files={},
        sample_metadata={},
        annotation=[
            Detection(
                class_name="car",
                boundingbox=BBoxAnnotation(x=0.1, y=0.1, w=0.3, h=0.3),
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

        def __iter__(self) -> Iterator[SimpleNamespace]:
            yield SimpleNamespace(
                images={"image": image}, labels={}, metadata={}
            )

    from luxonis_ml.data.loaders import label_converter
    from luxonis_ml.vizlab import (
        MaskOutline,
        RenderOptions,
        current_options,
        set_default_options,
    )
    from luxonis_ml.vizlab.viewer import Viewer as RealViewer

    monkeypatch.setattr(data_main, "check_exists", lambda *_args: None)
    monkeypatch.setattr(data_main, "LuxonisDataset", _Dataset)
    monkeypatch.setattr(data_main, "LuxonisLoader", _Loader)
    monkeypatch.setattr(
        label_converter,
        "loader_output_to_records",
        lambda *_args, **_kwargs: {"objects": record},
    )

    def run(*, fast: bool = False) -> tuple[MaskOutline, bool, bool]:
        monkeypatch.setattr(
            viewer_module,
            "Viewer",
            lambda: RealViewer(_FakeBackend(keys=[ord("q")])),
        )
        try:
            data_main.inspect("dataset", fast=fast)
            opts = current_options()
            return (
                opts.theme.style.mask_outline,
                opts.theme.style.shadow,
                opts.antialias,
            )
        finally:
            set_default_options(RenderOptions())

    # default: crisp, shadowed, anti-aliased
    assert run() == (MaskOutline.SMOOTH, True, True)
    # --fast: fill-only masks, no shadows, no shape anti-aliasing
    assert run(fast=True) == (MaskOutline.NONE, False, False)


def _compare_mocks(
    monkeypatch: pytest.MonkeyPatch,
    image: np.ndarray,
    real_viewer: Callable[..., object] | None = None,
) -> "tuple[_FakeBackend, list]":
    """Wire the ``compare`` command onto a fake dataset/loader/viewer.

    Returns the fake backend and a list capturing every ``with_panel`` call.
    ``real_viewer`` lets a caller pass the real `Viewer` class captured *before*
    any patch, so repeated calls do not re-import an already-patched name.
    """

    def _record() -> DatasetRecord:
        return DatasetRecord.model_construct(
            files={},
            sample_metadata={},
            annotation=[
                Detection(
                    class_name="car",
                    instance_id=1,
                    boundingbox=BBoxAnnotation(x=0.1, y=0.1, w=0.3, h=0.3),
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

        def __iter__(self) -> Iterator[SimpleNamespace]:
            yield SimpleNamespace(
                images={"image": image}, labels={}, metadata={}
            )

    from luxonis_ml.data.loaders import label_converter
    from luxonis_ml.vizlab import Image

    if real_viewer is None:
        from luxonis_ml.vizlab.viewer import Viewer

        real_viewer = Viewer

    panels: list[tuple[object, object]] = []

    def capture_panel(self: Image, data: object, **kwargs: object) -> Image:
        panels.append((kwargs.get("title"), data))
        return self

    backend = _FakeBackend(keys=[ord("q")])
    monkeypatch.setattr(data_main, "check_exists", lambda *_args: None)
    monkeypatch.setattr(data_main, "LuxonisDataset", _Dataset)
    monkeypatch.setattr(data_main, "LuxonisLoader", _Loader)
    monkeypatch.setattr(viewer_module, "Viewer", lambda: real_viewer(backend))
    monkeypatch.setattr(
        label_converter,
        "loader_output_to_records",
        lambda *_args, **_kwargs: {"objects": _record()},
    )
    monkeypatch.setattr(Image, "with_panel", capture_panel)
    return backend, panels


def test_compare_command_renders_verdict_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # `data compare gt preds` matches predictions against ground truth and shows a
    # verdict overlay with a metrics panel. Matching, rendering, and render_hits
    # run for real; only the window backend is faked.
    backend, panels = _compare_mocks(
        monkeypatch, np.zeros((40, 60, 3), dtype=np.uint8)
    )
    from luxonis_ml.vizlab import RenderOptions, set_default_options

    try:
        data_main.compare("ground_truth", "predictions")
    finally:
        set_default_options(RenderOptions())

    assert backend.shown == ["image"]  # the comparison frame was presented
    # Identical GT and predictions -> a single true positive, no false positives.
    metrics = next(data for title, data in panels if title == "Comparison")
    assert isinstance(metrics, dict)
    assert metrics["TP"] == 1
    assert metrics["FP"] == 0


def test_compare_command_supports_dual_and_triple_layouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from luxonis_ml.vizlab import RenderOptions, set_default_options
    from luxonis_ml.vizlab.viewer import Viewer as RealViewer

    for layout in ("dual", "triple"):
        backend, _ = _compare_mocks(
            monkeypatch, np.zeros((40, 60, 3), dtype=np.uint8), RealViewer
        )
        try:
            data_main.compare("gt", "preds", layout=layout)  # type: ignore[arg-type]
        finally:
            set_default_options(RenderOptions())
        assert backend.shown == [
            "image"
        ]  # the multi-panel frame was presented


def test_compare_command_errors_only_still_shows_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from luxonis_ml.vizlab import RenderOptions, set_default_options

    backend, _ = _compare_mocks(
        monkeypatch, np.zeros((40, 60, 3), dtype=np.uint8)
    )
    try:
        # Identical GT/preds -> a lone true positive is filtered out, but the
        # frame (with its metrics panel) is still presented.
        data_main.compare("gt", "preds", errors_only=True)
    finally:
        set_default_options(RenderOptions())
    assert backend.shown == ["image"]


def test_compare_command_summary_writes_confusion_figure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from luxonis_ml.vizlab import RenderOptions, set_default_options

    monkeypatch.chdir(tmp_path)
    backend, _ = _compare_mocks(
        monkeypatch, np.zeros((40, 60, 3), dtype=np.uint8)
    )
    try:
        data_main.compare("gt", "preds", summary=True, per_class=True)
    finally:
        set_default_options(RenderOptions())

    assert backend.shown == []  # headless: no interactive window
    assert (tmp_path / "gt_vs_preds_confusion.png").exists()
