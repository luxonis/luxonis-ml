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


def test_present_sample_metadata_splits_batch_into_labelled_samples() -> None:
    # A batch augmentation's merged metadata becomes one "sample N" group per
    # contributing input, dropping the duplicated top-level copy and the
    # machine-only input_index/sample_metadata wrapping.
    merged = {
        "record_id": 123,
        "source": "a.jpg",
        "batch_augmentation_metadata": [
            {"input_index": 0, "sample_metadata": {"record_id": 123}},
            {"input_index": 1, "sample_metadata": {"record_id": 456}},
        ],
    }
    assert data_main._present_sample_metadata(merged) == {
        "sample 1": {"record_id": 123},
        "sample 2": {"record_id": 456},
    }


def test_present_sample_metadata_collapses_single_input() -> None:
    merged = {
        "record_id": 7,
        "batch_augmentation_metadata": [
            {"input_index": 0, "sample_metadata": {"record_id": 7}}
        ],
    }
    assert data_main._present_sample_metadata(merged) == {"record_id": 7}


def test_present_sample_metadata_passes_non_batched_through() -> None:
    plain = {"record_id": 1, "source": "x.jpg"}
    assert data_main._present_sample_metadata(plain) is plain


def test_present_sample_metadata_flattens_single_source_filenames() -> None:
    # The common single-image record: the one-entry filenames dict collapses to
    # a "filename" Block field (its own labelled line), keeping the other fields.
    from luxonis_ml.vizlab import Block

    md = {"filenames": {"image": "frame_001.jpg"}, "record_id": 5}
    assert data_main._present_sample_metadata(md) == {
        "filename": Block("frame_001.jpg"),
        "record_id": 5,
    }


def test_present_sample_metadata_keeps_multi_source_filenames() -> None:
    # A true multi-image record keeps the full mapping (nothing to collapse).
    md = {"filenames": {"image": "a.jpg", "depth": "a.png"}, "record_id": 5}
    assert data_main._present_sample_metadata(md) == md


def test_present_sample_metadata_flattens_filenames_per_batch_sample() -> None:
    from luxonis_ml.vizlab import Block

    merged = {
        "batch_augmentation_metadata": [
            {
                "input_index": 0,
                "sample_metadata": {"filenames": {"image": "a.jpg"}},
            },
            {
                "input_index": 1,
                "sample_metadata": {"filenames": {"image": "b.jpg"}},
            },
        ],
    }
    assert data_main._present_sample_metadata(merged) == {
        "sample 1": {"filename": Block("a.jpg")},
        "sample 2": {"filename": Block("b.jpg")},
    }


def test_present_sample_metadata_labels_empty_inputs() -> None:
    merged = {
        "batch_augmentation_metadata": [
            {"input_index": 0, "sample_metadata": {"record_id": 1}},
            {"input_index": 1, "sample_metadata": {}},
        ],
    }
    assert data_main._present_sample_metadata(merged) == {
        "sample 1": {"record_id": 1},
        "sample 2": "(no metadata)",
    }


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
    monkeypatch.setattr(
        viewer_module, "Viewer", lambda **_k: RealViewer(backend)
    )
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
    monkeypatch.setattr(
        viewer_module, "Viewer", lambda **_k: RealViewer(backend)
    )
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

    def make_viewer(**_k: object) -> RealViewer:
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


def test_inspect_show_all_starts_with_decluttering_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Decluttering is on by default; --show-all starts the viewer with it off so
    # every detection is drawn from the first frame (the `d` key still toggles).
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

    created: list[RealViewer] = []

    monkeypatch.setattr(data_main, "check_exists", lambda *_args: None)
    monkeypatch.setattr(data_main, "LuxonisDataset", _Dataset)
    monkeypatch.setattr(data_main, "LuxonisLoader", _Loader)
    monkeypatch.setattr(
        label_converter,
        "loader_output_to_records",
        lambda *_args, **_kwargs: {"objects": record},
    )

    def declutter_after(*, show_all: bool) -> bool:
        created.clear()

        def make_viewer(**_k: object) -> RealViewer:
            viewer = RealViewer(_FakeBackend(keys=[ord("q")]))
            created.append(viewer)
            return viewer

        monkeypatch.setattr(viewer_module, "Viewer", make_viewer)
        data_main.inspect("dataset", show_all=show_all)
        return created[0].layers.declutter

    assert declutter_after(show_all=False) is True  # on by default
    assert declutter_after(show_all=True) is False  # --show-all turns it off


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
            lambda **_k: RealViewer(_FakeBackend(keys=[ord("q")])),
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
    monkeypatch.setattr(
        viewer_module, "Viewer", lambda **_k: real_viewer(backend)
    )
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


def _save_mocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wire fakes so ``inspect`` runs headless over one 60x40 car sample."""
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    record = DatasetRecord.model_construct(
        files={},
        sample_metadata={"weather": "clear"},
        annotation=[
            Detection(
                class_name="car",
                instance_id=1,
                boundingbox=BBoxAnnotation(x=0.1, y=0.1, w=0.3, h=0.3),
            )
        ],
        task_name="a",
    )
    records = {"a": record}

    class _Dataset:
        def __init__(self, *_a: object, **_k: object) -> None:
            pass

        def __len__(self) -> int:
            return 1

        def get_classes(self) -> dict[str, dict[str, int]]:
            return {"a": {"car": 0}}

        def get_class_names(self) -> dict[str, list[str]]:
            return {"a": ["car"]}

        def get_categorical_encodings(self) -> dict[str, object]:
            return {}

        def get_skeletons(self) -> dict[str, object]:
            return {}

    class _Loader:
        def __init__(self, *_a: object, **_k: object) -> None:
            self._augmentations = None

        def __iter__(self) -> Iterator[SimpleNamespace]:
            yield SimpleNamespace(
                images={"frame01.jpg": image}, labels={}, metadata={}
            )

    from luxonis_ml.data.loaders import label_converter

    monkeypatch.setattr(data_main, "check_exists", lambda *_a: None)
    monkeypatch.setattr(data_main, "LuxonisDataset", _Dataset)
    monkeypatch.setattr(data_main, "LuxonisLoader", _Loader)
    monkeypatch.setattr(
        label_converter,
        "loader_output_to_records",
        lambda *_a, **_k: records,
    )


def test_inspect_save_writes_svg_and_png(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from luxonis_ml.vizlab import RenderOptions, set_default_options

    _save_mocks(monkeypatch)
    try:
        data_main.inspect("ds", save=tmp_path / "svg", save_format="svg")
        data_main.inspect("ds", save=tmp_path / "png", save_format="png")
    finally:
        set_default_options(RenderOptions())

    svg = tmp_path / "svg" / "0000_frame01.svg"
    png = tmp_path / "png" / "0000_frame01.png"
    assert svg.read_bytes().startswith(b"<?xml")  # a vector document
    assert b"<image" in svg.read_bytes()  # with the photo embedded
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"  # a raster encode


def test_inspect_save_plain_drops_the_panel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import cv2

    from luxonis_ml.vizlab import RenderOptions, set_default_options

    _save_mocks(monkeypatch)
    try:
        data_main.inspect(
            "ds", save=tmp_path / "panel", legend=True, plain=False
        )
        data_main.inspect(
            "ds", save=tmp_path / "plain", legend=True, plain=True
        )
    finally:
        set_default_options(RenderOptions())

    paneled = cv2.imread(str(tmp_path / "panel" / "0000_frame01.png"))
    plain = cv2.imread(str(tmp_path / "plain" / "0000_frame01.png"))
    assert plain.shape[1] == 60  # just the source image, no panel
    assert paneled.shape[1] > plain.shape[1]  # the panel widened it
