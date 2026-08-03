"""End-to-end coverage for the ``data inspect`` command (thin viewer adapter)."""

import re
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from threading import Event, get_ident
from types import SimpleNamespace

import numpy as np
import pytest
from rich.console import Console

import luxonis_ml.data.__main__ as data_main
import luxonis_ml.vizlab.viewer as viewer_module
from luxonis_ml.data.utils.enums import BucketStorage
from luxonis_ml.data.utils.inspection import SampleFilterConfig
from luxonis_ml.ldf import (
    BBoxAnnotation,
    DatasetRecord,
    Detection,
    KeypointAnnotation,
)
from luxonis_ml.typing import (
    Labels,
    LoaderOutput,
    Params,
    TrackedAugmentations,
)


def _ignore_exists(
    _name: str,
    _bucket_storage: BucketStorage,
) -> None:
    """Stand in for the CLI's dataset-existence guard."""


def test_filter_config_has_flat_options_on_inspect_and_compare() -> None:
    _, inspect_arguments, inspect_ignored = data_main.app.parse_args(
        [
            "inspect",
            "dataset",
            "--task-name",
            "objects",
            "--task-name-mode",
            "exclude",
            "--metadata-filter",
            "camera.side",
            "left",
        ],
        exit_on_error=False,
    )
    _, compare_arguments, compare_ignored = data_main.app.parse_args(
        [
            "compare",
            "ground-truth",
            "predictions",
            "--class-name",
            "car",
            "--class-name-mode",
            "exclude",
            "--search",
            "0042",
        ],
        exit_on_error=False,
    )

    assert inspect_arguments.arguments["filters"] == SampleFilterConfig(
        task_name=["objects"],
        task_name_mode="exclude",
        metadata_filter=[("camera.side", "left")],
    )
    assert compare_arguments.arguments["filters"] == SampleFilterConfig(
        class_name=["car"],
        class_name_mode="exclude",
        search="0042",
    )
    assert inspect_ignored == {}
    assert compare_ignored == {}


@pytest.mark.parametrize("command", ["inspect", "compare"])
@pytest.mark.parametrize(
    ("flag", "expected"),
    [("--plain", True), ("--no-plain", False), (None, None)],
)
def test_plain_is_tri_state_on_both_commands(
    command: str, flag: str | None, expected: bool | None
) -> None:
    # 'None' is what lets --save decide: a clip defaults to plain, everything
    # else does not. Both spellings have to reach the command for that to work.
    argv = [command, "dataset", *(["other"] if command == "compare" else [])]
    _, arguments, _ = data_main.app.parse_args(
        [*argv, *([flag] if flag else [])], exit_on_error=False
    )
    # An unpassed option is absent rather than present-and-None, which is the
    # same thing as far as the command's default is concerned.
    assert arguments.arguments.get("plain") is expected


def _command_help(command: str) -> str:
    console = Console(record=True, width=120)
    data_main.app.help_print([command], console=console)
    return console.export_text()


def _help_panels(help_text: str) -> dict[str, str]:
    """Split rendered help into its group panels, keyed by title."""
    panels: dict[str, list[str]] = {}
    body: list[str] | None = None
    for line in help_text.splitlines():
        heading = re.match(r"╭─+ (.+?) ─+╮$", line)
        if heading is not None:
            body = panels.setdefault(str(heading.group(1)), [])
        elif line.startswith("╰"):
            body = None
        elif body is not None:
            body.append(line)
    return {name: "\n".join(body) for name, body in panels.items()}


def test_inspect_and_compare_help_group_related_options() -> None:
    inspect_help = _command_help("inspect")
    compare_help = _command_help("compare")

    for group in (
        "Dataset options",
        "Sample filters",
        "Augmentation options",
        "Visualization options",
        "Keypoint options",
        "Segmentation options",
        "Array options",
        "Viewer options",
        "Output options",
    ):
        assert group in inspect_help

    for group in (
        "Dataset options",
        "Sample filters",
        "Matching options",
        "Visualization options",
        "Keypoint options",
        "Segmentation options",
        "Reporting options",
    ):
        assert group in compare_help


@pytest.mark.parametrize("command", ["inspect", "compare"])
@pytest.mark.parametrize(
    ("flag", "group"),
    [
        ("--skeletons", "Keypoint options"),
        ("--keypoint-labels", "Keypoint options"),
        ("--show-background", "Segmentation options"),
    ],
)
def test_label_type_flags_live_in_their_own_panel(
    command: str, flag: str, group: str
) -> None:
    # A dataset with no keypoints (or masks) should be able to skip a whole
    # panel, which only works if the flags are not mixed into the general one.
    panels = _help_panels(_command_help(command))
    assert flag in panels[group]
    assert flag not in panels["Visualization options"]


def test_every_array_flag_lives_in_the_array_panel() -> None:
    panels = _help_panels(_command_help("inspect"))
    array_flags = set(re.findall(r"--array-[\w-]+", _command_help("inspect")))
    assert len(array_flags) >= 9
    for flag in array_flags:
        assert flag in panels["Array options"]
    assert "--array" not in panels["Visualization options"]


class _FakeBackend:
    """A headless `WindowBackend` recording shown windows, replaying keys."""

    def __init__(
        self,
        keys: list[int],
        screen: tuple[int, int] | None = None,
    ) -> None:
        self._keys = list(keys)
        self._screen = screen
        self.shown: list[str] = []

    def screen_size(self) -> tuple[int, int] | None:
        return self._screen

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
    merged: Params = {
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


def test_array_labels_keep_complete_nested_task_paths() -> None:
    # The one place deciding which labels are arrays and how --task-name scopes
    # them, so the renderer and the annotation-type filter cannot disagree.
    labels = {
        "parent/depth/array": np.zeros((2, 3)),
        "parent/flow/array": np.zeros((4, 5, 2)),
    }

    assert sorted(data_main._array_labels(labels)) == [
        "parent/depth",
        "parent/flow",
    ]
    assert sorted(
        data_main._array_labels(labels, frozenset({"parent/depth"}))
    ) == ["parent/depth"]
    assert sorted(
        data_main._array_labels(labels, frozenset({"parent/depth"}), "exclude")
    ) == ["parent/flow"]


def test_present_sample_metadata_collapses_single_input() -> None:
    merged: Params = {
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


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        # A saved frame carries the name of the image it came from, not the
        # window's -- one window shows every sample of the dataset in turn.
        ({"filenames": {"image": "frame_001.jpg"}}, "frame_001"),
        # Directories are dropped; the viewer wants a filename stem.
        ({"filenames": {"image": "train/2024/frame_001.png"}}, "frame_001"),
        # One frame tiles every source of a multi-image sample, so it is named
        # after all of them.
        ({"filenames": {"left": "a.jpg", "right": "b.jpg"}}, "a-b"),
        # Nothing to go on -> the viewer falls back to the window name.
        ({"filenames": {}}, None),
        ({"record_id": 5}, None),
    ],
)
def test_sample_stem_names_a_save_after_its_source_image(
    metadata: Params, expected: "str | None"
) -> None:
    assert data_main._sample_stem(metadata) == expected


def test_present_sample_metadata_keeps_multi_source_filenames() -> None:
    # A true multi-image record keeps the full mapping (nothing to collapse).
    md = {"filenames": {"image": "a.jpg", "depth": "a.png"}, "record_id": 5}
    assert data_main._present_sample_metadata(md) == md


def test_present_sample_metadata_flattens_filenames_per_batch_sample() -> None:
    from luxonis_ml.vizlab import Block

    merged: Params = {
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
    merged: Params = {
        "batch_augmentation_metadata": [
            {"input_index": 0, "sample_metadata": {"record_id": 1}},
            {"input_index": 1, "sample_metadata": {}},
        ],
    }
    assert data_main._present_sample_metadata(merged) == {
        "sample 1": {"record_id": 1},
        "sample 2": "(no metadata)",
    }


def test_per_instance_inspect_combines_instances_with_colors_and_tooltips(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    image = np.zeros((32, 48, 3), dtype=np.uint8)
    record = DatasetRecord.model_construct(
        files={},
        annotation=[
            Detection(
                class_name="car",
                instance_id=7,
                boundingbox=BBoxAnnotation(x=0.1, y=0.1, w=0.3, h=0.4),
                keypoints=KeypointAnnotation(
                    keypoints=[(0.2, 0.2, 2), (0.3, 0.3, 2)]
                ),
                metadata={"track_id": 41},
            ),
            Detection(
                class_name="car",
                instance_id=8,
                boundingbox=BBoxAnnotation(x=0.55, y=0.1, w=0.3, h=0.4),
                metadata={"track_id": 42},
            ),
        ],
        task_name="objects",
    )
    ignored_record = DatasetRecord.model_construct(
        files={},
        annotation=[
            Detection(
                class_name="bus",
                instance_id=9,
                boundingbox=BBoxAnnotation(x=0.2, y=0.6, w=0.3, h=0.3),
            )
        ],
        task_name="ignored",
    )

    class _Dataset:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def __len__(self) -> int:
            return 1

        def get_classes(self) -> dict[str, dict[str, int]]:
            return {"objects": {"car": 0}, "ignored": {"bus": 0}}

        def get_class_names(self) -> dict[str, list[str]]:
            return {"objects": ["car"], "ignored": ["bus"]}

        def get_task_names(self) -> list[str]:
            return ["objects", "ignored"]

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
                # What the loader records for an augmented sample: the applied
                # transforms keyed by name, with their runtime parameters.
                metadata={
                    "augmentations": TrackedAugmentations(
                        {"HorizontalFlip": {"p": 1.0}}
                    )
                },
            )

        def _init_augmentations(self, **_kwargs: object) -> object:
            return object()

    import luxonis_ml.vizlab.adapters.instances as instances_module
    from luxonis_ml.data.loaders import label_converter
    from luxonis_ml.vizlab import (
        LIGHT_THEME,
        Annotation,
        Frame,
        Hints,
        Palette,
        RenderOptions,
        Style,
        current_options,
        set_default_options,
    )
    from luxonis_ml.vizlab.adapters import InstanceDetection
    from luxonis_ml.vizlab.color import ColorLike
    from luxonis_ml.vizlab.layout.panel import PanelData
    from luxonis_ml.vizlab.viewer import Viewer as RealViewer

    panels: list[PanelData] = []
    converted: list[list[Annotation]] = []
    real_convert = instances_module.instances_to_annotations

    def capture_panel(
        self: Frame,
        data: PanelData,
        *,
        side: str = "right",
        width: float | None = None,
        title: str | None = None,
        style: Style | None = None,
        bg: ColorLike | None = None,
    ) -> Frame:
        panels.append(data)
        return self

    def capture_annotations(
        instances: Sequence[InstanceDetection],
        *,
        options: RenderOptions,
        palette: Palette,
    ) -> list[Annotation]:
        annotations = real_convert(
            instances,
            options=options,
            palette=palette,
        )
        converted.append(annotations)
        return annotations

    # One ordinary advance key is enough for the combined view. The former
    # stepping behavior would have opened one window per detection.
    backend = _FakeBackend(keys=[ord("x")])
    monkeypatch.setattr(data_main, "check_exists", _ignore_exists)
    monkeypatch.setattr(data_main, "LuxonisDataset", _Dataset)
    monkeypatch.setattr(data_main, "LuxonisLoader", _Loader)
    # inspect imports Viewer from the viewer package at call time, so patch it
    # there; the real viewer drives a headless fake backend.
    monkeypatch.setattr(
        viewer_module, "Viewer", lambda **_k: RealViewer(backend)
    )
    monkeypatch.setattr(
        label_converter,
        "loader_output_to_records",
        lambda *_args, **_kwargs: {
            "objects": record,
            "ignored": ignored_record,
        },
    )
    monkeypatch.setattr(
        instances_module,
        "instances_to_annotations",
        capture_annotations,
    )
    monkeypatch.setattr(Frame, "with_panel", capture_panel)

    aug_config = tmp_path / "augmentations.json"
    aug_config.write_text("[]")
    try:
        data_main.inspect(
            "dataset",
            aug_config=aug_config,
            per_instance=True,
            list_augmentations=True,
            theme="light",
            legend=True,
            filters=SampleFilterConfig(task_name=["objects"]),
        )
        # The chosen theme becomes the scope default (with the dataset palette
        # pinned onto it, so it is a light-background theme, not LIGHT_THEME itself).
        assert current_options().theme.background == LIGHT_THEME.background
    finally:
        set_default_options(RenderOptions())

    # One window per sample now, titled with the dataset, not one per source.
    assert backend.shown == ["dataset"]
    assert len(panels) == 1
    panel = panels[0]
    assert isinstance(panel, dict)
    # The name is the row; the parameters it sampled ride in its hover tooltip.
    assert panel["augmentations"] == Hints((("HorizontalFlip", {"p": 1.0}),))
    assert "controls" in panel
    assert "classes" not in panel

    assert len(converted) == 1
    first, second = converted[0]
    assert first.color is not None
    assert second.color is not None
    assert first.color != second.color

    first_tip = first.tooltip
    second_tip = second.tooltip
    assert first_tip is not None
    assert second_tip is not None
    assert first_tip.tint == first.color
    assert second_tip.tint == second.color
    assert first_tip.title == "car #7"
    assert first_tip.rows == (
        ("instance_id", "7"),
        ("class", "car"),
        ("task", "objects"),
        ("annotations", "bounding box, keypoints"),
        ("track_id", "41"),
    )
    assert first.children[0].color == first.color
    assert first.children[0].tooltip is first_tip
    assert second_tip.rows[0] == ("instance_id", "8")


@pytest.mark.parametrize(
    ("augmentations", "expected"),
    [
        # The loader's own provenance is bulky runtime parameters; without
        # --list-augmentations it would bury the record metadata.
        (TrackedAugmentations({"HorizontalFlip": {"p": 1.0}}), False),
        # A record's own "augmentations" field is ordinary metadata and stays.
        ({"note": "manual"}, True),
    ],
)
def test_inspect_hides_loader_augmentations_from_the_metadata_panel(
    monkeypatch: pytest.MonkeyPatch,
    augmentations: Params,
    expected: bool,
) -> None:
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    record = DatasetRecord.model_construct(
        files={},
        annotation=[
            Detection(
                class_name="car",
                instance_id=1,
                boundingbox=BBoxAnnotation(x=0.1, y=0.1, w=0.2, h=0.2),
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

        def get_task_names(self) -> list[str]:
            return ["objects"]

        def get_categorical_encodings(self) -> dict[str, object]:
            return {}

        def get_skeletons(self) -> dict[str, object]:
            return {}

    class _Loader:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self._augmentations = None

        def __iter__(self) -> Iterator[SimpleNamespace]:
            yield SimpleNamespace(
                images={"image": image},
                labels={},
                metadata={
                    "split": "train",
                    "augmentations": augmentations,
                },
            )

    from luxonis_ml.data.loaders import label_converter
    from luxonis_ml.vizlab import (
        Frame,
        RenderOptions,
        Style,
        set_default_options,
    )
    from luxonis_ml.vizlab.color import ColorLike
    from luxonis_ml.vizlab.layout.panel import PanelData
    from luxonis_ml.vizlab.viewer import Viewer as RealViewer

    panels: list[PanelData] = []

    def capture_panel(
        self: Frame,
        data: PanelData,
        *,
        side: str = "right",
        width: float | None = None,
        title: str | None = None,
        style: Style | None = None,
        bg: ColorLike | None = None,
    ) -> Frame:
        panels.append(data)
        return self

    backend = _FakeBackend(keys=[ord("q")])
    monkeypatch.setattr(data_main, "check_exists", _ignore_exists)
    monkeypatch.setattr(data_main, "LuxonisDataset", _Dataset)
    monkeypatch.setattr(data_main, "LuxonisLoader", _Loader)
    monkeypatch.setattr(
        viewer_module, "Viewer", lambda **_k: RealViewer(backend)
    )
    monkeypatch.setattr(
        label_converter,
        "loader_output_to_records",
        lambda *_args, **_kwargs: {"objects": record},
    )
    monkeypatch.setattr(Frame, "with_panel", capture_panel)

    try:
        data_main.inspect("dataset", list_augmentations=False)
    finally:
        set_default_options(RenderOptions())

    assert len(panels) == 1
    panel = panels[0]
    assert isinstance(panel, dict)
    # The rest of the record metadata is unaffected either way.
    assert panel["split"] == "train"
    assert ("augmentations" in panel) is expected


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

        def get_task_names(self) -> list[str]:
            return ["a", "b"]

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

    import luxonis_ml.vizlab.adapters.instances as instances_module
    from luxonis_ml.data.loaders import label_converter
    from luxonis_ml.vizlab import Annotation, Palette, RenderOptions
    from luxonis_ml.vizlab.adapters import ColorBy
    from luxonis_ml.vizlab.viewer import Viewer as RealViewer

    backend = _FakeBackend(keys=[ord("q")])
    filtered_tasks: list[list[str]] = []
    color_modes: list[ColorBy] = []
    real_blend = instances_module.blend_records_to_annotations
    real_colored = instances_module.records_to_colored_annotations

    def capture_blend(
        selected_records: Iterable[DatasetRecord],
        options: RenderOptions | None = None,
    ) -> list[Annotation]:
        records_list = list(selected_records)
        filtered_tasks.append([record.task_name for record in records_list])
        return real_blend(records_list, options)

    def capture_coloring(
        selected_records: Sequence[DatasetRecord],
        *,
        color_by: ColorBy,
        options: RenderOptions,
        identity_palette: Palette,
    ) -> list[Annotation]:
        color_modes.append(color_by)
        return real_colored(
            selected_records,
            color_by=color_by,
            options=options,
            identity_palette=identity_palette,
        )

    monkeypatch.setattr(data_main, "check_exists", _ignore_exists)
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
    monkeypatch.setattr(
        instances_module,
        "blend_records_to_annotations",
        capture_blend,
    )
    monkeypatch.setattr(
        instances_module,
        "records_to_colored_annotations",
        capture_coloring,
    )

    from luxonis_ml.vizlab import set_default_options

    try:
        data_main.inspect("dataset", legend=True)
        data_main.inspect(
            "dataset",
            legend=True,
            filters=SampleFilterConfig(task_name=["a"]),
        )
        data_main.inspect("dataset", legend=True, color_by="task")
        data_main.inspect(
            "dataset",
            legend=True,
            filters=SampleFilterConfig(
                task_name=["a"],
                task_name_mode="exclude",
            ),
        )
    finally:
        set_default_options(RenderOptions())

    # The first run uses the two-task grid. Filtering to one task takes the
    # single-record blend path. Include and exclude modes keep opposite tasks.
    assert backend.shown == ["dataset", "dataset", "dataset", "dataset"]
    assert filtered_tasks == [["a"], ["b"]]
    assert color_modes == ["class", "task", "class"]


def test_inspect_rejects_unknown_task_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Dataset:
        def __init__(
            self,
            name: str,
            *,
            bucket_storage: BucketStorage,
        ) -> None:
            pass

        def __len__(self) -> int:
            return 1

        def get_task_names(self) -> list[str]:
            return ["objects", "pose"]

    monkeypatch.setattr(data_main, "check_exists", _ignore_exists)
    monkeypatch.setattr(data_main, "LuxonisDataset", _Dataset)

    with pytest.raises(
        ValueError,
        match=(
            r"Unknown task name\(s\): 'missing'. "
            r"Available task names: 'objects', 'pose'."
        ),
    ):
        data_main.inspect(
            "dataset",
            filters=SampleFilterConfig(task_name=["missing"]),
        )


def test_inspect_rejects_conflicting_instance_color_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Dataset:
        def __init__(
            self,
            name: str,
            *,
            bucket_storage: BucketStorage,
        ) -> None:
            pass

        def __len__(self) -> int:
            return 1

    monkeypatch.setattr(data_main, "check_exists", _ignore_exists)
    monkeypatch.setattr(data_main, "LuxonisDataset", _Dataset)

    with pytest.raises(ValueError, match="--per-instance"):
        data_main.inspect(
            "dataset",
            per_instance=True,
            color_by="task",
        )


def test_inspect_rejects_unknown_class_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Dataset:
        def __init__(
            self,
            name: str,
            *,
            bucket_storage: BucketStorage,
        ) -> None:
            pass

        def __len__(self) -> int:
            return 1

        def get_class_names(self) -> dict[str, list[str]]:
            return {"objects": ["car", "person"]}

    monkeypatch.setattr(data_main, "check_exists", _ignore_exists)
    monkeypatch.setattr(data_main, "LuxonisDataset", _Dataset)

    with pytest.raises(
        ValueError,
        match=(
            r"Unknown class name\(s\): 'bus'. "
            r"Available class names: 'car', 'person'."
        ),
    ):
        data_main.inspect(
            "dataset",
            filters=SampleFilterConfig(class_name=["bus"]),
        )


def test_inspect_sample_filters_select_whole_matching_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = np.zeros((32, 48, 3), dtype=np.uint8)
    records = {
        0: DatasetRecord.model_construct(
            files={},
            annotation=[
                Detection(
                    class_name="person",
                    instance_id=1,
                    boundingbox=BBoxAnnotation(x=0.1, y=0.1, w=0.2, h=0.3),
                    metadata={"confidence": 0.95, "quality": "approved"},
                )
            ],
            task_name="objects",
        ),
        1: DatasetRecord.model_construct(
            files={},
            annotation=[
                Detection(
                    class_name="car",
                    instance_id=2,
                    boundingbox=BBoxAnnotation(x=0.2, y=0.2, w=0.3, h=0.3),
                    metadata={"confidence": 0.91, "quality": "approved"},
                ),
                # A nonmatching class remains visible because filters select
                # whole samples instead of pruning their annotations.
                Detection(
                    class_name="person",
                    instance_id=3,
                    boundingbox=BBoxAnnotation(x=0.6, y=0.2, w=0.2, h=0.3),
                ),
            ],
            task_name="objects",
        ),
    }
    samples = [
        LoaderOutput(
            images={"image": image},
            labels={"marker": np.array([0])},
            metadata={
                "filenames": {"image": "warehouse_0001.jpg"},
                "camera": {"side": "right"},
            },
        ),
        LoaderOutput(
            images={"image": image},
            labels={"marker": np.array([1])},
            metadata={
                "filenames": {"image": "warehouse_0042.jpg"},
                "camera": {"side": "left"},
            },
        ),
    ]

    class _Dataset:
        def __init__(
            self,
            name: str,
            *,
            bucket_storage: BucketStorage,
        ) -> None:
            pass

        def __len__(self) -> int:
            return len(samples)

        def get_classes(self) -> dict[str, dict[str, int]]:
            return {"objects": {"car": 0, "person": 1}}

        def get_class_names(self) -> dict[str, list[str]]:
            return {"objects": ["car", "person"]}

        def get_task_names(self) -> list[str]:
            return ["objects"]

        def get_categorical_encodings(self) -> dict[str, dict[str, int]]:
            return {}

        def get_skeletons(
            self,
        ) -> dict[str, tuple[list[str], list[tuple[int, int]]]]:
            return {}

    class _Loader:
        def __init__(
            self,
            dataset: _Dataset,
            *,
            view: list[str],
            update_mode: str,
        ) -> None:
            self._augmentations = None

        def __iter__(self) -> Iterator[LoaderOutput]:
            yield from samples

    import luxonis_ml.vizlab.adapters.instances as instances_module
    from luxonis_ml.data.loaders import label_converter
    from luxonis_ml.vizlab import (
        Annotation,
        Palette,
        RenderOptions,
        set_default_options,
    )
    from luxonis_ml.vizlab.adapters import ColorBy
    from luxonis_ml.vizlab.viewer import Viewer as RealViewer

    rendered_labels: list[list[str | None]] = []
    real_colored = instances_module.records_to_colored_annotations

    def capture_coloring(
        selected_records: Sequence[DatasetRecord],
        *,
        color_by: ColorBy,
        options: RenderOptions,
        identity_palette: Palette,
    ) -> list[Annotation]:
        annotations = real_colored(
            selected_records,
            color_by=color_by,
            options=options,
            identity_palette=identity_palette,
        )
        rendered_labels.append(
            [annotation.label for annotation in annotations]
        )
        return annotations

    backend = _FakeBackend(keys=[ord("q")])

    def make_viewer(
        *, hud: bool, save_dir: "str | Path | None" = None
    ) -> RealViewer:
        return RealViewer(backend, hud=hud, save_dir=save_dir)

    def convert_labels(
        labels: Labels,
        *,
        classes: dict[str, dict[str, int]],
        categorical_encodings: dict[str, dict[str, int]],
        render_background: bool,
    ) -> dict[str, DatasetRecord]:
        return {"objects": records[int(labels["marker"][0])]}

    monkeypatch.setattr(data_main, "check_exists", _ignore_exists)
    monkeypatch.setattr(data_main, "LuxonisDataset", _Dataset)
    monkeypatch.setattr(data_main, "LuxonisLoader", _Loader)
    monkeypatch.setattr(viewer_module, "Viewer", make_viewer)
    monkeypatch.setattr(
        label_converter,
        "loader_output_to_records",
        convert_labels,
    )
    monkeypatch.setattr(
        instances_module,
        "records_to_colored_annotations",
        capture_coloring,
    )

    try:
        data_main.inspect(
            "dataset",
            filters=SampleFilterConfig(
                class_name=["car"],
                annotation_type=["boundingbox"],
                metadata_filter=[
                    ("camera.side", "left"),
                    ("quality", "approved"),
                ],
                min_confidence=0.9,
                min_instances=2,
                max_instances=2,
                search="0042",
            ),
            list_augmentations=False,
            prefetch=1,
        )
    finally:
        set_default_options(RenderOptions())

    assert backend.shown == ["dataset"]
    assert rendered_labels == [["car", "person"]]


def test_inspect_prefetch_renders_the_next_frame_while_waiting_for_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    samples = [
        LoaderOutput(
            images={"image": image},
            labels={"marker": np.array([index])},
            metadata={"filenames": {"image": f"frame-{index}.jpg"}},
        )
        for index in range(2)
    ]

    class _Dataset:
        def __init__(
            self,
            name: str,
            *,
            bucket_storage: BucketStorage,
        ) -> None:
            pass

        def __len__(self) -> int:
            return len(samples)

        def get_classes(self) -> dict[str, dict[str, int]]:
            return {"objects": {"car": 0}}

        def get_class_names(self) -> dict[str, list[str]]:
            return {"objects": ["car"]}

        def get_categorical_encodings(self) -> dict[str, dict[str, int]]:
            return {}

        def get_skeletons(
            self,
        ) -> dict[str, tuple[list[str], list[tuple[int, int]]]]:
            return {}

    class _Loader:
        def __init__(
            self,
            dataset: _Dataset,
            *,
            view: list[str],
            update_mode: str,
        ) -> None:
            self._augmentations = None

        def __iter__(self) -> Iterator[LoaderOutput]:
            yield from samples

    from luxonis_ml.data.loaders import label_converter
    from luxonis_ml.vizlab import Frame, RenderOptions, set_default_options
    from luxonis_ml.vizlab.viewer import PreparedFrame
    from luxonis_ml.vizlab.viewer import Viewer as RealViewer

    second_frame_ready = Event()
    prepare_threads: list[int] = []
    consumer_thread = get_ident()
    original_prepare = RealViewer.prepare

    def tracked_prepare(
        self: RealViewer,
        frame: Frame,
    ) -> PreparedFrame:
        prepared = original_prepare(self, frame)
        prepare_threads.append(get_ident())
        if len(prepare_threads) == 2:
            second_frame_ready.set()
        return prepared

    class _WaitingBackend(_FakeBackend):
        def __init__(self) -> None:
            super().__init__([ord("x"), ord("q")])
            self._first_poll = True

        def poll_key(self, timeout_ms: int) -> int:
            if self._first_poll:
                self._first_poll = False
                assert second_frame_ready.wait(timeout=5.0)
            return super().poll_key(timeout_ms)

    backend = _WaitingBackend()

    def make_viewer(
        *, hud: bool, save_dir: "str | Path | None" = None
    ) -> RealViewer:
        return RealViewer(backend, hud=hud, save_dir=save_dir)

    def convert_labels(
        labels: Labels,
        *,
        classes: dict[str, dict[str, int]],
        categorical_encodings: dict[str, dict[str, int]],
        render_background: bool,
    ) -> dict[str, DatasetRecord]:
        return {"objects": record}

    monkeypatch.setattr(data_main, "check_exists", _ignore_exists)
    monkeypatch.setattr(data_main, "LuxonisDataset", _Dataset)
    monkeypatch.setattr(data_main, "LuxonisLoader", _Loader)
    monkeypatch.setattr(viewer_module, "Viewer", make_viewer)
    monkeypatch.setattr(RealViewer, "prepare", tracked_prepare)
    monkeypatch.setattr(
        label_converter,
        "loader_output_to_records",
        convert_labels,
    )

    try:
        data_main.inspect(
            "dataset",
            list_augmentations=False,
            plain=True,
            prefetch=1,
        )
    finally:
        set_default_options(RenderOptions())

    assert backend.shown == ["dataset", "dataset"]
    assert len(prepare_threads) == 2
    assert all(thread != consumer_thread for thread in prepare_threads)


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
    assert backend.shown == ["dataset", "dataset"]


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
            self._sample = SimpleNamespace(
                images={"image": image},
                labels={},
                metadata={"filenames": {"image": "frame.jpg"}},
            )

        def __iter__(self) -> Iterator[SimpleNamespace]:
            yield self._sample

        def __getitem__(self, index: int) -> SimpleNamespace:
            if index != 0:
                raise IndexError(index)
            return self._sample

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

    # One window per sample, titled with the ground-truth dataset.
    assert backend.shown == ["ground_truth"]
    # Identical GT and predictions -> a single true positive, no false positives.
    metrics = next(data for title, data in panels if title == "Comparison")
    assert isinstance(metrics, dict)
    assert metrics["TP"] == 1
    assert metrics["FP"] == 0


def test_compare_save_writes_a_clip_without_opening_a_viewer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Saving a comparison must be fully headless: no window, no screen probe.
    import cv2

    backend, _ = _compare_mocks(
        monkeypatch, np.zeros((40, 60, 3), dtype=np.uint8)
    )
    from luxonis_ml.vizlab import RenderOptions, set_default_options

    clip = tmp_path / "verdicts.mp4"
    try:
        data_main.compare("gt", "preds", save=clip, fps=4)
    finally:
        set_default_options(RenderOptions())

    assert backend.shown == []  # nothing was ever presented
    capture = cv2.VideoCapture(str(clip))
    assert capture.read()[0]
    capture.release()


def test_compare_save_clip_drops_the_metrics_panel_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Same rule as inspect: a clip is bare unless --no-plain asks for the panel.
    # The metrics panel is what carries precision/recall, so this is the one
    # place the default costs something -- hence checking both directions.
    _, panels = _compare_mocks(
        monkeypatch, np.zeros((40, 60, 3), dtype=np.uint8)
    )
    from luxonis_ml.vizlab import RenderOptions, set_default_options

    try:
        data_main.compare("gt", "preds", save=tmp_path / "bare.mp4")
        assert panels == []  # no metrics panel was attached at all
        data_main.compare(
            "gt", "preds", save=tmp_path / "full.mp4", plain=False
        )
    finally:
        set_default_options(RenderOptions())

    assert [title for title, _ in panels] == ["Comparison"]
    # No panel and no rounded surround leaves exactly the source width.
    assert _first_clip_frame(tmp_path / "bare.mp4").shape[1] == 60


def test_compare_save_writes_a_directory_of_stills(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A comparison frame is a `Frame`, which has no `save`; the directory form
    # only works because the writer is handed the scene the frame wraps.
    _compare_mocks(monkeypatch, np.zeros((40, 60, 3), dtype=np.uint8))
    from luxonis_ml.vizlab import RenderOptions, set_default_options

    try:
        data_main.compare("gt", "preds", save=tmp_path / "stills")
    finally:
        set_default_options(RenderOptions())

    written = sorted((tmp_path / "stills").iterdir())
    assert [path.name for path in written] == ["0000_image.png"]
    assert written[0].read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_compare_save_html_keeps_the_verdict_tooltips(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The baked-to-pixels path must not lose its hover regions.

    ``compare`` bakes some layouts to an image and reattaches the map with
    `Image.with_hitmap`, so the tooltips no longer live on annotations. The
    page has to resolve them from the scene anyway.
    """
    _compare_mocks(monkeypatch, np.zeros((40, 60, 3), dtype=np.uint8))
    from luxonis_ml.vizlab import RenderOptions, set_default_options

    try:
        data_main.compare(
            "gt", "preds", save=tmp_path / "pages", save_format="html"
        )
    finally:
        set_default_options(RenderOptions())

    written = sorted((tmp_path / "pages").iterdir())
    assert [path.name for path in written] == ["0000_image.html", "index.html"]
    page = (tmp_path / "pages" / "0000_image.html").read_text()
    assert page.lstrip().lower().startswith("<!doctype html")
    assert "data-tip" in page


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
        # The multi-panel frame was presented, in one window.
        assert backend.shown == ["gt"]


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
    assert backend.shown == ["gt"]


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


def test_compare_matches_by_filename_and_reports_unpaired_samples(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:

    image = np.zeros((40, 60, 3), dtype=np.uint8)

    def sample(filename: str, label: str) -> SimpleNamespace:
        return SimpleNamespace(
            images={"image": image},
            labels={"label": label},
            metadata={"filenames": {"image": filename}},
        )

    samples = {
        "gt": [
            sample("a.jpg", "car"),
            sample("b.jpg", "bus"),
            sample("missing.jpg", "car"),
        ],
        "pred": [
            sample("b.jpg", "bus"),
            sample("a.jpg", "car"),
            sample("extra.jpg", "car"),
        ],
    }

    class _Dataset:
        def __init__(self, name: str, **_kwargs: object) -> None:
            self.name = name

        def __len__(self) -> int:
            return len(samples[self.name])

        def get_classes(self) -> dict[str, dict[str, int]]:
            return {"objects": {"car": 0, "bus": 1}}

        def get_class_names(self) -> dict[str, list[str]]:
            return {"objects": ["car", "bus"]}

        def get_task_names(self) -> list[str]:
            return ["objects", "ignored"]

        def get_categorical_encodings(self) -> dict[str, object]:
            return {}

        def get_skeletons(self) -> dict[str, object]:
            return {}

    class _Loader:
        def __init__(self, dataset: _Dataset, **_kwargs: object) -> None:
            self._samples = samples[dataset.name]

        def __iter__(self) -> Iterator[SimpleNamespace]:
            yield from self._samples

        def __getitem__(self, index: int) -> SimpleNamespace:
            return self._samples[index]

    def record(label: str) -> DatasetRecord:
        return DatasetRecord.model_construct(
            files={},
            sample_metadata={},
            annotation=[
                Detection(
                    class_name=label,
                    instance_id=1,
                    boundingbox=BBoxAnnotation(x=0.1, y=0.1, w=0.3, h=0.3),
                )
            ],
            task_name="objects",
        )

    from luxonis_ml.data.loaders import label_converter
    from luxonis_ml.vizlab import Image, RenderOptions, set_default_options
    from luxonis_ml.vizlab.viewer import Viewer as RealViewer

    metrics: list[dict] = []

    def capture_panel(self: Image, data: object, **_kwargs: object) -> Image:
        assert isinstance(data, dict)
        metrics.append(data)
        return self

    monkeypatch.setattr(data_main, "check_exists", lambda *_args: None)
    monkeypatch.setattr(data_main, "LuxonisDataset", _Dataset)
    monkeypatch.setattr(data_main, "LuxonisLoader", _Loader)
    monkeypatch.setattr(
        label_converter,
        "loader_output_to_records",
        lambda labels, **_kwargs: {
            "objects": record(labels["label"]),
            "ignored": record(labels["label"]),
        },
    )
    monkeypatch.setattr(Image, "with_panel", capture_panel)
    monkeypatch.setattr(
        viewer_module,
        "Viewer",
        lambda **_kwargs: RealViewer(_FakeBackend(keys=[ord("x"), ord("q")])),
    )

    try:
        data_main.compare(
            "gt",
            "pred",
            filters=SampleFilterConfig(
                task_name=["ignored"],
                task_name_mode="exclude",
                class_name=["bus"],
                class_name_mode="exclude",
            ),
        )
    finally:
        set_default_options(RenderOptions())

    assert [item["TP"] for item in metrics] == [1]
    assert [item["class errors"] for item in metrics] == [0]
    output = capsys.readouterr().out
    assert "Missing prediction samples (1): image=missing.jpg" in output
    assert "Extra prediction samples (1): image=extra.jpg" in output


def _save_mocks(
    monkeypatch: pytest.MonkeyPatch,
    *,
    sources: Sequence[str] = ("frame01.jpg",),
    labels: "Labels | None" = None,
    detection_metadata: "dict[str, int | float | str] | None" = None,
) -> None:
    """Wire fakes so ``inspect`` runs headless over one 60x40 car sample.

    ``sources`` gives the sample several image sources (a stereo pair, say) and
    ``labels`` supplies raw loader labels such as ``{"stereo/array": ...}``.
    ``detection_metadata`` is opt-in because it is what gives the box a hover
    tooltip, which changes what every other caller renders.
    """
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    record = DatasetRecord.model_construct(
        files={},
        sample_metadata={"weather": "clear"},
        annotation=[
            Detection(
                class_name="car",
                instance_id=1,
                boundingbox=BBoxAnnotation(x=0.1, y=0.1, w=0.3, h=0.3),
                metadata=detection_metadata or {},
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
                images=dict.fromkeys(sources, image),
                labels=labels or {},
                metadata={},
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


@pytest.mark.parametrize("command", ["inspect", "compare"])
def test_save_format_accepts_html_on_the_command_line(command: str) -> None:
    """``--save-format html`` has to survive parsing, not just ``save()``.

    The writer and `Renderable.save` already dispatch on the extension, so a
    direct Python call works whatever the annotation says. Only the parser
    enforces the allowed set, so only a parse exercises the CLI surface.
    """
    args = ["ds"] if command == "inspect" else ["gt", "preds"]
    _, bound, _ = data_main.app.parse_args(
        [command, *args, "--save", "out", "--save-format", "html"],
        exit_on_error=False,
        print_error=False,
    )
    assert bound.arguments["save_format"] == "html"


@pytest.mark.parametrize("command", ["inspect", "compare"])
def test_save_format_still_rejects_an_unknown_format(command: str) -> None:
    args = ["ds"] if command == "inspect" else ["gt", "preds"]
    with pytest.raises(Exception, match=r"(?i)coercion|invalid|choice"):
        data_main.app.parse_args(
            [command, *args, "--save", "out", "--save-format", "webp"],
            exit_on_error=False,
            print_error=False,
        )


def test_inspect_save_writes_a_self_contained_html_page(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from luxonis_ml.vizlab import RenderOptions, set_default_options

    _save_mocks(monkeypatch, detection_metadata={"track": "7"})
    try:
        data_main.inspect("ds", save=tmp_path / "html", save_format="html")
    finally:
        set_default_options(RenderOptions())

    page = (tmp_path / "html" / "0000_frame01.html").read_text()
    assert page.lstrip().lower().startswith("<!doctype html")
    assert "<svg" in page  # the vector render, inlined
    # The point of the format: annotations stay hoverable in the saved file.
    # Assert on the hover layer itself — "data-tip" appearing anywhere in the
    # document is not evidence, as a stylesheet or script mentioning it also
    # satisfies that.
    layer = re.search(r'<g class="vl-hit"[^>]*>(.*?)</g>', page, re.DOTALL)
    assert layer is not None
    assert 'data-tip="0"' in layer.group(1)
    assert "track" in page  # and the card it points at carries the metadata
    # Self-contained: nothing is fetched from the network. Relative links to
    # sibling pages are fine — the directory travels as a unit.
    assert not re.search(r'(?:src|href)="(?:https?:)?//', page)


def test_inspect_save_html_writes_an_index_linking_every_page(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A directory of numbered pages needs one file you can actually open."""
    from luxonis_ml.vizlab import RenderOptions, set_default_options

    _save_mocks(monkeypatch, sources=("left.jpg", "right.jpg"))
    try:
        data_main.inspect("ds", save=tmp_path / "html", save_format="html")
    finally:
        set_default_options(RenderOptions())

    written = sorted(p.name for p in (tmp_path / "html").iterdir())
    index = (tmp_path / "html" / "index.html").read_text()
    pages = [name for name in written if name != "index.html"]
    assert pages  # the renders themselves
    for name in pages:
        assert f'href="{name}"' in index
    assert index.lstrip().lower().startswith("<!doctype html")
    # Self-contained and relative, so the directory can be moved or zipped.
    assert not re.search(r'(?:src|href)="(?:https?:)?//', index)


def test_html_pages_link_to_each_other_and_back_to_the_index(
    tmp_path: Path,
) -> None:
    """Each page carries prev/next/index; the ends stay in place but inert.

    Driven through the writer rather than the CLI because the fixture dataset
    holds one sample, and the linking only exists across several.
    """
    from luxonis_ml.vizlab import DARK_THEME, BBox, Image

    def scene(label: str) -> Image:
        return Image(np.zeros((30, 40, 3), dtype=np.uint8)).add(
            BBox(x=0.1, y=0.1, w=0.3, h=0.3).tag(label)
        )

    names = ["a.jpg", "b.jpg", "c.jpg"]
    data_main._write_renders(
        ((name, scene(name)) for name in names),
        tmp_path / "html",
        image_format="html",
        fps=5.0,
        background=DARK_THEME.background,
        theme=DARK_THEME,
        empty_note="nothing",
    )

    pages = ["0000_a.html", "0001_b.html", "0002_c.html"]
    assert sorted(p.name for p in (tmp_path / "html").iterdir()) == [
        *pages,
        "index.html",
    ]
    first, middle, last = (
        (tmp_path / "html" / name).read_text() for name in pages
    )
    for page in (first, middle, last):
        assert '<a href="index.html">home</a>' in page

    assert '<span class="step">&larr; prev</span>' in first
    assert 'href="0001_b.html" rel="next"' in first

    assert 'href="0000_a.html" rel="prev"' in middle
    assert 'href="0002_c.html" rel="next"' in middle

    assert 'href="0001_b.html" rel="prev"' in last
    assert '<span class="step">next &rarr;</span>' in last


def test_the_writer_holds_only_one_render_ahead(tmp_path: Path) -> None:
    """Linking forward must not mean materializing the whole stream.

    Each render can hold a decoded sample, so the writer pairs each item with
    the next rather than reading them all to learn how many there are.
    """
    alive: list[str] = []

    class _Tracked:
        def __init__(self, name: str) -> None:
            self.name = name
            alive.append(name)

        def render_html(self, **_: object) -> str:
            return f"<!DOCTYPE html><html><body>{self.name}</body></html>"

    from luxonis_ml.vizlab import DARK_THEME

    def stream() -> "Iterator[tuple[str, object]]":
        for index in range(6):
            # Never more than the held item plus the one just produced.
            assert len(alive) <= 2, alive
            yield f"{index}.jpg", _Tracked(f"{index}")
            alive.pop(0)

    data_main._write_renders(
        stream(),  # type: ignore[arg-type]
        tmp_path / "html",
        image_format="html",
        fps=5.0,
        background=DARK_THEME.background,
        theme=DARK_THEME,
        empty_note="nothing",
    )
    assert len(list((tmp_path / "html").iterdir())) == 7  # 6 pages + index


def test_a_single_page_still_gets_its_index_link(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Both ends inert is the correct rendering for a one-render directory."""
    from luxonis_ml.vizlab import RenderOptions, set_default_options

    _save_mocks(monkeypatch)
    try:
        data_main.inspect("ds", save=tmp_path / "html", save_format="html")
    finally:
        set_default_options(RenderOptions())

    page = (tmp_path / "html" / "0000_frame01.html").read_text()
    assert '<a href="index.html">home</a>' in page
    assert '<span class="step">&larr; prev</span>' in page
    assert '<span class="step">next &rarr;</span>' in page


def test_other_formats_get_no_index(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The index is an HTML affordance; a folder of ONGs already opens fine."""
    from luxonis_ml.vizlab import RenderOptions, set_default_options

    _save_mocks(monkeypatch)
    try:
        data_main.inspect("ds", save=tmp_path / "png", save_format="png")
    finally:
        set_default_options(RenderOptions())

    assert not (tmp_path / "png" / "index.html").exists()


def test_inspect_save_html_is_the_same_render_as_svg(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The page must wrap the vector render, not re-rasterize it."""
    from luxonis_ml.vizlab import RenderOptions, set_default_options

    _save_mocks(monkeypatch)
    try:
        data_main.inspect("ds", save=tmp_path / "svg", save_format="svg")
        data_main.inspect("ds", save=tmp_path / "html", save_format="html")
    finally:
        set_default_options(RenderOptions())

    svg = (tmp_path / "svg" / "0000_frame01.svg").read_text()
    page = (tmp_path / "html" / "0000_frame01.html").read_text()
    # Same geometry: whatever viewport the SVG drew at, the page inlines.
    viewbox = re.search(r'viewBox="([^"]+)"', page)
    assert viewbox is not None
    width = re.search(r'width="(\d+)', svg)
    assert width is not None
    assert viewbox.group(1).split()[2] == width.group(1)


def test_inspect_save_writes_a_clip_when_given_a_clip_extension(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The same --save option takes a directory or a single clip; the extension
    # is what tells them apart, so --save-format has nothing to say here.
    import cv2

    from luxonis_ml.vizlab import RenderOptions, set_default_options

    _save_mocks(monkeypatch)
    clip = tmp_path / "preview.mp4"
    try:
        data_main.inspect("ds", save=clip, fps=12)
    finally:
        set_default_options(RenderOptions())

    assert clip.stat().st_size > 0
    capture = cv2.VideoCapture(str(clip))
    assert capture.get(cv2.CAP_PROP_FPS) == 12
    capture.release()
    assert not (tmp_path / "preview").exists()  # no stray directory


def _first_clip_frame(path: Path) -> np.ndarray:
    """Decode a clip's opening frame."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    read, frame = capture.read()
    capture.release()
    assert read
    return frame


def test_inspect_save_clip_drops_the_panel_unless_asked_to_keep_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A clip has one fixed canvas but the panel's width follows each sample's
    # metadata, so --plain is the default there. --no-plain restores it, and
    # the directory form keeps the panel either way.
    import cv2

    from luxonis_ml.vizlab import RenderOptions, set_default_options

    _save_mocks(monkeypatch)
    try:
        data_main.inspect("ds", save=tmp_path / "bare.mp4")
        data_main.inspect("ds", save=tmp_path / "full.mp4", plain=False)
        data_main.inspect("ds", save=tmp_path / "stills")
    finally:
        set_default_options(RenderOptions())

    bare = _first_clip_frame(tmp_path / "bare.mp4")
    full = _first_clip_frame(tmp_path / "full.mp4")
    directory = cv2.imread(str(tmp_path / "stills" / "0000_frame01.png"))
    assert bare.shape[1] == 60  # the source image alone, no panel, no surround
    assert full.shape[1] > bare.shape[1]  # --no-plain put the panel back
    assert directory.shape[1] > bare.shape[1]  # a directory still gets one


def test_inspect_save_writes_an_animation_too(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from PIL import Image as PILImage

    from luxonis_ml.vizlab import RenderOptions, set_default_options

    _save_mocks(monkeypatch)
    clip = tmp_path / "preview.gif"
    try:
        data_main.inspect("ds", save=clip)
    finally:
        set_default_options(RenderOptions())

    with PILImage.open(clip) as animation:
        assert animation.format == "GIF"


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


def test_inspect_plain_drops_the_interactive_panel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from luxonis_ml.vizlab import Frame, RenderOptions, set_default_options
    from luxonis_ml.vizlab.viewer import Viewer as RealViewer

    _save_mocks(monkeypatch)
    panel_calls = 0

    def capture_panel(self: Frame, *_args: object, **_kwargs: object) -> Frame:
        nonlocal panel_calls
        panel_calls += 1
        return self

    monkeypatch.setattr(Frame, "with_panel", capture_panel)
    monkeypatch.setattr(
        viewer_module,
        "Viewer",
        lambda **_kwargs: RealViewer(_FakeBackend(keys=[ord("q")])),
    )
    try:
        data_main.inspect("ds", plain=True)
    finally:
        set_default_options(RenderOptions())

    assert panel_calls == 0


def test_inspect_auto_size_reserves_space_for_controls_panel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from luxonis_ml.vizlab import Frame, RenderOptions, set_default_options
    from luxonis_ml.vizlab.viewer import Viewer as RealViewer

    _save_mocks(monkeypatch)
    image_widths: list[int] = []

    def capture_panel(self: Frame, *_args: object, **_kwargs: object) -> Frame:
        image_widths.append(self.render().shape[1])
        return self

    monkeypatch.setattr(Frame, "with_panel", capture_panel)
    monkeypatch.setattr(
        viewer_module,
        "Viewer",
        lambda **_kwargs: RealViewer(
            _FakeBackend(keys=[ord("q")], screen=(1000, 800))
        ),
    )
    try:
        data_main.inspect("ds")
    finally:
        set_default_options(RenderOptions())

    assert image_widths == [500]


# --- multi-source tiling and array fields -----------------------------------


def _rendered_width(directory: Path) -> int:
    """Width of the single render a headless save wrote."""
    import cv2

    written = sorted(directory.iterdir())
    assert len(written) == 1, [p.name for p in written]
    return cv2.imread(str(written[0])).shape[1]


def _stereo_array() -> "Labels":
    """Build a loader-shaped array label matching the fake 60x40 sample."""
    return {
        "stereo/array": np.linspace(0.0, 10.0, 40 * 60).reshape(1, 1, 40, 60)
    }


def test_inspect_tiles_multiple_sources_into_one_render(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Two sources used to mean two windows -- which the viewer centres on the
    # same screen point -- and two saved files. Now they tile into one.
    from luxonis_ml.vizlab import RenderOptions, set_default_options

    _save_mocks(monkeypatch, sources=("left.jpg", "right.jpg"))
    try:
        data_main.inspect("ds", save=tmp_path / "stereo", plain=True)
    finally:
        set_default_options(RenderOptions())

    assert _rendered_width(tmp_path / "stereo") > 2 * 60  # both, side by side


def test_inspect_ignores_arrays_unless_asked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Most arrays are not pictures, so nothing changes without --array-viz.
    from luxonis_ml.vizlab import RenderOptions, set_default_options

    _save_mocks(monkeypatch, labels=_stereo_array())
    try:
        data_main.inspect("ds", save=tmp_path / "off", plain=True)
    finally:
        set_default_options(RenderOptions())

    assert _rendered_width(tmp_path / "off") == 60  # the lone source, untiled


def test_inspect_array_viz_adds_a_tile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from luxonis_ml.vizlab import RenderOptions, set_default_options

    _save_mocks(monkeypatch, labels=_stereo_array())
    try:
        data_main.inspect(
            "ds", save=tmp_path / "tiled", plain=True, array_viz=True
        )
    finally:
        set_default_options(RenderOptions())

    assert (
        _rendered_width(tmp_path / "tiled") > 60
    )  # the field got its own tile


def test_inspect_array_overlay_keeps_one_tile_and_targets_one_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # An overlay paints onto a source rather than adding a tile, and only onto
    # the reference view -- a disparity map does not describe the other one.
    import cv2

    from luxonis_ml.vizlab import RenderOptions, set_default_options

    _save_mocks(
        monkeypatch, sources=("left.jpg", "right.jpg"), labels=_stereo_array()
    )
    try:
        data_main.inspect(
            "ds",
            save=tmp_path / "over",
            plain=True,
            array_viz=True,
            array_mode="overlay",
        )
    finally:
        set_default_options(RenderOptions())

    written = sorted((tmp_path / "over").iterdir())
    rendered = cv2.imread(str(written[0]))
    # Two source tiles, no third: an overlay does not add one.
    assert rendered.shape[1] < 3 * 60
    left, right = rendered[:, :60], rendered[:, -60:]
    assert not np.array_equal(left, right)  # only the first source was painted


def test_inspect_array_options_require_array_viz(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Silently ignoring a refinement would render a frame that is not what was
    # asked for, so each one is rejected up front.
    _save_mocks(monkeypatch, labels=_stereo_array())
    for kwargs in (
        {"array_vmax": 291.0},
        {"array_ignore": 0.0},
        {"array_mode": "overlay"},
        {"array_colorbar": False},
        {"array_kind": [("stereo", "flow")]},
    ):
        with pytest.raises(ValueError, match="requires --array-viz"):
            data_main.inspect("ds", **kwargs)  # type: ignore[arg-type]


def test_inspect_array_kind_pins_how_a_task_is_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The stereo fixture is a plain 2-D field, which would infer as `scalar`.
    # Pinning it to `signed` has to win, and has to reach the built annotation.
    from luxonis_ml.vizlab import RenderOptions, set_default_options
    from luxonis_ml.vizlab.adapters import arrays as arrays_adapter

    built: list[object] = []
    real = arrays_adapter.array_annotations

    def capture(*args: object, **kwargs: object) -> object:
        result = real(*args, **kwargs)  # type: ignore[arg-type]
        built.extend(drawing.kind for drawing in result)
        return result

    monkeypatch.setattr(arrays_adapter, "array_annotations", capture)
    _save_mocks(monkeypatch, labels=_stereo_array())
    try:
        data_main.inspect(
            "ds",
            save=tmp_path / "pinned",
            plain=True,
            array_viz=True,
            array_kind=[("stereo", "signed")],
        )
    finally:
        set_default_options(RenderOptions())

    assert built == ["signed"]


def test_inspect_array_flags_reach_the_heatmap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The range and sentinel are what make two datasets comparable, so check
    # they arrive rather than only that something was drawn.
    from luxonis_ml.vizlab import RenderOptions, set_default_options
    from luxonis_ml.vizlab.adapters import arrays as arrays_adapter

    built: list[object] = []
    real = arrays_adapter.array_annotations

    def capture(*args: object, **kwargs: object) -> object:
        result = real(*args, **kwargs)  # type: ignore[arg-type]
        built.extend(drawing.field for drawing in result)
        return result

    # The command imports the bridge inside its body, so the patch has to land
    # on the source module rather than on a name in __main__.
    monkeypatch.setattr(arrays_adapter, "array_annotations", capture)
    _save_mocks(monkeypatch, labels=_stereo_array())
    try:
        data_main.inspect(
            "ds",
            save=tmp_path / "pinned",
            plain=True,
            array_viz=True,
            array_vmin=0.0,
            array_vmax=291.0,
            array_ignore=0.0,
            array_gradient="magma",
        )
    finally:
        set_default_options(RenderOptions())

    assert built, "no array field was built"
    field = built[0]
    assert (field.vmin, field.vmax, field.ignore_value) == (0.0, 291.0, 0.0)  # type: ignore[attr-defined]


def test_inspect_rejects_an_unknown_array_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _save_mocks(monkeypatch, labels=_stereo_array())
    with pytest.raises(KeyError, match="unknown gradient"):
        data_main.inspect("ds", array_viz=True, array_gradient="nope")


def test_inspect_syncs_has_arrays_onto_the_live_layer_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # `has_arrays` participates in LayerState equality, so without syncing it
    # from the sample snapshot the live state never compares equal and every
    # frame is re-rendered instead of using the prefetched one. That shows up
    # as "prefetch silently stopped helping", not as an error -- hence the test.
    from luxonis_ml.vizlab.viewer import Viewer as RealViewer

    created: list[RealViewer] = []
    backend = _FakeBackend(keys=[ord("q")])

    def make_viewer(**_k: object) -> RealViewer:
        viewer = RealViewer(backend)
        created.append(viewer)
        return viewer

    monkeypatch.setattr(viewer_module, "Viewer", make_viewer)
    _save_mocks(monkeypatch, labels=_stereo_array())

    from luxonis_ml.vizlab import RenderOptions, set_default_options

    try:
        data_main.inspect("ds", array_viz=True)
    finally:
        set_default_options(RenderOptions())

    layers = created[0].layers
    assert layers.has_arrays is True
    assert "a" in {control.key for control in layers.controls()}
