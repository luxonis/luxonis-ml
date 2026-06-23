from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from luxonis_ml.data import __main__ as data_cli
from luxonis_ml.data.utils.enums import BucketStorage


def test_inspect_displays_loader_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata: dict[str, object] = {
        "weather": "rain",
        "filepaths": {"image": "dataset/img_0.jpg"},
    }
    loader_kwargs: dict[str, Any] = {}
    visualized_metadata: list[dict[str, object] | None] = []

    class FakeDataset:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __len__(self) -> int:
            return 1

        def get_classes(self) -> dict[str, dict[str, int]]:
            return {}

        def get_categorical_encodings(self) -> dict[str, dict[str, int]]:
            return {}

    class FakeLoader:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            loader_kwargs.update(kwargs)

        def __iter__(
            self,
        ) -> Iterator[
            tuple[np.ndarray, dict[str, np.ndarray], dict[str, object]]
        ]:
            image = np.zeros((4, 4, 3), dtype=np.uint8)
            yield image, {}, metadata

    def fake_visualize(
        image: np.ndarray,
        source_name: str,
        annotations: dict[str, np.ndarray],
        classes: dict[str, dict[str, int]],
        blend_all: bool = False,
        categorical_encodings: dict[str, dict[str, int]] | None = None,
        metadata: dict[str, object] | None = None,
    ) -> np.ndarray:
        visualized_metadata.append(metadata)
        return image

    monkeypatch.setattr(data_cli, "check_exists", lambda *args, **kwargs: None)
    monkeypatch.setattr(data_cli, "LuxonisDataset", FakeDataset)
    monkeypatch.setattr(data_cli, "LuxonisLoader", FakeLoader)
    monkeypatch.setattr(data_cli, "visualize", fake_visualize)
    monkeypatch.setattr(data_cli.cv2, "namedWindow", lambda *args: None)
    monkeypatch.setattr(data_cli.cv2, "resizeWindow", lambda *args: None)
    monkeypatch.setattr(data_cli.cv2, "imshow", lambda *args: None)
    monkeypatch.setattr(data_cli.cv2, "waitKey", lambda: ord("q"))

    data_cli.inspect("dataset", bucket_storage=BucketStorage.LOCAL)

    assert loader_kwargs["add_filepaths_to_metadata"] is True
    assert visualized_metadata == [metadata]
