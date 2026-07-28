import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from luxonis_ml.data.utils.augmentations_collector import (
    AugmentationsCollector,
)


def _make_transform(
    name: str,
    params: dict[str, Any] | None = None,
    transforms: list[Any] | None = None,
) -> Any:
    namespace: dict[str, Any] = {}
    if params is not None:
        namespace["params"] = params
    if transforms is not None:
        namespace["transforms"] = transforms
    return type(name, (), namespace)()


def _wrap_transform(transform: Any) -> Callable[[], Any]:
    def wrapped_transform() -> Any:
        return transform

    return wrapped_transform


class _DummyAugmentations:
    def __init__(self):
        self.mixup = _make_transform("MixUp", {"stale": True})
        self.spatial_flip = _make_transform("HorizontalFlip", {"stale": True})
        self.pixel_flip = _make_transform("HorizontalFlip", {"stale": True})
        self.custom_only = _make_transform("CustomOnly", {"stale": True})
        self.resize_crop = _make_transform(
            "AtLeastOneBBoxRandomCrop", {"stale": True}
        )
        self.unapplied = _make_transform("Unapplied", {"stale": True})

        self._batch_transform = _make_transform(
            "BatchCompose", transforms=[self.mixup]
        )
        spatial_transform = _make_transform(
            "Compose", transforms=[self.spatial_flip, self.unapplied]
        )
        custom_transform = self.custom_only
        pixel_transform = _make_transform(
            "Compose", transforms=[self.pixel_flip]
        )
        resize_transform = _make_transform(
            "OneOf", transforms=[self.resize_crop]
        )

        self._spatial_transform = _wrap_transform(spatial_transform)
        self._custom_transform = _wrap_transform(custom_transform)
        self._pixel_transform = _wrap_transform(pixel_transform)
        self._resize_transform = _wrap_transform(resize_transform)
        self.input_batch = None
        self.output = object()

    def apply(self, input_batch: list[Any]) -> object:
        self.input_batch = input_batch
        self.mixup.params = {"applied": True}
        self.spatial_flip.params = {"applied": True}
        self.pixel_flip.params = {"applied": True}
        self.custom_only.params = {"applied": True}
        self.resize_crop.params = {"applied": True}
        return self.output


def test_collect_applied_transform_paths_collects_nested_paths():
    rotate = type("Rotate", (), {"params": {"shape": (32, 32, 3)}})()
    blur = type("Blur", (), {"params": {}})()
    inner_one_of = type(
        "OneOf",
        (),
        {"transforms": [blur, rotate]},
    )()
    horizontal_flip = type(
        "HorizontalFlip", (), {"params": {"shape": (32, 32, 3)}}
    )()
    root_compose = type(
        "Compose",
        (),
        {"transforms": [horizontal_flip, inner_one_of]},
    )()

    assert AugmentationsCollector.collect_applied_transform_paths(
        root_compose
    ) == [
        "HorizontalFlip",
        "OneOf/Rotate",
    ]


def test_flatten_config_augmentation_paths_handles_nested_transforms():
    config = [
        {"name": "HorizontalFlip", "params": {"p": 0.5}},
        {
            "name": "OneOf",
            "params": {
                "transforms": [
                    {"name": "Blur", "params": {"p": 1.0}},
                    {"name": "Sharpen", "params": {"p": 1.0}},
                ]
            },
        },
    ]

    assert AugmentationsCollector.flatten_config_augmentation_paths(
        config
    ) == [
        "HorizontalFlip",
        "OneOf",
        "OneOf/Blur",
        "OneOf/Sharpen",
    ]


def test_flatten_config_augmentation_paths_handles_deeply_nested_transforms():
    config = [
        {
            "name": "OneOf",
            "params": {
                "transforms": [
                    {
                        "name": "OneOf",
                        "params": {
                            "transforms": [
                                {"name": "Rotate", "params": {"p": 1.0}}
                            ]
                        },
                    }
                ]
            },
        }
    ]

    assert AugmentationsCollector.flatten_config_augmentation_paths(
        config
    ) == [
        "OneOf",
        "OneOf/OneOf",
        "OneOf/OneOf/Rotate",
    ]


def test_flatten_config_augmentation_paths_adds_resize_oneof_alias():
    config = [
        {
            "name": "AtLeastOneBBoxRandomCrop",
            "params": {
                "height": 32,
                "width": 32,
                "erosion_factor": 0.0,
                "p": 0.3,
            },
            "use_for_resizing": True,
        }
    ]

    assert AugmentationsCollector.flatten_config_augmentation_paths(
        config
    ) == [
        "AtLeastOneBBoxRandomCrop",
        "OneOf/AtLeastOneBBoxRandomCrop",
    ]


def test_load_augmentation_paths_accepts_in_memory_config():
    config = [
        {"name": "HorizontalFlip", "params": {"p": 0.5}},
        {
            "name": "OneOf",
            "params": {
                "transforms": [
                    {"name": "Blur", "params": {"p": 1.0}},
                ]
            },
        },
    ]

    assert AugmentationsCollector.load_augmentation_paths(config) == [
        "HorizontalFlip",
        "OneOf",
        "OneOf/Blur",
    ]


def test_collector_instruments_apply_and_tracks_unique_configured_paths():
    augmentations = _DummyAugmentations()
    collector = AugmentationsCollector(
        cast(Any, augmentations),
        [
            {"name": "MixUp"},
            {"name": "HorizontalFlip"},
            {"name": "CustomOnly", "params": {"p": 1.0}},
            {
                "name": "AtLeastOneBBoxRandomCrop",
                "params": {"p": 0.5},
                "use_for_resizing": True,
            },
        ],
    )

    applied = collector.get_applied_augmentations()
    applied.append("mutated")
    assert collector.get_applied_augmentations() == []

    input_batch = [object()]
    assert augmentations.apply(input_batch) is augmentations.output
    assert augmentations.input_batch == input_batch
    assert collector.get_applied_augmentations() == [
        "MixUp",
        "HorizontalFlip",
        "CustomOnly",
        "OneOf/AtLeastOneBBoxRandomCrop",
    ]
    assert augmentations.unapplied.params == {}


def test_load_augmentation_paths_accepts_json_and_yaml_files(
    tmp_path: Path,
):
    config = [
        {"name": "HorizontalFlip", "params": {"p": 0.5}},
        {
            "name": "OneOf",
            "params": {"transforms": [{"name": "Blur"}]},
        },
    ]
    json_path = tmp_path / "augmentations.json"
    json_path.write_text(json.dumps(config))
    yaml_path = tmp_path / "augmentations.yaml"
    yaml_path.write_text(
        """
        - name: HorizontalFlip
          params:
            p: 0.5
        - name: OneOf
          params:
            transforms:
              - name: Blur
        """
    )

    assert AugmentationsCollector.load_augmentation_paths(json_path) == [
        "HorizontalFlip",
        "OneOf",
        "OneOf/Blur",
    ]
    assert AugmentationsCollector.load_augmentation_paths(yaml_path) == [
        "HorizontalFlip",
        "OneOf",
        "OneOf/Blur",
    ]


def test_flatten_config_augmentation_paths_skips_invalid_items():
    config = [
        {"params": {"p": 1.0}},
        {"name": "InvalidParams", "params": "not a dict"},
        {
            "name": "ResizeInvalidParams",
            "params": "not a dict",
            "use_for_resizing": True,
        },
        {
            "name": "ResizeBoolProbability",
            "params": {"p": True},
            "use_for_resizing": True,
        },
        {
            "name": "ResizeStringProbability",
            "params": {"p": "often"},
            "use_for_resizing": True,
        },
    ]

    assert AugmentationsCollector.flatten_config_augmentation_paths(
        config
    ) == [
        "InvalidParams",
        "ResizeInvalidParams",
        "ResizeBoolProbability",
        "ResizeStringProbability",
    ]


def test_collect_applied_transform_paths_ignores_lambda_transform():
    lambda_transform = type("Lambda", (), {"params": {"shape": (32, 32, 3)}})()

    assert (
        AugmentationsCollector.collect_applied_transform_paths(
            lambda_transform
        )
        == []
    )
