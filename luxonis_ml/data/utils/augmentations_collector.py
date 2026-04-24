import inspect as pyinspect
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, cast

import yaml


class AugmentationsLike(Protocol):
    apply: Callable[..., object]
    batch_transform: Any
    spatial_transform: Callable[..., object]
    custom_transform: Callable[..., object]
    pixel_transform: Callable[..., object]
    resize_transform: Callable[..., object]


class AugmentationsCollector:
    def __init__(self, augmentations: object, aug_config_path: Path):
        self.augmentations = cast(AugmentationsLike, augmentations)
        self.configured_paths = set(
            self.load_augmentation_paths(aug_config_path)
        )
        self._applied_augmentations: list[str] = []
        self._original_apply = self.augmentations.apply
        self._tracked_transforms = self.get_tracked_transforms(
            self.augmentations
        )
        self._instrument()

    def get_applied_augmentations(self) -> list[str]:
        return self._applied_augmentations.copy()

    def _instrument(self) -> None:
        def capture_apply(input_batch: object) -> object:
            self._applied_augmentations.clear()
            for transform in self._tracked_transforms:
                self.reset_transform_params(transform)

            output = self._original_apply(input_batch)
            seen_paths: set[str] = set()
            for transform in self._tracked_transforms:
                for path in self.collect_applied_transform_paths(transform):
                    if (
                        path in self.configured_paths
                        and path not in seen_paths
                    ):
                        self._applied_augmentations.append(path)
                        seen_paths.add(path)
            return output

        self.augmentations.apply = capture_apply

    @staticmethod
    def load_augmentation_paths(aug_config_path: Path) -> list[str]:
        with open(aug_config_path) as file:
            if aug_config_path.suffix.lower() == ".json":
                config = json.load(file) or []
            else:
                config = yaml.safe_load(file) or []
        return AugmentationsCollector.flatten_config_augmentation_paths(config)

    @staticmethod
    def flatten_config_augmentation_paths(
        config: list[dict], parent_path: tuple[str, ...] = ()
    ) -> list[str]:
        paths: list[str] = []
        for item in config:
            name = item.get("name")
            if not isinstance(name, str):
                continue

            current_path = (*parent_path, name)
            paths.append("/".join(current_path))
            if AugmentationsCollector._is_probabilistic_resize_transform(item):
                paths.append("/".join((*parent_path, "OneOf", name)))

            params = item.get("params", {})
            if not isinstance(params, dict):
                continue

            nested_transforms = params.get("transforms")
            if not isinstance(nested_transforms, list):
                continue

            nested_items = [
                nested_item
                for nested_item in nested_transforms
                if isinstance(nested_item, dict)
            ]
            paths.extend(
                AugmentationsCollector.flatten_config_augmentation_paths(
                    nested_items, current_path
                )
            )
        return paths

    @staticmethod
    def _is_probabilistic_resize_transform(item: dict[str, Any]) -> bool:
        if not item.get("use_for_resizing", False):
            return False

        params = item.get("params", {})
        if not isinstance(params, dict):
            return False

        probability = params.get("p", 1.0)
        if isinstance(probability, bool) or not isinstance(
            probability, (int, float)
        ):
            return False

        return float(probability) < 1.0

    @staticmethod
    def get_tracked_transforms(augmentations: AugmentationsLike) -> list[Any]:
        return [
            augmentations.batch_transform,
            AugmentationsCollector.get_wrapped_transform(
                augmentations.spatial_transform
            ),
            AugmentationsCollector.get_wrapped_transform(
                augmentations.custom_transform
            ),
            AugmentationsCollector.get_wrapped_transform(
                augmentations.pixel_transform
            ),
            AugmentationsCollector.get_wrapped_transform(
                augmentations.resize_transform
            ),
        ]

    @staticmethod
    def get_wrapped_transform(fn: Callable[..., object]) -> Any:
        closure_vars = pyinspect.getclosurevars(fn)
        return closure_vars.nonlocals["transform"]

    @staticmethod
    def reset_transform_params(transform: Any) -> None:
        if hasattr(transform, "params"):
            transform.params = {}
        for child in getattr(transform, "transforms", []):
            AugmentationsCollector.reset_transform_params(child)

    @staticmethod
    def collect_applied_transform_paths(
        transform: Any, parent_path: tuple[str, ...] = ()
    ) -> list[str]:
        transform_name = type(transform).__name__
        child_transforms = getattr(transform, "transforms", None)

        if child_transforms is not None:
            next_path = (
                parent_path
                if transform_name in {"Compose", "BatchCompose"}
                else (*parent_path, transform_name)
            )
            paths: list[str] = []
            for child in child_transforms:
                paths.extend(
                    AugmentationsCollector.collect_applied_transform_paths(
                        child, next_path
                    )
                )
            return paths

        if transform_name == "Lambda":
            return []

        params = getattr(transform, "params", {})
        if not params:
            return []

        return ["/".join((*parent_path, transform_name))]
