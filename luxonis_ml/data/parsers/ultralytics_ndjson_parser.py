import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np
from loguru import logger

from luxonis_ml.data import BaseDataset, DatasetIterator
from luxonis_ml.data.utils.remote_file_downloader import (
    download_remote_file,
)

from .base_parser import BaseParser, ParserOutput


class UltralyticsNDJSONParser(BaseParser):
    """Parses Ultralytics NDJSON datasets into LDF."""

    @staticmethod
    def _resolve_ndjson_path(path: Path) -> Path | None:
        path = path.resolve()
        if path.is_file() and path.suffix.lower() == ".ndjson":
            return path
        if path.is_dir():
            matches = sorted(path.glob("*.ndjson"))
            if len(matches) == 1:
                return matches[0].resolve()
        return None

    @classmethod
    def _load_header(cls, path: Path) -> dict[str, Any] | None:
        ndjson_path = cls._resolve_ndjson_path(path)
        if ndjson_path is None:
            return None

        dataset_record = None
        has_image_record = False
        try:
            with open(ndjson_path, encoding="utf-8-sig") as file:
                for raw_line in file:
                    line = raw_line.strip()
                    if not line:
                        continue

                    record = json.loads(line)
                    if dataset_record is None:
                        dataset_record = record
                        continue

                    if record.get("type") == "image":
                        has_image_record = True
                        break
        except (OSError, json.JSONDecodeError):
            return None

        if (
            dataset_record is None
            or dataset_record.get("type") != "dataset"
            or "class_names" not in dataset_record
            or not has_image_record
        ):
            return None

        return dataset_record

    @staticmethod
    def _normalize_split_name(split_name: str | None) -> str:
        if split_name in {"train", "val", "test"}:
            return split_name
        if split_name in {"valid", "validation"}:
            return "val"
        if split_name is None:
            logger.warning(
                "Missing split in Ultralytics NDJSON record. Defaulting to 'train'."
            )
            return "train"

        logger.warning(
            f"Unknown split '{split_name}' in Ultralytics NDJSON record. "
            "Defaulting to 'train'."
        )
        return "train"

    @staticmethod
    def _get_class_names(
        class_names: list[str] | dict[str, str],
    ) -> dict[int, str]:
        if isinstance(class_names, list):
            return dict(enumerate(class_names))
        return {int(k): v for k, v in class_names.items()}

    @staticmethod
    def _fit_boundingbox(points: np.ndarray) -> dict[str, float]:
        x_min = float(np.min(points[:, 0]))
        y_min = float(np.min(points[:, 1]))
        x_max = float(np.max(points[:, 0]))
        y_max = float(np.max(points[:, 1]))
        return {
            "x": x_min,
            "y": y_min,
            "w": x_max - x_min,
            "h": y_max - y_min,
        }

    @classmethod
    def _download_image(
        cls, ndjson_path: Path, record: dict[str, Any]
    ) -> Path:
        file_name = Path(record["file"])
        url = record["url"]
        split_name = cls._normalize_split_name(record.get("split"))
        # Keep cached filenames deterministic while avoiding collisions
        # when different URLs share the same basename.
        url_hash = hashlib.blake2s(
            url.encode("utf-8"), digest_size=6
        ).hexdigest()
        suffix = file_name.suffix or Path(urlsplit(url).path).suffix
        destination = (
            ndjson_path.parent
            / f".{ndjson_path.stem}_cache"
            / split_name
            / f"{file_name.stem}-{url_hash}{suffix}"
        )
        return download_remote_file(url, destination, validate_image=True)

    @classmethod
    def _resolve_image_path(
        cls, ndjson_path: Path, record: dict[str, Any]
    ) -> Path:
        if record.get("url"):
            return cls._download_image(ndjson_path, record)

        file_path = Path(record["file"])
        if file_path.is_absolute():
            return file_path.resolve()
        return (ndjson_path.parent / file_path).resolve()

    def _build_record_stream(
        self, ndjson_path: Path
    ) -> tuple[DatasetIterator, dict[str, list[Path]], list[Path]]:
        header = self._load_header(ndjson_path)
        if header is None:
            raise ValueError(
                f"Invalid Ultralytics NDJSON dataset file: '{ndjson_path}'."
            )

        class_names = self._get_class_names(header["class_names"])
        kpt_shape = header.get("kpt_shape")
        added_by_split = {"train": [], "val": [], "test": []}
        seen_by_split = {"train": set(), "val": set(), "test": set()}
        added_images: list[Path] = []
        seen_images: set[Path] = set()

        def generator() -> DatasetIterator:
            with open(ndjson_path, encoding="utf-8-sig") as file:
                for raw_line in file:
                    line = raw_line.strip()
                    if not line:
                        continue

                    record = json.loads(line)
                    if record.get("type") != "image":
                        continue

                    image_path = self._resolve_image_path(ndjson_path, record)
                    split_name = self._normalize_split_name(
                        record.get("split")
                    )

                    if image_path not in seen_images:
                        seen_images.add(image_path)
                        added_images.append(image_path)
                    if image_path not in seen_by_split[split_name]:
                        seen_by_split[split_name].add(image_path)
                        added_by_split[split_name].append(image_path)

                    annotations = record.get("annotations") or {}
                    instance_id = 0
                    yielded_annotation = False

                    for box in annotations.get("boxes", []):
                        class_id, x_center, y_center, width, height = box
                        yielded_annotation = True
                        yield {
                            "file": str(image_path),
                            "annotation": {
                                "class": class_names[int(class_id)],
                                "instance_id": instance_id,
                                "boundingbox": {
                                    "x": float(x_center) - float(width) / 2,
                                    "y": float(y_center) - float(height) / 2,
                                    "w": float(width),
                                    "h": float(height),
                                },
                            },
                        }
                        instance_id += 1

                    for segment in annotations.get("segments", []):
                        class_id, *points = segment
                        points_array = np.array(points, dtype=float).reshape(
                            -1, 2
                        )
                        yielded_annotation = True
                        yield {
                            "file": str(image_path),
                            "annotation": {
                                "class": class_names[int(class_id)],
                                "instance_id": instance_id,
                                "boundingbox": self._fit_boundingbox(
                                    points_array
                                ),
                                "instance_segmentation": {
                                    "height": int(record["height"]),
                                    "width": int(record["width"]),
                                    "points": [
                                        (float(x), float(y))
                                        for x, y in points_array.tolist()
                                    ],
                                },
                            },
                        }
                        instance_id += 1

                    for pose in annotations.get("pose", []):
                        (
                            class_id,
                            x_center,
                            y_center,
                            width,
                            height,
                            *keypoints,
                        ) = pose
                        if kpt_shape is None:
                            if len(keypoints) % 3 != 0:
                                raise ValueError(
                                    "Ultralytics NDJSON pose annotations require "
                                    "`kpt_shape` in the dataset header when the "
                                    "keypoint dimensionality is not inferable."
                                )
                            n_kpts = len(keypoints) // 3
                            kpt_dim = 3
                        else:
                            n_kpts, kpt_dim = kpt_shape

                        keypoints_array = np.array(
                            keypoints, dtype=float
                        ).reshape(n_kpts, kpt_dim)
                        if kpt_dim == 2:
                            keypoints_array = np.concatenate(
                                [
                                    keypoints_array,
                                    np.ones((n_kpts, 1), dtype=float) * 2,
                                ],
                                axis=1,
                            )

                        yielded_annotation = True
                        yield {
                            "file": str(image_path),
                            "annotation": {
                                "class": class_names[int(class_id)],
                                "instance_id": instance_id,
                                "boundingbox": {
                                    "x": float(x_center) - float(width) / 2,
                                    "y": float(y_center) - float(height) / 2,
                                    "w": float(width),
                                    "h": float(height),
                                },
                                "keypoints": {
                                    "keypoints": [
                                        (float(x), float(y), int(v))
                                        for x, y, v in keypoints_array.tolist()
                                    ]
                                },
                            },
                        }
                        instance_id += 1

                    if not yielded_annotation:
                        yield {"file": str(image_path), "annotation": None}

        return generator(), added_by_split, added_images

    @staticmethod
    def validate_split(split_path: Path) -> dict[str, Any] | None:
        ndjson_path = UltralyticsNDJSONParser._resolve_ndjson_path(split_path)
        if ndjson_path is None:
            return None
        if UltralyticsNDJSONParser._load_header(ndjson_path) is None:
            return None
        return {"ndjson_path": ndjson_path}

    @classmethod
    def validate(cls, dataset_dir: Path) -> bool:
        return cls._load_header(dataset_dir) is not None

    def from_dir(
        self, dataset_dir: Path
    ) -> tuple[list[Path], list[Path], list[Path]]:
        ndjson_path = self._resolve_ndjson_path(dataset_dir)
        if ndjson_path is None:
            raise ValueError(
                f"Ultralytics NDJSON dataset file not found in '{dataset_dir}'."
            )

        generator, added_by_split, _added_images = self._build_record_stream(
            ndjson_path
        )
        self.dataset.add(self._wrap_generator(generator))
        return (
            added_by_split["train"],
            added_by_split["val"],
            added_by_split["test"],
        )

    def parse_dir(self, dataset_dir: Path, **kwargs) -> BaseDataset:
        split_ratios = kwargs.pop("split_ratios", None)
        is_counts = split_ratios is not None and all(
            isinstance(v, int) for v in split_ratios.values()
        )

        train, val, test = self.from_dir(dataset_dir, **kwargs)
        original_splits = {
            "train": train,
            "val": val,
            "test": test,
        }

        if split_ratios is None:
            self.dataset.make_splits(original_splits)
        elif is_counts:
            sampled = self._apply_counts_to_splits(
                original_splits,
                split_ratios,  # type: ignore[arg-type]
            )
            self.dataset.make_splits(sampled)
            self._remove_unsplit_records()
        else:
            logger.warning(
                "Using percentage-based split ratios will redistribute "
                "and shuffle all samples across splits. Original split "
                "boundaries from the NDJSON file will not be preserved."
            )
            self.dataset.make_splits(split_ratios)

        return self.dataset

    def from_split(self, ndjson_path: Path) -> ParserOutput:
        generator, _added_by_split, added_images = self._build_record_stream(
            ndjson_path
        )
        return generator, {}, added_images

    def parse_split(
        self,
        split: str | None = None,
        random_split: bool = True,
        split_ratios: dict[str, float | int] | None = None,
        **kwargs,
    ) -> BaseDataset:
        ndjson_path = kwargs.get("ndjson_path")
        if ndjson_path is None:
            raise ValueError("`ndjson_path` is required for NDJSON parsing.")

        generator, added_by_split, added_images = self._build_record_stream(
            ndjson_path
        )
        self.dataset.add(self._wrap_generator(generator))

        is_counts = split_ratios is not None and all(
            isinstance(v, int) for v in split_ratios.values()
        )

        if split is not None:
            self.dataset.make_splits({split: added_images})
        elif split_ratios is None:
            self.dataset.make_splits(added_by_split)
        elif is_counts:
            sampled = self._apply_counts_to_splits(
                added_by_split,
                split_ratios,  # type: ignore[arg-type]
            )
            self.dataset.make_splits(sampled)
            self._remove_unsplit_records()
        elif random_split:
            logger.warning(
                "Using percentage-based split ratios will redistribute "
                "and shuffle all samples across splits. Original split "
                "boundaries from the NDJSON file will not be preserved."
            )
            self.dataset.make_splits(split_ratios)

        return self.dataset
