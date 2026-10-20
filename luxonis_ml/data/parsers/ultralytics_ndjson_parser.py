import hashlib
import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np
from loguru import logger
from typing_extensions import override

from luxonis_ml.data import BaseDataset, DatasetIterator
from luxonis_ml.data.utils.enums import ParserIssue
from luxonis_ml.data.utils.remote_file_downloader import RemoteFileDownloader
from luxonis_ml.typing import PathType
from luxonis_ml.utils.path import (
    parse_manifest_path,
    resolve_manifest_path,
)

from .base_parser import BaseParser, ParserOutput, yolo_box_to_ldf


class UltralyticsNDJSONParser(BaseParser):
    """Parse Ultralytics NDJSON datasets into LDF.

    NDJSON records may carry their own split names. When no split is
    present on an image record, the image is assigned to ``"train"``.
    ``"valid"`` and ``"validation"`` are normalized to ``"val"``.

    An image record locates its image through two fields:

        - ``file`` names a local image. A relative path resolves against
          the directory that holds the ``.ndjson`` manifest. An absolute
          path is used as it is.
        - ``url`` names a remote image. The parser downloads it, and
          ``file`` then supplies only the cached file name. The parser
          appends a hash of the URL to that name and writes the image to
          a per-split cache directory.

    A record that holds a ``url`` therefore never reads a local ``file``
    path. One manifest can describe a fully remote dataset.
    """

    _remote_file_downloader = RemoteFileDownloader()

    @staticmethod
    @override
    def validate_split(split_path: Path) -> dict[str, Any] | None:
        ndjson_path = UltralyticsNDJSONParser._resolve_ndjson_path(split_path)
        if ndjson_path is None:
            return None
        if UltralyticsNDJSONParser._load_header(ndjson_path) is None:
            return None
        return {"ndjson_path": ndjson_path}

    @classmethod
    @override
    def validate(cls, dataset_dir: Path) -> bool:
        return cls._load_header(dataset_dir) is not None

    @override
    def from_dir(
        self, dataset_dir: Path, reuse_cached: bool = True, **kwargs
    ) -> tuple[list[Path], list[Path], list[Path]]:
        """Parse an Ultralytics NDJSON file into dataset records.

        Args:
            dataset_dir: Directory containing exactly one ``.ndjson`` file,
                or a direct path to an ``.ndjson`` file.
            reuse_cached: Whether to reuse cached remote images if they already exist.
            **kwargs: Parser-specific arguments.

        Returns:
            Added images for the train, validation, and test splits.

        Raises:
            ValueError: If no NDJSON dataset file can be resolved, the
                resolved file has an invalid header, a remote image
                download directory already exists, or pose annotations
                cannot infer keypoint dimensionality.

        """
        ndjson_path = self._resolve_ndjson_path(dataset_dir)
        if ndjson_path is None:
            raise ValueError(
                f"Ultralytics NDJSON dataset file not found in '{dataset_dir}'."
            )

        generator, added_by_split, _added_images = self._build_record_stream(
            ndjson_path, reuse_cached=reuse_cached
        )
        self._dataset.add(self._wrap_generator(generator))
        return (
            added_by_split["train"],
            added_by_split["val"],
            added_by_split["test"],
        )

    @override
    def parse_dir(
        self,
        dataset_dir: Path,
        reuse_cached: bool = True,
        **kwargs,
    ) -> BaseDataset:
        """Parse a full NDJSON dataset and preserve record-level splits.

        Args:
            dataset_dir: Directory containing exactly one ``.ndjson`` file,
                or a direct path to an ``.ndjson`` file.
            reuse_cached: Whether to reuse cached remote images if they already exist.
            kwargs: Parser-specific arguments. ``split_ratios`` may be
                supplied to resample split assignments.

        Returns:
            Dataset with parsed images and annotations.

        Raises:
            ValueError: If the NDJSON file cannot be resolved or parsed as
                a valid Ultralytics dataset.

        """
        self._reset_parser_issue_messages()
        split_ratios = kwargs.pop("split_ratios", None)
        is_counts = split_ratios is not None and all(
            isinstance(v, int) for v in split_ratios.values()
        )

        train, val, test = self.from_dir(
            dataset_dir,
            reuse_cached=reuse_cached,
            **kwargs,
        )
        original_splits: dict[str, Sequence[PathType]] = {
            "train": train,
            "val": val,
            "test": test,
        }

        if split_ratios is None:
            self._dataset.make_splits(original_splits)
        elif is_counts:
            sampled = self._apply_counts_to_splits(
                original_splits,
                split_ratios,  # type: ignore[arg-type]
            )
            self._dataset.make_splits(sampled)
            self._remove_unsplit_records()
        else:
            logger.warning(
                "Using percentage-based split ratios will redistribute "
                "and shuffle all samples across splits. Original split "
                "boundaries from the NDJSON file will not be preserved."
            )
            self._dataset.make_splits(split_ratios)

        return self._dataset

    @override
    def from_split(
        self,
        ndjson_path: Path,
        reuse_cached: bool = True,
    ) -> ParserOutput:
        """Parse a single Ultralytics NDJSON file.

        Args:
            ndjson_path: Path to an Ultralytics NDJSON file.
            reuse_cached: Whether to reuse cached remote images if they already exist.

        Returns:
            Parser output containing annotation records, empty skeleton
            metadata, and added images.

        Raises:
            ValueError: If the file has an invalid header, a remote image
                download directory already exists, or pose annotations
                cannot infer keypoint dimensionality.

        """
        generator, _added_by_split, added_images = self._build_record_stream(
            ndjson_path, reuse_cached=reuse_cached
        )
        return generator, {}, added_images

    @override
    def parse_split(
        self,
        split: str | None = None,
        random_split: bool = True,
        split_ratios: dict[str, float | int] | None = None,
        reuse_cached: bool = True,
        **kwargs,
    ) -> BaseDataset:
        """Parse an NDJSON file that represents a single parser input.

        Args:
            split: Optional split name to assign to all parsed images.
            random_split: Whether percentage ``split_ratios`` should
                resample all images.
            split_ratios: Optional ratios or counts. Float values are
                treated as ratios; integer values are treated as counts.
            reuse_cached: Whether to reuse cached remote images if they already exist.
            kwargs: Parser-specific arguments. Must include
                ``ndjson_path``.

        Returns:
            Dataset with parsed images and annotations.

        Raises:
            ValueError: If ``ndjson_path`` is missing, the file has an
                invalid header, a remote image download directory already
                exists, or pose annotations cannot infer keypoint
                dimensionality.

        """
        self._reset_parser_issue_messages()
        ndjson_path = kwargs.get("ndjson_path")
        if ndjson_path is None:
            raise ValueError("`ndjson_path` is required for NDJSON parsing.")

        generator, added_by_split, added_images = self._build_record_stream(
            ndjson_path, reuse_cached=reuse_cached
        )
        self._dataset.add(self._wrap_generator(generator))
        split_definitions: dict[str, Sequence[PathType]] = dict(added_by_split)

        is_counts = split_ratios is not None and all(
            isinstance(v, int) for v in split_ratios.values()
        )

        if split is not None:
            self._dataset.make_splits({split: added_images})
        elif split_ratios is None:
            self._dataset.make_splits(split_definitions)
        elif is_counts:
            sampled = self._apply_counts_to_splits(
                split_definitions,
                split_ratios,  # type: ignore[arg-type]
            )
            self._dataset.make_splits(sampled)
            self._remove_unsplit_records()
        elif random_split:
            logger.warning(
                "Using percentage-based split ratios will redistribute "
                "and shuffle all samples across splits. Original split "
                "boundaries from the NDJSON file will not be preserved."
            )
            self._dataset.make_splits(split_ratios)

        return self._dataset

    def _build_record_stream(
        self, ndjson_path: Path, reuse_cached: bool = True
    ) -> tuple[DatasetIterator, dict[str, list[Path]], list[Path]]:
        header = self._load_header(ndjson_path)
        if header is None:
            raise ValueError(
                f"Invalid Ultralytics NDJSON dataset file: '{ndjson_path}'."
            )

        class_names = self._get_class_names(header["class_names"])
        kpt_shape = header.get("kpt_shape")
        added_by_split: dict[str, list[Path]] = {
            "train": [],
            "val": [],
            "test": [],
        }
        added_images: list[Path] = []
        remote_image_dir = ndjson_path.parent / ndjson_path.stem

        def generator() -> DatasetIterator:
            seen_by_split: dict[str, set[Path]] = {
                "train": set(),
                "val": set(),
                "test": set(),
            }
            seen_images: set[Path] = set()
            remote_image_dir_checked = False
            for record in _image_records(ndjson_path):
                if record.get("url") and not remote_image_dir_checked:
                    _check_remote_image_dir(remote_image_dir, reuse_cached)
                    remote_image_dir_checked = True

                image_path = self._resolve_image_path(
                    ndjson_path,
                    record,
                    remote_image_dir=remote_image_dir,
                )
                if not record.get("url") and not image_path.exists():
                    self._warn_skipped_annotation(
                        ParserIssue.MISSING_IMAGE,
                        "referenced image file does not exist",
                        source=ndjson_path,
                        image=image_path,
                    )
                    continue
                split_name = self._normalize_split_name(record.get("split"))
                _add_unique(added_images, seen_images, image_path)
                _add_unique(
                    added_by_split[split_name],
                    seen_by_split[split_name],
                    image_path,
                )

                yield from _file_records(
                    str(image_path),
                    self._record_annotations(record, class_names, kpt_shape),
                )

        return generator(), added_by_split, added_images

    @classmethod
    def _record_annotations(
        cls,
        record: dict[str, Any],
        class_names: dict[int, str],
        kpt_shape: tuple[int, int] | None,
    ) -> Iterator[dict[str, Any]]:
        """Yield the LDF annotations of one image record.

        Boxes come first, then segments, then poses. The instance IDs
        count up across the three kinds.
        """
        annotations = record.get("annotations") or {}
        instance_id = 0
        for box in annotations.get("boxes", []):
            class_id, x_center, y_center, width, height = box
            yield {
                "class": class_names[int(class_id)],
                "instance_id": instance_id,
                "boundingbox": yolo_box_to_ldf(
                    x_center, y_center, width, height
                ),
            }
            instance_id += 1

        for class_id, *points in annotations.get("segments", []):
            points_array = np.array(points, dtype=float).reshape(-1, 2)
            yield {
                "class": class_names[int(class_id)],
                "instance_id": instance_id,
                "boundingbox": cls._fit_boundingbox(points_array),
                "instance_segmentation": {
                    "height": int(record["height"]),
                    "width": int(record["width"]),
                    "points": [
                        (float(x), float(y)) for x, y in points_array.tolist()
                    ],
                },
            }
            instance_id += 1

        for pose in annotations.get("pose", []):
            class_id, x_center, y_center, width, height, *flat = pose
            keypoints = _pose_keypoints(flat, kpt_shape)
            yield {
                "class": class_names[int(class_id)],
                "instance_id": instance_id,
                "boundingbox": yolo_box_to_ldf(
                    x_center, y_center, width, height
                ),
                "keypoints": {
                    "keypoints": [
                        (float(x), float(y), int(v))
                        for x, y, v in keypoints.tolist()
                    ]
                },
            }
            instance_id += 1

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

    @classmethod
    def _resolve_image_path(
        cls,
        ndjson_path: Path,
        record: dict[str, Any],
        *,
        remote_image_dir: Path,
    ) -> Path:
        if record.get("url"):
            return cls._download_image(
                record,
                remote_image_dir=remote_image_dir,
            )

        file_path = parse_manifest_path(record["file"])
        if file_path.is_absolute():
            return file_path.resolve()
        return resolve_manifest_path(ndjson_path.parent, record["file"])

    @classmethod
    def _download_image(
        cls,
        record: dict[str, Any],
        *,
        remote_image_dir: Path,
    ) -> Path:
        file_name = parse_manifest_path(record["file"])
        url = record["url"]
        split_name = cls._normalize_split_name(record.get("split"))
        url_hash = hashlib.blake2s(
            url.encode("utf-8"), digest_size=6
        ).hexdigest()
        suffix = file_name.suffix or Path(urlsplit(url).path).suffix
        destination = (
            remote_image_dir
            / split_name
            / f"{file_name.stem}-{url_hash}{suffix}"
        )
        return cls._remote_file_downloader.download(
            url, destination, validate_image=True
        )

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


def _image_records(ndjson_path: Path) -> Iterator[dict[str, Any]]:
    """Yield the image records of an NDJSON file, other lines skipped."""
    with open(ndjson_path, encoding="utf-8-sig") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("type") == "image":
                yield record


def _check_remote_image_dir(
    remote_image_dir: Path, reuse_cached: bool
) -> None:
    """Refuse or announce the reuse of a download directory that exists."""
    if not remote_image_dir.exists():
        return
    if not reuse_cached:
        raise ValueError(
            f"Remote NDJSON image directory "
            f"'{remote_image_dir}' already exists."
        )
    logger.warning(
        f"Reusing existing remote NDJSON image directory '{remote_image_dir}'."
    )


def _file_records(
    file: str, annotations: Iterator[dict[str, Any]]
) -> DatasetIterator:
    """Yield one record per annotation, or one record without labels."""
    yielded_annotation = False
    for annotation in annotations:
        yielded_annotation = True
        yield {"file": file, "annotation": annotation}
    if not yielded_annotation:
        yield {"file": file, "annotation": None}


def _add_unique(items: list[Path], seen: set[Path], item: Path) -> None:
    """Append an item the list does not hold yet."""
    if item not in seen:
        seen.add(item)
        items.append(item)


def _pose_keypoints(
    keypoints: list[Any], kpt_shape: tuple[int, int] | None
) -> np.ndarray:
    """Reshape a flat keypoint list into ``(x, y, visibility)`` rows.

    Without a ``kpt_shape`` the list must hold triples. Pairs get full
    visibility.

    Examples:
        >>> _pose_keypoints([0.1, 0.2, 2, 0.3, 0.4, 0], None).tolist()
        [[0.1, 0.2, 2.0], [0.3, 0.4, 0.0]]
        >>> _pose_keypoints([0.1, 0.2], (1, 2)).tolist()
        [[0.1, 0.2, 2.0]]
        >>> _pose_keypoints([0.1, 0.2], None)
        Traceback (most recent call last):
        ...
        ValueError: Ultralytics NDJSON pose annotations require `kpt_shape` ...

    """
    if kpt_shape is None:
        if len(keypoints) % 3 != 0:
            raise ValueError(
                "Ultralytics NDJSON pose annotations require "
                "`kpt_shape` in the dataset header when the "
                "keypoint dimensionality is not inferable."
            )
        n_kpts, kpt_dim = len(keypoints) // 3, 3
    else:
        n_kpts, kpt_dim = kpt_shape

    keypoints_array = np.array(keypoints, dtype=float).reshape(n_kpts, kpt_dim)
    if kpt_dim == 2:
        keypoints_array = np.concatenate(
            [keypoints_array, np.ones((n_kpts, 1), dtype=float) * 2],
            axis=1,
        )
    return keypoints_array
