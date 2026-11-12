"""Parser for datasets in the Ultralytics NDJSON format."""

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np
from loguru import logger
from typing_extensions import override

from luxonis_ml.data.utils.enums import ParserIssue
from luxonis_ml.data.utils.remote_file_downloader import RemoteFileDownloader
from luxonis_ml.utils.path import (
    parse_manifest_path,
    resolve_manifest_path,
)

from .parser_plugin import (
    Layout,
    ParseResult,
    ParserPlugin,
    SplitRecord,
    centered_box,
)


class UltralyticsNDJSONParser(ParserPlugin):
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

    dataset_types = (
        "ultralytics-ndjson",
        "ultralytics-ndjson-instancesegmentation",
        "ultralytics-ndjson-keypoints",
    )
    _remote_file_downloader = RemoteFileDownloader()

    @classmethod
    @override
    def detect(cls, source: Path) -> Layout | None:
        ndjson_path = cls._resolve_ndjson_path(source)
        if ndjson_path is None:
            return None

        header = cls._load_header(ndjson_path)
        if header is None:
            return None

        # Which splits a manifest uses is written on its image records, so
        # answering exactly would cost a walk of the whole file - the walk
        # the parse makes anyway. The layout claims the splits the format
        # defines and carries the manifest and header along.
        splits: dict[str | None, dict[str, Any]] = {
            split_name: {"ndjson_path": ndjson_path, "header": header}
            for split_name in ("train", "val", "test")
        }
        return Layout(splits)

    @override
    def parse(
        self,
        source: Path,
        layout: Layout,
        *,
        reuse_cached: bool = True,
        **kwargs: Any,
    ) -> ParseResult:
        del source, kwargs
        # Every split is read from the same manifest, so any entry of the
        # layout names it.
        manifest = next(iter(layout.splits.values()))
        return ParseResult(
            self._stream_records(
                manifest["ndjson_path"],
                manifest["header"],
                reuse_cached=reuse_cached,
            ),
            self._skeletons,
        )

    @override
    def enumerate_files(
        self,
        source: Path,
        layout: Layout,
        **kwargs: Any,
    ) -> dict[str | None, list[Path]] | None:
        """List each split's images without downloading any of them.

        Worth implementing here: the importer's fallback is a throwaway
        parse, which for this format would download every remote image and
        create the cache directory the real parse then refuses to write
        into. A remote image's destination is derived from its record, so
        it can be named without fetching anything.
        """
        del source, kwargs
        manifest = next(iter(layout.splits.values()))
        ndjson_path = manifest["ndjson_path"]
        base_dir = ndjson_path.parent
        remote_image_dir = base_dir / ndjson_path.stem

        enumerated: dict[str | None, list[Path]] = {}
        for record in self._iter_image_records(ndjson_path):
            image_path = self._resolve_image_path(
                base_dir,
                record,
                remote_image_dir=remote_image_dir,
                download=False,
            )
            # Drop what the parse drops: counting an image that is never
            # added would leave a count-based split short. The parse warns
            # about each one, so this stays silent.
            if not record.get("url") and not image_path.exists():
                continue
            # Likewise for the split warnings.
            split_name = self._normalize_split_name(
                record.get("split"), warn=False
            )
            enumerated.setdefault(split_name, []).append(image_path)
        return enumerated

    def _stream_records(
        self,
        ndjson_path: Path,
        header: dict[str, Any],
        *,
        reuse_cached: bool,
    ) -> Iterator[SplitRecord]:
        """Stream one manifest, tagging each record with its split.

        Args:
            ndjson_path: Manifest to read.
            header: The manifest's leading ``dataset`` record.
            reuse_cached: Whether an existing directory of downloaded
                images may be reused instead of failing.

        Yields:
            The split an image record names, and one record per
            annotation it carries. An image without annotations yields a
            single record with no annotation.

        """
        class_names = self._get_class_names(header["class_names"])
        kpt_shape = header.get("kpt_shape")
        base_dir = ndjson_path.parent
        remote_image_dir = base_dir / ndjson_path.stem
        remote_image_dir_checked = False

        for record in self._iter_image_records(ndjson_path):
            url = record.get("url")
            if url and not remote_image_dir_checked:
                _check_remote_image_dir(
                    remote_image_dir, reuse_cached=reuse_cached
                )
                remote_image_dir_checked = True

            image_path = self._resolve_image_path(
                base_dir,
                record,
                remote_image_dir=remote_image_dir,
            )
            if not url and not image_path.exists():
                self._warn_skipped_annotation(
                    ParserIssue.MISSING_IMAGE,
                    "referenced image file does not exist",
                    source=ndjson_path,
                    image=image_path,
                )
                continue

            split_name = self._normalize_split_name(record.get("split"))
            image_file = str(image_path)
            for file_record in _file_records(
                image_file,
                self._record_annotations(record, class_names, kpt_shape),
            ):
                yield split_name, file_record

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

        Yields:
            One annotation per box, segment or pose, with the class name,
            the instance ID and a bounding box. A segment also holds its
            polygon and the image size, and a pose its keypoints.

        """
        annotations = record.get("annotations") or {}
        instance_id = 0
        for box in annotations.get("boxes", []):
            class_id, x_center, y_center, width, height = box
            yield {
                "class": class_names[int(class_id)],
                "instance_id": instance_id,
                "boundingbox": centered_box(x_center, y_center, width, height),
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
                    "points": list(map(tuple, points_array.tolist())),
                },
            }
            instance_id += 1

        for pose in annotations.get("pose", []):
            class_id, x_center, y_center, width, height, *keypoints = pose
            keypoint_values = _pose_keypoints(keypoints, kpt_shape)
            yield {
                "class": class_names[int(class_id)],
                "instance_id": instance_id,
                "boundingbox": centered_box(x_center, y_center, width, height),
                "keypoints": {"keypoints": keypoint_values},
            }
            instance_id += 1

    @staticmethod
    def _iter_image_records(ndjson_path: Path) -> Iterator[dict[str, Any]]:
        """Yield image records one line at a time."""
        with ndjson_path.open(encoding="utf-8-sig") as file:
            for raw_line in file:
                line = raw_line.strip()
                if not line:
                    continue

                record = json.loads(line)
                if record.get("type") == "image":
                    yield record

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

    @staticmethod
    def _load_header(ndjson_path: Path) -> dict[str, Any] | None:
        """Return the leading ``dataset`` record of a manifest.

        Args:
            ndjson_path: Manifest to read, as resolved by
                `_resolve_ndjson_path`.

        Returns:
            The header, or ``None`` when the file cannot be read or does
            not describe an Ultralytics NDJSON dataset.

        """
        dataset_record = None
        has_image_record = False
        try:
            with ndjson_path.open(encoding="utf-8-sig") as file:
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
        base_dir: Path,
        record: dict[str, Any],
        *,
        remote_image_dir: Path,
        download: bool = True,
    ) -> Path:
        if record.get("url"):
            destination = cls._remote_image_path(
                record,
                remote_image_dir=remote_image_dir,
            )
            if not download:
                return destination
            return cls._remote_file_downloader.download(
                record["url"], destination, validate_image=True
            )

        file_path = parse_manifest_path(record["file"])
        if file_path.is_absolute():
            return file_path.resolve()
        return resolve_manifest_path(base_dir, record["file"])

    @classmethod
    def _remote_image_path(
        cls,
        record: dict[str, Any],
        *,
        remote_image_dir: Path,
    ) -> Path:
        """Where a record's remote image is, or would be, downloaded."""
        file_name = parse_manifest_path(record["file"])
        url = record["url"]
        # The caller warns about the record's split itself.
        split_name = cls._normalize_split_name(record.get("split"), warn=False)
        url_hash = hashlib.blake2s(
            url.encode("utf-8"), digest_size=6
        ).hexdigest()
        suffix = file_name.suffix or Path(urlsplit(url).path).suffix
        return (
            remote_image_dir
            / split_name
            / f"{file_name.stem}-{url_hash}{suffix}"
        )

    @staticmethod
    def _normalize_split_name(
        split_name: str | None, *, warn: bool = True
    ) -> str:
        if split_name in {"train", "val", "test"}:
            return split_name
        if split_name in {"valid", "validation"}:
            return "val"
        if not warn:
            return "train"
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
        """Fit a bounding box around a polygon."""
        x_min, y_min = points.min(axis=0).tolist()
        x_max, y_max = points.max(axis=0).tolist()
        return {
            "x": x_min,
            "y": y_min,
            "w": x_max - x_min,
            "h": y_max - y_min,
        }


def _check_remote_image_dir(
    remote_image_dir: Path, *, reuse_cached: bool
) -> None:
    """Refuse or announce the reuse of a download directory that exists."""
    if not remote_image_dir.exists():
        return
    if not reuse_cached:
        raise ValueError(
            f"Remote NDJSON image directory '{remote_image_dir}' already exists."
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


def _pose_keypoints(
    keypoints: list[Any], kpt_shape: tuple[int, int] | None
) -> list[tuple[float, float, int]]:
    """Reshape a flat keypoint list into ``(x, y, visibility)`` rows.

    Pairs get full visibility.

    Returns:
        One ``(x, y, visibility)`` triplet per keypoint.

    Raises:
        ValueError: If the length of ``keypoints`` does not fit
            ``kpt_shape`` or, without a ``kpt_shape``, is not a multiple
            of 3.

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
        # Annotations without a visibility column are fully
        # visible.
        return [(x, y, 2) for x, y in keypoints_array.tolist()]
    return [(x, y, int(v)) for x, y, v in keypoints_array.tolist()]
