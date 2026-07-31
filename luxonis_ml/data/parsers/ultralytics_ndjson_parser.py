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

from .parser_plugin import Layout, ParseResult, ParserPlugin, SplitRecord


class UltralyticsNDJSONParser(ParserPlugin):
    """Parse Ultralytics NDJSON datasets into LDF.

    NDJSON records may carry their own split names. When no split is
    present on an image record, the image is assigned to ``"train"``.
    ``"valid"`` and ``"validation"`` are normalized to ``"val"``.
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

        # Which splits a manifest uses is written on its image records,
        # so answering it exactly would cost a walk of the whole file —
        # the walk parsing already makes, tagging each record with the
        # split it names. The layout therefore claims the splits the
        # format defines, and carries the manifest and its header so that
        # neither is resolved a second time.
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
        # Every split of an NDJSON source is read from the same manifest,
        # so any entry of the layout names it.
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

        The fallback the importer would use instead is a throwaway parse,
        which for this format downloads every remote image and creates the
        cache directory that the real parse then refuses to write into.
        The destination of a remote image is derived from the record, so
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
            # The records the parse drops have to be dropped here too:
            # counting an image that is never added would leave a
            # count-based split short of what was asked for. The parse
            # warns about each one, so this stays silent.
            if not record.get("url") and not image_path.exists():
                continue
            # Parsing warns about every record's split; enumerating the
            # same records again should not repeat all of it.
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
                if remote_image_dir.exists():
                    if not reuse_cached:
                        raise ValueError(
                            f"Remote NDJSON image directory "
                            f"'{remote_image_dir}' already exists."
                        )
                    logger.warning(
                        f"Reusing existing remote NDJSON image "
                        f"directory '{remote_image_dir}'."
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
            annotations = record.get("annotations") or {}
            instance_id = 0
            yielded_annotation = False
            # The file name identifies the image, not the annotation:
            # stringifying it once per record keeps an image carrying
            # thirty instances from paying for thirty conversions.
            image_file = str(image_path)

            for box in annotations.get("boxes", []):
                class_id, x_center, y_center, width, height = box
                yielded_annotation = True
                yield (
                    split_name,
                    {
                        "file": image_file,
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
                    },
                )
                instance_id += 1

            for segment in annotations.get("segments", []):
                class_id, *points = segment
                points_array = np.array(points, dtype=float).reshape(-1, 2)
                yielded_annotation = True
                yield (
                    split_name,
                    {
                        "file": image_file,
                        "annotation": {
                            "class": class_names[int(class_id)],
                            "instance_id": instance_id,
                            "boundingbox": self._fit_boundingbox(points_array),
                            "instance_segmentation": {
                                "height": int(record["height"]),
                                "width": int(record["width"]),
                                # `tolist` on a `dtype=float` array hands
                                # back Python floats, so converting each
                                # coordinate again could not have changed
                                # one.
                                "points": list(
                                    map(tuple, points_array.tolist())
                                ),
                            },
                        },
                    },
                )
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

                keypoints_array = np.array(keypoints, dtype=float).reshape(
                    n_kpts, kpt_dim
                )
                if kpt_dim == 2:
                    # The appended visibility column was a constant
                    # ``2.0`` that ``int`` turned back into ``2``, so
                    # building and concatenating it only to cast it
                    # away produced this literal and nothing else.
                    keypoint_values = [
                        (x, y, 2) for x, y in keypoints_array.tolist()
                    ]
                else:
                    # As above, `tolist` already yields Python floats;
                    # only the visibility flag needs converting.
                    keypoint_values = [
                        (x, y, int(v)) for x, y, v in keypoints_array.tolist()
                    ]

                yielded_annotation = True
                yield (
                    split_name,
                    {
                        "file": image_file,
                        "annotation": {
                            "class": class_names[int(class_id)],
                            "instance_id": instance_id,
                            "boundingbox": {
                                "x": float(x_center) - float(width) / 2,
                                "y": float(y_center) - float(height) / 2,
                                "w": float(width),
                                "h": float(height),
                            },
                            "keypoints": {"keypoints": keypoint_values},
                        },
                    },
                )
                instance_id += 1

            if not yielded_annotation:
                yield split_name, {"file": image_file, "annotation": None}

    @staticmethod
    def _iter_image_records(ndjson_path: Path) -> Iterator[dict[str, Any]]:
        """Yield image records one line at a time."""
        with open(ndjson_path, encoding="utf-8-sig") as file:
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
        # One reduction per axis rather than one per corner. `min` and
        # `max` select an element instead of computing one, so reducing
        # both columns in a single pass returns the very same floats —
        # NaN propagation and the empty-polygon `ValueError` included —
        # while halving the NumPy calls and dropping the column slices.
        x_min, y_min = points.min(axis=0).tolist()
        x_max, y_max = points.max(axis=0).tolist()
        return {
            "x": x_min,
            "y": y_min,
            "w": x_max - x_min,
            "h": y_max - y_min,
        }
