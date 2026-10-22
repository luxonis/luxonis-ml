"""Export bounding boxes to the TensorFlow CSV format."""

import csv
import json
from pathlib import Path
from typing import Any, cast

import polars as pl
from PIL import Image
from typing_extensions import override

from luxonis_ml.data.exporters.base_exporter import BaseExporter
from luxonis_ml.data.exporters.exporter_utils import (
    PreparedLDF,
    check_group_file_correspondence,
    exporter_specific_annotation_warning,
    split_of_group,
)
from luxonis_ml.enums import DatasetType


class TensorflowCSVExporter(BaseExporter):
    """Export bounding boxes in the TensorFlow CSV format.

    Each split directory holds the images and an ``_annotations.csv``
    file with one row per box. A row gives the image size and the
    box corners in pixels.
    """

    def __init__(
        self,
        dataset_identifier: str,
        output_path: Path,
        max_partition_size_gb: float | None,
    ) -> None:
        """Create the exporter.

        Args:
            dataset_identifier: Name of the export directory.
            output_path: Directory where the export is written.
            max_partition_size_gb: Maximum partition size in GiB, or
                ``None`` to disable partitioning.

        """
        super().__init__(
            dataset_identifier, output_path, max_partition_size_gb
        )

    @staticmethod
    def get_split_names() -> dict[str, str]:
        """Return the directory name of each split.

        Returns:
            Split names mapped to directory names. The ``val`` split
            becomes ``valid``.

        """
        return {"train": "train", "val": "valid", "test": "test"}

    @override
    def supported_ann_types(self) -> list[str]:
        return DatasetType.TFCSV.supported_annotation_formats

    @override
    def export(self, prepared_ldf: PreparedLDF) -> None:
        check_group_file_correspondence(prepared_ldf)
        exporter_specific_annotation_warning(
            prepared_ldf, self.supported_ann_types()
        )

        rows_by_split: dict[str, list[dict[str, Any]]] = {
            v: [] for v in self.get_split_names().values()
        }

        df = prepared_ldf.processed_df
        grouped = df.group_by(["file", "group_id"], maintain_order=True)

        copied_files: set[Path] = set()

        for key, group_df in grouped:
            file_name, group_id = cast(tuple[str, Any], key)
            logical_split = split_of_group(prepared_ldf, group_id)
            split_name = self.get_split_names()[logical_split]

            file_path = Path(str(file_name))
            idx = self.image_indices.setdefault(
                file_path, len(self.image_indices)
            )
            new_name = f"{idx}{file_path.suffix}"

            with Image.open(file_path) as im:
                width, height = im.size

            per_image_rows = _csv_rows(group_df, new_name, width, height)

            # NOTE: We use a rough constant (64) to approximate the per-row CSV bytes that do NOT
            # depend on variable-length fields. Getting the true on-disk size here would require
            # serializing with csv.DictWriter using the exact dialect and encoding
            ann_size_est = sum(
                64 + len(r["class"]) + len(r["filename"])
                for r in per_image_rows
            )
            img_size = file_path.stat().st_size

            rows_by_split = self._maybe_roll_partition(
                rows_by_split, ann_size_est + img_size
            )

            if per_image_rows:
                rows_by_split[split_name].extend(per_image_rows)

            split_dir = self._get_data_path(
                self.output_path, split_name, self.part
            )
            split_dir.mkdir(parents=True, exist_ok=True)
            dest_img = split_dir / new_name

            if file_path not in copied_files:
                copied_files.add(file_path)
                if dest_img != file_path:
                    dest_img.write_bytes(file_path.read_bytes())
                self.current_size += img_size

        # Final write
        self._dump_annotations(rows_by_split, self.output_path, self.part)

    def _maybe_roll_partition(
        self,
        rows_by_split: dict[str, list[dict[str, Any]]],
        additional_size: int,
    ) -> dict[str, list[dict[str, Any]]]:
        if (
            self.max_partition_size
            and self.part is not None
            and (self.current_size + additional_size) > self.max_partition_size
        ):
            self._dump_annotations(rows_by_split, self.output_path, self.part)
            self.current_size = 0
            self.part += 1
            return {v: [] for v in self.get_split_names().values()}
        return rows_by_split

    @override
    def _dump_annotations(
        self,
        rows_by_split: dict[str, list[dict[str, Any]]],
        output_path: Path,
        part: int | None = None,
    ) -> None:
        base = (
            output_path / f"{self.dataset_identifier}_part{part}"
            if part is not None
            else output_path / self.dataset_identifier
        )
        base.mkdir(parents=True, exist_ok=True)

        for split_name, rows in rows_by_split.items():
            split_dir = base / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

            csv_path = split_dir / "_annotations.csv"
            fieldnames = [
                "filename",
                "width",
                "height",
                "class",
                "xmin",
                "ymin",
                "xmax",
                "ymax",
            ]
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in rows:
                    writer.writerow(r)

    @override
    def _get_data_path(
        self, output_path: Path, split_name: str, part: int | None = None
    ) -> Path:
        base = (
            output_path / f"{self.dataset_identifier}_part{part}"
            if part is not None
            else output_path / self.dataset_identifier
        )
        return base / split_name


def _csv_rows(
    group_df: pl.DataFrame, new_name: str, width: int, height: int
) -> list[dict[str, Any]]:
    """Return one CSV row per bounding box of an image, in pixels."""
    rows = []
    for row in group_df.iter_rows(named=True):
        if row["task_type"] != "boundingbox" or row["annotation"] is None:
            continue
        ann = json.loads(row["annotation"])
        cname = row["class_name"]
        if ann is None or not cname:
            continue

        x_tl = float(ann.get("x", 0.0))
        y_tl = float(ann.get("y", 0.0))
        w = float(ann.get("w", 0.0))
        h = float(ann.get("h", 0.0))

        rows.append(
            {
                "filename": new_name,
                "width": int(width),
                "height": int(height),
                "class": cname,
                "xmin": round(x_tl * width),
                "ymin": round(y_tl * height),
                "xmax": round((x_tl + w) * width),
                "ymax": round((y_tl + h) * height),
            }
        )
    return rows
