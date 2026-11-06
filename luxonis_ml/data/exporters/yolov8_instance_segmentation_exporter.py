"""Export instance segmentation to the YOLOv8 format."""

import json
from pathlib import Path
from typing import Any, cast

from typing_extensions import override

from luxonis_ml.data.exporters.exporter_utils import (
    PreparedLDF,
    annotation_to_polygons,
    check_group_file_correspondence,
    exporter_specific_annotation_warning,
    register_class,
    split_of_group,
)
from luxonis_ml.enums import DatasetType
from luxonis_ml.utils.path import path_to_posix

from .base_exporter import BaseExporter


class YoloV8InstanceSegmentationExporter(BaseExporter):
    """Export instance-segmentation annotations in YOLOv8 format.

    Attributes:
        class_to_id: Class-name to class-ID mapping.
        class_names: Class names ordered by class ID.

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
        self.class_to_id: dict[str, int] = {}
        self.class_names: list[str] = []

    def get_split_names(self) -> dict[str, str]:
        """Return the directory name of each split.

        Returns:
            The ``train``, ``val`` and ``test`` splits, each mapped to
            its own name.

        """
        return {"train": "train", "val": "val", "test": "test"}

    def _yaml_filename(self) -> str:
        return "dataset.yaml"

    @override
    def supported_ann_types(self) -> list[str]:
        return (
            DatasetType.YOLOV8INSTANCESEGMENTATION.supported_annotation_formats
        )

    @override
    def export(self, prepared_ldf: PreparedLDF) -> None:
        check_group_file_correspondence(prepared_ldf)
        exporter_specific_annotation_warning(
            prepared_ldf, self.supported_ann_types()
        )

        annotation_splits: dict[str, dict[str, list[str]]] = {
            k: {} for k in self.get_split_names().values()
        }

        df = prepared_ldf.processed_df
        grouped = df.group_by(["file", "group_id"], maintain_order=True)
        copied_files: set[Path] = set()

        for key, group_df in grouped:
            file_name, group_id = cast("tuple[str, Any]", key)
            logical_split = split_of_group(prepared_ldf, group_id)
            split = self.get_split_names()[logical_split]

            file_path = Path(str(file_name))
            idx = self.image_indices.setdefault(
                file_path, len(self.image_indices)
            )
            new_name = f"{idx}{file_path.suffix}"

            label_lines: list[str] = []

            for row in group_df.iter_rows(named=True):
                label_lines.extend(self._polygon_lines(row, file_path))

            ann_size_estimate = sum(len(s) + 1 for s in label_lines)
            img_size = file_path.stat().st_size
            annotation_splits = self._maybe_roll_partition(
                annotation_splits, ann_size_estimate + img_size
            )

            annotation_splits[split][new_name] = label_lines

            data_path = self._get_data_path(self.output_path, split, self.part)
            data_path.mkdir(parents=True, exist_ok=True)
            dest = data_path / new_name
            if file_path not in copied_files:
                copied_files.add(file_path)
                if dest != file_path:
                    dest.write_bytes(file_path.read_bytes())
                self.current_size += img_size

        self._dump_annotations(annotation_splits, self.output_path, self.part)

    def _polygon_lines(
        self, row: dict[str, Any], file_path: Path
    ) -> list[str]:
        """Return one YOLO segment line per polygon of an instance row."""
        if row["annotation"] is None:
            return []
        if row["task_type"] != "instance_segmentation":
            return []
        cname = row["class_name"]
        if not cname:
            return []

        cid = register_class(cname, self.class_to_id, self.class_names)
        polygons = annotation_to_polygons(
            json.loads(row["annotation"]), file_path
        )
        lines = []
        for poly in polygons:
            parts = [
                f"{min(max(x, 0.0), 1.0):.12f} {min(max(y, 0.0), 1.0):.12f}"
                for x, y in poly
            ]
            lines.append(f"{cid} " + " ".join(parts))
        return lines

    def _maybe_roll_partition(
        self,
        annotation_splits: dict[str, dict[str, list[str]]],
        additional_size: int,
    ) -> dict[str, dict[str, list[str]]]:
        if (
            self.max_partition_size
            and self.part is not None
            and (self.current_size + additional_size) > self.max_partition_size
        ):
            self._dump_annotations(
                annotation_splits, self.output_path, self.part
            )
            self.current_size = 0
            self.part += 1
            return {k: {} for k in self.get_split_names().values()}
        return annotation_splits

    @override
    def _dump_annotations(
        self,
        annotation_splits: dict[str, dict[str, list[str]]],
        output_path: Path,
        part: int | None = None,
    ) -> None:
        base = (
            output_path / f"{self.dataset_identifier}_part{part}"
            if part is not None
            else output_path / self.dataset_identifier
        )
        if all(len(images) == 0 for images in annotation_splits.values()):
            return

        for split_name in self.get_split_names().values():
            labels_dir = base / "labels" / split_name
            labels_dir.mkdir(parents=True, exist_ok=True)
            images_dir = base / "images" / split_name
            images_dir.mkdir(parents=True, exist_ok=True)

            for img_name, lines in annotation_splits.get(
                split_name, {}
            ).items():
                (labels_dir / f"{Path(img_name).stem}.txt").write_text(
                    "\n".join(lines), encoding="utf-8"
                )

        yaml_filename = self._yaml_filename()
        if yaml_filename:
            split_dirs = self.get_split_names()
            yaml_obj = {
                "train": path_to_posix(Path("images") / split_dirs["train"]),
                "val": path_to_posix(Path("images") / split_dirs["val"]),
                "test": path_to_posix(Path("images") / split_dirs["test"]),
                "nc": len(self.class_names),
                "names": self.class_names,
            }
            (base / yaml_filename).write_text(
                self._to_yaml(yaml_obj), encoding="utf-8"
            )

    @override
    def _get_data_path(
        self, output_path: Path, split: str, part: int | None = None
    ) -> Path:
        base = (
            output_path / f"{self.dataset_identifier}_part{part}"
            if part is not None
            else output_path / self.dataset_identifier
        )
        return base / "images" / split

    @staticmethod
    def _to_yaml(d: dict[str, Any]) -> str:
        lines: list[str] = []
        for k, v in d.items():
            lines.append(f"{k}: {v}")
        return "\n".join(lines) + "\n"
