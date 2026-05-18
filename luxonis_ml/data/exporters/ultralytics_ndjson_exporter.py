from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from PIL import Image

from luxonis_ml.data.exporters.exporter_utils import (
    PreparedLDF,
    annotation_to_polygons,
    check_group_file_correspondence,
    exporter_specific_annotation_warning,
    split_of_group,
)
from luxonis_ml.enums import DatasetType

from .base_exporter import BaseExporter

UltralyticsTask = Literal["detect", "segment", "pose"]


class UltralyticsNDJSONExporter(BaseExporter):
    def __init__(
        self,
        dataset_identifier: str,
        output_path: Path,
        max_partition_size_gb: float | None,
        *,
        dataset_type: DatasetType,
        ndjson_task: UltralyticsTask,
    ):
        super().__init__(
            dataset_identifier, output_path, max_partition_size_gb
        )
        self.dataset_type = dataset_type
        self.ndjson_task = ndjson_task
        self.class_to_id: dict[str, int] = {}
        self.class_names: list[str] = []
        self.keypoint_counts: set[int] = set()
        self._image_size_cache: dict[Path, tuple[int, int]] = {}

    def get_split_names(self) -> dict[str, str]:
        return {"train": "train", "val": "val", "test": "test"}

    def supported_ann_types(self) -> list[str]:
        if self.ndjson_task == "pose":
            return ["keypoints", "boundingbox"]
        return self.dataset_type.supported_annotation_formats

    def export(self, prepared_ldf: PreparedLDF) -> None:
        check_group_file_correspondence(prepared_ldf)
        exporter_specific_annotation_warning(
            prepared_ldf, self.supported_ann_types()
        )

        annotation_splits: dict[str, list[dict[str, Any]]] = {
            split: [] for split in self.get_split_names().values()
        }

        df = prepared_ldf.processed_df
        grouped = df.group_by(["file", "group_id"], maintain_order=True)
        copied_files: set[Path] = set()

        for key, group_df in grouped:
            file_name, group_id = cast(tuple[str, Any], key)
            logical_split = split_of_group(prepared_ldf, group_id)
            split = self.get_split_names()[logical_split]

            file_path = Path(str(file_name))
            idx = self.image_indices.setdefault(
                file_path, len(self.image_indices)
            )
            export_name = f"{idx}{file_path.suffix}"
            relative_file = Path(split) / export_name

            record = self._build_image_record(
                file_path=file_path,
                relative_file=relative_file,
                split=split,
                group_df=group_df,
            )

            ann_size_estimate = (
                len(json.dumps(record, separators=(",", ":"))) + 1
            )
            img_size = file_path.stat().st_size
            annotation_splits = self._maybe_roll_partition(
                annotation_splits, ann_size_estimate + img_size
            )

            annotation_splits[split].append(record)

            data_path = self._get_data_path(self.output_path, split, self.part)
            data_path.mkdir(parents=True, exist_ok=True)
            dest = data_path / export_name
            if file_path not in copied_files:
                copied_files.add(file_path)
                if dest != file_path:
                    dest.write_bytes(file_path.read_bytes())
                self.current_size += img_size

        self._dump_annotations(annotation_splits, self.output_path, self.part)

    def _build_image_record(
        self,
        file_path: Path,
        relative_file: Path,
        split: str,
        group_df: Any,
    ) -> dict[str, Any]:
        instances: dict[tuple[str, int], dict[str, Any]] = {}

        for row in group_df.iter_rows(named=True):
            task_type = row["task_type"]
            annotation = row["annotation"]
            class_name = row["class_name"]
            instance_id = row["instance_id"]

            if (
                annotation is None
                or task_type not in self.supported_ann_types()
                or not class_name
                or instance_id is None
                or instance_id < 0
            ):
                continue

            if class_name not in self.class_to_id:
                self.class_to_id[class_name] = len(self.class_to_id)
                self.class_names.append(class_name)

            instance = instances.setdefault(
                (class_name, int(instance_id)),
                {"class_id": self.class_to_id[class_name]},
            )
            instance[task_type] = json.loads(annotation)

        width, height = self._image_size(file_path)
        record: dict[str, Any] = {
            "type": "image",
            "file": relative_file.as_posix(),
            "width": width,
            "height": height,
            "split": split,
        }

        if annotations := self._build_annotations(instances, file_path):
            record["annotations"] = annotations

        return record

    def _build_annotations(
        self,
        instances: dict[tuple[str, int], dict[str, Any]],
        file_path: Path,
    ) -> dict[str, Any]:
        if self.ndjson_task == "detect":
            boxes = [
                self._bbox_to_line(
                    int(instance["class_id"]), instance["boundingbox"]
                )
                for instance in instances.values()
                if "boundingbox" in instance
            ]
            return {"boxes": boxes} if boxes else {}

        if self.ndjson_task == "segment":
            segments = [
                line
                for instance in instances.values()
                if "instance_segmentation" in instance
                for line in self._segments_to_lines(
                    int(instance["class_id"]),
                    instance["instance_segmentation"],
                    file_path,
                )
            ]
            return {"segments": segments} if segments else {}

        poses = [
            pose_line
            for instance in instances.values()
            if "keypoints" in instance
            for pose_line in [
                self._keypoints_to_line(
                    int(instance["class_id"]),
                    instance.get("boundingbox"),
                    instance["keypoints"],
                )
            ]
            if pose_line is not None
        ]
        return {"pose": poses} if poses else {}

    def _maybe_roll_partition(
        self,
        annotation_splits: dict[str, list[dict[str, Any]]],
        additional_size: int,
    ) -> dict[str, list[dict[str, Any]]]:
        has_data = any(annotation_splits[split] for split in annotation_splits)

        if (
            self.max_partition_size
            and self.part is not None
            and has_data
            and (self.current_size + additional_size) > self.max_partition_size
        ):
            self._dump_annotations(
                annotation_splits, self.output_path, self.part
            )
            self.current_size = 0
            self.part += 1
            return {split: [] for split in self.get_split_names().values()}
        return annotation_splits

    def _dump_annotations(
        self,
        annotation_splits: dict[str, list[dict[str, Any]]],
        output_path: Path,
        part: int | None = None,
    ) -> None:
        if all(len(records) == 0 for records in annotation_splits.values()):
            return

        base = (
            output_path / f"{self.dataset_identifier}_part{part}"
            if part is not None
            else output_path / self.dataset_identifier
        )

        for split_name in self.get_split_names().values():
            (base / split_name).mkdir(parents=True, exist_ok=True)

        header = self._dataset_header()
        lines = [json.dumps(header, separators=(",", ":"))]

        for split_name in self.get_split_names().values():
            for record in annotation_splits.get(split_name, []):
                lines.append(json.dumps(record, separators=(",", ":")))

        (base / "dataset.ndjson").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    def _get_data_path(
        self, output_path: Path, split: str, part: int | None = None
    ) -> Path:
        base = (
            output_path / f"{self.dataset_identifier}_part{part}"
            if part is not None
            else output_path / self.dataset_identifier
        )
        return base / split

    def _dataset_header(self) -> dict[str, Any]:
        header: dict[str, Any] = {
            "type": "dataset",
            "task": self.ndjson_task,
            "name": self.dataset_identifier,
            "description": "Converted from Luxonis export with relative local file paths.",
            "class_names": {
                str(class_id): class_name
                for class_id, class_name in enumerate(self.class_names)
            },
        }

        if self.ndjson_task == "pose" and len(self.keypoint_counts) == 1:
            header["kpt_shape"] = [next(iter(self.keypoint_counts)), 3]

        return header

    def _image_size(self, file_path: Path) -> tuple[int, int]:
        cached = self._image_size_cache.get(file_path)
        if cached is not None:
            return cached

        with Image.open(file_path) as image:
            width, height = image.size
        self._image_size_cache[file_path] = (width, height)
        return width, height

    @staticmethod
    def _bbox_to_line(
        class_id: int, bbox: dict[str, Any]
    ) -> list[float | int]:
        x = float(bbox["x"])
        y = float(bbox["y"])
        w = float(bbox["w"])
        h = float(bbox["h"])
        return [class_id, x + w / 2.0, y + h / 2.0, w, h]

    def _segments_to_lines(
        self,
        class_id: int,
        annotation: dict[str, Any],
        file_path: Path,
    ) -> list[list[float | int]]:
        if "points" in annotation:
            points = [
                (float(x), float(y)) for x, y in annotation["points"] or []
            ]
            if len(points) < 3:
                return []
            return [[class_id, *self._flatten_points(points)]]

        polygons = annotation_to_polygons(annotation, file_path)
        return [
            [class_id, *self._flatten_points(polygon)]
            for polygon in polygons
            if len(polygon) >= 3
        ]

    def _keypoints_to_line(
        self,
        class_id: int,
        bbox: dict[str, Any] | None,
        annotation: dict[str, Any],
    ) -> list[float | int] | None:
        keypoints = annotation.get("keypoints") or []
        if not keypoints:
            return None

        keypoint_array = np.array(keypoints, dtype=float)
        if keypoint_array.ndim != 2 or keypoint_array.shape[1] != 3:
            return None

        self.keypoint_counts.add(int(keypoint_array.shape[0]))

        if bbox is None:
            bbox = self._bbox_from_keypoints(keypoint_array)
            if bbox is None:
                return None

        line = self._bbox_to_line(class_id, bbox)
        for x, y, visibility in keypoint_array.tolist():
            line.extend([float(x), float(y), int(visibility)])
        return line

    @staticmethod
    def _bbox_from_keypoints(
        keypoints: np.ndarray,
    ) -> dict[str, float] | None:
        visible = keypoints[keypoints[:, 2] > 0][:, :2]
        points = visible if visible.size > 0 else keypoints[:, :2]
        if points.size == 0:
            return None

        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)
        return {
            "x": float(x_min),
            "y": float(y_min),
            "w": float(x_max - x_min),
            "h": float(y_max - y_min),
        }

    @staticmethod
    def _flatten_points(points: list[tuple[float, float]]) -> list[float]:
        flattened: list[float] = []
        for x, y in points:
            flattened.extend([float(x), float(y)])
        return flattened
