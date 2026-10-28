"""Export keypoints to the YOLOv8 pose format."""

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
from typing_extensions import override

from luxonis_ml.data.exporters.exporter_utils import (
    PreparedLDF,
    check_group_file_correspondence,
    exporter_specific_annotation_warning,
    register_class,
    split_of_group,
)
from luxonis_ml.enums import DatasetType
from luxonis_ml.utils.path import path_to_posix

from .base_exporter import BaseExporter


class YoloV8KeypointsExporter(BaseExporter):
    """Export keypoint annotations in YOLOv8 pose format.

    Attributes:
        class_to_id: Class-name to class-ID mapping.
        class_names: Class names ordered by class ID.
        kpt_shape: Optional ``(n_kpts, kpt_dim)`` pair written to
            ``dataset.yaml``.

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
        self.kpt_shape: tuple[int, int] | None = None  # (n_kpts, kpt_dim)

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
        return DatasetType.YOLOV8KEYPOINTS.supported_annotation_formats

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
            file_name, group_id = cast(tuple[str, Any], key)
            logical_split = split_of_group(prepared_ldf, group_id)
            split = self.get_split_names()[logical_split]

            file_path = Path(str(file_name))
            idx = self.image_indices.setdefault(
                file_path, len(self.image_indices)
            )
            new_name = f"{idx}{file_path.suffix}"

            label_lines: list[str] = []

            for row in group_df.iter_rows(named=True):
                line = self._keypoint_line(row)
                if line is not None:
                    label_lines.append(line)

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

    def _keypoint_line(self, row: dict[str, Any]) -> str | None:
        """Return the YOLO pose line of a keypoint row, or ``None``.

        The box of the line encloses the visible keypoints. A class whose
        keypoints are all rejected still gets an ID.

        Returns:
            The class ID, the ``x_center y_center w h`` box and an
            ``x y visibility`` triple for each keypoint, separated by
            spaces. ``None`` if the row is not a keypoint annotation,
            has no class name, or has no keypoint rows of 2 or 3
            values.

        """
        if row["annotation"] is None or row["task_type"] != "keypoints":
            return None
        cname = row["class_name"]
        if not cname:
            return None
        cid = register_class(cname, self.class_to_id, self.class_names)

        keypoints = json.loads(row["annotation"]).get("keypoints", None)
        if not keypoints:
            return None
        kp_arr = np.array(keypoints, dtype=float)
        if kp_arr.ndim != 2 or kp_arr.shape[1] not in (2, 3):
            return None

        n_kpts, kpt_dim = kp_arr.shape
        prev_n, prev_dim = self.kpt_shape or (0, 2)
        self.kpt_shape = (max(prev_n, n_kpts), max(prev_dim, kpt_dim))

        if kpt_dim == 2:
            vis = np.full((n_kpts, 1), 2.0, dtype=float)
            kp_arr = np.concatenate([kp_arr, vis], axis=1)  # -> (n_kpts, 3)
        kp_arr[:, 0] = np.clip(kp_arr[:, 0], 0.0, 1.0)
        kp_arr[:, 1] = np.clip(kp_arr[:, 1], 0.0, 1.0)

        bbox = " ".join(f"{v:.12f}" for v in _keypoints_bbox(kp_arr))
        return f"{cid} {bbox} {_flat_keypoints(kp_arr)}"

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
        if all(len(images) == 0 for images in annotation_splits.values()):
            return

        base = (
            output_path / f"{self.dataset_identifier}_part{part}"
            if part is not None
            else output_path / self.dataset_identifier
        )

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
            n_classes = len(self.class_names)
            kpt_shape = (
                self.kpt_shape if self.kpt_shape is not None else (0, 3)
            )

            yaml_obj: dict[str, Any] = {
                "train": path_to_posix(Path("images") / split_dirs["train"]),
                "val": path_to_posix(Path("images") / split_dirs["val"]),
                "test": path_to_posix(Path("images") / split_dirs["test"]),
                "nc": n_classes,
                "names": self.class_names,
                "kpt_shape": list(kpt_shape),
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


def _keypoints_bbox(kp_arr: np.ndarray) -> tuple[float, float, float, float]:
    """Return the ``(x_center, y_center, w, h)`` box around the keypoints.

    Only the visible keypoints count, unless none is visible. A collapsed
    box gets a tiny size.

    Returns:
        The box, with each value clipped to the range from 0 to 1.

    """
    visible_mask = kp_arr[:, 2] > 0
    pts = kp_arr[visible_mask][:, :2] if visible_mask.any() else kp_arr[:, :2]

    xmin, ymin = np.min(pts, axis=0)
    xmax, ymax = np.max(pts, axis=0)
    eps = 1e-6
    w = max(xmax - xmin, eps)
    h = max(ymax - ymin, eps)
    return (
        float(np.clip(xmin + w / 2.0, 0.0, 1.0)),
        float(np.clip(ymin + h / 2.0, 0.0, 1.0)),
        float(np.clip(w, 0.0, 1.0)),
        float(np.clip(h, 0.0, 1.0)),
    )


def _flat_keypoints(kp_arr: np.ndarray) -> str:
    """Format clipped ``(x, y, visibility)`` rows as one YOLO keypoint string."""
    parts = []
    for x, y, v in kp_arr.tolist():
        # YOLOv8 accepts integer vis (0/1/2).
        try:
            v_int = int(v)
        except (ValueError, OverflowError):
            v_int = 2
        parts.append(f"{x:.12f} {y:.12f} {v_int}")
    return " ".join(parts)
