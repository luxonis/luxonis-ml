from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from luxonis_ml.data.exporters.exporter_utils import (
    PreparedLDF,
    check_group_file_correspondence,
    exporter_specific_annotation_warning,
    split_of_group,
)

from .base_exporter import BaseExporter


class YoloV8KeypointsExporter(BaseExporter):
    def __init__(
        self,
        dataset_identifier: str,
        output_path: Path,
        max_partition_size_gb: float | None,
    ):
        super().__init__(
            dataset_identifier, output_path, max_partition_size_gb
        )
        self.class_to_id: dict[str, int] = {}
        self.class_names: list[str] = []
        self.kpt_shape: tuple[int, int] | None = None  # (n_kpts, kpt_dim)

    def get_split_names(self) -> dict[str, str]:
        return {"train": "train", "val": "val", "test": "test"}

    def _yaml_filename(self) -> str:
        return "dataset.yaml"

    def supported_ann_types(self) -> list[str]:
        return ["keypoints"]

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
                ttype = row["task_type"]
                ann_str = row["annotation"]
                cname = row["class_name"]

                if ann_str is None or ttype != "keypoints":
                    continue

                if cname and cname not in self.class_to_id:
                    self.class_to_id[cname] = len(self.class_to_id)
                    self.class_names.append(cname)
                if not cname or cname not in self.class_to_id:
                    continue

                ann = json.loads(ann_str)
                keypoints = ann.get("keypoints", None)
                if not keypoints:
                    continue

                kp_arr = np.array(keypoints, dtype=float)
                if kp_arr.ndim != 2 or kp_arr.shape[1] not in (2, 3):
                    continue

                n_kpts, kpt_dim = kp_arr.shape
                if self.kpt_shape is None:
                    self.kpt_shape = (n_kpts, 3 if kpt_dim == 3 else 2)
                else:
                    prev_n, prev_dim = self.kpt_shape
                    n_use = max(prev_n, n_kpts)
                    d_use = 3 if (prev_dim == 3 or kpt_dim == 3) else 2
                    self.kpt_shape = (n_use, d_use)

                if kpt_dim == 2:
                    vis = np.full((n_kpts, 1), 2.0, dtype=float)
                    kp_arr = np.concatenate(
                        [kp_arr, vis], axis=1
                    )  # -> (n_kpts, 3)

                kp_arr[:, 0] = np.clip(kp_arr[:, 0], 0.0, 1.0)
                kp_arr[:, 1] = np.clip(kp_arr[:, 1], 0.0, 1.0)

                # compute bbox from visible points (v>0), else all points
                visible_mask = kp_arr[:, 2] > 0
                pts = (
                    kp_arr[visible_mask][:, :2]
                    if visible_mask.any()
                    else kp_arr[:, :2]
                )

                if pts.size == 0:
                    # degenerate; skip instance
                    continue

                xmin, ymin = np.min(pts, axis=0)
                xmax, ymax = np.max(pts, axis=0)
                # ensure tiny box if collapsed
                eps = 1e-6
                w = max(xmax - xmin, eps)
                h = max(ymax - ymin, eps)
                x_center = xmin + w / 2.0
                y_center = ymin + h / 2.0

                # clamp bbox
                x_center = float(np.clip(x_center, 0.0, 1.0))
                y_center = float(np.clip(y_center, 0.0, 1.0))
                w = float(np.clip(w, 0.0, 1.0))
                h = float(np.clip(h, 0.0, 1.0))

                cid = self.class_to_id[cname]

                flat_kps: list[str] = []
                for x, y, v in kp_arr.tolist():
                    x_ = 0.0 if x < 0 else 1.0 if x > 1 else x
                    y_ = 0.0 if y < 0 else 1.0 if y > 1 else y
                    flat_kps.append(f"{x_:.12f}")
                    flat_kps.append(f"{y_:.12f}")
                    # YOLOv8 accepts integer vis (0/1/2). Cast safely.
                    try:
                        v_int = int(v)
                    except Exception:
                        v_int = 2
                    flat_kps.append(f"{v_int}")

                line = (
                    f"{cid} "
                    f"{x_center:.12f} {y_center:.12f} {w:.12f} {h:.12f} "
                    + " ".join(flat_kps)
                )
                label_lines.append(line)

            annotation_splits[split][new_name] = label_lines

            ann_size_estimate = sum(len(s) + 1 for s in label_lines)
            img_size = file_path.stat().st_size
            annotation_splits = self._maybe_roll_partition(
                annotation_splits, ann_size_estimate + img_size
            )

            data_path = self._get_data_path(self.output_path, split, self.part)
            data_path.mkdir(parents=True, exist_ok=True)
            dest = data_path / new_name
            if file_path not in copied_files:
                copied_files.add(file_path)
                if dest != file_path:
                    dest.write_bytes(file_path.read_bytes())
                self.current_size += img_size

        self._dump_annotations(annotation_splits, self.output_path, self.part)

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
                "train": str(Path("images") / split_dirs["train"]),
                "val": str(Path("images") / split_dirs["val"]),
                "test": str(Path("images") / split_dirs["test"]),
                "nc": n_classes,
                "names": self.class_names,
                "kpt_shape": list(kpt_shape),
            }

            (base / yaml_filename).write_text(
                self._to_yaml(yaml_obj), encoding="utf-8"
            )

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
