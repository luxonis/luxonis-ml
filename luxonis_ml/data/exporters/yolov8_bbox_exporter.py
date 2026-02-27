from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeAlias, cast

from luxonis_ml.data.exporters.exporter_utils import (
    PreparedLDF,
    check_group_file_correspondence,
    exporter_specific_annotation_warning,
    split_of_group,
)
from luxonis_ml.enums import DatasetType

from .base_exporter import BaseExporter

BBox: TypeAlias = tuple[int, float, float, float, float]


class YoloV8Exporter(BaseExporter):
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

    def get_split_names(self) -> dict[str, str]:
        return {"train": "train", "val": "val", "test": "test"}

    def _yaml_filename(self) -> str:
        return "dataset.yaml"

    def supported_ann_types(self) -> list[str]:
        return list(
            DatasetType.YOLOV8BOUNDINGBOX.supported_annotation_formats
        )

    def export(self, prepared_ldf: PreparedLDF) -> None:
        check_group_file_correspondence(prepared_ldf)
        exporter_specific_annotation_warning(
            prepared_ldf, self.supported_ann_types()
        )

        annotation_splits: dict[str, dict[str, list[tuple]]] = {
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

            label_lines: list[tuple] = []

            for row in group_df.iter_rows(named=True):
                ttype = row["task_type"]
                ann_str = row["annotation"]
                cname = row["class_name"]

                if ann_str is None:
                    continue

                if ttype == "boundingbox":
                    if cname and cname not in self.class_to_id:
                        self.class_to_id[cname] = len(self.class_to_id)
                        self.class_names.append(cname)

                    if not cname or cname not in self.class_to_id:
                        continue

                    data = json.loads(ann_str)
                    x = float(data.get("x", 0.0))
                    y = float(data.get("y", 0.0))
                    w = float(data.get("w", 0.0))
                    h = float(data.get("h", 0.0))
                    cid = self.class_to_id[cname]
                    label_lines.append((cid, x, y, w, h))

            ann_size_estimate = len(label_lines) * 32
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

    def _maybe_roll_partition(
        self,
        annotation_splits: dict[str, dict[str, list[tuple]]],
        additional_size: int,
    ) -> dict[str, dict[str, list[tuple]]]:
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
            return {k: {} for k in self.get_split_names().values()}
        return annotation_splits

    def _dump_annotations(
        self,
        annotation_splits: dict[str, dict[str, list[tuple]]],
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
                formatted: list[str] = []
                for line in lines:
                    cid = line[0]
                    if len(line) == 5:
                        # bbox: convert top-left (x,y,w,h) -> center (xc,yc,w,h)
                        _, x, y, w, h = line
                        xc = x + w / 2.0
                        yc = y + h / 2.0
                        formatted.append(
                            f"{cid} {xc:.12f} {yc:.12f} {w:.12f} {h:.12f}"
                        )

                (labels_dir / f"{Path(img_name).stem}.txt").write_text(
                    "\n".join(formatted), encoding="utf-8"
                )

        yaml_filename = self._yaml_filename()
        if yaml_filename:
            split_dirs = self.get_split_names()
            yaml_obj = {
                "train": str(Path("images") / split_dirs["train"]),
                "val": str(Path("images") / split_dirs["val"]),
                "test": str(Path("images") / split_dirs["test"]),
                "nc": len(self.class_names),
                "names": self.class_names,
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
