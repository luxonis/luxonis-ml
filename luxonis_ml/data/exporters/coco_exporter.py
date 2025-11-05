from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, cast

from PIL import Image

from luxonis_ml.data.exporters.base_exporter import BaseExporter
from luxonis_ml.data.exporters.exporter_utils import ExporterUtils, PreparedLDF
from luxonis_ml.data.utils import COCOFormat


class CocoExporter(BaseExporter):
    """Exporter for COCO dataset format (Roboflow/FiftyOne variants)."""

    def __init__(
        self,
        dataset_identifier: str,
        output_path: Path,
        max_partition_size_gb: float,
        format: COCOFormat = COCOFormat.ROBOFLOW,
    ):
        super().__init__(
            dataset_identifier, output_path, max_partition_size_gb
        )
        self.format = format
        splits = self.get_split_names()
        self.class_name_to_category_id: dict[str, dict[str, int]] = {
            s: {} for s in splits
        }
        self.last_category_id: dict[str, int] = {s: 1 for s in splits}
        self.image_registry: dict[str, dict[str, dict[str, Any]]] = {
            s: {} for s in splits
        }

    def get_split_names(self) -> dict[str, str]:
        if self.format == COCOFormat.ROBOFLOW:
            return {"train": "train", "val": "valid", "test": "test"}
        return {"train": "train", "val": "validation", "test": "test"}

    def transform(self, prepared_ldf: PreparedLDF) -> None:
        splits = self.get_split_names()
        annotation_splits: dict[str, dict[str, Any]] = {
            s: {"images": [], "categories": [], "annotations": []}
            for s in splits
        }
        ann_id_counter: dict[str, int] = {s: 1 for s in splits}
        ExporterUtils.check_group_file_correspondence(prepared_ldf)

        grouped = prepared_ldf.processed_df.group_by(
            ["file", "instance_id", "group_id"], maintain_order=True
        )
        copied_files: set[Path] = set()

        for key, entry in grouped:
            file_name, instance_id, group_id = cast(tuple[str, int, Any], key)

            split = ExporterUtils.split_of_group(prepared_ldf, group_id)
            file_path = Path(str(file_name))

            image_id, width, height, new_name = self._get_or_register_image(
                file_path, split, annotation_splits
            )

            ann = {
                "id": ann_id_counter[split],
                "image_id": image_id,
                "category_id": None,
                "bbox": [],
                "area": 0,
                "iscrowd": 0,
            }

            ann = [
                self._process_row(
                    row, split, annotation_splits, ann, width, height
                )
                for row in entry.iter_rows(named=True)
            ]

            annotation_splits[split]["annotations"].append(ann)
            ann_id_counter[split] += 1

            ann_size = sys.getsizeof(ann)
            img_size = file_path.stat().st_size
            annotation_splits = self._maybe_roll_partition(
                annotation_splits, ann_size + img_size
            )

            data_path = self._get_data_path(self.output_path, split, self.part)
            data_path.mkdir(parents=True, exist_ok=True)
            if file_path not in copied_files:
                copied_files.add(file_path)
                dest = data_path / new_name
                if dest != file_path:
                    dest.write_bytes(file_path.read_bytes())
                self.current_size += img_size

        self._dump_annotations(annotation_splits, self.output_path, self.part)

    def _maybe_roll_partition(
        self,
        annotation_splits: dict[str, dict[str, Any]],
        additional_size: int,
    ) -> dict[str, dict[str, Any]]:
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
            return {
                s: {"images": [], "categories": [], "annotations": []}
                for s in self.get_split_names()
            }
        return annotation_splits

    def _get_or_register_image(
        self,
        path: Path,
        split: str,
        annotation_splits: dict[str, dict[str, Any]],
    ) -> tuple[int, int, int, str]:
        idx = self.image_indices.setdefault(path, len(self.image_indices))
        new_name = f"{idx}{path.suffix}"
        if str(path) in self.image_registry[split]:
            img_data = self.image_registry[split][str(path)]
            return (
                img_data["id"],
                img_data["width"],
                img_data["height"],
                new_name,
            )

        width, height = Image.open(path).size
        image_id = len(annotation_splits[split]["images"]) + 1
        img_entry = {
            "id": image_id,
            "file_name": new_name,
            "height": height,
            "width": width,
        }
        annotation_splits[split]["images"].append(img_entry)
        self.image_registry[split][str(path)] = img_entry
        return image_id, width, height, new_name

    def _process_row(
        self,
        row: dict[str, Any],
        split: str,
        annotation_splits: dict[str, dict[str, Any]],
        ann: dict[str, Any],
        w: int,
        h: int,
    ) -> dict[str, Any]:
        ttype = row["task_type"]
        ann_str = row["annotation"]
        cname = row["class_name"]

        if cname and cname not in self.class_name_to_category_id[split]:
            cid = self.last_category_id[split]
            annotation_splits[split]["categories"].append(
                {"id": cid, "name": cname}
            )
            self.class_name_to_category_id[split][cname] = cid
            self.last_category_id[split] += 1

        if ttype == "classification" and cname:
            ann["category_id"] = self.class_name_to_category_id[split][cname]
            return ann

        if ann_str is None or ttype not in {"boundingbox"}:
            return ann

        data = json.loads(ann_str)

        if ttype == "boundingbox":
            self._fill_bbox(ann, data, w, h, split, cname)

        return ann

    def _fill_bbox(
        self,
        ann: dict[str, Any],
        data: dict[str, Any],
        w: int,
        h: int,
        split: str,
        cname: str,
    ) -> None:
        ann["category_id"] = (
            ann.get("category_id")
            or self.class_name_to_category_id[split][cname]
        )
        x, y, bw, bh = (
            data.get("x", 0.0),
            data.get("y", 0.0),
            data.get("w", 0.0),
            data.get("h", 0.0),
        )
        px, py, pw, ph = x * w, y * h, bw * w, bh * h
        ann["bbox"] = [px, py, pw, ph]
        ann["area"] = pw * ph

    def _dump_annotations(
        self,
        annotation_splits: dict[str, dict[str, Any]],
        output_path: Path,
        part: int | None = None,
    ) -> None:
        for split_key, split_data in annotation_splits.items():
            split_name = self.get_split_names().get(split_key, split_key)
            base = (
                output_path / f"{self.dataset_identifier}_part{part}"
                if part is not None
                else output_path / self.dataset_identifier
            )
            out_dir = base / split_name
            if self.format == COCOFormat.FIFTYONE:
                out_dir = out_dir / "labels"
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "_annotations.coco.json").write_text(
                json.dumps(split_data, indent=2), encoding="utf-8"
            )

    def _get_data_path(
        self, output_path: Path, split: str, part: int | None = None
    ) -> Path:
        split_name = self.get_split_names().get(split, split)
        base = (
            output_path / f"{self.dataset_identifier}_part{part}"
            if part is not None
            else output_path / self.dataset_identifier
        )
        data_path = base / split_name
        if self.format == COCOFormat.FIFTYONE:
            data_path = data_path / "data"
        return data_path
