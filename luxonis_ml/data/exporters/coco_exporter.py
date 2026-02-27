from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, cast

import pycocotools.mask as maskUtils
from loguru import logger
from PIL import Image

from luxonis_ml.data.exporters.base_exporter import BaseExporter
from luxonis_ml.data.exporters.exporter_utils import (
    PreparedLDF,
    check_group_file_correspondence,
    exporter_specific_annotation_warning,
    get_single_skeleton,
    split_of_group,
)
from luxonis_ml.data.utils import COCOFormat
from luxonis_ml.enums import DatasetType


class CocoExporter(BaseExporter):
    """Exporter for COCO dataset format (Roboflow/FiftyOne variants)."""

    def __init__(
        self,
        dataset_identifier: str,
        output_path: Path,
        max_partition_size_gb: float | None,
        format: COCOFormat = COCOFormat.ROBOFLOW,
        *,
        skeletons: dict[str, Any] | None = None,
    ):
        super().__init__(
            dataset_identifier, output_path, max_partition_size_gb
        )
        self.format = format
        self.skeletons = skeletons
        if self.skeletons is None:
            self.allow_keypoints = False
        elif len(self.skeletons) == 1:
            self.allow_keypoints = True
        else:
            self.allow_keypoints = False
            logger.warning(
                "Skipping keypoint annotations because COCO only supports a single keypoint export class."
                "To export multiple keypoint classes please use the Luxonis native export format"
            )

        splits = self.get_split_names()
        self.class_name_to_category_id: dict[str, dict[str, int]] = {
            s: {} for s in splits
        }
        self.last_category_id: dict[str, int] = dict.fromkeys(splits, 1)
        self.image_registry: dict[str, dict[str, dict[str, Any]]] = {
            s: {} for s in splits
        }

    def get_split_names(self) -> dict[str, str]:
        if self.format == COCOFormat.ROBOFLOW:
            return {"train": "train", "val": "valid", "test": "test"}
        return {"train": "train", "val": "validation", "test": "test"}

    def supported_ann_types(self) -> list[str]:
        return list(DatasetType.COCO.supported_annotation_formats)

    def export(self, prepared_ldf: PreparedLDF) -> None:
        check_group_file_correspondence(prepared_ldf)
        exporter_specific_annotation_warning(
            prepared_ldf, self.supported_ann_types()
        )

        splits = self.get_split_names()
        annotation_splits: dict[str, dict[str, Any]] = {
            s: {"images": [], "categories": [], "annotations": []}
            for s in splits
        }
        ann_id_counter: dict[str, int] = dict.fromkeys(splits, 1)

        grouped = prepared_ldf.processed_df.group_by(
            ["file", "instance_id", "group_id"], maintain_order=True
        )
        copied_files: set[Path] = set()

        for key, entry in grouped:
            file_name, _instance_id, group_id = cast(tuple[str, int, Any], key)

            split = split_of_group(prepared_ldf, group_id)
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
            has_valid_ann = False

            for row in entry.iter_rows(named=True):
                ann, row_has_ann = self._process_row(
                    row, split, annotation_splits, ann, width, height
                )
                has_valid_ann = has_valid_ann or row_has_ann

            img_size = file_path.stat().st_size
            if has_valid_ann:
                ann_size = sys.getsizeof(ann)
                annotation_splits = self._maybe_roll_partition(
                    annotation_splits, ann_size + img_size
                )

                annotation_splits[split]["annotations"].append(ann)
                ann_id_counter[split] += 1

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
    ) -> tuple[dict[str, Any], bool]:
        ttype = row["task_type"]
        ann_str = row["annotation"]
        cname = row["class_name"]

        if cname and cname not in self.class_name_to_category_id[split]:
            cid = self.last_category_id[split]

            cat_entry = {"id": cid, "name": cname}

            if self.allow_keypoints:
                kp_labels, kp_skeleton = get_single_skeleton(
                    self.allow_keypoints, self.skeletons
                )
                if kp_labels:
                    cat_entry["keypoints"] = kp_labels
                    cat_entry["skeleton"] = kp_skeleton

            annotation_splits[split]["categories"].append(cat_entry)
            self.class_name_to_category_id[split][cname] = cid
            self.last_category_id[split] += 1

        if ttype == "classification" and cname:
            ann["category_id"] = self.class_name_to_category_id[split][cname]
            return ann, True

        if ann_str is None:
            return ann, False

        if ttype == "boundingbox":
            data = json.loads(ann_str)
            self._fill_bbox(ann, data, w, h, split, cname)
            return ann, True

        if ttype == "instance_segmentation":
            data = json.loads(ann_str)
            self._fill_instance_segmentation(ann, data, split, cname)
            return ann, True

        if ttype == "keypoints" and self.allow_keypoints:
            data = json.loads(ann_str)
            self._fill_keypoints(ann, data, w, h, split, cname)
            return ann, True

        return ann, False

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

    def _fill_instance_segmentation(
        self,
        ann: dict[str, Any],
        data: dict[str, Any],
        split: str,
        cname: str,
    ) -> None:
        ann["category_id"] = (
            ann.get("category_id")
            or self.class_name_to_category_id[split][cname]
        )
        H = int(data["height"])
        W = int(data["width"])
        counts = data["counts"]
        ann["segmentation"] = {"size": [H, W], "counts": counts}
        ann["iscrowd"] = 0 if ann.get("iscrowd") is None else ann["iscrowd"]

        if counts is not None:
            rle_runtime = {
                "size": [H, W],
                "counts": counts.encode("utf-8"),
            }
            area = float(maskUtils.area(rle_runtime))  # type: ignore[arg-type]
            bbox = maskUtils.toBbox(rle_runtime).tolist()  # type: ignore[arg-type]
            ann["area"] = area
            if not ann.get("bbox"):
                ann["bbox"] = bbox
            return

        ann["area"] = H * W
        ann["bbox"] = [0.0, 0.0, float(W), float(H)]

    def _fill_keypoints(
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

        raw_kps = data.get("keypoints", [])
        kps_px: list[float] = []
        xs: list[float] = []
        ys: list[float] = []
        visible_count = 0

        for triplet in raw_kps:
            x_norm, y_norm, v = triplet
            x_px = float(x_norm) * float(w)
            y_px = float(y_norm) * float(h)

            kps_px.extend([x_px, y_px, int(v)])

            if int(v) > 0:
                xs.append(x_px)
                ys.append(y_px)
                visible_count += 1

        ann["keypoints"] = kps_px
        ann["num_keypoints"] = visible_count

        if visible_count > 0:
            x_min = float(min(xs))
            y_min = float(min(ys))
            x_max = float(max(xs))
            y_max = float(max(ys))
            bw = max(0.0, x_max - x_min)
            bh = max(0.0, y_max - y_min)
            if not ann.get("bbox"):
                ann["bbox"] = [x_min, y_min, bw, bh]
            ann["area"] = bw * bh
        else:
            ann.setdefault("bbox", [0.0, 0.0, 0.0, 0.0])
            ann.setdefault("area", 0.0)

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
