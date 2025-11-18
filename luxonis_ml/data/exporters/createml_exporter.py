from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from PIL import Image

from luxonis_ml.data.exporters.base_exporter import BaseExporter
from luxonis_ml.data.exporters.exporter_utils import (
    PreparedLDF,
    check_group_file_correspondence,
    exporter_specific_annotation_warning,
    split_of_group,
)


class CreateMLExporter(BaseExporter):
    def __init__(
        self,
        dataset_identifier: str,
        output_path: Path,
        max_partition_size_gb: float | None,
    ):
        super().__init__(
            dataset_identifier, output_path, max_partition_size_gb
        )

    @staticmethod
    def get_split_names() -> dict[str, str]:
        return {"train": "train", "val": "valid", "test": "test"}

    def supported_ann_types(self) -> list[str]:
        return ["boundingbox"]

    def export(self, prepared_ldf: PreparedLDF) -> None:
        check_group_file_correspondence(prepared_ldf)
        exporter_specific_annotation_warning(
            prepared_ldf, self.supported_ann_types()
        )

        anns_by_split: dict[str, dict[str, list[dict[str, Any]]]] = {
            v: {} for v in self.get_split_names().values()
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

            per_image_anns = self._collect_bounding_box_annotations(
                group_df=group_df, width=width, height=height
            )

            ann_size_est = self._estimate_annotation_bytes(
                new_name, per_image_anns
            )
            img_size = file_path.stat().st_size

            anns_by_split = self._maybe_roll_partition(
                anns_by_split, ann_size_est + img_size
            )

            anns_by_split[split_name][new_name] = per_image_anns

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

        self._dump_annotations(anns_by_split, self.output_path, self.part)

    @staticmethod
    def _estimate_annotation_bytes(
        img_name: str, anns: list[dict[str, Any]]
    ) -> int:
        payload = {"image": img_name, "annotations": anns}
        return len(
            (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        )

    def _collect_bounding_box_annotations(
        self,
        group_df: Any,
        width: int,
        height: int,
    ) -> list[dict[str, Any]]:
        per_image_anns: list[dict[str, Any]] = []

        for row in group_df.iter_rows(named=True):
            ttype = row["task_type"]
            ann_str = row["annotation"]
            cname = row["class_name"]

            if ttype != "boundingbox" or ann_str is None or not cname:
                continue

            data = json.loads(ann_str)
            x_tl = float(data.get("x", 0.0))
            y_tl = float(data.get("y", 0.0))
            w = float(data.get("w", 0.0))
            h = float(data.get("h", 0.0))

            x_px = x_tl * width
            y_px = y_tl * height
            w_px = w * width
            h_px = h * height
            cx_px = x_px + w_px / 2.0
            cy_px = y_px + h_px / 2.0

            per_image_anns.append(
                {
                    "label": cname,
                    "coordinates": {
                        "x": cx_px,
                        "y": cy_px,
                        "width": w_px,
                        "height": h_px,
                    },
                }
            )

        return per_image_anns

    def _maybe_roll_partition(
        self,
        anns_by_split: dict[str, dict[str, list[dict[str, Any]]]],
        additional_size: int,
    ) -> dict[str, dict[str, list[dict[str, Any]]]]:
        if (
            self.max_partition_size
            and self.part is not None
            and (self.current_size + additional_size) > self.max_partition_size
        ):
            self._dump_annotations(anns_by_split, self.output_path, self.part)
            self.current_size = 0
            self.part += 1
            return {v: {} for v in self.get_split_names().values()}
        return anns_by_split

    def _dump_annotations(
        self,
        anns_by_split: dict[str, dict[str, list[dict[str, Any]]]],
        output_path: Path,
        part: int | None = None,
    ) -> None:
        if not anns_by_split:
            return

        base = (
            output_path / f"{self.dataset_identifier}_part{part}"
            if part is not None
            else output_path / self.dataset_identifier
        )
        base.mkdir(parents=True, exist_ok=True)

        for split_name, per_image in anns_by_split.items():
            split_dir = base / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

            out_list: list[dict[str, Any]] = []
            for img_name, anns in per_image.items():
                out_list.append({"image": img_name, "annotations": anns})

            (split_dir / "_annotations.createml.json").write_text(
                json.dumps(out_list, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def _get_data_path(
        self, output_path: Path, split_name: str, part: int | None = None
    ) -> Path:
        base = (
            output_path / f"{self.dataset_identifier}_part{part}"
            if part is not None
            else output_path / self.dataset_identifier
        )
        return base / split_name
