from __future__ import annotations

import csv
import json
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, cast

import numpy as np
from PIL import Image

from luxonis_ml.data.exporters.base_exporter import BaseExporter
from luxonis_ml.data.exporters.exporter_utils import (
    PreparedLDF,
    check_group_file_correspondence,
    decode_rle_with_pycoco,
    exporter_specific_annotation_warning,
    split_of_group,
)
from luxonis_ml.enums import DatasetType


class SegmentationMaskDirectoryExporter(BaseExporter):
    def __init__(
        self,
        dataset_identifier: str,
        output_path: Path,
        max_partition_size_gb: float | None,
    ):
        super().__init__(
            dataset_identifier, output_path, max_partition_size_gb
        )
        self.split_class_maps: dict[str, OrderedDict[str, int]] = defaultdict(
            OrderedDict
        )
        self.BACKGROUND_NAME = " background"
        self.CLASS_COL = " Class"
        self.ID_COL = "id"

    def get_split_names(self) -> dict[str, str]:
        return {"train": "train", "val": "valid", "test": "test"}

    def supported_ann_types(self) -> list[str]:
        return list(DatasetType.SEGMASK.supported_annotation_formats)

    def _ensure_background(self, split: str) -> None:
        cmap = self.split_class_maps[split]
        if self.BACKGROUND_NAME not in cmap:
            # Insert background first so insertion order aligns with index = id
            cmap.clear()
            cmap[self.BACKGROUND_NAME] = 0

    def _class_id_for(self, split: str, class_name: str) -> int:
        self._ensure_background(split)
        cmap = self.split_class_maps[split]
        if class_name not in cmap:
            cmap[class_name] = len(cmap)
        return cmap[class_name]

    def _write_classes_csv(self, split: str, split_dir: Path) -> None:
        self._ensure_background(split)

        cmap = self.split_class_maps[split]

        # Ensure directory exists
        csv_path = split_dir / "_classes.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        items_by_id = sorted(cmap.items(), key=lambda kv: kv[1])

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([self.ID_COL, self.CLASS_COL])
            for name, cid in items_by_id:
                w.writerow([cid, name])

    def export(self, prepared_ldf: PreparedLDF) -> None:
        check_group_file_correspondence(prepared_ldf)
        exporter_specific_annotation_warning(
            prepared_ldf, self.supported_ann_types()
        )

        grouped = prepared_ldf.processed_df.group_by(
            ["file", "group_id"], maintain_order=True
        )

        copied_pairs: set[tuple[Path, str]] = set()

        for split in ("train", "val", "test"):
            split_dir = self._get_data_path(self.output_path, split, self.part)
            split_dir.mkdir(parents=True, exist_ok=True)

        for key, entry in grouped:
            file_name, group_id = cast(tuple[str, Any], key)
            file_path = Path(str(file_name))
            split = split_of_group(prepared_ldf, group_id)

            # Ensure background exists for this split up-front
            self._ensure_background(split)

            # Only semantic segmentation rows for the entire image (instance_id == -1)
            seg_rows = [
                row
                for row in entry.iter_rows(named=True)
                if row["task_type"] == "segmentation"
                and row["instance_id"] == -1
                and row["annotation"]
            ]
            if not seg_rows:
                continue

            idx = self.image_indices.setdefault(
                file_path, len(self.image_indices)
            )
            new_name = f"{idx}{file_path.suffix}"
            new_stem = Path(new_name).stem
            mask_name = f"{new_stem}_mask.png"

            split_dir = self._get_data_path(self.output_path, split, self.part)
            split_dir.mkdir(parents=True, exist_ok=True)

            dest_img = split_dir / new_name
            pair_key = (file_path, str(dest_img))
            if pair_key not in copied_pairs:
                if dest_img != file_path:
                    dest_img.write_bytes(file_path.read_bytes())
                copied_pairs.add(pair_key)

            combined: np.ndarray | None = None

            for row in seg_rows:
                cname = str(row["class_name"] or "")
                if not cname:
                    continue

                ann = json.loads(row["annotation"])
                m = decode_rle_with_pycoco(ann)
                h, w = m.shape

                if combined is None:
                    # Start with background (0) everywhere
                    combined = np.zeros((h, w), dtype=np.uint16)

                cid = self._class_id_for(split, cname)
                combined[m != 0] = cid

            if combined is not None:
                max_id = int(combined.max())
                if max_id > 255:
                    raise ValueError(
                        f"SegmentationMaskDirectoryExporter: class id {max_id} exceeds 255; "
                        f"the downstream parser expects 8-bit masks (cv2.IMREAD_GRAYSCALE). "
                        f"Reduce the number of classes or adjust the parser."
                    )

                # Always write 8-bit L to match parser behavior
                out_arr = combined.astype(np.uint8, copy=False)
                dest_mask = split_dir / mask_name
                Image.fromarray(out_arr, mode="L").save(dest_mask)

        for split in ("train", "val", "test"):
            split_dir = self._get_data_path(self.output_path, split, self.part)
            if split_dir.exists():
                # make sure background exists even if no classes were encountered
                self._ensure_background(split)
                self._write_classes_csv(split, split_dir)

    def _dump_annotations(
        self,
        annotation_splits: dict[str, dict[str, Any]],
        output_path: Path,
        part: int | None = None,
    ) -> None:
        return

    def _get_data_path(
        self, output_path: Path, split: str, part: int | None = None
    ) -> Path:
        split_name = self.get_split_names().get(split, split)
        base = (
            output_path / f"{self.dataset_identifier}_part{part}"
            if part is not None
            else output_path / self.dataset_identifier
        )
        return base / split_name
