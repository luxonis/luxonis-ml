from __future__ import annotations

import csv
import json
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, cast

import numpy as np
from PIL import Image
from pycocotools import mask as maskUtils  # <- use pycocotools

from luxonis_ml.data.exporters.base_exporter import BaseExporter
from luxonis_ml.data.exporters.exporter_utils import ExporterUtils, PreparedLDF


def _decode_rle_with_pycoco(ann: dict[str, Any]) -> np.ndarray:
    h = int(ann["height"])
    w = int(ann["width"])
    counts = ann["counts"]

    # pycocotools expects an RLE object with 'size' and 'counts'
    rle = {"size": [h, w], "counts": counts.encode("utf-8")}

    m = maskUtils.decode(rle)
    return np.array(m, dtype=np.uint8, order="C")


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

    def get_split_names(self) -> dict[str, str]:
        return {"train": "train", "val": "valid", "test": "test"}

    def _class_id_for(self, split: str, class_name: str) -> int:
        cmap = self.split_class_maps[split]
        if class_name not in cmap:
            cmap[class_name] = len(cmap) + 1
        return cmap[class_name]

    def _write_classes_csv(self, split: str, split_dir: Path) -> None:
        cmap = self.split_class_maps.get(split)
        if not cmap:
            return
        csv_path = split_dir / "_classes.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "name"])
            for name, cid in cmap.items():
                w.writerow([cid, name])

    def transform(self, prepared_ldf: PreparedLDF) -> None:
        ExporterUtils.check_group_file_correspondence(prepared_ldf)

        grouped = prepared_ldf.processed_df.group_by(
            ["file", "group_id"], maintain_order=True
        )

        copied_pairs: set[tuple[Path, str]] = set()

        for key, entry in grouped:
            file_name, group_id = cast(tuple[str, Any], key)
            file_path = Path(str(file_name))
            split = ExporterUtils.split_of_group(prepared_ldf, group_id)

            # Only segmentation rows (instance_id == -1)
            seg_rows = [
                row
                for row in entry.iter_rows(named=True)
                if row.get("task_type") == "segmentation"
                and row.get("instance_id") == -1
                and row.get("annotation")
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
                cname = str(row.get("class_name") or "")
                if not cname:
                    continue

                ann = row.get("annotation")
                ann = json.loads(ann)

                m = _decode_rle_with_pycoco(ann)  # uint8 {0,1}
                h, w = m.shape

                if combined is None:
                    combined = np.zeros((h, w), dtype=np.uint16)

                cid = self._class_id_for(split, cname)
                combined[m != 0] = cid

            if combined is not None:
                max_id = int(combined.max())
                if max_id <= 255:
                    out_arr = combined.astype(np.uint8, copy=False)
                    pil_mode = "L"
                else:
                    out_arr = combined
                    pil_mode = "I;16"

                dest_mask = split_dir / mask_name
                Image.fromarray(out_arr, mode=pil_mode).save(dest_mask)

        for split in ("train", "val", "test"):
            split_dir = self._get_data_path(self.output_path, split, self.part)
            if split_dir.exists():
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
