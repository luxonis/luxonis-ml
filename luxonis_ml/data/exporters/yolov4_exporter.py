from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeAlias, cast

from PIL import Image

from luxonis_ml.data.exporters.exporter_utils import (
    PreparedLDF,
    check_group_file_correspondence,
    exporter_specific_annotation_warning,
    split_of_group,
)

from .base_exporter import BaseExporter

BBox: TypeAlias = tuple[int, float, float, float, float]


class YoloV4Exporter(BaseExporter):
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

    # v4 uses "valid"
    def get_split_names(self) -> dict[str, str]:
        return {"train": "train", "val": "valid", "test": "test"}

    # no YAML for v4
    def _yaml_filename(self) -> str:
        return ""

    def supported_ann_types(self) -> list[str]:
        return ["boundingbox"]

    def export(self, prepared_ldf: PreparedLDF) -> None:
        check_group_file_correspondence(prepared_ldf)
        exporter_specific_annotation_warning(
            prepared_ldf, self.supported_ann_types()
        )

        annotation_splits: dict[str, dict[str, list[BBox]]] = {
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

            label_lines: list[BBox] = []

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

                # v4 ignores instance segmentation and keypoints entirely

            annotation_splits[split][new_name] = label_lines

            ann_size_estimate = len(label_lines) * 32
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
        annotation_splits: dict[str, dict[str, list[BBox]]],
        additional_size: int,
    ) -> dict[str, dict[str, list[BBox]]]:
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
        annotation_splits: dict[str, dict[str, list[BBox]]],
        output_path: Path,
        part: int | None = None,
    ) -> None:
        # v4 has its own flat format (no images/labels split dirs)
        self._dump_annotations_v4(annotation_splits, output_path, part)

    def _dump_annotations_v4(
        self,
        annotation_splits: dict[str, dict[str, list[BBox]]],
        output_path: Path,
        part: int | None = None,
    ) -> None:
        base_root = (
            output_path / f"{self.dataset_identifier}_part{part}"
            if part is not None
            else output_path / self.dataset_identifier
        )

        for split_name in self.get_split_names().values():
            split_dir = base_root / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

            (split_dir / "_classes.txt").write_text(
                "\n".join(self.class_names)
                + ("\n" if self.class_names else ""),
                encoding="utf-8",
            )

            lines_out: list[str] = []
            for img_name, lines in annotation_splits.get(
                split_name, {}
            ).items():
                img_path = split_dir / img_name
                with Image.open(img_path) as im:
                    width, height = im.size

                anns: list[str] = []
                for cid, x, y, w, h in lines:
                    x_min = x * width
                    y_min = y * height
                    x_max = (x + w) * width
                    y_max = (y + h) * height
                    anns.append(f"{x_min},{y_min},{x_max},{y_max},{cid}")

                lines_out.append(f"{img_name} {' '.join(anns)}")

            (split_dir / "_annotations.txt").write_text(
                "\n".join(lines_out) + ("\n" if lines_out else ""),
                encoding="utf-8",
            )

    def _get_data_path(
        self, output_path: Path, split: str, part: int | None = None
    ) -> Path:
        base = (
            output_path / f"{self.dataset_identifier}_part{part}"
            if part is not None
            else output_path / self.dataset_identifier
        )
        # v4 puts images directly under the split folder
        return base / split
