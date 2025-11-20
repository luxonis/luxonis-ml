from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from luxonis_ml.data.exporters.base_exporter import BaseExporter
from luxonis_ml.data.exporters.exporter_utils import (
    PreparedLDF,
    check_group_file_correspondence,
    exporter_specific_annotation_warning,
    split_of_group,
)


class DarknetExporter(BaseExporter):
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

        labels_by_split: dict[str, dict[str, list[str]]] = {
            v: {} for v in self.get_split_names().values()
        }

        df = prepared_ldf.processed_df
        grouped = df.group_by(["file", "group_id"], maintain_order=True)

        copied_files: set[Path] = set()

        for key, group_df in grouped:
            file_name, group_id = cast(tuple[str, Any], key)
            split_key = split_of_group(prepared_ldf, group_id)
            split_name = self.get_split_names()[split_key]

            file_path = Path(str(file_name))
            idx = self.image_indices.setdefault(
                file_path, len(self.image_indices)
            )
            new_name = f"{idx}{file_path.suffix}"
            new_stem = Path(new_name).stem

            label_lines = self._collect_darknet_bounding_box_labels(group_df)

            ann_size = sum(len(line) for line in label_lines)
            img_size = file_path.stat().st_size
            labels_by_split = self._maybe_roll_partition(
                labels_by_split, ann_size + img_size
            )

            labels_by_split[split_name][new_stem] = label_lines

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

        self._dump_annotations(labels_by_split, self.output_path, self.part)

    def _collect_darknet_bounding_box_labels(
        self,
        group_df: Any,
    ) -> list[str]:
        label_lines: list[str] = []

        for row in group_df.iter_rows(named=True):
            ttype = row["task_type"]
            ann_str = row["annotation"]
            cname = row["class_name"]

            if ttype != "boundingbox" or ann_str is None:
                continue

            # Register class if new
            if cname and cname not in self.class_to_id:
                self.class_to_id[cname] = len(self.class_to_id)
                self.class_names.append(cname)

            if not cname or cname not in self.class_to_id:
                continue

            data = json.loads(ann_str)
            x_tl = float(data.get("x", 0.0))
            y_tl = float(data.get("y", 0.0))
            w = float(data.get("w", 0.0))
            h = float(data.get("h", 0.0))

            cx = x_tl + w / 2.0
            cy = y_tl + h / 2.0

            cid = self.class_to_id[cname]
            label_lines.append(f"{cid} {cx:.12f} {cy:.12f} {w:.12f} {h:.12f}")

        return label_lines

    def _maybe_roll_partition(
        self,
        labels_by_split: dict[str, dict[str, list[str]]],
        additional_size: int,
    ) -> dict[str, dict[str, list[str]]]:
        if (
            self.max_partition_size
            and self.part is not None
            and (self.current_size + additional_size) > self.max_partition_size
        ):
            self._dump_annotations(
                labels_by_split, self.output_path, self.part
            )
            self.current_size = 0
            self.part += 1
            return {v: {} for v in self.get_split_names().values()}
        return labels_by_split

    def _dump_annotations(
        self,
        labels_by_split: dict[str, dict[str, list[str]]],
        output_path: Path,
        part: int | None = None,
    ) -> None:
        if all(len(imgs) == 0 for imgs in labels_by_split.values()):
            return

        base = (
            output_path / f"{self.dataset_identifier}_part{part}"
            if part is not None
            else output_path / self.dataset_identifier
        )
        base.mkdir(parents=True, exist_ok=True)

        for split_name, per_image in labels_by_split.items():
            split_dir = base / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

            for stem, lines in per_image.items():
                (split_dir / f"{stem}.txt").write_text(
                    "\n".join(lines) + ("\n" if lines else ""),
                    encoding="utf-8",
                )

            (split_dir / "_darknet.labels").write_text(
                "\n".join(self.class_names)
                + ("\n" if self.class_names else ""),
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
