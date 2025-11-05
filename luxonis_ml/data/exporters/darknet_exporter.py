from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from luxonis_ml.data.exporters.base_exporter import BaseExporter
from luxonis_ml.data.exporters.exporter_utils import ExporterUtils, PreparedLDF


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
        return {"train": "train", "val": "val", "test": "test"}

    def transform(self, prepared_ldf: PreparedLDF) -> None:
        ExporterUtils.check_group_file_correspondence(prepared_ldf)

        annotation_splits: dict[str, dict[str, list[str]]] = {
            k: {} for k in self.get_split_names()
        }
        split_image_lists: dict[str, list[str]] = {
            k: [] for k in self.get_split_names()
        }

        df = prepared_ldf.processed_df
        grouped = df.group_by(["file", "group_id"], maintain_order=True)

        copied_files: set[Path] = set()

        for key, group_df in grouped:
            file_name, group_id = cast(tuple[str, Any], key)
            split = ExporterUtils.split_of_group(prepared_ldf, group_id)

            file_path = Path(str(file_name))
            idx = self.image_indices.setdefault(
                file_path, len(self.image_indices)
            )
            new_name = f"{idx}{file_path.suffix}"
            new_stem = Path(new_name).stem

            label_lines: list[str] = []
            for row in group_df.iter_rows(named=True):
                ttype = row.get("task_type")
                ann_str = row.get("annotation")
                cname = row.get("class_name")

                # Only bounding boxes are supported by Darknet
                if ttype != "boundingbox" or ann_str is None:
                    continue

                if cname and cname not in self.class_to_id:
                    self.class_to_id[cname] = len(self.class_to_id)
                    self.class_names.append(cname)

                if not cname or cname not in self.class_to_id:
                    continue

                data = json.loads(ann_str)
                cx = float(data.get("x", 0.0))
                cy = float(data.get("y", 0.0))
                w = float(data.get("w", 0.0))
                h = float(data.get("h", 0.0))

                cid = self.class_to_id[cname]
                label_lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            annotation_splits[split][new_stem] = label_lines

            ann_size = sum(len(l_line) for l_line in label_lines)
            img_size = file_path.stat().st_size
            annotation_splits, split_image_lists = self._maybe_roll_partition(
                annotation_splits, split_image_lists, ann_size + img_size
            )

            data_path = self._get_data_path(self.output_path, split, self.part)
            data_path.mkdir(parents=True, exist_ok=True)
            dest = data_path / new_name

            if file_path not in copied_files:
                copied_files.add(file_path)
                if dest != file_path:
                    dest.write_bytes(file_path.read_bytes())
                self.current_size += img_size

            rel_img_path = str(Path("images") / split / new_name)
            split_image_lists[split].append(rel_img_path)

        self._dump_annotations(
            {
                "labels": annotation_splits,
                "lists": split_image_lists,
                "classes": self.class_names,
            },
            self.output_path,
            self.part,
        )

    def _maybe_roll_partition(
        self,
        annotation_splits: dict[str, dict[str, list[str]]],
        split_image_lists: dict[str, list[str]],
        additional_size: int,
    ) -> tuple[dict[str, dict[str, list[str]]], dict[str, list[str]]]:
        if (
            self.max_partition_size
            and self.part is not None
            and (self.current_size + additional_size) > self.max_partition_size
        ):
            self._dump_annotations(
                {
                    "labels": annotation_splits,
                    "lists": split_image_lists,
                    "classes": self.class_names,
                },
                self.output_path,
                self.part,
            )
            self.current_size = 0
            self.part += 1
            fresh_labels = {k: {} for k in self.get_split_names()}
            fresh_lists = {k: [] for k in self.get_split_names()}
            return fresh_labels, fresh_lists
        return annotation_splits, split_image_lists

    def _dump_annotations(
        self,
        annotations: dict[str, Any],
        output_path: Path,
        part: int | None = None,
    ) -> None:
        labels_by_split: dict[str, dict[str, list[str]]] = annotations[
            "labels"
        ]
        split_lists: dict[str, list[str]] = annotations["lists"]
        class_names: list[str] = annotations["classes"]

        base = (
            output_path / f"{self.dataset_identifier}_part{part}"
            if part is not None
            else output_path / self.dataset_identifier
        )
        base.mkdir(parents=True, exist_ok=True)

        for split_name in self.get_split_names().values():
            labels_dir = base / "labels" / split_name
            labels_dir.mkdir(parents=True, exist_ok=True)
            images_dir = base / "images" / split_name
            images_dir.mkdir(parents=True, exist_ok=True)

            for stem, lines in labels_by_split.get(split_name, {}).items():
                (labels_dir / f"{stem}.txt").write_text(
                    "\n".join(lines), encoding="utf-8"
                )

        lists_map = {
            "train": base / "train.txt",
            "val": base / "val.txt",
            "test": base / "test.txt",
        }
        for split_name, list_path in lists_map.items():
            items = split_lists.get(split_name, [])
            if items:
                list_path.write_text("\n".join(items) + "\n", encoding="utf-8")

        (base / "obj.names").write_text(
            "\n".join(class_names) + ("\n" if class_names else ""),
            encoding="utf-8",
        )
        data_lines = [
            f"classes={len(class_names)}",
            f"train={lists_map['train'].name}",
            f"valid={lists_map['val'].name}",
            f"names={(base / 'obj.names').name}",
            "backup=backup/",
        ]
        # include test only if present
        if (base / "test.txt").exists():
            data_lines.insert(3, f"test={lists_map['test'].name}")
        (base / "obj.data").write_text(
            "\n".join(data_lines) + "\n", encoding="utf-8"
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
