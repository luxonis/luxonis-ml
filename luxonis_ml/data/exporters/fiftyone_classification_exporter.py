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


class FiftyOneClassificationExporter(BaseExporter):
    """Output structure::

        <dataset_name>/
            train/
                data/
                    000001.jpg
                    000002.jpg
                    ...
                labels.json
            val/
                data/
                    ...
                labels.json
            test/
                data/
                    ...
                labels.json

    The labels.json has structure::

        E{lb}
            "classes": ["class1", "class2", ...],
            "labels": E{lb}
                "000001": 0,  # index into classes array
                "000002": 1,
                ...
            E{rb}
        E{rb}
    """

    def __init__(
        self,
        dataset_identifier: str,
        output_path: Path,
        max_partition_size_gb: float | None,
    ):
        super().__init__(
            dataset_identifier, output_path, max_partition_size_gb
        )
        self.class_to_idx: dict[str, int] = {}
        self.split_labels: dict[str, dict[str, int]] = {}
        self.split_image_counter: dict[str, int] = {}

    def get_split_names(self) -> dict[str, str]:
        return {"train": "train", "val": "validation", "test": "test"}

    def supported_ann_types(self) -> list[str]:
        return ["classification"]

    def export(self, prepared_ldf: PreparedLDF) -> None:
        check_group_file_correspondence(prepared_ldf)
        exporter_specific_annotation_warning(
            prepared_ldf, self.supported_ann_types()
        )

        for split in self.get_split_names():
            self.split_labels[split] = {}
            self.split_image_counter[split] = 0

        all_classes: set[str] = set()
        for row in prepared_ldf.processed_df.iter_rows(named=True):
            if (
                row["task_type"] == "classification"
                and row["instance_id"] == -1
            ):
                cname = row["class_name"]
                if cname:
                    all_classes.add(str(cname))

        sorted_classes = sorted(all_classes)
        self.class_to_idx = {
            cls: idx for idx, cls in enumerate(sorted_classes)
        }

        grouped = prepared_ldf.processed_df.group_by(
            ["file", "group_id"], maintain_order=True
        )

        copied_pairs: set[tuple[Path, str]] = set()

        for key, entry in grouped:
            file_name, group_id = cast(tuple[str, Any], key)
            file_path = Path(str(file_name))

            split = split_of_group(prepared_ldf, group_id)

            class_name: str | None = None
            for row in entry.iter_rows(named=True):
                if (
                    row["task_type"] == "classification"
                    and row["instance_id"] == -1
                ):
                    cname = row["class_name"]
                    if cname:
                        class_name = str(cname)
                        break  # Take first classification label

            if class_name is None:
                continue

            self.split_image_counter[split] += 1
            idx = self.split_image_counter[split]

            new_name = f"{idx:06d}{file_path.suffix}"

            target_dir = self._get_data_path(
                self.output_path, split, self.part
            )
            target_dir.mkdir(parents=True, exist_ok=True)

            dest = target_dir / new_name
            pair_key = (file_path, str(dest))

            if pair_key not in copied_pairs:
                copied_pairs.add(pair_key)
                if dest != file_path:
                    dest.write_bytes(file_path.read_bytes())

            # Store label mapping (without extension, just the padded number)
            label_key = f"{idx:06d}"
            self.split_labels[split][label_key] = self.class_to_idx[class_name]

        self._dump_annotations(
            {"classes": sorted_classes, "split_labels": self.split_labels},
            self.output_path,
            self.part,
        )

    def _dump_annotations(
        self,
        annotation_data: dict[str, Any],
        output_path: Path,
        part: int | None = None,
    ) -> None:
        classes = annotation_data["classes"]
        split_labels = annotation_data["split_labels"]

        for split_name, labels in split_labels.items():
            if not labels:
                continue

            save_name = self.get_split_names().get(split_name, split_name)
            base = (
                output_path / f"{self.dataset_identifier}_part{part}"
                if part is not None
                else output_path / self.dataset_identifier
            )
            split_path = base / (
                save_name if save_name is not None else str(split_name)
            )
            split_path.mkdir(parents=True, exist_ok=True)

            labels_data = {
                "classes": classes,
                "labels": labels,
            }
            (split_path / "labels.json").write_text(
                json.dumps(labels_data), encoding="utf-8"
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
        return base / split_name / "data"
