from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from luxonis_ml.data.exporters.base_exporter import BaseExporter
from luxonis_ml.data.exporters.exporter_utils import (
    PreparedLDF,
    check_group_file_correspondence,
    exporter_specific_annotation_warning,
    split_of_group,
)


class ClassificationDirectoryExporter(BaseExporter):
    def get_split_names(self) -> dict[str, str]:
        return {"train": "train", "val": "valid", "test": "test"}

    def supported_ann_types(self) -> list[str]:
        return ["classification"]

    def export(self, prepared_ldf: PreparedLDF) -> None:
        check_group_file_correspondence(prepared_ldf)
        exporter_specific_annotation_warning(
            prepared_ldf, self.supported_ann_types()
        )

        grouped = prepared_ldf.processed_df.group_by(
            ["file", "group_id"], maintain_order=True
        )

        copied_pairs: set[tuple[Path, str]] = (
            set()
        )  # (src, dest_abs_str) to avoid duplicate write in same pass

        for key, entry in grouped:
            file_name, group_id = cast(tuple[str, Any], key)
            file_path = Path(str(file_name))

            split = split_of_group(prepared_ldf, group_id)

            class_names: set[str] = set()
            for row in entry.iter_rows(named=True):
                if (
                    row["task_type"] == "classification"
                    and row["instance_id"] == -1
                ):
                    cname = row["class_name"]
                    if cname:
                        class_names.add(str(cname))

            if not class_names:
                continue

            idx = self.image_indices.setdefault(
                file_path, len(self.image_indices)
            )
            new_name = f"{idx}{file_path.suffix}"

            for cname in sorted(class_names):
                target_dir = (
                    self._get_data_path(self.output_path, split, self.part)
                    / cname
                )
                target_dir.mkdir(parents=True, exist_ok=True)

                dest = target_dir / new_name
                pair_key = (file_path, str(dest))

                if pair_key in copied_pairs:
                    continue
                copied_pairs.add(pair_key)

                if dest != file_path:
                    dest.write_bytes(file_path.read_bytes())

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
