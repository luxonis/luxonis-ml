from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from luxonis_ml.data.exporters.base_exporter import BaseExporter
from luxonis_ml.data.exporters.exporter_utils import (
    PreparedLDF,
    check_group_file_correspondence,
    exporter_specific_annotation_warning,
)


class FiftyOneClassificationExporter(BaseExporter):
    """Exports dataset to FiftyOne Classification format.

    This exporter produces a flat structure, ignoring any train/val/test
    splits defined in LDF. This matches the native FiftyOne Classification
    format which does not have built-in split support.

    Output structure::

        output_path/
        └── dataset_identifier/
            ├── data/
            │   ├── 0.jpg
            │   ├── 1.jpg
            │   └── ...
            └── labels.json

    The C{labels.json} file has the following structure::

        {
            "classes": ["class1", "class2", ...],
            "labels": {
                "0": 0,
                "1": 1,
                ...
            }
        }

    Where each key in C{labels} is the image filename (without extension)
    and the value is the index into the C{classes} list.

    @note: LDF splits (train/val/test) are ignored during export. All images
        are exported to a single flat structure. This ensures round-trip
        consistency with the flat FiftyOne Classification input format.
    """

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

        classes_set: set[str] = set()
        image_labels: dict[str, str] = {}

        for key, entry in grouped:
            file_name, _ = cast(tuple[str, Any], key)
            file_path = Path(str(file_name))

            for row in entry.iter_rows(named=True):
                if (
                    row["task_type"] == "classification"
                    and row["instance_id"] == -1
                ):
                    cname = row["class_name"]
                    if cname:
                        classes_set.add(str(cname))
                        idx = self.image_indices.setdefault(
                            file_path, len(self.image_indices)
                        )
                        new_name = str(idx)
                        image_labels[new_name] = str(cname)

                        data_dir = self._get_data_path(
                            self.output_path, "", self.part
                        )
                        data_dir.mkdir(parents=True, exist_ok=True)

                        dest = data_dir / f"{new_name}{file_path.suffix}"
                        if dest != file_path and not dest.exists():
                            dest.write_bytes(file_path.read_bytes())
                        break

        classes = sorted(classes_set)
        class_to_idx = {c: i for i, c in enumerate(classes)}

        labels_dict = {
            img_name: class_to_idx[class_name]
            for img_name, class_name in image_labels.items()
        }

        annotations = {"classes": classes, "labels": labels_dict}
        self._dump_annotations(annotations, self.output_path, self.part)

    def _dump_annotations(
        self,
        annotations: dict[str, Any],
        output_path: Path,
        part: int | None = None,
    ) -> None:
        base = (
            output_path / f"{self.dataset_identifier}_part{part}"
            if part is not None
            else output_path / self.dataset_identifier
        )
        base.mkdir(parents=True, exist_ok=True)
        labels_file = base / "labels.json"
        with open(labels_file, "w") as f:
            json.dump(annotations, f, indent=2)

    def _get_data_path(
        self, output_path: Path, split: str, part: int | None = None
    ) -> Path:
        base = (
            output_path / f"{self.dataset_identifier}_part{part}"
            if part is not None
            else output_path / self.dataset_identifier
        )
        return base / "data"
