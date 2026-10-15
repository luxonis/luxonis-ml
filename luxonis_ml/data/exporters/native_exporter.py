"""Export to the native LDF format."""

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import polars as pl
from semver.version import Version

from luxonis_ml.data.exporters.base_exporter import BaseExporter
from luxonis_ml.data.exporters.exporter_utils import (
    PreparedLDF,
    split_of_group,
)
from luxonis_ml.data.exporters.ldf_downgrade import LDFDowngrader
from luxonis_ml.data.utils.constants import LDF_VERSION
from luxonis_ml.enums import DatasetType
from luxonis_ml.ldf import (
    DatasetRecord,
    Keypoint,
    KeypointMetadata,
    KeypointState,
)
from luxonis_ml.utils.path import path_to_posix


class NativeExporter(BaseExporter):
    """Exporter for native LDF directory format.

    Native export writes split directories with copied media and an
    ``annotations.json`` file. Each annotation entry follows the same
    record-level shape accepted by `NativeParser` and `LuxonisDataset.add`.

    ``sample_metadata`` is exported as a JSON object next to ``file`` or
    ``files``. It is **record-level metadata**, not an annotation label.

    The export root also holds a ``metadata.json`` version stamp, such as
    ``{"ldf_version": "3.0.0"}``. It is not the full `Metadata` model a
    dataset keeps in its own storage.

    A keypoint record holds every keypoint of its task, in the order of the
    task, with its ``sigma`` and its ``connected_to``. A keypoint that the
    instance does not name is ``"unlabeled"``. The import thus builds the
    same `KeypointMetadata`.

    Passing an older ``ldf_version`` strips the fields that version does
    not know -- exporting LDF 2.1 or 2.0 writes the keypoints as lists
    without their names and task fields, and LDF 2.0 also omits
    ``sample_metadata``. See `LDFDowngrader`.

    Example:
        .. code-block:: json

            {
              "file": "images/0.jpg",
              "task_name": "detection",

              "sample_metadata": {
                "record_id": 123,
                "camera": "left",
                "tags": ["night", "warehouse"]
              },

              "annotation": {
                "instance_id": 0,
                "class": "person",
                "boundingbox": {
                  "x": 0.1,
                  "y": 0.2,
                  "w": 0.3,
                  "h": 0.4
                }
              }
            }

    """

    def __init__(
        self,
        dataset_identifier: str,
        output_path: Path,
        max_partition_size_gb: float | None,
        *,
        ldf_version: Version | None = None,
    ):
        super().__init__(
            dataset_identifier, output_path, max_partition_size_gb
        )
        self.ldf_version = ldf_version or LDF_VERSION
        self._downgrade = LDFDowngrader(self.ldf_version)

    @staticmethod
    def get_split_names() -> dict[str, str]:
        return {"train": "train", "val": "val", "test": "test"}

    def supported_ann_types(self) -> list[str]:
        return DatasetType.NATIVE.supported_annotation_formats

    def export(self, prepared_ldf: PreparedLDF) -> None:
        annotation_splits: dict[str, list[dict[str, Any]]] = {
            k: [] for k in self.get_split_names()
        }
        grouped_image_sources = prepared_ldf.grouped_image_sources
        grouped_df = prepared_ldf.processed_df.group_by(
            "group_id", maintain_order=True
        )
        copied_files: set[Path] = set()

        for group_key, group_df in grouped_df:
            group_id = (
                group_key[0] if isinstance(group_key, tuple) else group_key
            )
            split = split_of_group(prepared_ldf, group_id)

            matched_df = grouped_image_sources.filter(
                pl.col("group_id") == group_id
            )
            group_files = matched_df.get_column("file").to_list()
            group_source_names = matched_df.get_column("source_name").to_list()

            records = [
                self._process_row(
                    row,
                    group_source_names,
                    group_files,
                    prepared_ldf.keypoint_metadata,
                )
                for row in group_df.iter_rows(named=True)
            ]

            ann_size = sum(sys.getsizeof(r) for r in records)
            img_size = sum(Path(f).stat().st_size for f in group_files)
            annotation_splits = self._maybe_roll_partition(
                annotation_splits, ann_size + img_size
            )

            data_path = self._get_data_path(self.output_path, split, self.part)
            data_path.mkdir(parents=True, exist_ok=True)

            for f in group_files:
                p = Path(f)
                if p not in copied_files:
                    copied_files.add(p)
                    idx = self.image_indices[p]
                    shutil.copy(p, data_path / f"{idx}{p.suffix}")
                    self.current_size += p.stat().st_size

            annotation_splits[split].extend(records)

        self._dump_annotations(annotation_splits, self.output_path, self.part)
        self._downgrade.log_summary()

    def _maybe_roll_partition(
        self,
        annotation_splits: dict[str, list[dict[str, Any]]],
        additional_size: int,
    ) -> dict[str, list[dict[str, Any]]]:
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
            return {k: [] for k in self.get_split_names()}
        return annotation_splits

    def _process_row(
        self,
        row: dict[str, Any],
        group_source_names: list[str],
        group_files: list[str],
        keypoint_metadata: dict[str, KeypointMetadata],
    ) -> dict[str, Any]:
        task_name = row["task_name"]
        class_name = row["class_name"]
        instance_id = row["instance_id"]
        task_type = row["task_type"]
        ann_str = row["annotation"]

        source_to_file = {
            name: path_to_posix(
                Path("images")
                / f"{self.image_indices.setdefault(Path(f), len(self.image_indices))}{Path(f).suffix}"
            )
            for name, f in zip(group_source_names, group_files, strict=True)
        }

        multi_source = len(source_to_file) > 1
        record: dict[str, Any] = {
            ("files" if multi_source else "file"): (
                source_to_file
                if multi_source
                else source_to_file[group_source_names[0]]
            ),
            "task_name": task_name,
            "sample_metadata": DatasetRecord.decode_metadata(
                row.get("sample_metadata")
            ),
        }

        if ann_str is not None:
            data = json.loads(ann_str)
            ann: dict[str, Any] = {
                "instance_id": instance_id,
                "class": class_name,
            }
            if task_type == "keypoints":
                ann[task_type] = {
                    "keypoints": _record_keypoints(
                        keypoint_metadata[task_name], data["keypoints"]
                    )
                }
            elif task_type in (
                "boundingbox",
                "segmentation",
                "instance_segmentation",
            ):
                ann[task_type] = data
            elif task_type.startswith("metadata/"):
                ann["metadata"] = {task_type[9:]: data}
            record["annotation"] = ann

        return self._downgrade(record)

    def _dump_annotations(
        self,
        annotation_splits: dict[str, list[dict[str, Any]]],
        output_path: Path,
        part: int | None = None,
    ) -> None:
        base = self._base_path(output_path, part)
        base.mkdir(parents=True, exist_ok=True)
        # Called once per partition, so every part carries its own stamp.
        (base / "metadata.json").write_text(
            json.dumps({"ldf_version": str(self.ldf_version)}, indent=4),
            encoding="utf-8",
        )

        for split_name, items in annotation_splits.items():
            save_name = self.get_split_names().get(split_name, split_name)
            split_path = base / save_name
            split_path.mkdir(parents=True, exist_ok=True)
            (split_path / "annotations.json").write_text(
                json.dumps(items, indent=4), encoding="utf-8"
            )

    def _base_path(self, output_path: Path, part: int | None) -> Path:
        """Return the export root of a partition."""
        name = (
            self.dataset_identifier
            if part is None
            else f"{self.dataset_identifier}_part{part}"
        )
        return output_path / name

    def _get_data_path(
        self, output_path: Path, split: str, part: int | None = None
    ) -> Path:
        return self._base_path(output_path, part) / split / "images"


def _record_keypoints(
    task_keypoints: KeypointMetadata, stored: dict[str, list[float]]
) -> dict[str, dict[str, Any]]:
    """Give a stored row every keypoint and task field of its task."""
    return {
        name: Keypoint(
            **spec.model_dump(), x=x, y=y, state=KeypointState(state)
        ).model_dump(mode="json", exclude_defaults=True)
        for name, spec in task_keypoints.keypoints.items()
        for x, y, state in [stored.get(name, (0.0, 0.0, 0))]
    }
