from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any, TypeAlias, cast

from PIL import Image

from luxonis_ml.data.exporters.exporter_utils import ExporterUtils, PreparedLDF

from .base_exporter import BaseExporter


class YOLOFormat(Enum):
    V4 = "v4"
    V6 = "v6"
    V8 = "v8"


BBox: TypeAlias = tuple[int, float, float, float, float]


class YoloExporter(BaseExporter):
    def __init__(
        self,
        dataset_identifier: str,
        output_path: Path,
        max_partition_size_gb: float | None,
        *,
        version: YOLOFormat = YOLOFormat.V8,
    ):
        super().__init__(
            dataset_identifier, output_path, max_partition_size_gb
        )
        self.class_to_id: dict[str, int] = {}
        self.class_names: list[str] = []
        self.version = version

    def get_split_names(self) -> dict[str, str]:
        if self.version is YOLOFormat.V8:
            return {"train": "train", "val": "val", "test": "test"}
        return {"train": "train", "val": "valid", "test": "test"}

    def _yaml_filename(self) -> str:
        if self.version is YOLOFormat.V6:
            return "data.yaml"
        if self.version is YOLOFormat.V8:
            return "dataset.yaml"
        return ""

    def transform(self, prepared_ldf: PreparedLDF) -> None:
        ExporterUtils.check_group_file_correspondence(prepared_ldf)

        annotation_splits: dict[str, dict[str, list[BBox]]] = {
            k: {} for k in self.get_split_names().values()
        }

        df = prepared_ldf.processed_df
        grouped = df.group_by(["file", "group_id"], maintain_order=True)

        copied_files: set[Path] = set()

        for key, group_df in grouped:
            file_name, group_id = cast(tuple[str, Any], key)
            logical_split = ExporterUtils.split_of_group(
                prepared_ldf, group_id
            )
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
                elif (
                    ttype == "instance_segmentation"
                    and self.version == YOLOFormat.V8
                ):
                    data = json.loads(ann_str)
                    height = data.get("height")
                    width = data.get("width")
                    rle = data.get("counts")

                    if not cname:
                        continue

                    if cname not in self.class_to_id:
                        self.class_to_id[cname] = len(self.class_to_id)
                        self.class_names.append(cname)

                    cid = self.class_to_id[cname]

                    polygons = ExporterUtils.rle_to_yolo_polygon(
                        rle, height, width
                    )

                    for poly in polygons:
                        if len(poly) < 6:
                            continue

                        label_lines.append((cid, *poly))

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
        if self.version is YOLOFormat.V4:
            self._dump_annotations_v4(annotation_splits, output_path, part)
            return

        base = (
            output_path / f"{self.dataset_identifier}_part{part}"
            if part is not None
            else output_path / self.dataset_identifier
        )

        for split_name in self.get_split_names().values():
            labels_dir = base / "labels" / split_name
            labels_dir.mkdir(parents=True, exist_ok=True)
            images_dir = base / "images" / split_name
            images_dir.mkdir(parents=True, exist_ok=True)

            for img_name, lines in annotation_splits.get(
                split_name, {}
            ).items():
                if self.version in (YOLOFormat.V6, YOLOFormat.V8):
                    formatted = []
                    for line in lines:
                        cid = line[0]
                        if (
                            len(line) == 5
                        ):  # bounding box annotations are always len 5
                            _, x, y, w, h = line
                            if self.version in (YOLOFormat.V6, YOLOFormat.V8):
                                xc = x + w / 2.0
                                yc = y + h / 2.0
                                formatted.append(
                                    f"{cid} {xc:.12f} {yc:.12f} {w:.12f} {h:.12f}"
                                )
                            else:
                                formatted.append(
                                    f"{cid} {x:.12f} {y:.12f} {w:.12f} {h:.12f}"
                                )

                        elif (
                            len(line) > 5
                        ):  #  instance segmentation annotations
                            coords = " ".join([f"{v:.12f}" for v in line[1:]])
                            formatted.append(f"{cid} {coords}")
                else:
                    formatted = [
                        f"{cid} {x:.12f} {y:.12f} {w:.12f} {h:.12f}"
                        for (cid, x, y, w, h) in lines
                    ]

                (labels_dir / f"{Path(img_name).stem}.txt").write_text(
                    "\n".join(formatted), encoding="utf-8"
                )

        yaml_filename = self._yaml_filename()
        if yaml_filename:
            split_dirs = self.get_split_names()
            yaml_obj = {
                "train": str(Path("images") / split_dirs["train"]),
                "val": str(Path("images") / split_dirs["val"]),
                "test": str(Path("images") / split_dirs["test"]),
                "nc": len(self.class_names),
                "names": self.class_names,
            }
            (base / yaml_filename).write_text(
                self._to_yaml(yaml_obj), encoding="utf-8"
            )

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
        if self.version is YOLOFormat.V4:
            return base / split
        return base / "images" / split

    @staticmethod
    def _to_yaml(d: dict[str, Any]) -> str:
        lines: list[str] = []
        for k, v in d.items():
            lines.append(f"{k}: {v}")
        return "\n".join(lines) + "\n"
