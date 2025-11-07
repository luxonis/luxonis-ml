from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast
from xml.etree.ElementTree import Element, ElementTree, SubElement

from defusedxml import minidom
from PIL import Image

from luxonis_ml.data.exporters.base_exporter import BaseExporter
from luxonis_ml.data.exporters.exporter_utils import ExporterUtils, PreparedLDF


class VOCExporter(BaseExporter):
    @staticmethod
    def get_split_names() -> dict[str, str]:
        return {"train": "train", "val": "valid", "test": "test"}

    def transform(self, prepared_ldf: PreparedLDF) -> None:
        ExporterUtils.check_group_file_correspondence(prepared_ldf)

        per_split_data: dict[str, dict[str, dict[str, Any]]] = {
            v: {} for v in self.get_split_names().values()
        }

        df = prepared_ldf.processed_df
        grouped = df.group_by(["file", "group_id"], maintain_order=True)

        copied_files: set[Path] = set()

        for key, group_df in grouped:
            file_name, group_id = cast(tuple[str, Any], key)
            split_key = ExporterUtils.split_of_group(prepared_ldf, group_id)
            split_name = self.get_split_names()[split_key]

            src_path = Path(str(file_name))
            idx = self.image_indices.setdefault(
                src_path, len(self.image_indices)
            )
            new_name = f"{idx}{src_path.suffix}"
            new_stem = Path(new_name).stem

            with Image.open(src_path) as im:
                W, H = im.size
                C = len(im.getbands()) if hasattr(im, "getbands") else 3

            objects: list[dict[str, Any]] = []
            for row in group_df.iter_rows(named=True):
                if row.get("task_type") != "boundingbox":
                    continue
                ann_str = row.get("annotation")
                cname = row.get("class_name")
                if not ann_str or not cname:
                    continue

                data = json.loads(ann_str)
                xn = float(data.get("x", 0.0))
                yn = float(data.get("y", 0.0))
                wn = float(data.get("w", 0.0))
                hn = float(data.get("h", 0.0))

                xmin = int(round(xn * W))
                ymin = int(round(yn * H))

                # widths/heights: round once (no double rounding)
                w_px = max(1, int(round(wn * W)))
                h_px = max(1, int(round(hn * H)))

                # build EXCLUSIVE max from min + size
                xmax = xmin + w_px  # exclusive right edge
                ymax = ymin + h_px  # exclusive bottom edge

                # clamp: allow touching the edge (exclusive bound can be == W/H)
                xmin = max(0, min(xmin, W - 1))
                ymin = max(0, min(ymin, H - 1))
                xmax = max(xmin + 1, min(xmax, W))
                ymax = max(ymin + 1, min(ymax, H))

                objects.append(
                    {
                        "name": str(cname),
                        "bbox": (xmin, ymin, xmax, ymax),
                        "pose": "Unspecified",
                        "truncated": 0,
                        "difficult": 0,
                    }
                )

            per_split_data[split_name][new_stem] = {
                "filename": new_name,
                "abs_path": src_path.resolve(),
                "size": (W, H, C),
                "objects": objects,
            }

            img_size = src_path.stat().st_size
            approx_xml_size = self._estimate_xml_size(objects)
            per_split_data = self._maybe_roll_partition(
                per_split_data, img_size + approx_xml_size
            )

            split_dir = self._get_data_path(
                self.output_path, split_name, self.part
            )
            split_dir.mkdir(parents=True, exist_ok=True)
            dest_img = split_dir / new_name
            if src_path not in copied_files:
                copied_files.add(src_path)
                if dest_img != src_path:
                    dest_img.write_bytes(src_path.read_bytes())
                self.current_size += img_size

        self._dump_annotations(per_split_data, self.output_path, self.part)

    def _maybe_roll_partition(
        self,
        per_split_data: dict[str, dict[str, dict[str, Any]]],
        additional_size: int,
    ) -> dict[str, dict[str, dict[str, Any]]]:
        if (
            self.max_partition_size
            and self.part is not None
            and (self.current_size + additional_size) > self.max_partition_size
        ):
            self._dump_annotations(per_split_data, self.output_path, self.part)
            self.current_size = 0
            self.part += 1
            return {v: {} for v in self.get_split_names().values()}
        return per_split_data

    @staticmethod
    def _estimate_xml_size(objects: list[dict[str, Any]]) -> int:
        """Approximate byte-count.

        The alternative is to serialize, pretty-print and discard every
        time just to compute the true xml size.
        """
        per_object = 140  # ~lines for name/bbox/flags
        header = 300  # folder/filename/path/size etc.
        return header + per_object * max(1, len(objects))

    def _dump_annotations(
        self,
        per_split_data: dict[str, dict[str, dict[str, Any]]],
        output_path: Path,
        part: int | None = None,
    ) -> None:
        if not per_split_data:
            return

        base = (
            output_path / f"{self.dataset_identifier}_part{part}"
            if part is not None
            else output_path / self.dataset_identifier
        )
        base.mkdir(parents=True, exist_ok=True)

        for split_name, per_image in per_split_data.items():
            split_dir = base / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

            for stem, meta in per_image.items():
                xml_str = self._build_voc_xml_string(
                    folder=split_name,
                    filename=meta["filename"],
                    abs_path=meta["abs_path"],
                    size=meta["size"],  # (W,H,C)
                    objects=meta["objects"],
                )
                (split_dir / f"{stem}.xml").write_text(
                    xml_str, encoding="utf-8"
                )

    def _build_voc_xml_string(
        self,
        folder: str,
        filename: str,
        abs_path: Path,
        size: tuple[int, int, int],
        objects: list[dict[str, Any]],
    ) -> str:
        W, H, C = size

        ann = Element("annotation")
        SubElement(ann, "folder").text = folder
        SubElement(ann, "filename").text = filename
        SubElement(ann, "path").text = str(abs_path)

        src = SubElement(ann, "source")
        SubElement(src, "database").text = "Unknown"

        size_el = SubElement(ann, "size")
        SubElement(size_el, "width").text = str(W)
        SubElement(size_el, "height").text = str(H)
        SubElement(size_el, "depth").text = str(C)

        SubElement(ann, "segmented").text = "0"

        for obj in objects:
            xmin, ymin, xmax, ymax = obj["bbox"]
            o = SubElement(ann, "object")
            SubElement(o, "name").text = obj["name"]
            SubElement(o, "pose").text = obj.get("pose", "Unspecified")
            SubElement(o, "truncated").text = str(
                int(bool(obj.get("truncated", 0)))
            )
            SubElement(o, "difficult").text = str(
                int(bool(obj.get("difficult", 0)))
            )
            bb = SubElement(o, "bndbox")
            SubElement(bb, "xmin").text = f"{xmin:.12f}"
            SubElement(bb, "ymin").text = f"{ymin:.12f}"
            SubElement(bb, "xmax").text = f"{xmax:.12f}"
            SubElement(bb, "ymax").text = f"{ymax:.12f}"

        # pretty print with XML declaration
        xml_bytes = self._etree_to_pretty_bytes(ann)
        return xml_bytes.decode("utf-8")

    @staticmethod
    def _etree_to_pretty_bytes(elem: Element) -> bytes:
        buf = bytearray()
        tmp = ElementTree(elem)
        from io import BytesIO

        raw = BytesIO()
        tmp.write(raw, encoding="utf-8", xml_declaration=True)
        pretty = minidom.parseString(raw.getvalue()).toprettyxml(indent="  ")
        buf.extend(pretty.encode("utf-8"))
        return bytes(buf)

    def _get_data_path(
        self, output_path: Path, split_name: str, part: int | None = None
    ) -> Path:
        base = (
            output_path / f"{self.dataset_identifier}_part{part}"
            if part is not None
            else output_path / self.dataset_identifier
        )
        return base / split_name
