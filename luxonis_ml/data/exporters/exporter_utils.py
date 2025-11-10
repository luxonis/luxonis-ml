import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import polars as pl
from loguru import logger
from pycocotools import mask

if TYPE_CHECKING:
    from luxonis_ml.data.datasets.luxonis_dataset import LuxonisDataset
    from luxonis_ml.data.exporters.base_exporter import BaseExporter


class PreparedLDF:
    """Lightweight container for LDF data, ready for export."""

    def __init__(
        self,
        splits: dict[str, Any],
        processed_df: pl.DataFrame,
        grouped_image_sources: pl.DataFrame,
    ):
        self.splits = splits
        self.processed_df = processed_df
        self.grouped_image_sources = grouped_image_sources

    @classmethod
    def from_dataset(cls, ldf: "LuxonisDataset") -> "PreparedLDF":
        """Prepare a dataset for export into the LDF representation."""
        splits = ldf.get_splits()
        if splits is None:
            raise ValueError("Cannot export dataset without splits")

        df = ldf._load_df_offline(raise_when_empty=True).with_row_count(
            "row_idx"
        )

        def resolve_path(
            img_path: str | Path, uuid: str, media_path: str
        ) -> str:
            p = Path(img_path)
            if p.exists():
                return str(p)
            fallback = Path(media_path) / f"{uuid}{p.suffix}"
            if not fallback.exists():
                raise FileNotFoundError(f"Missing image: {fallback}")
            return str(fallback)

        df = df.with_columns(
            pl.struct(["file", "uuid"])
            .map_elements(
                lambda r: resolve_path(
                    r["file"], r["uuid"], str(ldf.media_path)
                ),
                return_dtype=pl.Utf8,
            )
            .alias("file")
        )

        grouped_image_sources = df.select(
            "group_id", "source_name", "file"
        ).unique()

        annotated = df.filter(pl.col("annotation").is_not_null())
        annotated_groups = annotated.select("group_id").unique()["group_id"]
        noann_first = df.filter(
            ~pl.col("group_id").is_in(annotated_groups)
        ).unique(subset=["group_id"], keep="first")

        processed_df = (
            pl.concat([annotated, noann_first]).sort("row_idx").drop("row_idx")
        )

        return cls(
            splits=splits,
            processed_df=processed_df,
            grouped_image_sources=grouped_image_sources,
        )


@dataclass(frozen=True)
class ExporterSpec:
    cls: type["BaseExporter"]
    kwargs: dict


class ExporterUtils:
    @staticmethod
    def check_group_file_correspondence(prepared_ldf: PreparedLDF) -> None:
        df = prepared_ldf.processed_df
        group_to_files = df.group_by("group_id").agg(
            pl.col("file").n_unique().alias("file_count")
        )
        invalid_groups = group_to_files.filter(pl.col("file_count") > 1)
        assert invalid_groups.is_empty(), (
            "Each annotation instance must correspond to exactly one file. "
            f"Found groups with multiple files: {invalid_groups['group_id'].to_list()}"
            f"To export multiple files (e.g. RGB, depth) per instance, export to Native format"
        )

    @staticmethod
    def exporter_specific_annotation_warning(
        prepared_ldf: PreparedLDF, supported_ann_types: list[str]
    ) -> None:
        df = prepared_ldf.processed_df

        present_task_types = (
            df.select(pl.col("task_type").drop_nans().drop_nulls().unique())
            .to_series()
            .to_list()
        )

        unsupported = [
            name
            for name in present_task_types
            if name not in supported_ann_types
        ]

        for name in unsupported:
            logger.warning(
                f"Found unsupported annotation type '{name}'; skipping this annotation type."
            )

    @staticmethod
    def split_of_group(prepared_ldf: PreparedLDF, group_id: Any) -> str:
        split = next(
            (s for s, ids in prepared_ldf.splits.items() if group_id in ids),
            None,
        )
        assert split is not None, "group must belong to a split"
        return split

    @staticmethod
    def create_zip_output(
        max_partition_size: float | None,
        output_path: Path,
        part: int | None,
        dataset_identifier: str,
    ) -> Path | list[Path]:
        archives: list[Path] = []

        if max_partition_size is not None and part is not None:
            for i in range(part + 1):
                folder = output_path / f"{dataset_identifier}_part{i}"
                if folder.exists():
                    archive_file = shutil.make_archive(
                        str(folder), "zip", root_dir=folder
                    )
                    archives.append(Path(archive_file))
        else:
            folder = output_path / dataset_identifier
            if folder.exists():
                archive_file = shutil.make_archive(
                    str(folder), "zip", root_dir=folder
                )
                archives.append(Path(archive_file))

        return archives if len(archives) > 1 else archives[0]

    @staticmethod
    def get_single_skeleton(
        allow_keypoints: bool, skeletons: dict[str, Any] | None = None
    ) -> tuple[list[str], list[list[int]]]:
        """Returns (labels, skeleton_edges_1_based) for the single
        skeleton.

        Edges are converted to 1-based indices per COCO spec.
        """
        if not allow_keypoints or skeletons is None:
            return [], []
        if isinstance(skeletons, dict):
            sk = next(iter(skeletons.values()))
        else:  # list
            sk = skeletons[0]
        labels = list(sk.get("labels", []))
        edges = sk.get("edges", [])
        # COCO expects 1-based indices in skeleton
        skeleton_1_based = [[a + 1, b + 1] for a, b in edges]
        return labels, skeleton_1_based

    @staticmethod
    def rle_to_yolo_polygon(rle: str, height: int, width: int) -> list:
        # Decode RLE to binary mask
        m = mask.decode({"size": [height, width], "counts": rle})

        # Each contour = one polygon
        contours, _ = cv2.findContours(
            m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        polygons = []
        for contour in contours:
            contour = contour.squeeze()
            if len(contour.shape) != 2:
                continue
            polygon = []
            for x, y in contour:
                polygon.extend([x / width, y / height])
            polygons.append(polygon)

        return polygons

    @staticmethod
    def _bbox_from_poly(
        coords: list[float],
    ) -> tuple[float, float, float, float]:
        xs = coords[0::2]
        ys = coords[1::2]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        return x_min, y_min, (x_max - x_min), (y_max - y_min)

    @staticmethod
    def _iou_xywh(
        a: tuple[float, float, float, float],
        b: tuple[float, float, float, float],
    ) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh
        inter_w = max(0.0, min(ax2, bx2) - max(ax, bx))
        inter_h = max(0.0, min(ay2, by2) - max(ay, by))
        inter = inter_w * inter_h
        if inter <= 0.0:
            return 0.0
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0.0 else 0.0

    @staticmethod
    def decode_rle_with_pycoco(ann: dict[str, Any]) -> np.ndarray:
        h = int(ann["height"])
        w = int(ann["width"])
        counts = ann["counts"]

        # pycocotools expects an RLE object with 'size' and 'counts'
        rle = {"size": [h, w], "counts": counts.encode("utf-8")}

        m = mask.decode(rle)  # type: ignore[arg-type]
        return np.array(m, dtype=np.uint8, order="C")

    def _normalize(
        self, xs: list[float], ys: list[float], w: float, h: float
    ) -> list[float]:
        out: list[float] = []
        for x, y in zip(xs, ys, strict=True):
            out.append(max(0.0, min(1.0, x / w)))
            out.append(max(0.0, min(1.0, y / h)))
        return out
