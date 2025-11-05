import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

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
            "Each group_id must correspond to exactly one file. "
            f"Found groups with multiple files: {invalid_groups['group_id'].to_list()}"
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
        allow_keypoints: bool, skeletons: dict
    ) -> tuple[list[str], list[list[int]]]:
        """Returns (labels, skeleton_edges_1_based) for the **single**
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
