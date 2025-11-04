from pathlib import Path
from typing import Any

import polars as pl

from luxonis_ml.data import LuxonisDataset


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
    def from_dataset(cls, ldf: LuxonisDataset) -> "PreparedLDF":
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
