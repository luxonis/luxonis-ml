import json
import shutil
from pathlib import Path

import polars as pl


class PreparedLDF:
    """Lightweight container for normalized LDF data."""

    def __init__(self, splits, grouped_df, grouped_image_sources):
        self.splits = splits
        self.grouped_df = grouped_df
        self.grouped_image_sources = grouped_image_sources
        self.image_indices = {}


def prepare_ldf_export(ldf) -> PreparedLDF:
    """Shared LDF preprocessing logic for all exporters."""
    splits = ldf.get_splits()
    if splits is None:
        raise ValueError("Cannot export dataset without splits")

    df = ldf._load_df_offline(raise_when_empty=True)

    df = df.with_row_count("row_idx").with_columns(
        pl.col("row_idx").min().over("file").alias("first_occur")
    )

    def resolve_path(img_path: str, uuid: str, media_path: str) -> str:
        img_path = Path(img_path)
        if img_path.exists():
            return str(img_path)
        ext = img_path.suffix.lstrip(".")
        fallback = Path(media_path) / f"{uuid}.{ext}"
        if not fallback.exists():
            raise FileNotFoundError(f"Missing image: {fallback}")
        return str(fallback)

    df = df.with_columns(
        pl.struct(["file", "uuid"])
        .map_elements(
            lambda row: resolve_path(
                row["file"], row["uuid"], str(ldf.media_path)
            ),
            return_dtype=pl.Utf8,
        )
        .alias("file")
    )

    grouped_image_sources = df.select(
        "group_id", "source_name", "file"
    ).unique()

    df = (
        df.with_columns(
            [
                pl.col("annotation").is_not_null().alias("has_annotation"),
                pl.col("group_id")
                .cumcount()
                .over("group_id")
                .alias("first_occur"),
            ]
        )
        .pipe(
            lambda df: (
                df.filter(pl.col("has_annotation")).vstack(
                    df.filter(
                        ~pl.col("group_id").is_in(
                            df.filter(pl.col("has_annotation"))
                            .select("group_id")
                            .unique()["group_id"]
                        )
                    ).unique(subset=["group_id"], keep="first")
                )
            )
        )
        .sort(["row_idx"])
        .select(
            [
                col
                for col in df.columns
                if col not in ["has_annotation", "row_idx", "first_occur"]
            ]
        )
    )

    grouped = df.group_by("group_id", maintain_order=True)

    return PreparedLDF(
        splits=splits,
        grouped_df=grouped,
        grouped_image_sources=grouped_image_sources,
    )


def dump_annotations(annotations, output_path, identifier, part=None):
    for split_name, annotation_data in annotations.items():
        if part is not None:
            split_path = output_path / f"{identifier}_part{part}" / split_name
        else:
            split_path = output_path / identifier / split_name
        split_path.mkdir(parents=True, exist_ok=True)
        with open(split_path / "annotations.json", "w") as f:
            json.dump(annotation_data, f, indent=4)


def create_zip_output(
    dataset_identifier, max_partition_size, part, output_path
):
    archives = []
    if max_partition_size:
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
