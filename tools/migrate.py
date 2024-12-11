import json
from typing import Optional

import polars as pl
import typer

from luxonis_ml.data import LuxonisDataset
from luxonis_ml.data.utils.constants import LDF_VERSION
from luxonis_ml.utils import environ


def migrate_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    # Version 2.0
    if "class_name" in df.columns:
        return df

    return (
        df.rename({"class": "class_name"})
        .with_columns(
            [
                pl.col("task").alias("task_type"),
                pl.col("task").alias("task_name"),
                pl.lit("image").alias("source_name"),
            ]
        )
        .select(
            [
                "file",
                "source_name",
                "task_name",
                "created_at",
                "class_name",
                "instance_id",
                "task_type",
                "annotation",
                "uuid",
            ]
        )
    )


def main(
    dataset_name: Optional[str] = None, team_id: Optional[str] = None
) -> None:
    team_id = team_id or environ.LUXONISML_TEAM_ID
    if dataset_name is None:
        datasets = LuxonisDataset.list_datasets(
            team_id, list_incompatible=True
        )
    else:
        datasets = [dataset_name]

    for dataset_name in datasets:
        base_path = (
            environ.LUXONISML_BASE_PATH
            / "data"
            / team_id
            / "datasets"
            / dataset_name
        )
        for parquet_file in (base_path / "annotations").glob("*.parquet"):
            df = pl.read_parquet(parquet_file)
            new_df = migrate_dataframe(df)
            new_df.write_parquet(parquet_file)

        metadata_path = base_path / "metadata" / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["ldf_version"] = str(LDF_VERSION)
        metadata_path.write_text(json.dumps(metadata))
        typer.echo(f"Migration complete for '{dataset_name}'.")

    typer.echo(f"All datasets migrated to version {LDF_VERSION}")


if __name__ == "__main__":
    typer.run(main)
