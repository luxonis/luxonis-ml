"""The LDF models moved to `luxonis_ml.ldf`; old import paths must still work."""


def test_annotation_shim_reexports_ldf():
    from luxonis_ml import ldf
    from luxonis_ml.data.datasets.annotation import (
        BBoxAnnotation,
        Category,
        DatasetRecord,
        Detection,
        ParquetRecord,
        load_annotation,
    )

    assert Detection is ldf.Detection
    assert DatasetRecord is ldf.DatasetRecord
    assert Category is ldf.Category
    assert load_annotation is ldf.load_annotation
    assert BBoxAnnotation is ldf.BBoxAnnotation
    assert ParquetRecord is ldf.ParquetRecord


def test_datasets_and_data_namespace_reexports():
    from luxonis_ml import ldf
    from luxonis_ml.data import Category
    from luxonis_ml.data.datasets import DatasetRecord, Detection

    assert Detection is ldf.Detection
    assert DatasetRecord is ldf.DatasetRecord
    assert Category is ldf.Category


def test_parquet_record_compat():
    from luxonis_ml import ldf
    from luxonis_ml.data.utils import ParquetRecord as FromUtils
    from luxonis_ml.data.utils.parquet import ParquetRecord as FromParquet

    assert FromUtils is FromParquet is ldf.ParquetRecord
