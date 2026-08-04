"""The LDF models moved to `luxonis_ml.ldf`; old import paths must still work."""

from luxonis_ml import ldf
from luxonis_ml.data import Category, datasets
from luxonis_ml.data.datasets import annotation
from luxonis_ml.data.utils import ParquetRecord, parquet


def test_annotation_module_reexports_everything():
    assert annotation.__all__ == ldf.__all__
    for name in ldf.__all__:
        assert getattr(annotation, name) is getattr(ldf, name)


def test_namespaces_reexport_the_moved_names():
    assert datasets.Detection is ldf.Detection
    assert datasets.DatasetRecord is ldf.DatasetRecord
    assert Category is ldf.Category
    assert ParquetRecord is parquet.ParquetRecord is ldf.ParquetRecord
