"""The LDF models moved to `luxonis_ml.ldf`; old import paths must still work."""

import re
from pathlib import Path

import luxonis_ml
from luxonis_ml import ldf
from luxonis_ml.data import Category, datasets
from luxonis_ml.data.datasets import annotation
from luxonis_ml.data.utils import ParquetRecord, parquet

LDF_CLASS_REFERENCE = re.compile(r"luxonis_ml\.ldf\.([A-Z]\w*)")


def test_annotation_module_reexports_everything():
    assert annotation.__all__ == ldf.__all__
    for name in ldf.__all__:
        assert getattr(annotation, name) is getattr(ldf, name)


def test_namespaces_reexport_the_moved_names():
    assert datasets.Detection is ldf.Detection
    assert datasets.DatasetRecord is ldf.DatasetRecord
    assert Category is ldf.Category
    assert ParquetRecord is parquet.ParquetRecord is ldf.ParquetRecord


def test_references_to_ldf_classes_resolve():
    """The keypoint rename left `migration.py` on the old class name.

    It pointed at `luxonis_ml.ldf.Skeleton`, which became
    `KeypointMetadata`. Pydoctor drops a link target it cannot resolve,
    so the published docs named a type that nobody can import.
    """
    package = Path(luxonis_ml.__file__).parent
    for path in sorted(package.rglob("*.py")):
        # Python source is UTF-8. Without this, the read takes the locale
        # encoding, and `__main__.py` breaks the test on Windows.
        source = path.read_text(encoding="utf-8")
        for name in LDF_CLASS_REFERENCE.findall(source):
            assert name in ldf.__all__, f"{path}: luxonis_ml.ldf.{name}"
