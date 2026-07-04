"""Public entry point for Luxonis NN Archive creation and inspection.

The `luxonis_ml.nn_archive` package collects the schema models and helper
functions used to create NN Archive tar files. An archive stores
``config.json`` alongside compiled model executables so model metadata,
input/output definitions, tensor layouts, and parser-head configuration can
travel with the executable artifact.


Example:
    Build a validated archive with the package-level imports.

    .. code-block:: python

        from luxonis_ml.nn_archive import ArchiveGenerator

        generator = ArchiveGenerator(
            archive_name="model",
            save_path="artifacts",
            cfg_dict=config,
            executables_paths=["model.blob"],
        )
        archive_path = generator.make_archive()

See:
    `luxonis_ml.nn_archive.config_building_blocks` for the input, output,
    metadata, and parser-head schemas used by `Model`.

"""

from .archive_generator import ArchiveGenerator
from .config import Config
from .model import Model
from .utils import infer_layout, is_nn_archive

__all__ = [
    "ArchiveGenerator",
    "Config",
    "Model",
    "infer_layout",
    "is_nn_archive",
]
