import tarfile
from pathlib import Path

from luxonis_ml.utils.filesystem import PathType


def is_nn_archive(path: PathType) -> bool:
    path = Path(path)

    if not path.is_file():
        return False

    if not tarfile.is_tarfile(path):
        return False

    with tarfile.open(path, "r") as tar:
        if "config.json" not in tar.getnames():
            return False

    return True
