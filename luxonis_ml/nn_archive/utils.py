import tarfile
from pathlib import Path

from luxonis_ml.utils.filesystem import PathType


def is_nn_archive(path: PathType) -> bool:
    """Check if the given path is a valid NN archive file.

    @type path: PathType
    @param path: Path to the file to check.
    @rtype: bool
    @return: True if the file is a valid NN archive file, False otherwise.
    """
    path = Path(path)

    if not path.is_file():
        return False

    if not tarfile.is_tarfile(path):
        return False

    with tarfile.open(path, "r") as tar:
        if "config.json" not in tar.getnames():
            return False

    return True
