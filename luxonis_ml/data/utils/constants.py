from typing import Final

from semver.version import Version

LDF_VERSION: Final[Version] = Version.parse(
    "3.0", optional_minor_and_patch=True
)
