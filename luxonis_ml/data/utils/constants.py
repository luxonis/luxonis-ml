from typing import Final

from semver.version import Version

LDF_VERSION: Final[Version] = Version.parse(
    "2.0", optional_minor_and_patch=True
)
