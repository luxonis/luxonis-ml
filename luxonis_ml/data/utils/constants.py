from typing import Final

from semver.version import Version

LDF_VERSION: Final[Version] = Version.parse(
    "2.0", optional_minor_and_patch=True
)
"""The version of the Luxonis Data Format used by this library.

Mismatch in major version numbers indicates incompatibility. Minor
version numbers indicate new non-breaking features.
"""
