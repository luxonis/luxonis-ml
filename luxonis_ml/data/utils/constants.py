from typing import Final

from semver.version import Version

LDF_VERSION: Final[Version] = Version.parse(
    "3.0", optional_minor_and_patch=True
)
"""The LDF version this installation writes.

LDF 3.0 groups a record's detections by task name, so one record can carry a
whole sample. The parquet rows did not move, but the record contract did.
"""
