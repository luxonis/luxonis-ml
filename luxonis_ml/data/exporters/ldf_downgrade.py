"""Rewriting exported native records to older LDF versions.

`DatasetRecord` forbids extra fields, so a record written by a newer
luxonis-ml fails to validate on an older install -- LDF 2.1 added
``sample_metadata``, which nothing older accepts. Exporting to an older
version therefore drops every field introduced above it.

Annotations forbid extra fields as well, so the same holds one level
down: LDF 2.2 added the keypoint ``skeleton``.
"""

from collections import Counter
from typing import Any, Final

from loguru import logger
from semver.version import Version

from luxonis_ml.data.utils.constants import LDF_VERSION


def _parse(version: str) -> Version:
    return Version.parse(version, optional_minor_and_patch=True)


#: Record fields keyed by the LDF version that introduced them. Add an
#: entry when a new version adds a field to `DatasetRecord`.
_ADDED_FIELDS: Final[dict[str, Version]] = {
    "sample_metadata": _parse("2.1"),
}

#: The same for fields of a single annotation, keyed by the task type
#: that holds them and the field name.
_ADDED_ANNOTATION_FIELDS: Final[dict[tuple[str, str], Version]] = {
    ("keypoints", "skeleton"): _parse("2.2"),
}

#: LDF versions the native exporter can write, newest first.
SUPPORTED_EXPORT_VERSIONS: Final[tuple[Version, ...]] = (
    LDF_VERSION,
    _parse("2.1"),
    _parse("2.0"),
)


def resolve_export_version(version: str | Version | None) -> Version:
    """Resolve and validate a requested native export version.

    Args:
        version: Requested LDF version. Minor and patch parts are
            optional, so both ``"2.0"`` and ``"2.0.0"`` are accepted.
            ``None`` selects the current `LDF_VERSION`.

    Returns:
        The resolved version.

    Raises:
        ValueError: If the version is malformed, is newer than this
            installation writes, or is not one the native exporter can
            produce.

    """
    if version is None:
        return LDF_VERSION

    supported = ", ".join(str(v) for v in SUPPORTED_EXPORT_VERSIONS)
    invalid = (
        f"Invalid LDF version '{version}'. Supported versions: {supported}."
    )

    try:
        resolved = _parse(str(version))
    except ValueError as e:
        raise ValueError(invalid) from e

    if resolved > LDF_VERSION:
        raise ValueError(
            f"Cannot export LDF {resolved}: this installation of luxonis-ml "
            f"writes LDF {LDF_VERSION} at the newest."
        )
    if resolved not in SUPPORTED_EXPORT_VERSIONS:
        raise ValueError(invalid)
    return resolved


class LDFDowngrader:
    """Strips exported native records down to an older LDF version.

    Records are built at the current `LDF_VERSION` and passed through a
    downgrader on their way out, which keeps version-specific knowledge in
    one place and lets discarded data be counted as it goes.

    Attributes:
        target_version: LDF version the records are rewritten to.

    """

    def __init__(self, target_version: Version):
        self.target_version = target_version
        self._to_drop = [
            field
            for field, added_in in _ADDED_FIELDS.items()
            if added_in > target_version
        ]
        self._to_drop_from_annotation = [
            path
            for path, added_in in _ADDED_ANNOTATION_FIELDS.items()
            if added_in > target_version
        ]
        self._dropped: Counter[str] = Counter()
        self._n_records = 0

    def __call__(self, record: dict[str, Any]) -> dict[str, Any]:
        """Rewrite one exported record in place and return it."""
        self._n_records += 1
        for field in self._to_drop:
            # An empty value is no loss, so drop it but do not report it.
            if record.pop(field, None):
                self._dropped[field] += 1

        annotation = record.get("annotation")
        if isinstance(annotation, dict):
            for task_type, field in self._to_drop_from_annotation:
                task = annotation.get(task_type)
                if isinstance(task, dict) and task.pop(field, None):
                    self._dropped[f"{task_type}.{field}"] += 1
        return record

    def log_summary(self) -> None:
        """Warn about populated data the downgrade discarded."""
        for field, n_dropped in self._dropped.items():
            logger.warning(
                f"Exporting to LDF {self.target_version} drops '{field}' "
                f"from {n_dropped} of {self._n_records} records."
            )
