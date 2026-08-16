"""Rewriting exported native records to older LDF versions.

`DatasetRecord` forbids extra fields, so a record written by a newer
luxonis-ml fails to validate on an older install -- LDF 2.1 added
``sample_metadata``, which nothing older accepts. Exporting to an older
version therefore drops every field introduced above it.

Annotations forbid extra fields as well, so the same holds one level
down: LDF 2.2 added ``edges``, ``flip_pairs`` and ``sigmas`` to a keypoint
annotation.

LDF 3.0 changed the shape rather than the fields: it groups the detections
of a record by task name. An older version reads the flat
``file``/``task_name``/``annotation`` record, so the downgrade rebuilds it.
"""

from collections import Counter
from collections.abc import Mapping
from typing import Any, Final, TypeGuard

from loguru import logger
from semver.version import Version

from luxonis_ml.data.utils.constants import LDF_VERSION
from luxonis_ml.typing import Params, ParamValue


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
    ("keypoints", "edges"): _parse("2.2"),
    ("keypoints", "flip_pairs"): _parse("2.2"),
    ("keypoints", "sigmas"): _parse("2.2"),
}

#: The version that started to key the keypoints by name. An older one
#: reads them as a plain list.
_KEYPOINT_NAMES_ADDED_IN: Final[Version] = _parse("2.2")

#: The version that grouped a record's detections by task name. Anything
#: older reads the flat ``file``/``task_name``/``annotation`` shape.
_TASK_KEYED_RECORDS: Final[Version] = _parse("3.0")

#: LDF versions the native exporter can write, newest first.
SUPPORTED_EXPORT_VERSIONS: Final[tuple[Version, ...]] = (
    LDF_VERSION,
    _parse("2.2"),
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
        self._keeps_keypoint_names = target_version >= _KEYPOINT_NAMES_ADDED_IN
        self._flatten_record = target_version < _TASK_KEYED_RECORDS
        self._dropped: Counter[str] = Counter()
        self._n_records = 0

    def __call__(self, record: dict[str, Any]) -> dict[str, Any]:
        """Rewrite one exported record in place and return it."""
        self._n_records += 1
        if self._flatten_record:
            record = self._flatten(record)
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
            self._strip_keypoint_names(annotation)
        return record

    @staticmethod
    def _flatten(record: dict[str, Any]) -> dict[str, Any]:
        """Rewrite a record into the flat shape older LDF versions read.

        The exporter writes one detection per record, so the task-keyed
        mapping always holds a single task and at most one detection.
        """
        flattened: dict[str, Any] = {}
        media = record.pop("media", None)
        if isinstance(media, dict):
            flattened["files"] = media
        elif media is not None:
            flattened["file"] = media

        annotation = record.pop("annotation", None)
        if _is_task_keyed(annotation):
            task_name, detections = next(iter(annotation.items()))
            flattened["task_name"] = task_name
            if detections:
                flattened["annotation"] = detections[0]
        elif annotation:
            # Already flat, so it holds the one detection of the record.
            flattened["annotation"] = annotation

        # The remaining fields keep the order the record had.
        flattened.update(record)
        return flattened

    def log_summary(self) -> None:
        """Warn about populated data the downgrade discarded."""
        for field, n_dropped in self._dropped.items():
            logger.warning(
                f"Exporting to LDF {self.target_version} drops '{field}' "
                f"from {n_dropped} of {self._n_records} records."
            )

    def _strip_keypoint_names(self, annotation: dict[str, Any]) -> None:
        """Turn named keypoints back into a positional list.

        LDF 2.2 keys the keypoints by name. An older version reads them as
        a plain list, so a mapping fails to validate there.
        """
        if self._keeps_keypoint_names:
            return
        keypoints = annotation.get("keypoints")
        if not isinstance(keypoints, dict):
            return
        values = keypoints.get("keypoints")
        if isinstance(values, dict):
            keypoints["keypoints"] = list(values.values())
            self._dropped["keypoints.names"] += 1


def _is_task_keyed(
    annotation: ParamValue,
) -> TypeGuard[Mapping[str, list[Params]]]:
    """Whether an annotation payload groups its detections by task name.

    LDF 3.0 keys the detections of a record by task name. A record written
    by an older version carries a single detection here, whose values are
    annotation payloads rather than lists of them.
    """
    return (
        isinstance(annotation, Mapping)
        and bool(annotation)
        and all(
            isinstance(detections, list) for detections in annotation.values()
        )
    )
