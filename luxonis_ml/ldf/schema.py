"""The dataset-level schema a single record cannot describe on its own.

A record names its classes, but not the IDs a dataset assigns them, and a
sample with no annotation names nothing at all. Converting between records and
loader arrays therefore needs the schema of the dataset the sample came from:
which tasks exist, which class takes which ID, how many keypoints a task has,
and how categorical metadata is encoded.

`LuxonisLoader` attaches it to every sample under the reserved
`SCHEMA_METADATA_KEY` of `LoaderOutput.metadata`, so `LoaderOutput.to_ldf` can
rebuild a record without being handed the dataset. It is a read-only copy:
the dataset stays the only writer, and the write paths strip the key.
"""

from functools import cached_property

from luxonis_ml.typing import BaseModelExtraForbid, Params

from .annotation import KeypointMetadata

__all__ = ["SCHEMA_METADATA_KEY", "DatasetSchema"]

SCHEMA_METADATA_KEY = "schema"
"""Reserved `LoaderOutput.metadata` key holding the `DatasetSchema`."""


class DatasetSchema(BaseModelExtraForbid):
    """Dataset-level information that a single record does not carry.

    Attributes:
        tasks: Task types grouped by task name. This is what makes a sample
            with no annotation a negative for every task rather than for none.
        classes: Class IDs grouped by task name and class name.
        keypoint_metadata: Keypoint declarations grouped by task name.
        categorical_encodings: Integer encodings of categorical metadata,
            keyed by the full metadata task.
        n_keypoints: Number of keypoints grouped by task name.

    Example:
        >>> schema = DatasetSchema(
        ...     tasks={"detection": ["boundingbox", "classification"]},
        ...     classes={"detection": {"car": 0, "truck": 1}},
        ... )
        >>> schema.class_id("detection", "truck")
        1
        >>> schema.n_classes("detection")
        2

    """

    tasks: dict[str, list[str]] = {}
    classes: dict[str, dict[str, int]] = {}
    keypoint_metadata: dict[str, KeypointMetadata] = {}
    categorical_encodings: dict[str, dict[str, int]] = {}
    n_keypoints: dict[str, int] = {}

    @cached_property
    def as_metadata(self) -> Params:
        """The schema as the plain data that travels in sample metadata.

        Every sample of one dataset shares this dictionary rather than a copy
        of it, so attaching the schema costs nothing per sample. Treat it as
        read-only.
        """
        return self.model_dump()

    def class_id(self, task_name: str, class_name: str) -> int:
        """Return the ID a task assigns to a class name.

        Args:
            task_name: Task the class belongs to.
            class_name: Name of the class.

        Returns:
            The class ID.

        Raises:
            ValueError: If the task does not define the class.

        """
        try:
            return self.classes[task_name][class_name]
        except KeyError:
            raise ValueError(
                f"Task '{task_name}' does not define the class '{class_name}'."
            ) from None

    def class_name(self, task_name: str, class_id: int) -> str | None:
        """Return the class a task gives an ID to.

        Args:
            task_name: Task the class belongs to.
            class_id: ID of the class.

        Returns:
            The class name, or ``None`` when the task has no such ID.

        """
        for name, id_ in self.classes.get(task_name, {}).items():
            if id_ == class_id:
                return name
        return None

    def n_classes(self, task_name: str) -> int:
        """Return how many classes a task defines.

        Args:
            task_name: Task to count the classes of.

        Returns:
            The number of classes, and :math:`0` for an unknown task.

        """
        return len(self.classes.get(task_name, {}))
