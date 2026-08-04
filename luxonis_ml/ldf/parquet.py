"""Parquet row schema of the Luxonis Data Format.

Annotations are flattened into one `ParquetRecord` per label before they are
written. `luxonis_ml.data.utils.parquet.ParquetFileManager` does the writing.
"""

from typing import TypedDict


class ParquetRecord(TypedDict):
    """Single annotation row written to parquet.

    Attributes:
        file: Image or source file path.
        source_name: Source component name.
        task_name: Task name.
        class_name: Optional class name.
        instance_id: Optional instance identifier.
        task_type: Optional task type.
        annotation: Optional serialized annotation JSON.
        sample_metadata: Serialized JSON object for **record-level metadata**.
            Empty metadata is stored as ``"{}"``.

    Example:
        .. python::

            {
                "file": "/data/images/frame_001.jpg",
                "source_name": "image",
                "task_name": "detection",
                "class_name": "person",
                "instance_id": 0,
                "task_type": "boundingbox",
                "annotation": '{"x":0.1,"y":0.2,"w":0.3,"h":0.4}',
                "sample_metadata": '{"record_id":123,"camera":"left"}',
            }

    """

    file: str
    source_name: str
    task_name: str
    class_name: str | None
    instance_id: int | None
    task_type: str | None
    annotation: str | None
    sample_metadata: str
