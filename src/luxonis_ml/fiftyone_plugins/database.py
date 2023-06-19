from fiftyone.core.fields import (
    DateTimeField,
    ListField,
    ObjectIdField,
    StringField,
    FloatField,
    IntField,
    DictField,
    BooleanField
)

from fiftyone.core.odm.document import Document
from fiftyone.core.odm.dataset import DatasetDocument

class LuxonisDatasetDocument(Document):
    """Backing document for LuxonisDataset."""

    dataset_id = ObjectIdField(db_field="_dataset_id", required=True)
    dataset_id_str = StringField(required=True)
    team_name = StringField(required=True)
    dataset_name = StringField(required=True)
    path = StringField(required=True)
    bucket_type = StringField(required=True)
    current_version = FloatField()

class LuxonisSourceDocument(Document):
    """Backing document for LuxonisDataset sources."""

    luxonis_dataset_id = ObjectIdField(db_field="_luxonis_dataset_id", required=True)
    name = StringField(required=True)
    source_type = StringField()
    component_names = ListField(StringField())
    component_htypes = ListField(IntField())
    component_itypes = ListField(IntField())

class VersionDocument(Document):
    """Backing document for LuxonisDataset versions."""

    # id = ObjectIdField()
    number = FloatField(required=True)
    dataset_id = ObjectIdField(db_field="_dataset_id", required=True)
    dataset_id_str = StringField(required=True)
    created_at = DateTimeField(required=True)
    samples = ListField(StringField())
    note = StringField()

class TransactionDocument(Document):
    """Backing document for LuxonisDataset transactions."""

    dataset_id = ObjectIdField(db_field="_dataset_id", required=True)
    created_at = DateTimeField(required=True)
    executed = BooleanField(required=True)
    action = StringField(required=True)
    sample_id = ObjectIdField()
    field = StringField()
    value = DictField()
    component = StringField()
    version = FloatField()
