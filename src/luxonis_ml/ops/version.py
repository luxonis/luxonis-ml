import fiftyone.core.odm as foo
import luxonis_ml.fiftyone_plugins as fop
from bson import ObjectId
from datetime import datetime

class LuxonisVersion:

    def __init__(self, dataset, version_number=None, samples=None, note=None):
        """
        dataset: LuxonisDataset instance
        version_number: number referencing dataset version
        samples: list of fiftyone sample IDs
        """

        self.conn = foo.get_db_conn()

        self.version_number = dataset.version if version_number is None else version_number
        dataset_id = dataset.fo_dataset._doc.id
        self.dataset_id_str = dataset_id.binary.hex()

        if not self._exists():

            if samples is None:
                raise Exception("samples must be defined for creating a new version")

            version_doc = fop.VersionDocument(
                number=self.version_number,
                dataset_id=dataset_id,
                dataset_id_str=self.dataset_id_str,
                created_at=datetime.utcnow(),
                samples=samples,
                note=note
            )
            
            version_doc.save(upsert=True)

        self.doc = self._find()

    def _exists(self):
        return bool(list(self.conn.version_document.find({ "$and": [{"dataset_id_str": self.dataset_id_str}, {"number": self.version_number}] }).limit(1)))

    def _find(self):
        return self.conn.version_document.find({"dataset_id_str": self.dataset_id_str, "number": self.version_number})[0]

    def get_samples(self):
        return self.doc['samples']
