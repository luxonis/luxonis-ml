import fiftyone as fo
import fiftyone.core.odm as foo
import luxonis_ml.data.fiftyone_plugins as fop
import luxonis_ml.data.utils.data_utils as data_utils
from luxonis_ml.data.utils.exceptions import *
from fiftyone import ViewField as F
import os, subprocess, shutil
from pathlib import Path
import cv2
import json
import boto3
from google.cloud import storage
import logging, time
from tqdm import tqdm
from enum import Enum
import numpy as np
from datetime import datetime
import pymongo
from bson.objectid import ObjectId
from typing import List, Dict, Tuple, Optional
from .utils.s3_utils import *
from .utils.gcs_utils import *

LDF_VERSION = "0.1.0"


class HType(Enum):
    """Individual file type"""

    IMAGE = 1
    JSON = 2


class IType(Enum):
    """Image type for IMAGE HType"""

    BGR = 1
    MONO = 2
    DISPARITY = 3
    DEPTH = 4


class LDFTransactionType(Enum):
    """The type of transaction"""

    END = "END"
    ADD = "ADD"
    UPDATE = "UPDATE"
    DELETE = "DELETE"


class BucketType(Enum):
    """Whether storage is internal or external"""

    INTERNAL = "internal"
    EXTERNAL = "external"


class BucketStorage(Enum):
    """Underlying object storage for a bucket"""

    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"
    AZURE_BLOB = "azure"


class LDFComponent:
    """Abstract class for a piece of media within a source.
    Most commonly, this abstracts an image sensor."""

    htypes = [HType.IMAGE, HType.JSON]
    itypes = [IType.BGR, IType.MONO, IType.DISPARITY, IType.DEPTH]

    def __init__(self, name: str, htype: HType, itype: Optional[IType] = None) -> None:
        if htype not in self.htypes:
            raise Exception(f"{htype} is not a valid HType!")
        self.name = name
        self.htype = htype
        if htype == HType.IMAGE:
            if itype is not None:
                if itype not in self.itypes:
                    raise Exception(f"{itype} is not a valid IType!")
                self.itype = itype
            else:
                raise Exception("itype parameter is required for image component")


class LDFSource:
    """Abstracts the structure of a dataset and which components/media are included"""

    types = ["real_oak", "synthetic_oak", "custom"]

    def __init__(
        self,
        name: str,
        components: List[LDFComponent],
        main_component: Optional[str] = None,
        source_type: Optional[str] = None,
    ) -> None:
        self.name = name
        self.components = {component.name: component for component in components}
        if main_component is not None:
            self.main_component = main_component
        else:
            self.main_component = list(self.components.keys())[
                0
            ]  # make first component main component by default
        if source_type is not None:
            self.source_type = source_type
        else:
            self.source_type = "custom"

    def add_component(self, component):
        self.components += component


class LuxonisDataset:
    """Luxonis Dataset Format (LDF). Used to define datasets in the Luxonis MLOps ecosystem"""

    @staticmethod
    def create(
        dataset_name: str,
        team_id: str = os.getenv("LUXONISML_TEAM_ID", "offline"),
        team_name: str = os.getenv("LUXONISML_TEAM_NAME", "offline"),
    ) -> str:
        """
        Creates new LDF or returns dataset ID for existing LDF

        dataset_name:
            Name of the dataset
        team_id:
            Optional unique team identifier for the cloud
        team_name:
            Optional team name for the cloud
        """

        conn = foo.get_db_conn()
        res = list(
            conn.luxonis_dataset_document.find(
                {"$and": [{"team_id": team_id}, {"dataset_name": dataset_name}]}
            )
        )
        if len(res) >= 2:
            raise Exception("More than 1 dataset exists under this name!")

        if len(res):
            return str(res[0]["_id"])
        else:
            dataset_doc = fop.LuxonisDatasetDocument(
                team_id=team_id,
                team_name=team_name,
                dataset_name=dataset_name,
                ldf_version=LDF_VERSION,
            )
            dataset_doc = dataset_doc.save(upsert=True)

            dataset_id = str(dataset_doc.id)

            return dataset_id

    def __init__(
        self,
        dataset_name: str,
        team_id: str = os.getenv("LUXONISML_TEAM_ID", "offline"),
        team_name: str = os.getenv("LUXONISML_TEAM_NAME", "offline"),
        bucket_type: BucketType = BucketType.INTERNAL,
        bucket_storage: BucketStorage = BucketStorage.LOCAL,
    ) -> None:
        """
        Initializes LDF

        dataset_name:
            Name of the dataset
        team_id:
            Optional unique team identifier for the cloud
        team_name:
            Optional team name for the cloud
        bucket_type:
            Whether to use external cloud buckets
        bucket_storage:
            Underlying bucket storage from local (no cloud), S3, or GCS
        """

        self.conn = foo.get_db_conn()

        self.base_path = os.getenv(
            "LUXONISML_BASE_PATH", str(Path.home() / "luxonis_ml")
        )

        self.team_id = team_id
        self.team_name = team_name
        self.dataset_name = dataset_name
        self.bucket_type = bucket_type
        self.bucket_storage = bucket_storage
        if not isinstance(self.bucket_type, BucketType):
            raise Exception(f"Must use a valid BucketType!")
        if not isinstance(self.bucket_storage, BucketStorage):
            raise Exception(f"Must use a valid BucketStorage!")

        credentials_cache_file = str(Path(self.base_path) / "credentials.json")
        if os.path.exists(credentials_cache_file):
            with open(credentials_cache_file) as file:
                self.creds = json.load(file)
        else:
            self.creds = {}

        self.version = 0.0
        self.tasks = ["class", "boxes", "segmentation", "keypoints"]
        self.compute_heatmaps = True  # TODO: could make this configurable

        log_format = "%(asctime)s [%(levelname)s] %(message)s"
        self.log_level = os.environ.get("LUXONISML_LEVEL", "INFO").upper()
        logging.basicConfig(level=self.log_level, format=log_format)
        self.logger = logging.getLogger(__name__)
        file_handler = logging.FileHandler("ldf_debug.log")
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(log_format)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.last_time = None

    def __enter__(self):
        res = list(
            self.conn.luxonis_dataset_document.find(
                {
                    "$and": [
                        {"team_id": self.team_id},
                        {"dataset_name": self.dataset_name},
                    ]
                }
            )
        )

        if len(res):
            assert len(res) == 1
            self.dataset_doc = fop.LuxonisDatasetDocument.objects.get(
                team_id=self.team_id, dataset_name=self.dataset_name
            )
            self.dataset_id = str(self.dataset_doc.id)
            self.full_name = self.full_name = f"{self.team_id}-{self.dataset_id}"

            self._doc_to_class()

            self._init_path()

            self.dataset_doc.save(upsert=True)

        else:
            raise Exception("Dataset not found!")

        if self.full_name in fo.list_datasets():
            self.fo_dataset = fo.load_dataset(self.full_name)
        else:
            self.fo_dataset = fo.Dataset(self.full_name)
        self.fo_dataset.persistent = True

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.fo_dataset.persistent = True
        except Exception as e:
            self.logger.warning(f"Dataset persistence on exit failed for reason {e}")
        self._class_to_doc()
        self.dataset_doc.save(upsert=True)

    def _doc_to_class(self) -> None:
        """Converts mongo document to class attributes"""

        if hasattr(self.dataset_doc, "path") and self.dataset_doc.path is not None:
            self.path = self.dataset_doc.path
        if (
            hasattr(self.dataset_doc, "bucket_type")
            and self.dataset_doc.bucket_type is not None
        ):
            self.bucket_type = BucketType(self.dataset_doc.bucket_type)
        if (
            hasattr(self.dataset_doc, "bucket_storage")
            and self.dataset_doc.bucket_storage is not None
        ):
            self.bucket_storage = BucketStorage(self.dataset_doc.bucket_storage)
        if (
            hasattr(self.dataset_doc, "dataset_version")
            and self.dataset_doc.dataset_version is not None
        ):
            self.version = float(self.dataset_doc.dataset_version)

        doc = list(self._get_source())
        if len(doc):
            doc = doc[0]
            components = [
                LDFComponent(name, HType(htype), IType(itype))
                for name, htype, itype in list(
                    zip(
                        doc["component_names"],
                        doc["component_htypes"],
                        doc["component_itypes"],
                    )
                )
            ]
            main_component = components[0].name
            self.source = LDFSource(
                name=doc["name"],
                components=components,
                main_component=main_component,
                source_type=doc["source_type"],
            )
        else:
            self.source = None

    def _class_to_doc(self) -> None:
        """Converts class attributes to mongo document"""

        self.dataset_doc.fo_dataset_id = self.fo_dataset._doc.id
        self.dataset_doc.path = self.path
        self.dataset_doc.bucket_type = self.bucket_type.value
        self.dataset_doc.bucket_storage = self.bucket_storage.value
        self.dataset_doc.dataset_version = str(self.version)

    def _get_credentials(self, key: str) -> str:
        """Gets secret credentials from credentials file or ENV variables"""

        if key in self.creds.keys():
            return self.creds[key]
        else:
            return os.getenv(key, None)

    def _get_source(self) -> pymongo.cursor.Cursor:
        """Queries to find source document tied to dataset"""

        return self.conn.luxonis_source_document.find(
            {"$and": [{"_luxonis_dataset_id": self.dataset_doc.id}]}
        ).limit(1)

    def _init_boto3_client(self) -> None:
        """Initializes connection to S3 bucket"""

        self.client = boto3.client(
            "s3",
            endpoint_url=self._get_credentials("AWS_S3_ENDPOINT_URL"),
            aws_access_key_id=self._get_credentials("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=self._get_credentials("AWS_SECRET_ACCESS_KEY"),
        )
        self.bucket = self.path.split("//")[1].split("/")[0]
        self.bucket_path = self.path.split(self.bucket + "/")[-1]

    def _init_gcs_client(self) -> None:
        """Initializes connection to GCS bucket"""

        self.bucket = self.path.split("//")[1].split("/")[0]
        self.bucket_path = self.path.split(self.bucket + "/")[-1]
        self.client = storage.Client()

    def _init_path(self) -> None:
        """Configures local path or bucket directory"""

        if self.bucket_storage.value == "local":
            self.path = str(
                Path(self.base_path)
                / "data"
                / self.team_id
                / "datasets"
                / self.dataset_id
            )
            os.makedirs(self.path, exist_ok=True)
        elif self.bucket_storage.value == "s3":
            self.path = f"s3://{self._get_credentials('AWS_BUCKET')}/{self.team_id}/datasets/{self.dataset_id}"
            self._init_boto3_client()
        elif self.bucket_storage.value == "gcs":
            self.path = f"gs://{self._get_credentials('GCS_BUCKET')}/{self.team_id}/datasets/{self.dataset_id}"
            self._init_gcs_client()

    def _save_version(self, samples: List[fo.Sample], note: str) -> None:
        """Creates the fiftyone view equivalent to dataset version"""

        version_doc = fop.VersionDocument(
            number=self.version,
            dataset_id=self.fo_dataset._doc.id,
            dataset_id_str=str(self.fo_dataset._doc.id),
            created_at=datetime.utcnow(),
            samples=samples,
            note=note,
        )
        version_doc.save()

        version_view = self.fo_dataset[samples]
        self.fo_dataset.save_view(f"version_{self.version}", version_view)

    def _log_time(self, note: Optional[str] = None, final: bool = False) -> None:
        """Helper to log the time taken within a function to INFO"""

        if self.logger.isEnabledFor(logging.DEBUG):
            if self.last_time is None:
                self.last_time = time.time()
            else:
                new_time = time.time()
                self.logger.debug(f"{note} took {(new_time-self.last_time)*1000} ms")
                self.last_time = new_time
                if final:
                    self.last_time = None

    def _get_sample_collection(self) -> Optional[str]:
        """Gets the name of the mongo collection storing samples"""

        res = list(self.conn.datasets.find({"_id": ObjectId(self.fo_dataset._doc.id)}))
        if len(res):
            return res[0]["sample_collection_name"]
        else:
            return None

    def _update_from_instance_ids(
        self,
        instance_ids: List[str],
        attribute: str,
        value: str,
        main_only: bool = False,
    ) -> List[ObjectId]:
        """Updates an attribute of a group of samples given their instance IDs"""

        try:
            sample_collection = self._get_sample_collection()
        except:
            raise Exception("Cannot find sample collection name")

        if not main_only:
            samples = self.conn[sample_collection].find(
                {"instance_id": {"$in": instance_ids}}
            )
            instance_ids = set()
            for sample in samples:
                gid = sample[self.source.name]["_id"]
                group_instance_ids = list(
                    self.conn[sample_collection].find(
                        {f"{self.source.name}._id": gid}, {"instance_id": 1}
                    )
                )
                for iid in group_instance_ids:
                    instance_ids.add(iid["instance_id"])
            instance_ids = list(instance_ids)

        sample_ids = list(
            self.conn[sample_collection].find(
                {"instance_id": {"$in": instance_ids}}, {"_id": 1}
            )
        )
        sample_ids = [sid["_id"] for sid in sample_ids]

        self.conn[sample_collection].update_many(
            {"_id": {"$in": sample_ids}},
            {"$set": {attribute: value}},
        )

        return sample_ids

    def create_source(
        self,
        name: Optional[str] = None,
        oak_d_default: bool = False,
        custom_components: Optional[List[LDFComponent]] = None,
    ) -> LDFSource:
        """
        Creates the underlying source of the dataset, defining

        name:
            Name of the source
        oak_d_default:
            Use the default source structure for OAK-D
        custom_components:
            A list of LDFComponents for a custom source
        """

        if oak_d_default:
            if name is None:
                name = "oak_d"
            components = [
                LDFComponent("left", htype=HType.IMAGE, itype=IType.MONO),
                LDFComponent("right", htype=HType.IMAGE, itype=IType.MONO),
                LDFComponent("color", htype=HType.IMAGE, itype=IType.BGR),
                LDFComponent("disparity", htype=HType.IMAGE, itype=IType.DISPARITY),
                LDFComponent("depth", htype=HType.IMAGE, itype=IType.DEPTH),
            ]

            source = LDFSource(
                name, components, main_component="color", source_type="real_oak"
            )

        elif name is not None and custom_components is not None:
            source = LDFSource(name, custom_components)
        else:
            raise Exception("name and custom_components required for custom sources")

        if self.dataset_doc.id is None:
            self._class_to_doc()
            self.dataset_doc.save(safe=True, upsert=True)
        res = self._get_source()
        if len(list(res)):  # exists
            self.logger.warning(f"Updating from a previously saved source!")
            self.conn.luxonis_source_document.delete_many(
                {"_luxonis_dataset_id": self.dataset_doc.id}
            )
        luxonis_dataset_id = self.dataset_doc.id
        source_doc = fop.LuxonisSourceDocument(
            luxonis_dataset_id=luxonis_dataset_id,
            name=source.name,
            source_type=source.source_type,
            component_names=[c.name for _, c in source.components.items()],
            component_htypes=[c.htype.value for _, c in source.components.items()],
            component_itypes=[c.itype.value for _, c in source.components.items()],
        )
        source_doc.save(upsert=True)

        self.source = source

        self.fo_dataset.add_group_field(
            self.source.name, default=self.source.main_component
        )

        return source

    def set_classes(self, classes: List[str], task: Optional[str] = None) -> None:
        """
        Sets the names of classes for the dataset. This can be across all CV tasks or certain tasks

        classes:
            List of class names to set
        task:
            Optionally specify the task where these classes apply. This can be from the options in self.tasks
        """

        if task is not None:
            if task not in self.tasks:
                raise Exception(f"Task {task} is not a supported task")
            self.fo_dataset.classes[task] = classes
        else:
            for task in self.tasks:
                self.fo_dataset.classes[task] = classes

    def set_mask_targets(self, mask_targets: Dict) -> None:
        """
        For segmentation, this sets how the integers in a segmentaiton mask map to classes.

        mask_targets:
            Dictionary mapping from integer keys to class names. For example,
            {
                1: "apple",
                2: "orange",
                3: "banana"
            }
        """

        if 0 in mask_targets.keys():
            raise Exception("Cannot set 0! This is assumed to be the background class")
        self.fo_dataset.mask_targets = {"segmentation": mask_targets}

    def set_skeleton(self, skeleton: Dict) -> None:
        """
        Sets a skeleton definition for datasets using keypoits

        skeleton:
            Dictionary with keys
            labels:
                List of strings for the name of each keypoint
            edges:
                List of length 2 lists of keypoint indices denoting how keypoints are connected
            The following is an example
            {
                "labels": ["nose", "left ear", "right ear],
                "edges": [[0,1],[0,2]]
            }

        NOTE: this only supports one class, which seems to be a fiftyone limitation
        """

        self.fo_dataset.skeletons.update(
            {
                "keypoints": fo.KeypointSkeleton(
                    labels=skeleton["labels"], edges=skeleton["edges"]
                )
            }
        )
        self.fo_dataset.save()

    def sync_from_cloud(self) -> None:
        """Downloads media from cloud bucket"""

        if self.bucket_storage.value == "local":
            self.logger.warning("This is a local dataset! Cannot sync")
        elif self.bucket_storage.value == "s3":
            sync_from_s3(
                non_streaming_dir=str(Path(self.base_path) / "data"),
                bucket=self.bucket,
                bucket_dir=str(Path(self.team_id) / "datasets" / self.dataset_id),
                endpoint_url=self._get_credentials("AWS_S3_ENDPOINT_URL"),
            )
        elif self.bucket_storage.value == "gcs":
            sync_from_gcs(
                non_streaming_dir=str(Path(self.base_path) / "data"),
                bucket=self.bucket,
                bucket_dir=str(Path(self.team_id) / "datasets" / self.dataset_id),
            )
        else:
            raise NotImplementedError

    def get_classes(self) -> Tuple[List[str], Dict]:
        """Gets overall classes in the dataset and classes according to CV task"""

        classes = set()
        classes_by_task = {}
        for task in self.fo_dataset.classes:
            if len(self.fo_dataset.classes[task]):
                classes_by_task[task] = self.fo_dataset.classes[task]
                for cls in self.fo_dataset.classes[task]:
                    classes.add(cls)
        mask_classes = set()
        if "segmentation" in self.fo_dataset.mask_targets.keys():
            for key in self.fo_dataset.mask_targets["segmentation"]:
                mask_classes.add(self.fo_dataset.mask_targets["segmentation"][key])
        classes_by_task["segmentation"] = list(mask_classes)
        classes = list(classes.union(mask_classes))
        classes.sort()

        return classes, classes_by_task

    def get_classes_count(self) -> Dict:
        """Returns dictionary with number of occurances for each class. If no class label is present returns empty dict."""

        try:
            count_dict = self.fo_dataset.count_values("class.classifications.label")
        except:
            self.logger.warning(
                "No 'class' label present in the dataset. Returning empty dictionary."
            )
            count_dict = {}
        return count_dict

    def delete_dataset(self) -> None:
        """Deletes the entire dataset in mongo, aside from the images"""

        res = list(
            self.conn.luxonis_dataset_document.find(
                {
                    "$and": [
                        {"team_id": self.team_id},
                        {"_id": ObjectId(self.dataset_id)},
                    ]
                }
            )
        )
        if len(res):
            ldf_doc = res[0]

            self.conn.luxonis_source_document.delete_many(
                {"_luxonis_dataset_id": ldf_doc["_id"]}
            )
            self.conn.version_document.delete_many(
                {"_dataset_id": ldf_doc["_dataset_id"]}
            )
            self.conn.transaction_document.delete_many({"_dataset_id": ldf_doc["_id"]})
            self.conn.luxonis_dataset_document.delete_many(
                {
                    "$and": [
                        {"team_id": self.team_id},
                        {"_id": ObjectId(self.dataset_id)},
                    ]
                }
            )
        fo.delete_dataset(self.full_name)

    def _make_transaction(
        self,
        action: LDFTransactionType,
        sample_id: Optional[ObjectId] = None,
        field: Optional[str] = None,
        value: Optional[str] = None,
        component: Optional[str] = None,
    ) -> ObjectId:
        """Creates a transaction mongo document for a modification to the dataset"""

        if field == "segmentation":
            mask = value
            value = {"segmentation": "mask"}
        else:
            mask = None
        transaction_doc = fop.TransactionDocument(
            dataset_id=self.dataset_doc.id,
            created_at=datetime.utcnow(),
            executed=False,
            action=action.value,
            sample_id=sample_id,
            field=field,
            value={"value": value},
            mask=mask,
            component=component,
            version=-1,  # encodes no version yet assigned
        )
        transaction_doc = transaction_doc.save(upsert=True)

        return transaction_doc.id

    def _check_transactions_to_execute(self) -> Optional[List]:
        """Returns a list of transactions to execute if transactions need to be executed."""

        attribute, value = "executed", False

        transactions = list(
            self.conn.transaction_document.find(
                {"_dataset_id": self.dataset_doc.id}
            ).sort("created_at", pymongo.ASCENDING)
        )
        if len(transactions):
            if transactions[-1]["action"] != LDFTransactionType.END.value:
                i = -1
                while (
                    i != -len(transactions) - 1 and transactions[i][attribute] == value
                ):
                    self.conn.transaction_document.delete_many(
                        {"_id": transactions[i]["_id"]}
                    )
                    i -= 1
                return None
            else:
                i = -2
                while (
                    i != -len(transactions) - 1 and transactions[i][attribute] == value
                ):
                    i -= 1
                return transactions[i + 1 :]
        else:
            return None

    def _check_transactions_to_version(self) -> List:
        """Returns a list of transactions to version if transactions need to be versioned."""

        transactions = list(
            self.conn.transaction_document.find(
                {"_dataset_id": self.dataset_doc.id, "version": -1}
            ).sort("created_at", pymongo.ASCENDING)
        )
        return transactions

    def _execute_transaction(self, tid: ObjectId) -> None:
        """Mark a transaction as executed"""

        self.conn.transaction_document.update_one(
            {"_id": tid}, {"$set": {"executed": True}}
        )

    def _unexecute_transaction(self, tid: ObjectId) -> None:
        """Mark a transaction as not executed"""

        self.conn.transaction_document.update_one(
            {"_id": tid}, {"$set": {"executed": False}}
        )

    def _version_transaction(self, tid: ObjectId) -> None:
        """Mark a transaction as versioned"""

        self.conn.transaction_document.update_one(
            {"_id": tid}, {"$set": {"version": self.version}}
        )

    def _unversion_transaction(self, tid: ObjectId) -> None:
        """Mark a transaction as not versioned"""

        self.conn.transaction_document.update_one(
            {"_id": tid}, {"$set": {"version": -1}}
        )

    def _incr_version(self, media_change: bool, field_change: bool) -> None:
        """Increments the dataset version based on media changes or annotation changes"""

        if media_change:  # major change
            self.version += 1
        if field_change:  # minor change
            self.version += 0.1
        self.version = round(self.version, 1)

    def _decr_version(self, media_change: bool, field_change: bool) -> None:
        """Decrements the dataset version based on media changes or annotation changes"""

        if media_change:  # major change
            self.version -= 1
        if field_change:  # minor change
            self.version -= 0.1
        self.version = round(self.version, 1)

    def _add_filter(
        self, additions: List[Dict], from_bucket: bool = False
    ) -> Optional[Tuple[Dict, bool, bool]]:
        """Filters out any already existing additions to the dataset and makes transactions for changes to execute"""

        self._log_time()

        source = self.source
        components = self.source.components

        latest_view = self.fo_dataset.match(
            (F("latest") == True) | (F("version") == -1)
        )
        filepaths = {component_name: [] for component_name in components}
        for component_name in self.fo_dataset.group_slices:
            self.fo_dataset.group_slice = component_name
            paths = [sample["filepath"] for sample in latest_view]
            filepaths[component_name] = list(
                np.array(
                    [f"{path.split('/')[-2]}/{path.split('/')[-1]}" for path in paths]
                )
            )

        self.logger.info("Checking for additions or modifications...")

        transactions = []
        transaction_to_additions = {}
        media_change = False
        field_change = False
        filepath = None

        self._log_time("Filter setup", final=True)

        try:
            items = enumerate(additions)
            if not self.logger.isEnabledFor(logging.DEBUG) and self.logger.isEnabledFor(
                logging.INFO
            ):
                items = tqdm(items, total=len(additions))

            if from_bucket:
                if self.bucket_storage.value == "s3":
                    raise NotImplementedError
                elif self.bucket_storage.value == "gcs":
                    paths_from_gcs(self, additions)
                elif self.bucket_storage.value == "azure":
                    raise NotImplementedError

            for i, addition in items:
                self._log_time()
                # change the filepath for all components

                if not from_bucket:
                    for component_name in addition.keys():
                        try:
                            filepath = addition[component_name]["filepath"]
                        except:
                            raise AdditionsStructureException(
                                "Additions must be List[dict] with {'<component_name>': {'filepath':...}}. The <component_name> and filepath are required"
                            )
                        additions[i][component_name]["_old_filepath"] = filepath
                        if not data_utils.is_modified_filepath(self, filepath):
                            try:
                                hashpath, hash = data_utils.generate_hashname(filepath)
                            except:
                                raise AdditionNotFoundException(
                                    f"{filepath} does not exist"
                                )
                            additions[i][component_name]["instance_id"] = hash
                            additions[i][component_name]["_new_image_name"] = hashpath
                        granule = data_utils.get_granule(
                            filepath, addition, component_name
                        )
                        new_filepath = f"/{self.team_id}/datasets/{self.dataset_id}/{component_name}/{granule}"
                        additions[i][component_name]["filepath"] = new_filepath

                group = fo.Group(name=source.name)

                self._log_time("Filter addition setup")

                # check for ADD or UPDATE cases in the dataset
                for component_name in addition.keys():
                    filepath = addition[component_name]["filepath"]
                    granule = data_utils.get_granule(filepath, addition, component_name)
                    new_filepath = f"/{self.team_id}/datasets/{self.dataset_id}/{component_name}/{granule}"
                    candidate = f"{component_name}/{granule}"
                    if candidate not in filepaths[component_name]:
                        # ADD case
                        media_change = True
                        additions[i][component_name]["_group"] = group

                        self._log_time("Filter ADD find case")

                        if "class" in additions[i][component_name].keys():
                            data_utils.assert_classification_format(
                                self, additions[i][component_name]["class"]
                            )
                        if "boxes" in additions[i][component_name].keys():
                            data_utils.assert_boxes_format(
                                self, additions[i][component_name]["boxes"]
                            )
                        if "segmentation" in additions[i][component_name].keys():
                            data_utils.assert_segmentation_format(
                                self, additions[i][component_name]["segmentation"]
                            )
                        if "keypoints" in additions[i][component_name].keys():
                            data_utils.assert_keypoints_format(
                                self, additions[i][component_name]["keypoints"]
                            )

                        self._log_time("Filter ADD check annotation format")

                        tid = self._make_transaction(
                            LDFTransactionType.ADD,
                            sample_id=None,
                            field="filepath",
                            value=new_filepath,
                            component=component_name,
                        )
                        tid = str(tid)
                        transactions.append(tid)
                        transaction_to_additions[tid] = i

                        self._log_time("Filter ADD make transaction", final=True)
                    else:
                        # check for UPDATE
                        self.fo_dataset.group_slice = component_name
                        sample_view = self.fo_dataset.match(
                            F("filepath") == new_filepath
                        )
                        # find the most up to date sample
                        max_version = -np.inf
                        for sample in sample_view:
                            if sample.version == -1:
                                latest_sample = sample
                                max_version = sample.version
                                break
                            if sample.version > max_version:
                                latest_sample = sample
                                max_version = sample.version

                        self._log_time("Filter UPDATE find case")

                        changes = data_utils.check_fields(
                            self, latest_sample, addition, component_name
                        )

                        self._log_time("Filter UPDATE check fields")

                        for change in changes:
                            field, value = list(change.items())[0]
                            field_change = True
                            if field == "split":
                                group = data_utils.get_group_from_sample(
                                    self, latest_sample
                                )
                                for component_name in group:
                                    tid = self._make_transaction(
                                        LDFTransactionType.UPDATE,
                                        sample_id=group[component_name]["id"],
                                        field=field,
                                        value=value,
                                        component=component_name,
                                    )
                                    transactions.append(str(tid))
                            else:
                                tid = self._make_transaction(
                                    LDFTransactionType.UPDATE,
                                    sample_id=latest_sample._id,
                                    field=field,
                                    value=value,
                                    component=component_name,
                                )
                                transactions.append(str(tid))

                        self._log_time("Filter UPDATE make transaction", final=True)

            if media_change or field_change:
                self._make_transaction(LDFTransactionType.END)

                return transaction_to_additions, media_change, field_change
            else:
                return None

        except BaseException as e:
            for tid in transactions:
                self.conn.transaction_document.delete_many({"_id": ObjectId(tid)})
            try:
                if data_utils.is_modified_filepath(self, filepath):
                    filepath = additions[i][component_name]["_old_filepath"]
            except:
                pass
            raise DataTransactionException(filepath, type(e).__name__, str(e))

    def _add_extract(
        self, additions: List[Dict], from_bucket: bool = False
    ) -> List[Dict]:
        """Adds media to organized local or bucket storage"""

        self.logger.info("Extracting dataset media...")
        self._log_time()

        components = self.source.components
        if self.bucket_storage.value == "local":
            local_cache = str(
                Path(self.base_path)
                / "data"
                / self.team_id
                / "datasets"
                / self.dataset_id
            )
        else:
            local_cache = str(Path(self.base_path) / "tmp")
        os.makedirs(local_cache, exist_ok=True)
        for component_name in components:
            os.makedirs(str(Path(local_cache) / component_name), exist_ok=True)

        sync = False

        items = enumerate(additions)
        if not self.logger.isEnabledFor(logging.DEBUG) and self.logger.isEnabledFor(
            logging.INFO
        ):
            items = tqdm(items, total=len(additions))

        self._log_time("Extract setup", final=True)

        for i, addition in items:
            self._log_time()

            add_heatmaps = {}
            for component_name in addition.keys():
                if component_name not in components.keys():
                    raise Exception(
                        f"Component {component_name} is not present in source"
                    )
                component = addition[component_name]
                if "filepath" not in component.keys():
                    raise Exception("Must specify filepath for every component!")

                filepath = component["_old_filepath"]
                granule = data_utils.get_granule(filepath, addition, component_name)
                local_path = str(Path(local_cache) / component_name / granule)

                if self.bucket_storage.value == "local":
                    if os.path.exists(local_path):
                        self._log_time(
                            "Extract addition local check existence", final=True
                        )
                        continue
                    self._log_time("Extract addition local check existence")
                elif self.bucket_storage.value == "s3":
                    s3_path = component["filepath"][1:]
                    if check_s3_file_existence(
                        self.bucket,
                        s3_path,
                        self._get_credentials("AWS_S3_ENDPOINT_URL"),
                    ):
                        self._log_time(
                            "Extract addition s3 check existence", final=True
                        )
                        continue
                    self._log_time("Extract addition s3 check existence")
                    sync = True

                if not from_bucket:
                    if not data_utils.is_modified_filepath(self, filepath):
                        cmd = f"cp {filepath} {local_path}"
                        subprocess.check_output(cmd, shell=True)
                        self._log_time("Extract addition local cp", final=True)
                    else:
                        self.logger.warning(
                            f"Skipping extraction for {filepath} as path is likely corrupted due to memory assignment"
                        )

                    if self.compute_heatmaps and (
                        components[component_name].itype == IType.DISPARITY
                        or components[component_name].itype == IType.DEPTH
                    ):
                        self._log_time()

                        heatmap_component = f"{component_name}_heatmap"
                        os.makedirs(
                            str(Path(local_cache) / heatmap_component), exist_ok=True
                        )
                        im = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
                        if im.dtype == np.uint16:
                            im = im / 8
                        heatmap = (im / 96 * 255).astype(np.uint8)
                        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                        cv2.imwrite(
                            str(Path(local_cache) / heatmap_component / granule),
                            heatmap,
                        )
                        new_filepath = f"/{self.team_id}/datasets/{self.dataset_id}/{heatmap_component}/{granule}"
                        add_heatmaps[heatmap_component] = new_filepath

                        self._log_time(
                            "Extract addition heatmap computation", final=True
                        )

            if self.compute_heatmaps and not from_bucket and sync:
                for heatmap_component in add_heatmaps:
                    additions[i][heatmap_component] = {}
                    additions[i][heatmap_component]["filepath"] = add_heatmaps[
                        heatmap_component
                    ]

        self._log_time()

        if self.bucket_storage.value != "local":
            if from_bucket:
                if self.bucket_storage.value == "gcs":
                    copy_to_gcs(self, additions)
                else:
                    raise NotImplementedError
            else:
                if self.bucket_storage.value == "s3" and not from_bucket:
                    sync_to_s3(
                        bucket=self.bucket,
                        s3_dir=self.bucket_path,
                        local_dir=local_cache,
                        endpoint_url=self._get_credentials("AWS_S3_ENDPOINT_URL"),
                    )
                else:
                    sync_to_gcs(
                        bucket=self.bucket,
                        gcs_dir=self.bucket_path,
                        local_dir=local_cache,
                    )

        elif self.bucket_storage.value == "gcs" and not from_bucket:
            sync_to_gcs(
                bucket=self.bucket, gcs_dir=self.bucket_path, local_dir=local_cache
            )

        if self.bucket_storage.value != "local":
            shutil.rmtree(local_cache)

        self._log_time("Extract S3 sync", final=True)

        return additions

    def _add_execute(
        self,
        additions: Optional[List[Dict]] = None,
        transaction_to_additions: Optional[Dict] = None,
        annotated: bool = True,
    ) -> None:
        """Executes changes to annotations and other dataset fields in mongo"""

        self.logger.info("Executing changes to dataset...")

        source = self.source
        samples = []
        copied_samples = []

        transaction = None
        transactions = self._check_transactions_to_execute()

        try:
            if transactions is None:
                self.logger.info("There are no changes to the dataset to execute!")
                return

            items = transactions
            if not self.logger.isEnabledFor(logging.DEBUG) and self.logger.isEnabledFor(
                logging.INFO
            ):
                items = tqdm(transactions, total=len(transactions))

            for transaction in items:
                if transaction["action"] == LDFTransactionType.ADD.value:
                    if additions is None or transaction_to_additions is None:
                        raise Exception(
                            "additions and transaction_to_additions required for adding data"
                        )

                    if transaction_to_additions.get(str(transaction["_id"])) is None:
                        raise TransactionNotFoundException(
                            f"{str(transaction['_id'])} not found matching an addition"
                        )

                    addition = additions[
                        transaction_to_additions[str(transaction["_id"])]
                    ]
                    component_name = transaction["component"]
                    component = addition[component_name]
                    group = component["_group"]
                    sample = fo.Sample(
                        filepath=component["filepath"],
                        version=-1.0,
                        latest=False,
                        annotated=annotated,
                        tid=transaction["_id"].binary.hex(),
                        created_at=datetime.utcnow(),
                    )
                    sample[source.name] = group.element(component_name)

                    for ann in component.keys():
                        if ann == "class" and component["class"] is not None:
                            sample["class"] = data_utils.construct_class_label(
                                self, component["class"]
                            )
                        elif ann == "boxes" and component["boxes"] is not None:
                            sample["boxes"] = data_utils.construct_boxes_label(
                                self, component["boxes"]
                            )
                        elif (
                            ann == "segmentation"
                            and component["segmentation"] is not None
                        ):
                            sample[
                                "segmentation"
                            ] = data_utils.construct_segmentation_label(
                                self, component["segmentation"]
                            )
                        elif ann == "keypoints" and component["keypoints"] is not None:
                            sample["keypoints"] = data_utils.construct_keypoints_label(
                                self, component["keypoints"]
                            )
                        elif ann.startswith("_"):
                            continue  # ignore temporary attributes
                        else:
                            sample[ann] = component[ann]

                    if (
                        self.compute_heatmaps
                        and f"{component_name}_heatmap" in addition.keys()
                    ):
                        sample["heatmap"] = fo.Heatmap(
                            map_path=addition[f"{component_name}_heatmap"]["filepath"]
                        )

                    samples.append(sample)

                elif transaction["action"] == LDFTransactionType.UPDATE.value:
                    self.fo_dataset.group_slice = transaction["component"]
                    old_sample = self.fo_dataset[transaction["sample_id"]]

                    # check if there is already a working example with filepath and version as -1
                    sample = self.fo_dataset.match(
                        (F("filepath") == old_sample.filepath) & (F("version") == -1)
                    )
                    if len(sample):
                        # update the existing sample
                        for sample in sample:
                            break
                        if transaction["field"] in self.tasks:
                            if transaction["field"] == "class":
                                sample[
                                    transaction["field"]
                                ] = data_utils.construct_class_label(
                                    self, transaction["value"]["value"]
                                )
                            elif transaction["field"] == "boxes":
                                sample[
                                    transaction["field"]
                                ] = data_utils.construct_boxes_label(
                                    self, transaction["value"]["value"]
                                )
                            elif transaction["field"] == "segmentation":
                                sample[
                                    transaction["field"]
                                ] = data_utils.construct_segmentation_label(
                                    self, transaction["mask"]
                                )
                            elif transaction["field"] == "keypoints":
                                sample[
                                    transaction["field"]
                                ] = data_utils.construct_keypoints_label(
                                    self, transaction["value"]["value"]
                                )
                        else:
                            sample[transaction["field"]] = transaction["value"]["value"]
                        sample.save()
                    else:
                        # create a new sample which is a copy of an old sample
                        sample_dict = old_sample.to_dict()
                        sample = fo.Sample.from_dict(sample_dict)
                        if transaction["field"] in self.tasks:
                            if transaction["field"] == "class":
                                sample[
                                    transaction["field"]
                                ] = data_utils.construct_class_label(
                                    self, transaction["value"]["value"]
                                )
                            elif transaction["field"] == "boxes":
                                sample[
                                    transaction["field"]
                                ] = data_utils.construct_boxes_label(
                                    self, transaction["value"]["value"]
                                )
                            elif transaction["field"] == "segmentation":
                                sample[
                                    transaction["field"]
                                ] = data_utils.construct_segmentation_label(
                                    self, transaction["mask"]
                                )
                            elif transaction["field"] == "keypoints":
                                sample[
                                    transaction["field"]
                                ] = data_utils.construct_keypoints_label(
                                    self, transaction["value"]["value"]
                                )
                        else:
                            sample[transaction["field"]] = transaction["value"]["value"]

                        sample["tid"] = transaction["_id"].binary.hex()
                        sample["latest"] = False
                        sample["version"] = -1.0
                        sample["created_at"] = datetime.strptime(
                            sample["created_at"]["$date"], "%Y-%m-%dT%H:%M:%S.%fZ"
                        )
                        self.fo_dataset.add_sample(sample)
                        copied_samples.append(sample.id)

                self._execute_transaction(transaction["_id"])

        except BaseException as e:
            for sid in copied_samples:  # delete any copied samples
                del self.fo_dataset[sid]
            for t in transactions:  # set execution to False for all transactions
                self._unexecute_transaction(t["_id"])
            raise DataExecutionException(transaction, type(e).__name__, str(e))

        if len(samples):
            self.logger.info("Adding samples to Fiftyone")
            self.fo_dataset.add_samples(samples)

    def _add_defaults(self, additions: List[Dict]) -> None:
        """Adds any default attributes to the additions which may be missing"""

        for i, addition in enumerate(additions):
            for component_name in addition.keys():
                if "split" not in addition[component_name].keys():
                    addition[component_name]["split"] = "train"

    def add(
        self,
        additions: List[Dict],
        update_only: bool = False,
        from_bucket: bool = False,
        add_defaults: bool = True,
        annotated: bool = True,
    ) -> None:
        """
        Function to add data to LDF

        additions:
            A list of dictionaries describing each Voxel51 sample.
            Each dict contains keys of components
            The value of each component can contain the following keys:
                filepath (media)   : path to image, video, point cloud, or voxel
                -----
                class              : name of class for an entire image (Image Classification)
                boxes              : list of Nx5 numpy arrays [class, xmin, ymin, width, height] (Object Detection)
                segmentation       : numpy array where pixels correspond to integer values of classes
                keypoints          : list of classes and (x,y) keypoints
        update_only:
            Skip the media extraction step since if there are no new images
        from_bucket:
            True if adding images to a cloud bucket instead of locally
        add_defaults:
            Add default values, such as split. Useful when calling the method the first time. Best if set to false on updates.
        annotated:
            Whether annotations are included with the additions
        """

        if from_bucket and self.bucket_storage.value == "local":
            raise Exception("from_bucket must be False for local dataset!")

        self._check_transactions_to_execute()  # will ensure any interrupted transactions are clear

        post_filter = False

        try:
            # add defaults before
            if add_defaults:
                self._add_defaults(additions)

            filter_result = self._add_filter(additions, from_bucket)
            if filter_result is None:
                self.logger.info("No additions or modifications")
                return
            else:
                transaction_to_additions, media_change, field_change = filter_result
                post_filter = True

            if not update_only:
                additions = self._add_extract(
                    additions, transaction_to_additions, from_bucket
                )

            self._add_execute(additions, transaction_to_additions, annotated)

        except BaseException as e:
            # This will not handle cases where a network connection is interrupted, as cleanup requires a network connection
            self.logger.error(
                "-------- Cleaning up... please to not interrupt the program! --------"
            )
            if post_filter:
                # Additionally delete all transactions that were saved by _add_filter but not executed
                # All other cleanup is handled in _add_filter and _add_execute
                self.conn.transaction_document.delete_many(
                    {"_dataset_id": self.dataset_doc.id, "executed": False}
                )
            raise e

    def delete(self, instance_ids: List[str]) -> None:
        """
        Function to delete data by instance ID

        instance_ids:
            A list of instance IDs as strings
        """

        # TODO: add a try/except to gracefully handle any errors

        self._check_transactions_to_execute()  # will clear transactions any interrupted transactions

        sample_ids = self._update_from_instance_ids(instance_ids, "latest", False)

        for sid in sample_ids:
            tid = self._make_transaction(
                LDFTransactionType.DELETE,
                sample_id=sid,
            )
            self._execute_transaction(tid)

        tid = self._make_transaction(LDFTransactionType.END)
        self._execute_transaction(tid)

    def annotate(self, instance_ids: List[str]) -> None:
        """Marks a set of images as annotated"""

        self._update_from_instance_ids(instance_ids, "annotated", True)

    def create_version(
        self,
        note: str,
    ) -> None:
        """
        Creates a saved version of the dataset for training reproducibility

        note:
            A message associated with the version to help remember it
        """

        def get_current_sample(sample_collection, transaction):
            result = list(
                self.conn[sample_collection].find(
                    {
                        "tid": transaction["_id"].binary.hex(),
                        "filepath": {"$regex": f"/{transaction['component']}/[^/]*$"},
                    }
                )
            )
            if len(result):
                return result[0]["_id"], result[0]["annotated"]
            else:
                return None, None

        self.logger.info("Creating new version...")
        self._log_time()
        transaction = None
        version_changed, samples_added, samples_deprecated = False, False, False
        versioned_transactions = []

        try:
            try:
                sample_collection = self._get_sample_collection()
            except:
                raise Exception("Cannot find sample collection name")
            transactions = self._check_transactions_to_version()
            if transactions is None:
                self.logger.info("There are no changes to the dataset to version!")
                return

            contains_add = (
                True
                if np.sum(
                    [t["action"] == LDFTransactionType.ADD.value for t in transactions]
                )
                else False
            )
            contains_update = (
                True
                if np.sum(
                    [
                        t["action"] == LDFTransactionType.UPDATE.value
                        for t in transactions
                    ]
                )
                else False
            )
            contains_delete = (
                True
                if np.sum(
                    [
                        t["action"] == LDFTransactionType.DELETE.value
                        for t in transactions
                    ]
                )
                else False
            )
            media_change = contains_add or contains_delete
            field_change = contains_update

            if not media_change and not field_change:
                self.logger.info("No changes to version!")
                return

            self._incr_version(media_change, field_change)
            version_changed = True

            add_samples = set()
            deprecate_samples = set()

            items = transactions
            if not self.logger.isEnabledFor(logging.DEBUG) and self.logger.isEnabledFor(
                logging.INFO
            ):
                items = tqdm(items, total=len(transactions))

            self._log_time("Version setup", final=True)

            for transaction in items:
                self._log_time()

                if transaction["action"] in [
                    LDFTransactionType.END.value,
                    LDFTransactionType.DELETE.value,
                ]:
                    self._version_transaction(transaction["_id"])
                    versioned_transactions.append(transaction["_id"])
                    continue

                if transaction["action"] == LDFTransactionType.ADD.value:
                    sample_id, annotated = get_current_sample(
                        sample_collection, transaction
                    )
                    if sample_id is None:
                        # This case should ideally not happen unless there is a problem with rollback
                        # Another possibility would be to delete the transaction and throw a warning instead
                        raise Exception(
                            f"Sample is none for transaction ADD with tid {transaction['_id'].binary.hex()}"
                        )
                    if annotated:
                        add_samples.add(sample_id)

                    self._log_time("Version ADD")

                elif transaction["action"] == LDFTransactionType.UPDATE.value:
                    sample_id, annotated = get_current_sample(
                        sample_collection, transaction
                    )
                    if sample_id is None:
                        # This is fine to ignore, as it means a new sample which has multiple ADD and/or UPDATE
                        # The updates are already executed and we are just finding the latest
                        self._version_transaction(transaction["_id"])
                        versioned_transactions.append(transaction["_id"])
                        continue
                    else:
                        if annotated:
                            add_samples.add(sample_id)
                            deprecate_samples.add(transaction["sample_id"])

                    self._log_time("Version UPDATE")

                if annotated:
                    self._version_transaction(transaction["_id"])
                    versioned_transactions.append(transaction["_id"])

                self._log_time("Version transaction", final=True)

            if not len(add_samples) and not len(deprecate_samples):
                self.logger.info("No annotated samples to version!")

            self._log_time()

            add_samples = [ObjectId(sample_id) for sample_id in add_samples]
            self.conn[sample_collection].update_many(
                {"_id": {"$in": add_samples}},
                {"$set": {"latest": True, "version": self.version}},
            )
            samples_added = True

            self._log_time("Version add_samples")

            deprecate_samples = [ObjectId(sample_id) for sample_id in deprecate_samples]
            self.conn[sample_collection].update_many(
                {"_id": {"$in": deprecate_samples}},
                {"$set": {"latest": False}},
            )
            samples_deprecated = True

            self._log_time("Version deprecate_samples")

            result = self.conn[sample_collection].find({"latest": True})
            version_samples = [res["_id"].binary.hex() for res in result]
            self._log_time("Version gather version samples")
            self._save_version(version_samples, note)
            self._log_time("Version save version")

        except BaseException as e:
            self.logger.error(
                "-------- Cleaning up... please to not interrupt the program! --------"
            )

            version_view = f"version_{self.version}"
            if self.fo_dataset.has_saved_view(version_view):
                self.fo_dataset.delete_saved_view(version_view)
            if version_changed:
                self._decr_version(media_change, field_change)
            for t in versioned_transactions:
                self._unversion_transaction(t)
            if samples_added:
                self.conn[sample_collection].update_many(
                    {"_id": {"$in": add_samples}},
                    {"$set": {"latest": False, "version": self.version}},
                )
            if samples_deprecated:
                self.conn[sample_collection].update_many(
                    {"_id": {"$in": deprecate_samples}},
                    {"$set": {"latest": True}},
                )

            raise DataVersionException(transaction, type(e).__name__, str(e))

    def create_view(
        self,
        name: str,
        expr: fo.core.expressions.ViewExpression,
        version: Optional[str] = None,
    ) -> fo.core.view.DatasetView:
        """
        Creates a view of a dataset from any fiftyone query (ViewExpression)

        name:
            Name of the view
        expr:
            ViewExpression defining the dataset query
            example expr: (F("sunny") ==  True) && (F("scene") == "outdoors")
        version:
            Optionally apply the ViewExpression under a specific version
        """

        if version is None:
            version = self.version

        version_view = self.fo_dataset.load_saved_view(f"version_{version}")

        view = version_view.match(expr)

        self.fo_dataset.save_view(name, view)

        return view

    def export_to_json(self, version):
        pass
