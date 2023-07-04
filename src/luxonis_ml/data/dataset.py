import fiftyone as fo
import fiftyone.core.odm as foo
import luxonis_ml.data.fiftyone_plugins as fop
import luxonis_ml.data.utils.data_utils as data_utils
from fiftyone import ViewField as F
import os, subprocess, shutil
from pathlib import Path
import glob
import warnings
import cv2
from PIL import Image
import pickle
import json
import boto3
import glob
from tqdm import tqdm
from enum import Enum
import numpy as np
from datetime import datetime
import pymongo
from bson.objectid import ObjectId
from .version import LuxonisVersion

class HType(Enum):
    """ Individual file type """
    IMAGE = 1
    JSON = 2

class IType(Enum):
    """ Image type for IMAGE HType """
    BGR = 1
    MONO = 2
    DISPARITY = 3
    DEPTH = 4

class LDFTransactionType(Enum):
    """ The type of transaction """
    END = "END"
    ADD = "ADD"
    UPDATE = "UPDATE"
    DELETE = "DELETE"

class LDFComponent:
    htypes = [HType.IMAGE, HType.JSON]
    itypes = [IType.BGR, IType.MONO, IType.DISPARITY, IType.DEPTH]

    def __init__(self, name, htype, itype=None):
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
    types = ['real_oak', 'synthetic_oak', 'custom']

    def __init__(self, name, components, main_component=None, source_type=None):
        self.name = name
        self.components = {component.name:component for component in components}
        if main_component is not None:
            self.main_component = main_component
        else:
            self.main_component = list(self.components.keys())[0] # make first component main component by default
        if source_type is not None:
            self.source_type = source_type
        else:
            self.source_type = 'custom'

    def add_component(self, component):
        self.components += component

class LuxonisDataset:
    """
    Uses the LDF (Luxonis Dataset Format)

    The goal is to standardize an arbitrary number of devices and sensor configurations, synthetic OAKs, and other sources
    """

    @staticmethod
    def create(
        team_id,
        team_name,
        dataset_name,
    ):
        
        dataset_doc = fop.LuxonisDatasetDocument(
            team_id=team_id,
            team_name=team_name,
            dataset_name=dataset_name
        )
        dataset_doc = dataset_doc.save(upsert=True)
        
        return str(dataset_doc.id)

    def __init__(
        self, 
        team_id, 
        dataset_id, 
        team_name=None,
        dataset_name=None,
        bucket_type='local', 
        override_bucket_type=False
    ):
        """
        team name: team under which you can find all datasets
        dataset name: name of the dataset
        bucket_type: underlying storage for images, which can be local or an AWS bucket
        override_bucket_type: option to change underlying storage from saved setting in DB
        """

        self.conn = foo.get_db_conn()

        self.team_id = team_id
        self.dataset_id = dataset_id
        self.team_name = team_name
        self.dataset_name = dataset_name
        self.full_name = f"{self.team_id}-{self.dataset_id}"
        self.bucket_type = bucket_type
        self.bucket_choices = ['local', 'aws']
        self.override_bucket_type = override_bucket_type
        if self.bucket_type not in self.bucket_choices:
            raise Exception(f"Bucket type {self.bucket_type} is not supported!")

        credentials_cache_file = f'{str(Path.home())}/.cache/luxonis_ml/credentials.json'
        if os.path.exists(credentials_cache_file):
            with open(credentials_cache_file) as file:
                self.creds = json.load(file)
        else:
            self.creds = {}

        self._init_path()
        self.version = 0.0
        self.tasks = ['class', 'boxes', 'segmentation', 'keypoints']
        self.compute_heatmaps = True # TODO: could make this configurable

    def __enter__(self):

        if self.full_name in fo.list_datasets():
            self.fo_dataset = fo.load_dataset(self.full_name)
        else:
            self.fo_dataset = fo.Dataset(self.full_name)
            self.conn.luxonis_dataset_document.update_one
        self.fo_dataset.persistent = True

        res = list(self.conn.luxonis_dataset_document.find(
            { "$and": [{"team_id": self.team_id}, {"_id": ObjectId(self.dataset_id)}] }
        ))

        if len(res):
            assert len(res) == 1
            self.dataset_doc = fop.LuxonisDatasetDocument.objects.get(
                team_id=self.team_id,
                id=ObjectId(self.dataset_id)
            )

            tmp_bucket_type = self.bucket_type

            self._doc_to_class()

            if self.override_bucket_type:
                self.bucket_type = tmp_bucket_type

            self._init_path()

        else:
            raise Exception("Dataset not found!")

        return self

    def __exit__(self, exc_type, exc_value, traceback):

        self._class_to_doc()
        self.dataset_doc.save(upsert=True)

    def _doc_to_class(self):

        if hasattr(self.dataset_doc, 'path') and self.dataset_doc.path is not None:
            self.path = self.dataset_doc.path
        if hasattr(self.dataset_doc, 'bucket_type') and self.dataset_doc.bucket_type is not None:
            self.bucket_type = self.dataset_doc.bucket_type
        if hasattr(self.dataset_doc, 'current_version') and self.dataset_doc.current_version is not None:
            self.version = self.dataset_doc.current_version

        doc = list(self._get_source())
        if len(doc):
            doc = doc[0]
            components = [
                LDFComponent(name, HType(htype), IType(itype)) \
                for name, htype, itype in list(zip(doc['component_names'], doc['component_htypes'], doc['component_itypes']))
            ]
            main_component = components[0].name
            self.source = LDFSource(
                name=doc['name'],
                components=components,
                main_component=main_component,
                source_type=doc['source_type']
            )
        else:
            self.source = None

    def _class_to_doc(self):

        self.dataset_doc.fo_dataset_id = self.fo_dataset._doc.id
        self.dataset_doc.path = self.path
        self.dataset_doc.bucket_type = self.bucket_type
        self.dataset_doc.current_version = self.version

    def _get_credentials(self, key):
        if key in self.creds.keys():
            return self.creds[key]
        else:
            return os.environ[key]

    def _get_source(self):
        return self.conn.luxonis_source_document.find(
            { "$and": [{"_luxonis_dataset_id": self.dataset_doc.id}] }
        ).limit(1)

    def _init_boto3_client(self):
        self.client = boto3.client('s3',
                        endpoint_url = self._get_credentials('AWS_S3_ENDPOINT_URL'),
                        aws_access_key_id = self._get_credentials('AWS_ACCESS_KEY_ID'),
                        aws_secret_access_key = self._get_credentials('AWS_SECRET_ACCESS_KEY')
                      )
        self.bucket = self.path.split('//')[1].split('/')[0]
        self.bucket_path = self.path.split(self.bucket+'/')[-1]
        # self._create_directory(self.bucket_path)

    def _init_path(self):
        if self.bucket_type == 'local':
            self.path = f"{str(Path.home())}/.cache/luxonis_ml/data/{self.team_id}/datasets/{self.dataset_id}"
            os.makedirs(self.path, exist_ok=True)
        elif self.bucket_type == 'aws':
            self.path = f"s3://{self._get_credentials('AWS_BUCKET')}/{self.team_id}/datasets/{self.dataset_id}"
            self._init_boto3_client()

    def _create_directory(self, path, clear_contents=False):
        if self.bucket_type == 'local':
            os.makedirs(path, exist_ok=True)
        elif self.bucket_type == 'aws':
            resp = self.client.list_objects(Bucket=self.bucket, Prefix=f"{self.bucket_path}/{path}/", Delimiter='/', MaxKeys=1)
            if 'Contents' not in resp:
                self.client.put_object(Bucket=self.bucket, Key=f"{self.bucket_path}/{path}/")
            elif clear_contents:
                for content in resp['Contents']:
                    self.client.delete_object(Bucket=self.bucket, Key=content['Key'])

    def _save_version(self, samples, note):

        version = LuxonisVersion(self, samples=samples, note=note)
        samples = version.get_samples()

        version_view = self.fo_dataset[samples]
        self.fo_dataset.save_view(f"version_{self.version}", version_view)

    def create_source(self,
        name=None,
        oak_d_default=False,
        # calibration=None,
        # calibration_mapping=None,
        custom_components=None):
        """
        name (str): optional rename for standard sources, required for custom sources
        calibration (dict): DepthAI calibration file for real device (TODO: or Unity devices)
        calibration_mapping (dict): for calibration files from non-standard products,
            maps camera ID to a LDFComponent for the sensor and ID pairs to disparities
        custom_components (list): list of LDFComponents for a custom source

        TODO: more features with calibration
        """

        if oak_d_default:
            if name is None: name = "oak_d"
            components = [
                LDFComponent('left', htype=HType.IMAGE, itype=IType.MONO),
                LDFComponent('right', htype=HType.IMAGE, itype=IType.MONO),
                LDFComponent('color', htype=HType.IMAGE, itype=IType.BGR),
                LDFComponent('disparity', htype=HType.IMAGE, itype=IType.DISPARITY),
                LDFComponent('depth', htype=HType.IMAGE, itype=IType.DEPTH)
            ]

            source = LDFSource(name, components, main_component='color', source_type='real_oak')

        elif name is not None and custom_components is not None:
            source = LDFSource(name, custom_components)
        else:
            raise Exception("name and custom_components required for custom sources")

        if self.dataset_doc.id is None:
            self._class_to_doc()
            self.dataset_doc.save(safe=True, upsert=True)
        res = self._get_source()
        if len(list(res)): # exists
            warnings.warn(f"Updating from a previously saved source!")
            self.conn.luxonis_source_document.delete_many(
                { "_luxonis_dataset_id": self.dataset_doc.id }
            )
        luxonis_dataset_id = self.dataset_doc.id
        source_doc = fop.LuxonisSourceDocument(
            luxonis_dataset_id=luxonis_dataset_id,
            name=source.name,
            source_type=source.source_type,
            component_names=[c.name for _,c in source.components.items()],
            component_htypes=[c.htype.value for _,c in source.components.items()],
            component_itypes=[c.itype.value for _,c in source.components.items()],
        )
        source_doc.save(upsert=True)

        self.source = source

        self.fo_dataset.add_group_field(self.source.name, default=self.source.main_component)

        return source

    def launch_app(self):

        session = fo.launch_app(dataset)

    def set_classes(self, classes, task=None):
        if task is not None:
            if task not in self.tasks:
                raise Exception(f'Task {task} is not a supported task')
            self.fo_dataset.classes[task] = classes
        else:
            for task in self.tasks:
                self.fo_dataset.classes[task] = classes

    def set_mask_targets(self, mask_targets):
        if 0 in mask_targets.keys():
            raise Exception('Cannot set 0! This is assumed to be the background class')
        self.fo_dataset.mask_targets = {
            'segmentation': mask_targets
        }

    def set_skeleton(self, skeleton):
        """
        class_name (str): name of class to add fo skeleton for
        skeleton (dict):
            labels (list): list of strings for what each keypoint represents
            egdes (list): list of length length-2 lists of keypoint indices denoting
                          how keypoints are connected

        NOTE: this only supports one class, which seems to be a fiftyone limitation
        """

        self.fo_dataset.skeletons.update({
            'keypoints': fo.KeypointSkeleton(
                labels=skeleton['labels'],
                edges=skeleton['edges']
            )
        })
        self.fo_dataset.save()

    def sync_from_cloud(self):

        self.non_streaming_dir = f"{str(Path.home())}/.cache/luxonis_ml/data/{self.team_id}/datasets/{self.dataset_id}"
        os.makedirs(self.non_streaming_dir, exist_ok=True)

        if self.bucket_type == 'local':
            print("This is a local dataset! Cannot sync")
        elif self.bucket_type == 'aws':
            print("Syncing from cloud...")
            cmd = f"aws s3 sync s3://{self.bucket}/{self.team_id}/datasets/{self.dataset_id} \
                    {self.non_streaming_dir} \
                    --endpoint-url={self._get_credentials('AWS_S3_ENDPOINT_URL')}"
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            while True:
                output = process.stdout.readline()
                if output == b'' and process.poll() is not None:
                    break
                if output:
                    print(output.strip().decode())

    def get_classes(self):

        classes = set()
        classes_by_task = {}
        for task in self.fo_dataset.classes:
            if len(self.fo_dataset.classes[task]):
                classes_by_task[task] = self.fo_dataset.classes[task]
                for cls in self.fo_dataset.classes[task]:
                    classes.add(cls)
        mask_classes = set()
        if 'segmentation' in self.fo_dataset.mask_targets.keys():
            for key in self.fo_dataset.mask_targets['segmentation']:
                mask_classes.add(self.fo_dataset.mask_targets['segmentation'][key])
        classes_by_task['segmentation'] = list(mask_classes)
        classes = list(classes.union(mask_classes))
        classes.sort()

        return classes, classes_by_task

    def get_classes_count(self):
        """ Returns dictionary with number of occurances for each class. If no class label is present returns empty dict."""
        try:
            count_dict = self.fo_dataset.count_values("class.classifications.label")
        except:
            warnings.warn("No 'class' label present in the dataset. Returning empty dictionary.")
            count_dict = {}
        return count_dict

    def delete_dataset(self):

        """
        Deletes the entire dataset, aside from the images
        """

        res = list(self.conn.luxonis_dataset_document.find(
            { "$and": [{"team_id": self.team_id}, {"_id": ObjectId(self.dataset_id)}] }
        ))
        if len(res):
            ldf_doc = res[0]

            self.conn.luxonis_source_document.delete_many(
                { "_luxonis_dataset_id": ldf_doc["_id"] }
            )
            self.conn.version_document.delete_many(
                { "_dataset_id": ldf_doc["_dataset_id"] }
            )
            self.conn.transaction_document.delete_many(
                { "_dataset_id": ldf_doc["_id"] }
            )
            self.conn.luxonis_dataset_document.delete_many(
                { "$and": [{"team_id": self.team_id}, {"_id": ObjectId(self.dataset_id)}] }
            )
        fo.delete_dataset(self.full_name)

    def _make_transaction(
        self,
        action,
        sample_id=None,
        field=None,
        value=None,
        component=None
    ):

        transaction_doc = fop.TransactionDocument(
            dataset_id=self.dataset_doc.id,
            created_at=datetime.utcnow(),
            executed=False,
            action=action.value,
            sample_id=sample_id,
            field=field,
            value={'value':value},
            component=component,
            version=-1 # encodes no version yet assigned
        )
        transaction_doc = transaction_doc.save(upsert=True)

        return transaction_doc.id

    def _check_transactions(self, for_versioning=False):

        if for_versioning:
            attribute, value = 'version', -1
        else:
            attribute, value = 'executed', False

        transactions = list(self.conn.transaction_document.find(
            { '_dataset_id': self.dataset_doc.id }
        ).sort('created_at', pymongo.ASCENDING))
        if len(transactions):
            if transactions[-1]['action'] != LDFTransactionType.END.value:
                i = -1
                while i != -len(transactions)-1 and \
                    transactions[i][attribute] == value:

                    self.conn.transaction_document.delete_many(
                        { '_id': transactions[i]['_id'] }
                    )
                    i -= 1
                return None
            else:
                i = -2
                while i != -len(transactions)-1 and \
                    transactions[i][attribute] == value:
                    i -= 1
                return transactions[i+1:]
        else:
            return None

    def _execute_transaction(self, tid):

        self.conn.transaction_document.update_one(
            { '_id': tid}, { '$set': { 'executed': True } }
        )

    def _version_transaction(self, tid):

        self.conn.transaction_document.update_one(
            { '_id': tid}, { '$set': { 'version': self.version } }
        )

    def _incr_version(self, media_change, field_change):
        if media_change: # major change
            self.version += 1
        if field_change: # minor change
            self.version += 0.1
        self.version = round(self.version, 1)

    def _add_filter(self, additions):
        """
        Filters out any additions to the dataset already existing
        """

        source = self.source
        components = self.source.components

        latest_view = self.fo_dataset.match(
            (F("latest") == True) | (F("version") == -1)
        )
        filepaths = {component_name: [] for component_name in components}
        for component_name in self.fo_dataset.group_slices:
            self.fo_dataset.group_slice = component_name
            paths = [sample['filepath'] for sample in latest_view]
            filepaths[component_name] = list(np.array([f"{path.split('/')[-2]}/{path.split('/')[-1]}" for path in paths]))

        print("Checking for additions or modifications...")

        transaction_to_additions = {}
        media_change = False
        field_change = False

        for i, addition in tqdm(enumerate(additions), total=len(additions)):
            # change the filepath for all components
            for component_name in addition.keys():
                filepath = addition[component_name]['filepath']
                additions[i][component_name]['_old_filepath'] = filepath
                granule = data_utils.get_granule(filepath, addition, component_name)
                new_filepath = f"/{self.team_id}/datasets/{self.dataset_id}/{component_name}/{granule}"
                additions[i][component_name]['filepath'] = new_filepath

            group = fo.Group(name=source.name)

            # check for ADD or UPDATE cases in the dataset
            for component_name in addition.keys():
                filepath = addition[component_name]['filepath']
                granule = data_utils.get_granule(filepath, addition, component_name)
                new_filepath = f"/{self.team_id}/datasets/{self.dataset_id}/{component_name}/{granule}"
                candidate = f"{component_name}/{granule}"
                if candidate not in filepaths[component_name]:
                    # ADD case
                    media_change = True
                    additions[i][component_name]['_group'] = group
                    tid = self._make_transaction(
                        LDFTransactionType.ADD,
                        sample_id=None,
                        field='filepath',
                        value=new_filepath,
                        component=component_name
                    )
                    transaction_to_additions[tid] = i
                else:
                    # check for UPDATE
                    self.fo_dataset.group_slice = component_name
                    sample_view = self.fo_dataset.match(
                        F("filepath") == new_filepath
                    )
                    # find the most up to date sample
                    max_version = -np.inf
                    for sample in sample_view:
                        if sample.version > max_version:
                            latest_sample = sample
                            max_version = sample.version

                    changes = data_utils.check_fields(
                        self,
                        latest_sample,
                        addition,
                        component_name
                    )
                    for change in changes:
                        field, value = list(change.items())[0]
                        field_change = True
                        tid = self._make_transaction(
                            LDFTransactionType.UPDATE,
                            sample_id=latest_sample._id,
                            field=field,
                            value=value,
                            component=component_name
                        )

        if media_change or field_change:
            self._make_transaction(
                LDFTransactionType.END
            )

            return transaction_to_additions, media_change, field_change
        else:
            return None

    def _add_extract(self, additions, from_bucket):
        """
        Filters out any additions to the dataset already existing
        """

        print("Extracting dataset media...")

        components = self.source.components
        if self.bucket_type == 'local':
            local_cache = f'{str(Path.home())}/.cache/luxonis_ml/data/{self.team_id}/datasets/{self.dataset_id}'
        elif self.bucket_type == 'aws':
            local_cache = f'{str(Path.home())}/.cache/luxonis_ml/tmp'
        os.makedirs(local_cache, exist_ok=True)
        for component_name in components:
            os.makedirs(f'{local_cache}/{component_name}', exist_ok=True)

        for i, addition in tqdm(enumerate(additions), total=len(additions)):

            add_heatmaps = {}
            for component_name in addition.keys():
                if component_name not in components.keys():
                    raise Exception(f"Component {component_name} is not present in source")
                component = addition[component_name]
                if 'filepath' not in component.keys():
                    raise Exception("Must specify filepath for every component!")

                filepath = component['_old_filepath']
                granule = data_utils.get_granule(filepath, addition, component_name)
                # new_filepath = f"/{self.team_id}/datasets/{self.dataset_id}/{component_name}/{granule}"
                # additions[i][component_name]['filepath'] = new_filepath

                if from_bucket:
                    old_prefix = filepath.split(f"s3://{self.bucket}/")[-1]
                    new_prefix = f'{self.bucket_path}/{component_name}/{granule}'
                    self.client.copy_object(
                        Bucket=self.bucket,
                        Key=new_prefix,
                        CopySource={'Bucket': self.bucket, 'Key': old_prefix}
                    )
                else:
                    cmd = f"cp {filepath} {local_cache}/{component_name}/{granule}"
                    subprocess.check_output(cmd, shell=True)

                if self.compute_heatmaps and not from_bucket and \
                (components[component_name].itype == IType.DISPARITY or \
                components[component_name].itype == IType.DEPTH):

                    heatmap_component = f"{component_name}_heatmap"
                    os.makedirs(f'{local_cache}/{heatmap_component}', exist_ok=True)
                    im = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
                    if im.dtype == np.uint16:
                        im = im/8
                    heatmap = (im/96*255).astype(np.uint8)
                    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    cv2.imwrite(f'{local_cache}/{heatmap_component}/{granule}', heatmap)
                    new_filepath = f"/{self.team_id}/datasets/{self.dataset_id}/{heatmap_component}/{granule}"
                    add_heatmaps[heatmap_component] = new_filepath

            if self.compute_heatmaps and not from_bucket:
                for heatmap_component in add_heatmaps:
                    additions[i][heatmap_component] = {}
                    additions[i][heatmap_component]['filepath'] = add_heatmaps[heatmap_component]

        if self.bucket_type == 'aws' and not from_bucket:
            print("Syncing to S3 bucket...")
            cmd = f"aws s3 sync {local_cache} \
                    s3://{self.bucket}/{self.team_id}/datasets/{self.dataset_id} \
                    --endpoint-url={self._get_credentials('AWS_S3_ENDPOINT_URL')}"
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            while True:
                output = process.stdout.readline()
                if output == b'' and process.poll() is not None:
                    break
                if output:
                    print(output.strip().decode())

        if self.bucket_type != 'local':
            shutil.rmtree(local_cache)

        return additions

    def _add_execute(self, additions=None, transaction_to_additions=None):

        source = self.source
        samples = []

        transactions = self._check_transactions()

        if transactions is None:
            raise Exception('There are no changes to the dataset to execute!')

        for transaction in transactions:

            if transaction['action'] == LDFTransactionType.ADD.value:

                if additions is None or transaction_to_additions is None:
                    raise Exception("additions and transaction_to_additions required for adding data")

                addition = additions[transaction_to_additions[transaction['_id']]]
                component_name = transaction['component']
                component = addition[component_name]
                group = component['_group']
                sample = fo.Sample(
                    filepath=component['filepath'],
                    version=self.version,
                    latest=True
                )
                sample[source.name] = group.element(component_name)

                for ann in component.keys():
                    if ann == 'class' and component['class'] is not None:
                        sample['class'] = data_utils.construct_class_label(self, component['class'])
                    elif ann == 'boxes' and component['boxes'] is not None:
                        sample['boxes'] = data_utils.construct_boxes_label(self, component['boxes'])
                    elif ann == 'segmentation' and component['segmentation'] is not None:
                        sample['segmentation'] = data_utils.construct_segmentation_label(self, component['segmentation'])
                    elif ann == 'keypoints' and component['keypoints'] is not None:
                        sample['keypoints'] = data_utils.construct_keypoints_label(self, component['keypoints'])
                    elif ann.startswith('_'):
                        continue # ignore temporary attributes
                    else:
                        sample[ann] = component[ann]

                if 'split' not in component.keys():
                    sample['split'] = 'train' # default split

                if self.compute_heatmaps and f'{component_name}_heatmap' in addition.keys():
                    sample['heatmap'] = fo.Heatmap(map_path=addition[f'{component_name}_heatmap']['filepath'])

                sample['tid'] = transaction['_id'].binary.hex()
                sample['latest'] = False
                sample['version'] = -1.0
                samples.append(sample)

            elif transaction['action'] == LDFTransactionType.UPDATE.value:

                self.fo_dataset.group_slice = transaction['component']
                old_sample = self.fo_dataset[transaction['sample_id']]

                # check if there is already a working example with filepath and version as -1
                sample = self.fo_dataset.match(
                    (F('filepath') == old_sample.filepath) & \
                    (F('version') == -1)
                )
                if len(sample):
                    # update the existing sample
                    for sample in sample:
                        break
                    if transaction['field'] in self.tasks:
                        if transaction['field'] == 'class':
                            sample[transaction['field']] = data_utils.construct_class_label(self, transaction['value']['value'])
                        elif transaction['field'] == 'boxes':
                            sample[transaction['field']] = data_utils.construct_boxes_label(self, transaction['value']['value'])
                        elif transaction['field'] == 'segmentation':
                            sample[transaction['field']] = data_utils.construct_segmentation_label(self, transaction['value']['value'])
                        elif transaction['field'] == 'keypoints':
                            sample[transaction['field']] = data_utils.construct_keypoints_label(self, transaction['value']['value'])
                    else:
                        sample[transaction['field']] = transaction['value']['value']
                    sample.save()
                else:
                    # create a new sample which is a copy of an old sample
                    sample_dict = old_sample.to_dict()
                    sample = fo.Sample.from_dict(sample_dict)
                    if transaction['field'] in self.tasks:
                        if transaction['field'] == 'class':
                            sample[transaction['field']] = data_utils.construct_class_label(self, transaction['value']['value'])
                        elif transaction['field'] == 'boxes':
                            sample[transaction['field']] = data_utils.construct_boxes_label(self, transaction['value']['value'])
                        elif transaction['field'] == 'segmentation':
                            sample[transaction['field']] = data_utils.construct_segmentation_label(self, transaction['value']['value'])
                        elif transaction['field'] == 'keypoints':
                            sample[transaction['field']] = data_utils.construct_keypoints_label(self, transaction['value']['value'])
                    else:
                        sample[transaction['field']] = transaction['value']['value']

                    sample['tid'] = transaction['_id'].binary.hex()
                    sample['latest'] = False
                    sample['version'] = -1.0
                    self.fo_dataset.add_sample(sample)

            self._execute_transaction(transaction['_id'])

        if len(samples):
            self.fo_dataset.add_samples(samples)

    def add(
        self,
        additions,
        media_exists=False,
        from_bucket=False,
    ):
        """
        Function to add data and automatically log transactions

        additions: a list of dictionaries describing each Voxel51 sample
            Each dict contains keys of components
            The value of each component can contain the following keys:
                filepath (media) : path to image, video, point cloud, or voxel
                -----
                class              : name of class for an entire image (Image Classification)
                boxes              : list of Nx5 numpy arrays [class, xmin, ymin, width, height] (Object Detection)
                segmentation       : numpy array where pixels correspond to integer values of classes
                keypoints          : list of classes and (x,y) keypoints
        from_bucket: True if adding images to a cloud bucket instead of locally
        """

        if from_bucket and self.bucket_type == 'local':
            raise Exception('from_bucket must be False for local dataset!')

        self._check_transactions() # will clear transactions any interrupted transactions

        # TODO: add a try/except to gracefully handle any errors

        filter_result = self._add_filter(additions)
        if filter_result is None:
            print('No additions or modifications')
            return
        else:
            transaction_to_additions, media_change, field_change = filter_result

        if not media_exists:
            additions = self._add_extract(additions, from_bucket)

        self._add_execute(additions, transaction_to_additions)

    def delete(self, deletions):
        """
        Function to delete data by sample ID

        deletions: a list of sample IDs as strings
        """

        # TODO: add a try/except to gracefully handle any errors

        self._check_transactions() # will clear transactions any interrupted transactions

        for delete_id in deletions:

            # assume we want to delete all components in a group
            sample = self.fo_dataset[delete_id]
            gid = sample[self.source.name]['id']
            group = self.fo_dataset.get_group(gid)
            for component_name in group:
                sample = self.fo_dataset[group[component_name]['id']]
                tid = self._make_transaction(
                    LDFTransactionType.DELETE,
                    sample_id=sample.id,
                    component=component_name
                )
                sample.latest = False
                sample.save()
                self._execute_transaction(tid)

        tid = self._make_transaction(LDFTransactionType.END)
        self._execute_transaction(tid)

    def create_version(self, note):

        def get_current_sample(transaction):
            sample = self.fo_dataset.match(
                F("tid") == transaction['_id'].binary.hex()
            )
            if not len(sample):
                return None
            for sample in sample:
                break
            return sample

        def get_previous_sample(transaction):
            sample = self.fo_dataset[transaction['sample_id']]
            return sample

        # TODO: exception handling

        transactions = self._check_transactions(for_versioning=True)
        if transactions is None:
            raise Exception('There are no changes to the dataset to version!')

        contains_add = True if np.sum([t['action'] == LDFTransactionType.ADD.value for t in transactions]) else False
        contains_update = True if np.sum([t['action'] == LDFTransactionType.UPDATE.value for t in transactions]) else False
        contains_delete = True if np.sum([t['action'] == LDFTransactionType.DELETE.value for t in transactions]) else False
        media_change = contains_add or contains_delete
        field_change = contains_update
        self._incr_version(media_change, field_change)

        add_samples = set()
        deprecate_samples = set()
        for transaction in transactions:

            if transaction['action'] != LDFTransactionType.END.value:
                self.fo_dataset.group_slice = transaction['component']

            if transaction['action'] == LDFTransactionType.ADD.value:
                sample = get_current_sample(transaction)
                add_samples.add(sample['id'])

            elif transaction['action'] == LDFTransactionType.UPDATE.value:
                sample = get_current_sample(transaction)
                if sample is None: # a new sample which has had both ADD and UPDATE
                    # sample = get_previous_sample(transaction)
                    continue
                else:
                    add_samples.add(sample['id'])
                    prev_sample = get_previous_sample(transaction)
                    deprecate_samples.add(prev_sample['id'])

            self._version_transaction(transaction['_id'])

        for sample_id in add_samples:
            sample = self.fo_dataset[sample_id]
            sample['latest'] = True
            sample['version'] = self.version
            sample.save()
        for sample_id in deprecate_samples:
            sample = self.fo_dataset[sample_id]
            sample['latest'] = False
            sample.save()

        version_samples = []
        for component_name in self.source.components:
            self.fo_dataset.group_slice = component_name
            latest_view = self.fo_dataset.match(
                F("latest") == True
            )
            version_samples += [sample.id for sample in latest_view]
        self._save_version(version_samples, note)

    def create_view(self, name, expr, version=None):

        if version is None:
            version = self.version

        version_view = self.fo_dataset.load_saved_view(f'version_{version}')

        view = version_view.match(expr)

        self.fo_dataset.save_view(name, view)

        return view
