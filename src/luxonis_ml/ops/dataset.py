#!/usr/bin/env python3

import fiftyone as fo
import fiftyone.core.odm as foo
import luxonis_ml.fiftyone_plugins as fop
import luxonis_ml.ops.utils.data_utils as data_utils
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

    def __init__(self, team_name, dataset_name, bucket_type='local', override_bucket_type=False):
        """
        team name: team under which you can find all datasets
        dataset name: name of the dataset
        bucket_type: underlying storage for images, which can be local or an AWS bucket
        override_bucket_type: option to change underlying storage from saved setting in DB
        """

        self.conn = foo.get_db_conn()

        self.team = team_name
        self.name = dataset_name
        self.full_name = f"{self.team}-{self.name}"
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
        self.version = 0
        self.tasks = ['class', 'boxes', 'segmentation', 'keypoints']
        self.compute_heatmaps = True # TODO: could make this configurable

    def __enter__(self):

        if self.full_name in fo.list_datasets():
            self.fo_dataset = fo.load_dataset(self.full_name)
        else:
            self.fo_dataset = fo.Dataset(self.full_name)
        self.fo_dataset.persistent = True

        res = list(self.conn.luxonis_dataset_document.find(
            { "$and": [{"team_name": self.team}, {"dataset_name": self.name}] }
        ))

        if len(res): # already exists
            assert len(res) == 1
            self.dataset_doc = fop.LuxonisDatasetDocument.objects.get(
                team_name=self.team,
                dataset_name=self.name
            )

            tmp_bucket_type = self.bucket_type

            self._doc_to_class()

            if self.override_bucket_type:
                self.bucket_type = tmp_bucket_type

            self._init_path()

        else: # create new
            dataset_id = self.fo_dataset._doc.id
            dataset_id_str = dataset_id.binary.hex()
            self.dataset_doc = fop.LuxonisDatasetDocument(
                dataset_id=dataset_id,
                dataset_id_str=dataset_id_str,
                team_name=self.team,
                dataset_name=self.name,
                path=self.path,
                bucket_type='aws',
                current_version=self.version,
            )

            self._init_path()
            self.source = None # assuming a single dataset can only have one source

        return self

    def __exit__(self, exc_type, exc_value, traceback):

        self._class_to_doc()
        self.dataset_doc.save(safe=True, upsert=True)

    def _doc_to_class(self):

        self.path = self.dataset_doc.path
        self.bucket_type = self.dataset_doc.bucket_type
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
            self.path = f"{str(Path.home())}/.cache/luxonis_ml/data/{self.team}/datasets/{self.name}"
            os.makedirs(self.path, exist_ok=True)
        elif self.bucket_type == 'aws':
            self.path = f"s3://{self._get_credentials('AWS_BUCKET')}/{self.team}/datasets/{self.name}"
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

        self.non_streaming_dir = f"{str(Path.home())}/.cache/luxonis_ml/data/{self.team}/datasets/{self.name}"
        os.makedirs(self.non_streaming_dir, exist_ok=True)

        if self.bucket_type == 'local':
            print("This is a local dataset! Cannot sync")
        elif self.bucket_type == 'aws':
            print("Syncing from cloud...")
            cmd = f"aws s3 sync s3://{self.bucket}/{self.team}/datasets/{self.name} \
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

    def delete(self):
        """
        Deletes the entire dataset, aside from the images
        """

        res = list(self.conn.luxonis_dataset_document.find(
            { "$and": [{"team_name": self.team}, {"dataset_name": self.name}] }
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
                { "$and": [{"team_name": self.team}, {"dataset_name": self.name}] }
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
            current_version=self.version
        )
        transaction_doc = transaction_doc.save(upsert=True)

        return transaction_doc.id

    def _check_transactions(self):

        # TODO: maybe this should look for the last non-executed transaction instead of END

        transactions = list(self.conn.transaction_document.find(
            { '_dataset_id': self.dataset_doc.id }
        ).sort('created_at', pymongo.ASCENDING))
        if len(transactions):
            if transactions[-1]['action'] != LDFTransactionType.END.value:
                i = -1
                while transactions[i]['action'] != LDFTransactionType.END.value \
                    and i != -len(transactions)-1:

                    self.conn.transaction_document.delete_many(
                        { '_id': transactions[i]['_id'] }
                    )
                    i -= 1
                return None
            else:
                i = -2
                while transactions[i]['action'] != LDFTransactionType.END.value \
                    and i != -len(transactions):
                    i -= 1
                return transactions[i+1:]
        else:
            return None

    def _incr_version(self, media_change, field_change):
        if media_change: # major change
            self.version += 1
        if field_change: # minor change
            self.version += 0.1

    def _add_filter(self, additions):
        """
        Filters out any additions to the dataset already existing
        """

        source = self.source
        components = self.source.components

        latest_view = self.fo_dataset.match(
            F("latest") == True
        )
        filepaths = [sample['filepath'] for sample in latest_view]
        filepaths = np.array([f"{path.split('/')[-2]}/{path.split('/')[-1]}" for path in filepaths])

        print("Checking for additions or modifications...")

        transaction_to_additions = {}
        media_change = False
        field_change = False

        for i, addition in tqdm(enumerate(additions), total=len(additions)):
            for component_name in addition.keys():
                filepath = addition[component_name]['filepath']
                granule = data_utils.get_granule(filepath, addition, component_name)
                new_filepath = f"/{self.team}/datasets/{self.name}/{component_name}/{granule}"
                candidate = f"{component_name}/{granule}"
                if candidate not in list(filepaths):
                    # ADD case
                    media_change = True
                    tid = self._make_transaction(
                        LDFTransactionType.ADD,
                        sample_id=None,
                        field='filepath',
                        value=new_filepath
                    )
                    transaction_to_additions[tid] = i
                    break
                else:
                    # check for UPDATE
                    mount_path = f"/{self.team}/datasets/{self.name}/{component_name}/{granule}"
                    sample_view = self.fo_dataset.match( # TODO: could probably make this more efficient
                        F("filepath") == mount_path
                    )
                    # find the most up to date sample
                    max_version = -1
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
                        field, value = change.items()
                        field_change = True
                        tid = self._make_transaction(
                            LDFTransactionType.UPDATE,
                            sample_id=latest_sample._id,
                            field=field,
                            value=value,
                            component=component_name
                        )
                        transaction_to_additions[tid] = i

        # additions = filtered
        # if len(additions) == 0:
        #     print('No new additions!')
        #     return None
        # else:
        #     version_samples = []
        #     for i, sample in tqdm(enumerate(latest_view), total=len(latest_view)):
        #         if updated[i]:
        #             sample.latest = False
        #             sample.save()
        #         else:
        #             version_samples.append(sample.id)

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
            local_cache = f'{str(Path.home())}/.cache/luxonis_ml/data/{self.team}/datasets/{self.name}'
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

                filepath = component['filepath']
                granule = data_utils.get_granule(filepath, addition, component_name)
                new_filepath = f"/{self.team}/datasets/{self.name}/{component_name}/{granule}"
                additions[i][component_name]['filepath'] = new_filepath

                if from_bucket:
                    new_prefix = f'{self.bucket_path}/{component_name}/{granule}'
                    self.client.copy_object(
                        Bucket=self.bucket,
                        Key=new_prefix,
                        CopySource={'Bucket': self.bucket, 'Key': filepath}
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
                    new_filepath = f"/{self.team}/datasets/{self.name}/{heatmap_component}/{granule}"
                    add_heatmaps[heatmap_component] = new_filepath

            if self.compute_heatmaps and not from_bucket:
                for heatmap_component in add_heatmaps:
                    additions[i][heatmap_component] = {}
                    additions[i][heatmap_component]['filepath'] = add_heatmaps[heatmap_component]

        if self.bucket_type == 'aws' and not from_bucket:
            print("Syncing to S3 bucket...")
            cmd = f"aws s3 sync {local_cache} \
                    s3://{self.bucket}/{self.team}/datasets/{self.name} \
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

    def _add_execute(self, additions, transaction_to_additions):

        source = self.source
        group = fo.Group(name=source.name)
        samples = []
        latest_view = self.fo_dataset.match(
            F("latest") == True
        )
        sample_ids = np.array([sample['_id'] for sample in latest_view])
        samples = [sample for sample in latest_view]

        transactions = self._check_transactions()

        if transactions is None:
            raise Exception('There are no changes to the dataset to execute!')

        for transaction in transactions:
            if transaction['action'] == LDFTransactionType.ADD.value:
                addition = additions[transaction_to_additions[transaction['_id']]]
                component_names = [component_name for component_name in addition.keys() if not component_name.endswith('_heatmap')]
                for component_name in component_names:

                    component = addition[component_name]

                    sample = fo.Sample(filepath=component['filepath'], version=self.version, latest=True)
                    sample[source.name] = group.element(component_name)

                    for ann in component.keys():
                        if ann == 'class':
                            sample['class'] = fo.Classification(label=component['class'])
                        elif ann == 'boxes':
                            sample['boxes'] = fo.Detections(detections=[
                                fo.Detection(
                                    label=box[0] if isinstance(box[0], str) else self.fo_dataset.classes['boxes'][int(box[0])],
                                    bounding_box=box[1:5]
                                )
                                for box in component['boxes']
                            ])
                        elif ann == 'segmentation':
                            sample['segmentation'] = fo.Segmentation(mask=component['segmentation'])
                        elif ann == 'keypoints':
                            sample['keypoints'] = fo.Keypoints(keypoints=[
                                fo.Keypoint(
                                    label=kp[0] if isinstance(kp[0], str) else self.fo_dataset.classes['keypoints'][int(kp[0])],
                                    points=kp[1]
                                )
                                for kp in component['keypoints']
                            ])
                        elif ann == 'new_image_name':
                            continue # ignore this as an attribute
                        else:
                            sample[ann] = component[ann]

                    if 'split' not in component.keys():
                        sample['split'] = 'train' # default split

                    if self.compute_heatmaps and f'{component_name}_heatmap' in addition.keys():
                        sample['heatmap'] = fo.Heatmap(map_path=addition[f'{component_name}_heatmap']['filepath'])

                    samples.append(sample)

            elif transaction['action'] == LDFTransactionType.UPDATE.value:

                addition = additions[transaction_to_additions[transaction['_id']]]
                idx = np.where(sample == transaction['sample_id'])
                previous_sample = samples[idx]
                # TODO: change only the field specified in the update transaction

            elif transaction['action'] == LDFTransactionType.END.value:
                # TODO: compute version_samples
                new_ids = self.fo_dataset.add_samples(samples)
                return

    def add(
        self,
        additions,
        from_bucket=False,
    ):
        """
        Function to add data and automatically version the data

        dataset: LuxonisDataset instance
        dict: a list of dictionaries describing each Voxel51 sample
            Each dict contains keys of components
            The value of each component can contain the following keys:
                filepath (media) : path to image, video, point cloud, or voxel
                -----
                class              : name of class for an entire image (Image Classification)
                boxes              : list of Nx6 numpy arrays [class, xmin, ymin, width, height] (Object Detection)
                segmentation       : numpy array where pixels correspond to integer values of classes
                keypoints          : list of classes and (x,y) keypoints
        """

        if from_bucket and self.bucket_type == 'local':
            raise Exception('from_bucket must be False for local dataset!')

        self._check_transactions() # will clear transactions any interrupted transactions

        # TODO: add a try/except to gracefully handle any errors
        # try:

        filter_result = self._add_filter(additions)
        if filter_result is None:
            print('No additions or modifications')
            return
        else:
            transaction_to_additions, media_change, field_change = filter_result

        self._incr_version(media_change, field_change)

        additions = self._add_extract(additions, from_bucket)

        version_samples = self._add_execute(additions, transaction_to_additions)

        # self._save_version(version_samples, note)

        # except Exception as e:
