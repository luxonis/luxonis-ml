#!/usr/bin/env python3

import os, subprocess, shutil
import io
from pathlib import Path
import glob
import warnings
from hashlib import sha256
import cv2
from PIL import Image
import pickle
import json
from pathlib import Path
import boto3
import glob
import tarfile
from tqdm import tqdm
from enum import Enum
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from pandasql import sqldf

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

class Bough(Enum):
    """ A larger branch """
    RAW = "raw"
    PROCESSED = "processed"
    WEBDATASET = "webdataset"

class LDFComponent:
    htypes = [HType.IMAGE, HType.JSON]
    itypes = [IType.BGR, IType.MONO, IType.DISPARITY, IType.DEPTH]

    def __init__(self, name, htype, image_compression='png', itype=None):
        if htype not in self.htypes:
            raise Exception(f"{htype} is not a valid HType!")
        self.name = name
        self.htype = htype
        if htype == HType.IMAGE:
            self.compression = image_compression
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
        self.components['json'] = LDFComponent('json', htype=HType.JSON) # always have the JSON component
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

    def __init__(self, local_path=None, s3_path=None, artifact_repo=None, bough=Bough.PROCESSED):
        """
        s3_path: path to dataset on S3 bucket. Must start with s3://
        local_path: optional but recommended path to local copy for faster processing
        """

        if s3_path is None and local_path is None:
            raise Exception("S3 path or local path required to initialize LDF!")

        self._init_path(s3_path, local_path)

        self.artifact_repo = artifact_repo
        self.bough = bough

        credentials_cache_file = f'{str(Path.home())}/.cache/luxonis_ml/credentials.json'
        if os.path.exists(credentials_cache_file):
            with open(credentials_cache_file) as file:
                self.creds = json.load(file)
        else:
            self.creds = {}

        if not self.s3: # LOCAL path
            if not os.path.exists(self.path):
                os.mkdir(self.path)
            if self.path.startswith('./'):
                self.path = self.path.replace('./', f"{os.getcwd()}/")
            elif not self.path.startswith('/'):
                self.path = f"{os.getcwd()}/{self.path}"

        self._reset_status() # in this case it serves as an init

    def __enter__(self):

        if not self.s3:
            exists = True if os.path.exists(f"{self.path}/.cache/") else False
        else:
            resp = self.client.list_objects(Bucket=self.bucket, Prefix=f'{self.bucket_path}/.cache/', Delimiter='/', MaxKeys=1)
            exists = 'Contents' in resp

        if not exists:
            self.name = os.path.basename(os.path.normpath(os.getcwd()))
            self._create_bough()

            self.sources = {}
            self._create_directory(".cache")
            self.calibration_set = set()
            self.classes = []
            self.classes_by_task = {}
            self.keypoint_definitions = {}
            self.df = pd.DataFrame(columns=["basename", "split"])

        else:
            local_path = None if self.path.startswith('s3://') else self.path
            tmp_artifact_repo = self.artifact_repo
            tmp_s3_path = self.s3_path
            tmp_creds = self.creds

            if not self.s3:
                with open(f"{self.path}/.cache/cache.pickle", "rb") as file:
                    self.__dict__.update(pickle.load(file).__dict__)

                self._init_path(self.s3_path, local_path) # ensure path is not affected by cache if we switch between local/S3

                pq_file = f"{self.path}/metadata.parquet"
                self.df = pd.read_parquet(pq_file)

            else:
                s3_obj = self.client.get_object(Bucket=self.bucket, Key=f'{self.bucket_path}/.cache/cache.pickle')
                pickle_bytes = s3_obj['Body'].read()
                self.__dict__.update(pickle.loads(pickle_bytes).__dict__)
                self._init_boto3_client()

                self._init_path(self.s3_path, local_path) # ensure path is not affected by cache if we switch between local/S3

                # parquet with S3 initialization not implemented
                raise NotImplementedError()

            self._init_path(self.s3_path, local_path) # ensure path is not affected by cache if we switch between local/S3
            if tmp_artifact_repo is not None: # only overwrite artifact_repo if it is specifically provided
                self.artifact_repo = tmp_artifact_repo
            if tmp_s3_path is not None:
                self.s3_path = tmp_s3_path
            self.creds = tmp_creds

            if self.s3_path and not self.s3_path.startswith('s3://'):
                raise Exception("s3_path must start with s3:// !")

            # ensure paths for the desired bough exist
            self._create_bough()

        self._init_boto3_client()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.client = None

        if not self.s3:
            cache_file = f"{self.path}/.cache/cache.pickle"
            # self._modify_file(cache_file)
            with open(cache_file, "wb") as file:
                pickle.dump(self, file)

            pq_file = f"{self.path}/metadata.parquet"
            self.df.to_parquet(pq_file)
        else:
            self_bytes = pickle.dumps(self)
            self._init_boto3_client()
            self.client.put_object(Body=self_bytes, Bucket=self.bucket, Key=f"{self.bucket_path}/.cache/cache.pickle")

            # parquet with S3 initialization not implemented
            raise NotImplementedError()

    def _get_credentials(self, key):
        if key in self.creds.keys():
            return self.creds[key]
        else:
            return os.environ[key]

    def _init_boto3_client(self):
        if self.s3_path:
            self.client = boto3.client('s3',
                            endpoint_url = self._get_credentials('AWS_S3_ENDPOINT_URL'),
                            aws_access_key_id = self._get_credentials('AWS_ACCESS_KEY_ID'),
                            aws_secret_access_key = self._get_credentials('AWS_SECRET_ACCESS_KEY')
                          )
            self.bucket = self.s3_path.split('//')[1].split('/')[0]
            self.bucket_path = self.s3_path.split(self.bucket+'/')[-1]
        else:
            self.client, self.bucket, self.bucket_path = None, None, None

    def _init_path(self, s3_path, local_path):
        self.s3_path = s3_path
        if local_path is not None:
            self.path = local_path
        else:
            self.path = s3_path
        if self.s3_path is None:
            self.s3 = False
        else:
            self.s3 = True if self.path.startswith('s3://') else False

    def _create_directory(self, path, clear_contents=False):
        if not self.s3:
            full_path = f"{self.path}/{path}"
            if not os.path.exists(full_path):
                os.mkdir(full_path)
            elif clear_contents:
                shutil.rmtree(full_path)
                os.mkdir(full_path)
        else:
            resp = self.client.list_objects(Bucket=self.bucket, Prefix=f"{self.bucket_path}/{path}/", Delimiter='/', MaxKeys=1)
            if 'Contents' not in resp:
                self.client.put_object(Bucket=self.bucket, Key=f"{self.bucket_path}/{path}/")
            elif clear_contents:
                for content in resp['Contents']:
                    self.client.delete_object(Bucket=self.bucket, Key=content['Key']) # TODO: test after creating webdataset

    def _create_bough(self):
        self._create_directory(f"{self.bough.value}")
        if self.bough != Bough.WEBDATASET:
            self._create_directory(f"{self.bough.value}/calibration")
            self._create_directory(f"{self.bough.value}/ldf")

    def _add_file(self, path):
        path = path.replace(self.path, '').replace(self.path+'/', '') # ensure path is relative
        bough = self.bough.value
        if path in self.removed_files[bough]:
            self.removed_files[bough].remove(path)
        elif path not in self.added_files[bough]:
            self.added_files[bough].append(path)

    def _remove_file(self, path):
        path = path.replace(self.path, '').replace(self.path+'/', '') # ensure path is relative
        bough = self.bough.value
        if path in self.added_files[bough]:
            self.added_files[bough].remove(path)
        elif path not in self.removed_files[bough]:
            self.removed_files[bough].append(path)

    def _modify_file(self, path):
        path = path.replace(self.path, '').replace(self.path+'/', '') # ensure path is relative
        bough = self.bough.value
        if path in self.added_files[bough]:
            self.added_files[bough].remove(path)
        elif path not in self.modified_files[bough]:
            self.modified_files[bough].append(path)

    def _add_class(self, ann):

        class_name = ann['class_name']
        if class_name not in self.classes:
            self.classes.append(class_name)

        for task in ann:
            if task in ['class_name', 'class_id']:
                continue
            if task not in self.classes_by_task.keys():
                self.classes_by_task[task] = []
            if class_name not in self.classes_by_task[task]:
                self.classes_by_task[task].append(class_name)

        # ensure we use a "global" class index to be compatible with multiple sources
        class_id = self.classes.index(class_name)
        return class_id

    def _add_keypoint_definition(self, class_id, definition):
        if class_id not in self.keypoint_definitions.keys():
            self.keypoint_definitions[class_id] = definition

    def _reset_status(self):
        self.added_files = {b.value:[] for b in Bough}
        self.removed_files = {b.value:[] for b in Bough}
        self.modified_files = {b.value:[] for b in Bough}

    def _add_df_column(self, column_name):
        missing_data = [None for _ in range(len(self.df))]
        self.df[column_name] = missing_data

    def _add_df_row(self, basename, split, calibration, **kwargs):
        combined_dict = {
            'basename': basename,
            'split': split,
            'calibration': calibration
        }
        for key in kwargs:
            combined_dict[key] = kwargs[key]

        insert_columns = []
        for col in self.df.columns:
            if col not in combined_dict.keys():
                insert_columns.append(None)
            else:
                insert_columns.append(combined_dict[col])

        idx = self.df.index[self.df['basename'] == basename].tolist()
        if len(idx):
            self.df.loc[idx[0]] = insert_columns
        else:
            self.df.loc[len(self.df)] = insert_columns

    def _update_df_row(self, basename, **kwargs):
        idx = self.df.index[self.df['basename'] == basename].tolist()
        insert_columns = self.df.loc[idx[0]].to_dict()
        for col in insert_columns:
            if col in kwargs.keys():
                insert_columns[col] = kwargs[col]
        self.df.loc[idx[0]] = insert_columns

    def _query_df(self, query):
        df = self.df
        return sqldf(query, locals())

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

            # TODO: implement for OAK-1, OAK-D SR, other models

        elif name is not None and custom_components is not None:
            source = LDFSource(name, custom_components)
        else:
            raise Exception("name and custom_components required for custom sources")

        if name not in self.sources.keys():
            self.sources[name] = source
        else:
            warnings.warn(f"A source already exists under the name {name} -- not adding")

        return source

    def remove_source(self, source_name):
        """ Removes a source from the dataset and all associated files """
        if source_name in self.sources.keys():
            del self.sources[source_name]

            if not self.s3:
                source_files = glob.glob(f"{self.path}/{self.bough.value}/ldf/*.{source_name}.*")
                for path in source_files:
                    basename = path.split('/')[-1].split('.')[0]
                    self.df = self.df[self.df.basename != basename]
                    os.remove(path)
                    self._remove_file(path)
                self._modify_file('metadata.parquet')
            else:
                raise NotImplementedError()
        else:
            warnings.warn(f"Cannot remove source {source_name}. Source does not exist!")


    def add_data(self, source_name, data, calibration=None, split='train', **kwargs):
        """
        source_name: name of the LDFSource to add data to
        calibration: if using hardware, add calibration file here
        split: split for ML training
        kwargs: can pass any additional wanted metadata
        """

        # check that source and data match the dataset
        if source_name in self.sources.keys():
            source = self.sources[source_name]
            for component_name in data:
                if component_name not in source.components.keys():
                    raise Exception(f"Component {component_name} does not exist in source {source_name}!")
        else:
            raise Exception(f"Source {source_name} does not exist!")

        hash_bytes = pickle.dumps(data[source.main_component])
        basename = sha256(hash_bytes).hexdigest()

        if 'json' not in data.keys():
            data['json'] = {'metadata': {}} # ensure we have some JSON information, even without image annotations
        elif 'metadata' not in data['json'].keys():
            data['json']['metadata'] = {}

        if calibration is not None:
            calibration_bytes = pickle.dumps(calibration)
            calibration_name = sha256(calibration_bytes).hexdigest()

            if calibration_name not in self.calibration_set:
                if not self.s3:
                    path = f'{self.path}/{self.bough.value}/calibration/{calibration_name}.json'
                    with open(path, 'w') as file:
                        json.dump(calibration, file)
                    self._add_file(path)
                else:
                    self.client.put_object(
                        Body=str(calibration),
                        Bucket=self.bucket,
                        Key=f'{self.bucket_path}/{self.bough.value}/calibration/{calibration_name}.json',
                        ContentType=f'text/json'
                    )
                self.calibration_set.add(calibration_name)
        else:
            calibration_name = None

        non_json_components = list(data.keys())
        non_json_components.remove('json')

        for component_name in non_json_components:
            component = source.components[component_name]

            if component_name not in data['json'].keys():
                data['json'][component_name] = {}

            if component.htype == HType.IMAGE:
                data['json'][component_name]['image_metadata'] = {
                    'height': data[component_name].shape[0],
                    'width': data[component_name].shape[1]
                }

                ext = component.compression
                if not self.s3:
                    path = f'{self.path}/{self.bough.value}/ldf/{basename}.{source_name}.{component_name}.{ext}'
                    cv2.imwrite(path, data[component_name])
                    self._add_file(path)
                else:
                    image_bytes = io.BytesIO()
                    Image.fromarray(data[component_name]).save(image_bytes, ext)
                    image_bytes = image_bytes.getvalue()
                    self.client.put_object(
                        Body=image_bytes,
                        Bucket=self.bucket,
                        Key=f'{self.bucket_path}/{self.bough.value}/ldf/{basename}.{source_name}.{component_name}.{ext}',
                        ContentType=f'image/{ext}'
                    )

        # Add JSON component
        component_name = 'json'
        json_dict = data[component_name]
        if calibration is not None:
            if 'calibration' not in self.df.columns:
                self._add_df_column('calibration')
            json_dict['metadata']['calibration'] = calibration_name

        for attribute in kwargs:
            if attribute not in self.df.columns:
                self._add_df_column(attribute)
            json_dict['metadata'][attribute] = kwargs[attribute]

        for component in non_json_components:
            if component in json_dict.keys() and 'annotations' in json_dict[component].keys():
                for i, ann in enumerate(json_dict[component]['annotations']):
                    if 'class_name' not in ann.keys():
                        raise Exception("class_name is required in an annotation")
                    class_id = self._add_class(ann)
                    json_dict[component]['annotations'][i]['class'] = class_id

        if not self.s3:
            path = f'{self.path}/{self.bough.value}/ldf/{basename}.{source_name}.json'
            with open(path, 'w') as file:
                json.dump(data[component_name], file)
            self._add_file(path)
        else:
            self.client.put_object(
                Body=str(data[component_name]),
                Bucket=self.bucket,
                Key=f'{self.bucket_path}/{self.bough.value}/ldf/{basename}.json',
                ContentType=f'text/json'
            )

        self._add_df_row(basename, split, calibration_name, **kwargs)
        self._modify_file('metadata.parquet')


    def remove_data(self, ids):
        """ Removes a list of training examples from the dataset by hash ID """

        if not self.s3:
            for remove_id in ids:
                remove_files = glob.glob(f"{self.path}/{self.bough.value}/ldf/{remove_id}.*")
                for path in remove_files:
                    basename = path.split('/')[-1].split('.')[0]
                    self.df = self.df[self.df.basename != basename]
                    os.remove(path)
                    self._remove_file(path)
            self._modify_file('metadata.parquet')
        else:
            raise NotImplementedError()

    def update_annotations(self, basename, ann_data):
        json_path = glob.glob(f"{self.path}/{self.bough.value}/ldf/{basename}.*.json")[0]
        with open(json_path) as file:
            json_dict = json.load(file)

        for component in ann_data:
            for i, ann in enumerate(ann_data[component]['annotations']):
                if 'class_name' not in ann.keys():
                    raise Exception("class_name is required in an annotation")
                class_id = self._add_class(ann)
                ann_data[component]['annotations'][i]['class'] = class_id

        for component in ann_data:
            if component in json_dict.keys():
                json_dict[component]['annotations'] = ann_data[component]['annotations']
            else:
                warnings.warn(f"Component {component} not found for {basename}")

        with open(json_path, 'w') as file:
            json.dump(json_dict, file)

    def update_metadata(self, basename, **kwargs):
        json_path = glob.glob(f"{self.path}/{self.bough.value}/ldf/{basename}.*.json")[0]
        with open(json_path) as file:
            json_dict = json.load(file)

        for attribute in kwargs:
            if attribute not in self.df.columns:
                self._add_df_column(attribute)
            json_dict['metadata'][attribute] = kwargs[attribute]

        self._update_df_row(basename, **kwargs)

    def to_webdataset(self, view_name, query, sources=None, components=None, shard_size=20):

        if self.bough == Bough.WEBDATASET:
            webdataset_checked_out = True
            tmp_bough = Bough.PROCESSED
        else:
            webdataset_checked_out = False
            tmp_bough = self.bough
            self.bough = Bough.WEBDATASET
            self._create_bough()

        if not self.s3:
            prev_dir = os.getcwd()
            os.chdir(f"{self.path}/{tmp_bough.value}/ldf")
        else:
            prefix = f'{self.bucket_path}/{self.name}/ldf/'
            resp = self.client.list_objects(Bucket=self.bucket, Prefix=prefix, Delimiter='/')
            files = [content['Key'].split(prefix)[-1] for content in resp['Contents']]

        try:
            basenames = list(self._query_df(query)['basename'])
            file_list = []
            tar_count = 0
            for i, basename in tqdm(enumerate(basenames)):
                if i == 0: # extract all possible extensions based on one basename
                    possible_extensions = []
                    for filename in glob.glob(f"{basename}.*"):
                        possible_extensions.append(filename.replace(basename,""))
                for extension in possible_extensions:
                    file_list.append(f"{basename}{extension}")

                if (i+1) % shard_size == 0 or (i+1) == len(basenames):
                    tar_count_str = str(tar_count).zfill(6)

                    if not self.s3:
                        path = f'{self.path}/{self.bough.value}/{view_name}_{tar_count_str}.tar'
                        with tarfile.open(f'{self.path}/{self.bough.value}/{view_name}_{tar_count_str}.tar', 'w') as file:
                            for item in file_list:
                                file.add(item)
                        self._add_file(path)
                    else:
                        # TODO: optimize this, it's not really usable
                        tar_fileobj = io.BytesIO()
                        # with tarfile.open(fileobj=tar_fileobj, mode="w|") as tar:
                        with tarfile.open('tmp.tar', mode="w|") as tar:
                            for item in file_list:
                                s3_obj = self.client.get_object(Bucket=self.bucket, Key=f'{self.bucket_path}/{self.name}/ldf/{item}')
                                obj_bytes = s3_obj['Body'].read()
                                tf = tarfile.TarInfo(item)
                                tf.size = len(obj_bytes)
                                tar.addfile(tf, io.BytesIO(obj_bytes))

                        with open('tmp.tar', 'rb') as file:
                            self.client.put_object(
                                Body=file,
                                Bucket=self.bucket,
                                Key=f'{self.bucket_path}/{self.bough.value}/{view_name}_{tar_count_str}.tar',
                            )

                    file_list = []
                    tar_count += 1
        except Exception as e: # make sure we catch any errors to change back directories
            if not self.s3:
                os.chdir(prev_dir)
            print(f"Error in webdataset conversion: {e}")
            import traceback
            traceback.print_exc()

        if not self.s3:
            os.chdir(prev_dir)

        if not webdataset_checked_out:
            self.bough = tmp_bough

    def add_metadata(self, query, metadata_fn=None, **kwargs):
        """
        query: SQL query to get the basenames of data to change
        metadata_fn: optional function that takes as input the JSON file dict and returns a dictionary of metadata to add
        kwargs: any metadata to add that doesn't require special computation
        """
        # note: make sure we update metadata in the JSON as well as parquet table
        pass

    def update_annotation(self, query, update_fn):
        """
        query: SQL query to get the basenames of data to change
        update_fn: optional function that takes as input the JSON file dict and returns an update JSON dict
        """
        pass
