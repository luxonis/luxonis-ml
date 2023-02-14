#!/usr/bin/env python3

import os, subprocess, glob
import json
import lakefs_client
from lakefs_client import models
from lakefs_client.client import LakeFSClient
from lakefs_client.model.object_stage_creation import ObjectStageCreation
from lakefs_client.model.commit_creation import CommitCreation
from .dataset import Bough
from tqdm import tqdm
import random, string, hashlib
import time
import hashlib
import pandas as pd
import pickle

class Client:

    def __init__(self, dataset):
        """
        dataset: LuxonisDataset to retrieve data from
        """

        self.dataset = dataset
        self.name = self.dataset.artifact_repo
        if self.name is None:
            raise Exception("Set artifact/lakefs repo with: luxonis_ml dataset init --lakefs [repo name]")
        self.path = dataset.path
        self.s3 = dataset.s3

        configuration = lakefs_client.Configuration()
        configuration.username = self.dataset._get_credentials('LAKEFS_ACCESS_KEY')
        configuration.password = self.dataset._get_credentials('LAKEFS_SECRET_ACCESS_KEY')
        configuration.host = self.dataset._get_credentials('LAKEFS_ENDPOINT_URL')
        self.lakefs_client = LakeFSClient(configuration)

class LuxonisDatasetArtifact(Client):

    def __init__(self, dataset):
        super(LuxonisDatasetArtifact, self).__init__(dataset)

    def _get_file_checksum(self, path):
        with open(path, 'rb') as file:
            file_hash = hashlib.md5()
            while chunk := file.read(8192):
                file_hash.update(chunk)
            checksum = file_hash.hexdigest()
        return checksum

    def _stage_object(self, path, file_metadata):
        object_stage_creation = ObjectStageCreation(
            physical_address=file_metadata[path]['s3_path'],
            checksum=file_metadata[path]['checksum'],
            size_bytes=file_metadata[path]['size'],
        )
        return object_stage_creation

    def checkout(self, branch=None, commit=None):

        if branch is None and commit is None:
            raise Exception("Must provide either branch or commit number!")

        branches = [result['id'] for result in self.lakefs_client.branches.list_branches(self.name).results]

        if commit is not None:

            try:
                self.lakefs_client.commits.get_commit(self.name, commit)
            except:
                raise Exception("Failed to find LakeFS commit")

            candidates = []
            for check_branch in branches:
                response = self.lakefs_client.commits.log_branch_commits(self.name, check_branch)
                for result in response['results']:
                    if commit == result['id']:
                        candidates.append(check_branch)
            if len(candidates) > 1 and branch is None:
                raise Exception(f"Commit found on multiple branches: {candidates} - please specify the desired branch!")
            elif branch is not None:
                if branch not in candidates:
                    raise Exception(f"Specified branch {branch} is not associated with this commit!")
            else:
                branch = candidates[0]

            branch_bough = branch.split('_')[1]
            self.dataset.bough = Bough(branch_bough)
            self.dataset.branch = branch
            self.dataset.commit_id = commit
            ref = commit

        elif branch is not None:
            if not branch.startswith('_'):
                branch = f"_{branch}"
            branch_bough = branch.split('_')[1]
            bough_values = [b.value for b in Bough]
            if branch_bough not in bough_values:
                branch = f"_{self.dataset.bough.value}{branch}"
            elif branch_bough != self.dataset.bough.value:
                i = bough_values.index(branch_bough)
                self.dataset.bough = [b for b in Bough][i]
                self.dataset._create_bough()
                print(f"Switched to bough {self.dataset.bough.value}")

            self.dataset.branch = branch

            if branch not in branches:
                self.dataset.commit_id = self.lakefs_client.branches.create_branch(repository=self.name, branch_creation=models.BranchCreation(name=branch, source=self.dataset.bough.value))
            else:
                self.dataset.commit_id = self.lakefs_client.branches.get_branch(repository=self.name, branch=branch)['commit_id']
            ref = branch

        # compare against local storage to find which files are added, modified, or removed
        self.dataset._reset_status()

        ls = self.lakefs_client.objects.list_objects(self.name, ref).to_dict()
        results = ls['results']
        while ls['pagination']['has_more']:
            ls = self.lakefs_client.objects.list_objects(self.name, ref, after=ls['pagination']['next_offset']).to_dict()
            results += ls['results']
        path_to_checksum = {f"{self.dataset.bough.value}/{response['path']}": response['checksum'] for response in results}

        cache_path = f"{self.dataset.bough.value}/.cache/cache.pickle"
        metadata_path = f"{self.dataset.bough.value}/metadata.parquet"
        local_paths_to_check = list(glob.iglob(f"{self.dataset.bough.value}/**/*.*", recursive=True)) + [metadata_path]

        for path in local_paths_to_check:
            if path in path_to_checksum.keys():
                local_checksum = self._get_file_checksum(path if path != metadata_path else "metadata.parquet")
                if local_checksum != path_to_checksum[path]:
                    self.dataset._modify_file(path if path != metadata_path else "metadata.parquet")
            else:
                self.dataset._add_file(path if path != metadata_path else "metadata.parquet")

        for path in path_to_checksum:
            if path == metadata_path: path = "metadata.parquet"
            if path == cache_path: path = ".cache/cache.pickle"
            if not os.path.exists(f"{self.path}/{path}"):
                self.dataset._remove_file(path)


    def push(self, message=None):
        if hasattr(self.dataset, 'branch'):
            t0 = time.time()

            if message is None: message = "luxonis_ml push: default commit message"

            random_str = ''.join(random.choices(string.ascii_letters+string.digits+string.punctuation, k=100))
            ldf_commit_number = hashlib.sha1(bytes(random_str, 'utf-8')).hexdigest()

            added_and_modified = self.dataset.added_files[self.dataset.bough.value] + self.dataset.modified_files[self.dataset.bough.value]

            if len(added_and_modified) > 0:
                print("Preparing local data")
                added_and_modified.append('.cache/cache.pickle') # always push cache

                path_to_info = {}
                for path in tqdm(added_and_modified):
                    if path.startswith(self.dataset.bough.value):
                        split_by = f"{self.dataset.bough.value}/"
                    else:
                        split_by = f"/{self.dataset.bough.value}/"
                    lakefs_path = path.split(split_by)[-1]
                    relative_path = f"{self.dataset.bough.value}/{lakefs_path}"
                    checksum = self._get_file_checksum(path)
                    size = os.path.getsize(path)
                    s3_path = f"s3://{self.dataset.bucket}/lakefs/{self.name}/_ldf_commits/{ldf_commit_number}/{path.split(split_by)[-1]}"
                    path_to_info[lakefs_path] = {
                        's3_path': s3_path,
                        'checksum': checksum,
                        'size': size
                    }

                    if path in ['.cache/cache.pickle', 'metadata.parquet']:
                        relative_path = path

                    local_dir = relative_path.split(relative_path.split('/')[-1])[0]
                    if not local_dir.startswith(self.dataset.bough.value):
                        local_dir = f"{self.dataset.bough.value}/{local_dir}"
                    os.makedirs(f"tmp/{local_dir}", exist_ok=True)
                    cmd = f"cp {relative_path} tmp/{local_dir}"
                    subprocess.check_output(cmd, shell=True)

                print("Uploading data to S3 - this could take some time...")
                cmd = f"b2 sync tmp/{self.dataset.bough.value} b2://{self.dataset.bucket}/lakefs/{self.name}/_ldf_commits/{ldf_commit_number}/"
                subprocess.check_output(cmd, shell=True)

                folders = os.listdir(f"tmp/{self.dataset.bough.value}")
                cmd = f"rm -rf tmp/"
                subprocess.check_output(cmd, shell=True)

                print("Staging in LakeFS...")
                for path in tqdm(path_to_info):
                    staged_object = self._stage_object(path, path_to_info)
                    self.lakefs_client.objects.stage_object(self.name, self.dataset.branch, path, staged_object)

                commit_creation = CommitCreation(message=message)
                try:
                    response = self.lakefs_client.commits.commit(self.name, self.dataset.branch, commit_creation)
                    self.dataset.commit_id = response['id']
                except:
                    raise Exception("No commit - no changes!")

            if len(self.dataset.removed_files[self.dataset.bough.value]) > 0:
                print("Removing data")
                for path in tqdm(self.dataset.removed_files[self.dataset.bough.value]):
                    if path.startswith(self.dataset.bough.value):
                        split_by = f"{self.dataset.bough.value}/"
                    else:
                        split_by = f"/{self.dataset.bough.value}/"
                    lakefs_path = path.split(split_by)[-1]
                    cmd = f"lakectl fs rm lakefs://{self.name}/{self.dataset.branch}/{lakefs_path}"
                    subprocess.check_output(cmd, shell=True)

                commit_creation = models.CommitCreation(message=message)
                response = self.lakefs_client.commits.commit(self.name, self.dataset.branch, commit_creation)
                self.dataset.commit_id = response['id']

            self.dataset._reset_status()

            t1 = time.time()
            print(f"Took {(t1-t0)/60} minutes!")

        else:
            raise Exception("Must checkout a branch before pushing!")

    def pull(self, s3_pull=False, rclone_s3_remote_name='b2'):

        if hasattr(self.dataset, 'commit_id'):
            t0 = time.time()

            if s3_pull:
                print("Running rclone to sync with S3...")
                bucket_path = self.dataset.s3_path.split('s3://')[-1]
                cmd = f"rclone sync --s3-force-path-style=true lakefs:{self.name}/{self.dataset.commit_id}/ {rclone_s3_remote_name}:{bucket_path}/{self.dataset.bough.value}"
                subprocess.check_output(cmd, shell=True)
                # TODO: fix pull location of cache and metadata
            else:
                print("Running rclone to sync locally...")
                cmd = f"rclone sync --s3-force-path-style=true lakefs:{self.name}/{self.dataset.commit_id}/ {self.dataset.bough.value}"
                subprocess.check_output(cmd, shell=True)

                cache_path = f"{self.dataset.bough.value}/.cache/cache.pickle"
                metadata_path = f"{self.dataset.bough.value}/metadata.parquet"
                if os.path.exists(cache_path):
                    cmd = f"mv {cache_path} .cache/cache.pickle"
                    subprocess.check_output(cmd, shell=True)
                if os.path.exists(metadata_path):
                    cmd = f"mv {metadata_path} metadata.parquet"
                    subprocess.check_output(cmd, shell=True)

                # ensure data gets updated from the pull
                with open(f"{self.path}/.cache/cache.pickle", "rb") as file:
                    self.dataset.__dict__.update(pickle.load(file).__dict__)
                pq_file = f"{self.dataset.path}/metadata.parquet"
                self.dataset.df = pd.read_parquet(pq_file)
                # TODO: some way to handle the case of being set back a commit?

            t1 = time.time()
            print(f"Took {(t1-t0)/60} minutes!")


class LuxonisModelArtifact(Client):

    def __init__(self, dataset):
        super(LuxonisModelArtifact, self).__init__(dataset)
