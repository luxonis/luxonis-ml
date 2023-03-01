#!/usr/bin/env python3

from luxonis_ml.ops import *
import argparse
import sys, os
import warnings
import json
from pathlib import Path

def _config():
    AWS_ACCESS_KEY_ID = input("AWS Access Key: ")
    AWS_SECRET_ACCESS_KEY = input("AWS Secret Access Key: ")
    AWS_S3_ENDPOINT_URL = input("AWS Endpoint URL: ")
    LAKEFS_ACCESS_KEY = input("LakeFS Access Key: ")
    LAKEFS_SECRET_ACCESS_KEY = input("LakeFS Secret Access Key: ")
    LAKEFS_ENDPOINT_URL = input("LakeFS Endpoint URL: ")

    cache_dir = f'{str(Path.home())}/.cache/luxonis_ml'
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    cache_file = f"{cache_dir}/credentials.json"
    credentials = {
        "AWS_ACCESS_KEY_ID": AWS_ACCESS_KEY_ID,
        "AWS_SECRET_ACCESS_KEY": AWS_SECRET_ACCESS_KEY,
        "AWS_S3_ENDPOINT_URL": AWS_S3_ENDPOINT_URL,
        "LAKEFS_ACCESS_KEY": LAKEFS_ACCESS_KEY,
        "LAKEFS_SECRET_ACCESS_KEY": LAKEFS_SECRET_ACCESS_KEY,
        "LAKEFS_ENDPOINT_URL": LAKEFS_ENDPOINT_URL,
    }

    with open(cache_file, 'w') as file:
        json.dump(credentials, file)

    print(f"Credentials saved to {cache_file}!")

def _dataset_init(args):

    name = os.path.basename(os.path.normpath(os.getcwd()))
    if args.s3_path:
        if name != os.path.basename(os.path.normpath(args.s3_path)):
            raise Exception(f"Please match end of s3 prefix with local directory name {name}")
    if args.lakefs_repo:
        if name != args.lakefs_repo:
            raise Exception(f"Please match lakefs repo with local directory name {name}")

    if os.path.exists(f"{os.getcwd()}/.cache"):
        warnings.warn(f"Warning: LDF already exists in this directory")

    with LuxonisDataset(local_path=os.getcwd(), s3_path=args.s3_path, artifact_repo=args.lakefs_repo) as dataset:
        pass

    print("Initialization successful!")

def _dataset_set_s3(args):

    name = os.path.basename(os.path.normpath(os.getcwd()))
    if name != os.path.basename(os.path.normpath(args.s3_path)):
        raise Exception(f"Please match end of s3 prefix with local directory name {name}")

    with LuxonisDataset(local_path=os.getcwd(), s3_path=args.s3_path) as dataset:
        pass

def _dataset_set_lakefs_repo(args):

    name = os.path.basename(os.path.normpath(os.getcwd()))
    if name != args.lakefs_repo:
        raise Exception(f"Please match lakefs repo with local directory name {name}")

    with LuxonisDataset(local_path=os.getcwd(), artifact_repo=args.lakefs_repo) as dataset:
        pass

def _dataset_checkout(args):

    if args.branch is None and args.commit is None:
        raise Exception("Must provide either branch (--branch) or commit number (--commit)!")

    with LuxonisDataset(local_path=os.getcwd()) as dataset:
        artifact = LuxonisDatasetArtifact(dataset)
        artifact.checkout(branch=args.branch, commit=args.commit)

        if args.commit is not None:
            print(f"Checked out commit {dataset.commit_id}")
        else:
            print(f"Checked out branch {dataset.branch}")

def _dataset_status(args):

    with LuxonisDataset(local_path=os.getcwd()) as dataset:

        artifact = LuxonisDatasetArtifact(dataset)
        # calling checkout on current branch will update status
        artifact.checkout(branch=dataset.branch, commit=dataset.commit_id)

        print(f"Branch: {dataset.branch}")
        print(f"Commit: {dataset.commit_id}", end='\n\n')
        if args.verbose:
            if len(dataset.added_files[dataset.bough.value]):
                print(f"Added files: {dataset.added_files[dataset.bough.value]}", end="\n\n")
            if len(dataset.modified_files[dataset.bough.value]):
                print(f"Modified files: {dataset.modified_files[dataset.bough.value]}", end="\n\n")
            if len(dataset.removed_files[dataset.bough.value]):
                print(f"Removed files: {dataset.removed_files[dataset.bough.value]}", end="\n\n")
        else:
            print(f"{len(dataset.added_files[dataset.bough.value])} files added")
            print(f"{len(dataset.modified_files[dataset.bough.value])} files modified")
            print(f"{len(dataset.removed_files[dataset.bough.value])} files removed")

def _dataset_pull(args):

    with LuxonisDataset(local_path=os.getcwd()) as dataset:
        artifact = LuxonisDatasetArtifact(dataset)
        artifact.pull(s3_pull=args.pull_to_s3, rclone_s3_remote_name=args.rclone_remote)

def _dataset_push(args):

    with LuxonisDataset(local_path=os.getcwd()) as dataset:
        artifact = LuxonisDatasetArtifact(dataset)
        artifact.push(message=args.message)

def _dataset_webdataset(args):

    with LuxonisDataset(local_path=os.getcwd()) as dataset:
        dataset.to_webdataset(view_name=args.view, query=args.query)

def main():

    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest='main')
    parser_config = subparsers.add_parser('config', help='Configure API keys for luxonis_ml')
    parser_dataset = subparsers.add_parser('dataset', help='Dataset programs to work with LDF')

    dataset_subparsers = parser_dataset.add_subparsers(dest='dataset')

    parser_dataset_init = dataset_subparsers.add_parser('init', help='Initialize a dataset')
    parser_dataset_init.add_argument('-s3', '--s3_path', type=str, help='S3 path where LDF is stored', default=None)
    parser_dataset_init.add_argument('-lfs', '--lakefs_repo', type=str, help='Name of LakeFS repo', default=None)

    parser_dataset_set_s3 = dataset_subparsers.add_parser('set_s3', help='Set the S3 prefix for streaming')
    parser_dataset_set_s3.add_argument('-s3', '--s3_path', type=str, help='S3 path where LDF is stored', default=None)

    parser_dataset_set_lakefs_repo = dataset_subparsers.add_parser('set_lakefs_repo', help='Set the LakeFS repo for dataset versioning')
    parser_dataset_set_lakefs_repo.add_argument('-lfs', '--lakefs_repo', type=str, help='Name of LakeFS repo', default=None)

    parser_dataset_checkout = dataset_subparsers.add_parser('checkout', help='Checkout a LakeFS branch or commit')
    parser_dataset_checkout.add_argument('-b', '--branch', type=str, help='LakeFS branch to checkout', default=None)
    parser_dataset_checkout.add_argument('-c', '--commit', type=str, help='LakeFS branch to checkout', default=None)

    parser_dataset_status = dataset_subparsers.add_parser('status', help='Show branch, commit, and changed files')
    parser_dataset_status.add_argument('-v', '--verbose', action='store_true', help='Show all changed files')

    parser_dataset_pull = dataset_subparsers.add_parser('pull', help='Pull dataset version from LakeFS')
    parser_dataset_pull.add_argument('-s3', '--pull_to_s3', action='store_true', help='Pull to the S3 bucket instead of locally (for streaming)')
    parser_dataset_pull.add_argument('-rr', '--rclone_remote', type=str, help='Name of the rclone remote store for S3 bucket', default='b2')

    parser_dataset_push = dataset_subparsers.add_parser('push', help='Push local dataset changes to LakeFS')
    parser_dataset_push.add_argument('-m', '--message', type=str, help='LakeFS commit message', required=False)

    parser_dataset_webdataset = dataset_subparsers.add_parser('webdataset', help='Convert local dataset to WebDataset format')
    parser_dataset_webdataset.add_argument('-v', '--view', type=str, help='View of the dataset and/or name of the webdataset', required=True)
    parser_dataset_webdataset.add_argument('-q', '--query', type=str, help='SQL query in format: SELECT basename FROM df WHERE [condition]', required=True)

    args = parser.parse_args()

    if args.main == 'config':
        _config()
    elif args.dataset == 'init':
        _dataset_init(args)
    elif args.dataset == 'set_s3':
        _dataset_set_s3(args)
    elif args.dataset == 'set_lakefs_repo':
        _dataset_set_lakefs_repo(args)
    elif args.dataset == 'checkout':
        _dataset_checkout(args)
    elif args.dataset == 'status':
        _dataset_status(args)
    elif args.dataset == 'pull':
        _dataset_pull(args)
    elif args.dataset == 'push':
        _dataset_push(args)
    elif args.dataset == 'webdataset':
        _dataset_webdataset(args)
