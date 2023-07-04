#!/usr/bin/env python3

from luxonis_ml.data import *
import argparse
import os
import json
from pathlib import Path

def _config():
    AWS_BUCKET = input("AWS Bucket: ")
    AWS_ACCESS_KEY_ID = input("AWS Access Key: ")
    AWS_SECRET_ACCESS_KEY = input("AWS Secret Access Key: ")
    AWS_S3_ENDPOINT_URL = input("AWS Endpoint URL: ")
    MONGO_URI = input("MongoDB URI: ")
    LABELSTUDIO_URL = input("label-studio URL: ")
    LABELSTUDIO_KEY = input("label-studio API Key: ")

    cache_dir = f'{str(Path.home())}/.cache/luxonis_ml'
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    cache_file = f"{cache_dir}/credentials.json"
    credentials = {
        "AWS_BUCKET": AWS_BUCKET,
        "AWS_ACCESS_KEY_ID": AWS_ACCESS_KEY_ID,
        "AWS_SECRET_ACCESS_KEY": AWS_SECRET_ACCESS_KEY,
        "AWS_S3_ENDPOINT_URL": AWS_S3_ENDPOINT_URL,
        "MONGO_URI": MONGO_URI,
        "LABELSTUDIO_URL": LABELSTUDIO_URL,
        "LABELSTUDIO_KEY": LABELSTUDIO_KEY
    }

    with open(cache_file, 'w') as file:
        json.dump(credentials, file)

    fo_config = {
        "database_dir": "null",
        "database_uri": MONGO_URI
    }
    fo_config_file = f"{str(Path.home())}/.fiftyone/config.json"

    with open(fo_config_file, 'w') as file:
        json.dump(fo_config, file)

    print(f"Credentials saved to {cache_file}!")

def _dataset_sync(args):
    with LuxonisDataset(args.team, args.dataset_name) as dataset:
        dataset.sync_from_cloud()

def main():

    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest='main')
    parser_config = subparsers.add_parser('config', help='Configure API keys for luxonis_ml')
    parser_dataset = subparsers.add_parser('dataset', help='Dataset programs to work with LDF')

    dataset_subparsers = parser_dataset.add_subparsers(dest='dataset')

    parser_dataset_sync = dataset_subparsers.add_parser('sync', help='Sync dataset media from cloud')
    parser_dataset_sync.add_argument('-t', '--team', type=str, help='Team name', default=None)
    parser_dataset_sync.add_argument('-n', '--dataset_name', type=str, help='Name of dataset', default=None)

    args = parser.parse_args()

    if args.main == 'config':
        _config()
    elif args.dataset == 'sync':
        _dataset_sync(args)
