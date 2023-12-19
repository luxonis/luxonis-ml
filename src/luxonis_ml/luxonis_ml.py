#!/usr/bin/env python3

import argparse
import os
import json
from pathlib import Path
import numpy as np
from tabulate import tabulate


def _config():
    AWS_BUCKET = input("AWS Bucket: ")
    AWS_ACCESS_KEY_ID = input("AWS Access Key: ")
    AWS_SECRET_ACCESS_KEY = input("AWS Secret Access Key: ")
    AWS_S3_ENDPOINT_URL = input("AWS Endpoint URL: ")
    MONGO_URI = input("MongoDB URI: ")
    LABELSTUDIO_URL = input("label-studio URL: ")
    LABELSTUDIO_KEY = input("label-studio API Key: ")

    cache_dir = f"{str(Path.home())}/.cache/luxonis_ml"
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
        "LABELSTUDIO_KEY": LABELSTUDIO_KEY,
    }

    with open(cache_file, "w") as file:
        json.dump(credentials, file)

    fo_config = {"database_dir": "null", "database_uri": MONGO_URI}
    fo_config_file = f"{str(Path.home())}/.fiftyone/config.json"

    with open(fo_config_file, "w") as file:
        json.dump(fo_config, file)

    print(f"Credentials saved to {cache_file}!")


def _dataset_sync(args):
    raise NotImplementedError


def _debug_performance(args):
    with open(args.log) as file:
        lines = file.readlines()

    items, times = [], []
    for line in lines:
        if line.endswith("ms\n"):
            time = line.split(" ")[-2]
            item = line.split(time)[0].split("[DEBUG] ")[1]
            items.append(item)
            times.append(float(time))

    items = np.array(items)
    times = np.array(times)
    data_dict = {}
    for uitem in np.unique(items):
        data_dict[uitem] = times[items == uitem]

    data = []
    for key in data_dict:
        data.append(
            [
                key,
                np.sum(data_dict[key]),
                np.mean(data_dict[key]),
                np.std(data_dict[key], ddof=1),
            ]
        )
    headers = ["Item", "Total Time (ms)", "Mean Time (ms)", "Std Time (ms)"]
    table = tabulate(data, headers, tablefmt="grid")
    print(table)


def main():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="main")
    subparsers.add_parser("config", help="Configure API keys for luxonis_ml")
    parser_dataset = subparsers.add_parser(
        "dataset", help="Dataset programs to work with LDF"
    )
    parser_debug = subparsers.add_parser("debug", help="Debug functions")

    dataset_subparsers = parser_dataset.add_subparsers(dest="dataset")
    debug_subparsers = parser_debug.add_subparsers(dest="debug")

    parser_dataset_sync = dataset_subparsers.add_parser(
        "sync", help="Sync dataset media from cloud"
    )
    parser_dataset_sync.add_argument(
        "-t", "--team_id", type=str, help="Team ID", default=None
    )
    parser_dataset_sync.add_argument(
        "-d", "--dataset_id", type=str, help="Dataset ID", default=None
    )

    parser_debug_performance = debug_subparsers.add_parser(
        "performance", help="Compute performance statistics from a log file"
    )
    parser_debug_performance.add_argument(
        "-l", "--log", type=str, help="Path to log file", default=None
    )

    args = parser.parse_args()

    if args.main == "config":
        _config()
    elif hasattr(args, "dataset") and args.dataset == "sync":
        _dataset_sync(args)
    elif hasattr(args, "debug") and args.debug == "performance":
        _debug_performance(args)
