from datetime import datetime
from cron_validator import CronValidator
import os
import sys

# Get the directory of the current file and add it to the sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from dag_factory import DAGFactory
from airflow.providers.mongo.hooks.mongo import MongoHook

VERSION='v1.0.3'
REPEAT_EVERY = ["once", "hour", "day", "week", "month", "year"]

# DAG Factory
factory = DAGFactory()

# Function to fetch active configs from MongoDB
def fetch_active_configs():
    mongo_conn_id = 'mongo_rh'
    hook = MongoHook(conn_id=mongo_conn_id)
    documents = hook.find(mongo_db='robothub', mongo_collection='configs', query={"status": "active"})

    configs = []
    for doc in documents:
        # Check if repeat_every is a valid cron expression
        if 'repeat_every' in doc['data'] and doc['data']['repeat_every'] is not None:
            repeat = doc['data']['repeat_every']
            try:
                assert CronValidator.parse(repeat) is not None
            except AssertionError:
                print(f"WARNING: Invalid cron expression for config {doc['config_name']}: {repeat}")
                continue

        configs.append((doc['config_name'], doc['data']['repeat_every']))
    
    return configs

# Fetch active configurations
active_configs = fetch_active_configs()

# Loop over dynamic config and auto-generate DAGs for each client
for config_name, repeat in active_configs:
    dag_id = f"robothub_dag_{config_name}_{VERSION}"
    print(dag_id)
    robothub_dag = factory.create('RobotHubIngest', config_name, repeat).build()
    globals()[dag_id] = robothub_dag