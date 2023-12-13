from datetime import datetime
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
        # Skip if the from_date is in the future
        if 'from_date' in doc['data'] and doc['data']['from_date'] is not None:
            date = doc['data']['from_date']
            if date > datetime.now():
                print(f"WARNING: from_date {date} is in the future for config {doc['config_name']}")
                continue
        
        # Skip if the to_date is in the past
        if 'to_date' in doc['data'] and doc['data']['to_date'] is not None:
            date = doc['data']['to_date']
            if date < datetime.now():
                print(f"WARNING: to_date {date} is in the past for config {doc['config_name']}")
                # Warning: you might did not want to put a past date
                # You might want to delete the config or set status to inactive
                continue
        
        # Set the repeat value
        repeat = "once"
        if 'repeat_every' in doc['data']:
            if doc['data']['repeat_every'] is None:
                repeat = "once"
            elif doc['data']['repeat_every'] in REPEAT_EVERY:
                repeat = doc['data']['repeat_every']
            else:
                # if the repeat_every value is invalid, set it to None - the DAG will not be scheduled
                repeat = None
                print(f"WARNING: Invalid repeat_every value {doc['data']['repeat_every']} for config {doc['config_name']}")
        
        configs.append((doc['config_name'], repeat))
    
    return configs

# Fetch active configurations
active_configs = fetch_active_configs()

# Loop over dynamic config and auto-generate DAGs for each client
for config_name, repeat in active_configs:
    dag_id = f"robothub_dag_{config_name}_{VERSION}"
    print(dag_id)
    robothub_dag = factory.create('RobotHubIngest', config_name, repeat).build()
    globals()[dag_id] = robothub_dag