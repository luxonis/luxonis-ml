import os
import sys

# Get the directory of the current file and add it to the sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from dag_factory import DAGFactory
from airflow.providers.mongo.hooks.mongo import MongoHook

VERSION='v1.0.3'

# DAG Factory
factory = DAGFactory()

# Function to fetch active configs from MongoDB
def fetch_active_configs():
    mongo_conn_id = 'mongo_rh'
    hook = MongoHook(conn_id=mongo_conn_id)
    documents = hook.find(mongo_db='robothub', mongo_collection='configs', query={"status": "active"})
    return [doc['config_name'] for doc in documents]

# Fetch active configurations
active_configs = fetch_active_configs()

# Loop over dynamic config and auto-generate DAGs for each client
for config_name in active_configs:
    dag_id = f"robothub_dag_{config_name}_{VERSION}"
    print(dag_id)
    robothub_dag = factory.create('RobotHubIngest', config_name).build()
    globals()[dag_id] = robothub_dag