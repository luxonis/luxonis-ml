import asyncio
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta

from airflow import DAG
from airflow.models import Variable
from airflow.decorators import dag, task
from airflow.providers.mongo.hooks.mongo import MongoHook

from luxonis_ml.robothub.config_rh import RHConfig
from luxonis_ml.robothub.filter_rh import RH_Downloader
from luxonis_ml.robothub.convert_ldf import LDF_Converter

# Define the default arguments for the DAG
default_args = {
    'owner': 'user',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

# Instantiate the DAG
@dag(
    'robothub_dag',
    default_args=default_args,
    description='A DAG to process RobotHub data',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2023, 9, 11),
    catchup=False
)
def robothub_dag():

    @task
    def fetch_config_from_mongo(mongo_conn_id, config_name):
        schema_name = 'config_schema_rh'

        # Use the connection ID you set up in the Airflow UI
        hook = MongoHook(conn_id=mongo_conn_id)
        config_collection = hook.get_collection("configs", mongo_db="robothub")

        # Fetch the configuration
        config_entry = config_collection.find_one(filter={"config_name": config_name})
        if config_entry is None:
            raise ValueError(f"Config {config_name} not found in MongoDB!")
        config_data = config_entry["data"]

        # Fetch the schema
        schema_entry = config_collection.find_one(filter={"config_name": schema_name})
        if schema_entry is None:
            raise ValueError(f"Schema {schema_name} not found in MongoDB!")
        schema_data = schema_entry["data"]

        rh_config = RHConfig(config_data, schema_data)
        return rh_config.to_dict()
    
    @task
    def get_detections(serialized_data, rh_token):
        rh_config = RHConfig.from_dict(serialized_data)
        dest_dir = './tmp/images'

        rh_downloader = RH_Downloader(rh_token, rh_config, dest_dir)
        
        detections = rh_downloader.get_all_detections()
        filtered_detections = rh_downloader.filter_detections(detections)
        rh_downloader.save_detections_info(filtered_detections)
        
        return filtered_detections
    
    @task
    def get_num_batches(filtered_detections, max_img_limit):
        num_batches = (len(filtered_detections) + max_img_limit - 1) // max_img_limit
        return list(range(num_batches))

    # Task to download images
    @task
    def download_images(batch_num, filtered_detections, serialized_data, rh_token, max_img_limit):
        rh_config = RHConfig.from_dict(serialized_data)
        dest_dir = './tmp/images'

        rh_downloader = RH_Downloader(rh_token, rh_config, dest_dir)

        start = batch_num * max_img_limit
        end = (batch_num + 1) * max_img_limit
        if start >= len(filtered_detections) or start < 0 or end < 0:
            return
        subset_detections = filtered_detections[start:end]

        asyncio.run(rh_downloader.download_images(subset_detections))

    # Task to convert images to LDF
    @task
    def convert_to_ldf(serialized_data):
        rh_config = RHConfig.from_dict(serialized_data)
        dest_dir = './tmp/images'
        ldf_converter = LDF_Converter(rh_config, dest_dir)
        ldf_converter.detections_to_ldf()
    
    # Variables
    rh_token = Variable.get("RH_TOKEN")
    mongo_conn_id = Variable.get("MONGO_CONN_ID")
    max_img_limit = int(Variable.get("MAX_IMG_LIMIT"))

    # Tasks
    fetched_config = fetch_config_from_mongo(mongo_conn_id, 'minimal_rh')
    detections_result = get_detections(fetched_config, rh_token)
    num_batches_list = get_num_batches(detections_result, max_img_limit)

    download_tasks = download_images.partial(
        filtered_detections=detections_result,
        serialized_data=fetched_config,
        rh_token=rh_token,
        max_img_limit=max_img_limit
    ).expand(batch_num=num_batches_list)

    converted_ldf = convert_to_ldf(fetched_config)

    # Dependencies
    fetched_config >> detections_result >> num_batches_list >> download_tasks >> converted_ldf


# Assign the DAG to a variable to indicate it should be discovered by Airflow
dynamic_robothub_dag = robothub_dag()

if __name__ == "__main__":
    from airflow.utils.state import State
    dynamic_robothub_dag.clear(dag_run_state=State.NONE)
    dynamic_robothub_dag.run()

