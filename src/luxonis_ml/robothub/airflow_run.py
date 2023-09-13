import os
from dotenv import load_dotenv
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
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
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}
params = {
    'config_path': 'simple_rh', # './configs/simple_rh.yaml',
    'schema_path': './configs/config_schema_rh.json',
    'env_path': './.env',
    'dest_dir': './tmp/images',
}

# Instantiate the DAG
dag = DAG(
    'robothub_dag',
    default_args=default_args,
    params=params,
    description='A DAG to process RobotHub data',
    schedule_interval=timedelta(minutes=50),  # This can be adjusted based on your needs
    start_date=datetime(2023, 9, 11),
    catchup=False
)

def fetch_config_from_mongo(**kwargs):
    ti = kwargs['ti']
    config_path = kwargs['params']['config_path']
    schema_path = kwargs['params']['schema_path']

    # Use the connection ID you set up in the Airflow UI
    hook = MongoHook(conn_id="mongo_rh")
    config_collection = hook.get_collection("configs", mongo_db="robothub")

    # Fetch the configuration
    config_entry = config_collection.find_one(filter={"config_name": config_path})
    if config_entry is None:
        raise ValueError(f"Config {config_path} not found in MongoDB!")
    config_data = config_entry["data"]

    rh_config = RHConfig("", schema_path, config_data)
    ti.xcom_push(key='rh_config', value=rh_config.to_dict())

fetch_mongo_config_task = PythonOperator(
    task_id='mongo_fetch',
    python_callable=fetch_config_from_mongo,
    provide_context=True,
    dag=dag,
)

# Task to load environment variables
def load_env_vars(**kwargs):
    env_path = kwargs['params']['env_path']
    load_dotenv(env_path)
    rh_token = os.getenv('TOKEN')
    return rh_token

load_env_vars_task = PythonOperator(
    task_id='load_env_vars',
    python_callable=load_env_vars,
    dag=dag,
)

# Task to download images
def download_images(**kwargs):
    ti = kwargs['ti']
    serialized_data = ti.xcom_pull(task_ids='mongo_fetch', key='rh_config')
    rh_config = RHConfig.from_dict(serialized_data)
    dest_dir = kwargs['params']['dest_dir']
    rh_token = ti.xcom_pull(task_ids='load_env_vars')
    rh_downloader = RH_Downloader(rh_token, rh_config, dest_dir)
    rh_downloader.run_async_download()

download_images_task = PythonOperator(
    task_id='download_images',
    python_callable=download_images,
    provide_context=True,
    dag=dag,
)

# Task to convert images to LDF
def convert_to_ldf(**kwargs):
    ti = kwargs['ti']
    serialized_data = ti.xcom_pull(task_ids='mongo_fetch', key='rh_config')
    rh_config = RHConfig.from_dict(serialized_data)
    dest_dir = kwargs['params']['dest_dir']
    ldf_converter = LDF_Converter(rh_config, dest_dir)
    ldf_converter.detections_to_ldf()

convert_to_ldf_task = PythonOperator(
    task_id='convert_to_ldf',
    python_callable=convert_to_ldf,
    provide_context=True,
    dag=dag,
)

# Set the task dependencies
fetch_mongo_config_task >> load_env_vars_task >> download_images_task >> convert_to_ldf_task 

if __name__ == "__main__":
    dag.cli()

