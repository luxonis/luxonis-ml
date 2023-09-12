from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from dotenv import load_dotenv
from luxonis_ml.robothub.config_rh import RHConfig
from luxonis_ml.robothub.filter_rh import RH_Downloader
from luxonis_ml.robothub.convert_ldf import LDF_Converter
import os

# Define the default arguments for the DAG
default_args = {
    'owner': 'user',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}
params = {
    'config_path': './simple_rh.yaml',
    'schema_path': './config_schema_rh.json',
    'env_path': './.env',
    'dest_dir': './tmp/images',
}

# Instantiate the DAG
dag = DAG(
    'robothub_dag',
    default_args=default_args,
    params=params,
    description='A DAG to process RobotHub data',
    schedule_interval=timedelta(minutes=10),  # This can be adjusted based on your needs
    start_date=datetime(2023, 9, 11),
    catchup=False
)

# Task to read the configuration
def read_config(**kwargs):
    ti = kwargs['ti']
    config_path = kwargs['params']['config_path']
    schema_path = kwargs['params']['schema_path']
    rh_config = RHConfig(config_path, schema_path)
    ti.xcom_push(key='rh_config', value=rh_config.to_dict())

read_config_task = PythonOperator(
    task_id='read_config',
    python_callable=read_config,
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
    serialized_data = ti.xcom_pull(task_ids='read_config', key='rh_config')
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
    serialized_data = ti.xcom_pull(task_ids='read_config', key='rh_config')
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

final_bash_test = BashOperator(
    task_id='final_bash_test',
    bash_command='echo "This is the final task!"',
    dag=dag,
)

# Set the task dependencies
read_config_task >> load_env_vars_task >> download_images_task >> convert_to_ldf_task >> final_bash_test

if __name__ == "__main__":
    dag.cli()

