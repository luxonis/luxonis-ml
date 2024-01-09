import asyncio
from datetime import datetime, timedelta

from airflow import DAG
from airflow.models import Variable, DagRun
from airflow.decorators import dag, task_group, task
from airflow.providers.mongo.hooks.mongo import MongoHook

from luxonis_ml.utils import LuxonisFileSystem

import os
import sys
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config_rh import RHConfig
from filter_rh import RH_Downloader
# from weav_api import WeaviateAPI
# from convert_ldf import LDF_Converter


# Instantiate the DAG
class RobotHubIngest():
	def __init__(self, config_name, repeat):
		self.config_name = config_name
		self.repeat = repeat # cron expression

		self.default_args = {
				'owner': 'user',
				'depends_on_past': False,
				'email_on_failure': False,
				'email_on_retry': False,
				'retries': 1,
				'retry_delay': timedelta(minutes=4),
		}

	def build(self):
		@dag(
			 'robothub_dag_'+self.config_name,
			 default_args=self.default_args,
			 description='A DAG to process RobotHub data',
			 schedule_interval=self.repeat,
			 start_date=datetime(2023, 9, 11), #it doesn't matter, as long as it's in the past
			 catchup=False
		)
		def robothub_dag():
			@task
			def fetch_config_from_mongo(mongo_conn_id, config_name):
				# Use the connection ID you set up in the Airflow UI
				hook = MongoHook(conn_id=mongo_conn_id)
				config_collection = hook.get_collection("configs", mongo_db="robothub")

				# Fetch the configuration
				config_entry = config_collection.find_one(filter={"config_name": config_name})
				if config_entry is None:
					raise ValueError(f"Config {config_name} not found in MongoDB!")
				config_data = config_entry["data"]

				# extract the schema name from the config
				schema_name = config_data['schema_name']

				# Fetch the schema
				schema_collection = hook.get_collection("schemas", mongo_db="robothub")
				schema_entry = schema_collection.find_one(filter={"schema_name": schema_name})
				if schema_entry is None:
					raise ValueError(f"Schema {schema_name} not found in MongoDB!")
				schema_data = schema_entry["data"]

				rh_config = RHConfig(config_data, schema_data)
				return rh_config.to_dict()

			@task
			def get_detections(config_data, rh_token, gcs):
				rh_config = RHConfig.from_dict(config_data)
				
				lfs = LuxonisFileSystem("gs://" + gcs["bucket"])
				dest_dir = gcs["prefix"]
				rh_downloader = RH_Downloader(rh_token, rh_config, dest_dir, lfs)
				
				detections = rh_downloader.get_all_detections()
				filtered_detections = rh_downloader.filter_detections(detections)
				detections_file = rh_downloader.save_detections_info(filtered_detections)
				
				return detections_file

			@task
			def get_num_batches(detections_file, max_img_limit, gcs):
				lfs = LuxonisFileSystem("gs://" + gcs["bucket"])
				saved_detections_buffer = lfs.read_to_byte_buffer(detections_file)
				saved_detections_bytes = saved_detections_buffer.read()
				saved_detections_str = saved_detections_bytes.decode('utf-8')
				saved_detections = json.loads(saved_detections_str)
			
				num_batches = (len(saved_detections) + max_img_limit - 1) // max_img_limit
				return list(range(num_batches))
			
			@task_group
			def image_embedding_tasks(batch_num, detections_file, config_data, rh_token, max_img_limit, gcs):
				# Task to download images
				@task
				def download_images(batch_num, detections_file, config_data, rh_token, max_img_limit, gcs):
					rh_config = RHConfig.from_dict(config_data)

					lfs = LuxonisFileSystem("gs://" + gcs["bucket"])
					saved_detections_buffer = lfs.read_to_byte_buffer(detections_file)
					saved_detections_bytes = saved_detections_buffer.read()
					saved_detections_str = saved_detections_bytes.decode('utf-8')
					saved_detections = json.loads(saved_detections_str)

					rh_downloader = RH_Downloader(rh_token, rh_config, gcs["prefix"] + '/images', lfs)

					start = batch_num * max_img_limit
					end = (batch_num + 1) * max_img_limit
					if start >= len(saved_detections) or start < 0 or end < 0:
						return
					subset_detections = saved_detections[start:end]

					asyncio.run(rh_downloader.download_images(subset_detections))

					return [d['framePath'] for d in subset_detections]
				
				@task
				def get_embeddings(image_paths, gcs):
					from typing import List, Tuple
					from PIL import Image
					from io import BytesIO
					import numpy as np
					
					import torch
					# from torchvision import transforms
					import torchvision.transforms as transforms
					import onnxruntime as ort

					def get_image_tensors_from_gcs(
						image_paths: List[str],
						transform: transforms.Compose,
						lfs: LuxonisFileSystem,
					) -> torch.Tensor:
						tensors = []
						for path in image_paths:
							buffer = lfs.read_to_byte_buffer(remote_path=path).getvalue()
							try:
								image = Image.open(BytesIO(buffer)).convert('RGB')
							except:
								print("Error occured while processing image: ", path)
								continue
							tensor = transform(image)
							tensors.append(tensor)
						return torch.stack(tensors)

					def extract_embeddings_onnx(
						image_paths: List[str],
						ort_session: ort.InferenceSession,
						transform: transforms.Compose,
						lfs: LuxonisFileSystem,
						output_layer_name: str = "/Flatten_output_0",
						batch_size: int = 64,
					) -> torch.Tensor:
						embeddings = []

						for i in range(0, len(image_paths), batch_size):
							batch_paths = image_paths[i:i + batch_size]
							batch_tensors = get_image_tensors_from_gcs(batch_paths, transform, lfs)

							# Extract embeddings using ONNX
							ort_inputs = {ort_session.get_inputs()[0].name: batch_tensors.numpy()}
							ort_outputs = ort_session.run([output_layer_name], ort_inputs)[0]
							embeddings.extend(torch.from_numpy(ort_outputs).squeeze())

						return torch.stack(embeddings)

					# Assuming lsf is an instance of LuxonisFileSystem
					lfs = LuxonisFileSystem("gs://" + gcs["bucket"]) 

					local_model = "./emb_model.onnx"
					lfs.get_file(gcs["model_path"], local_model)

					# Define the transform
					transform = transforms.Compose([
						transforms.Resize((224, 224)),
						transforms.ToTensor(),
						transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
					])

					# Batch process images from GCS
					provider = (
						['CUDAExecutionProvider'] 
						if torch.cuda.is_available() 
						and 'CUDAExecutionProvider' in ort.get_available_providers() 
						else None
					)
					ort_session = ort.InferenceSession(local_model, providers=provider)

					embeddings = extract_embeddings_onnx(
						image_paths,
						ort_session,
						transform,
						lfs,
						output_layer_name="/Flatten_output_0",
						batch_size=64,
					)
					embeddings = embeddings.numpy()
					np.savetxt("embeddings.txt", embeddings)
					remote_path = gcs["prefix"] + '/embeddings.txt'
					lfs.put_file("embeddings.txt", remote_path)
					return remote_path

				@task
				def emb_to_weaviate(image_paths, embeddings_file_path, gcs):
					import numpy as np
					from weav_api import WeaviateAPI
					import weaviate

					# Get embeddings
					lfs = LuxonisFileSystem("gs://" + gcs["bucket"])
					local_embeddings_file_path = embeddings_file_path.split('/')[-1]
					lfs.get_file(embeddings_file_path, local_embeddings_file_path)
					embeddings = np.loadtxt(local_embeddings_file_path)
					
					# Get uuids
					uuids = [img.split('/')[-1].split('.')[0] for img in image_paths]

					# Connect Weaviate
					client = weaviate.connect_to_local()
					w_api = WeaviateAPI(client, self.config_name)

					# Insert embeddings
					w_api.insert_embeddings(uuids, embeddings.tolist())
				
				img_path = download_images(
					batch_num, 
					detections_file, 
					config_data, 
					rh_token, 
					max_img_limit,
					gcs
				)
				emb_out = get_embeddings(img_path, gcs)
				eq = emb_to_weaviate(img_path, emb_out, gcs)
				img_path >> emb_out >> eq
			
			@task
			def smart_select(config_data, detections_file, gcs):
				import os
				import json
				import numpy as np
				import weaviate
				from weav_api import WeaviateAPI
				from luxonis_ml.embeddings.methods.representative import calculate_similarity_matrix, find_representative_kmedoids

				rh_config = RHConfig.from_dict(config_data)
				cfg = rh_config.get_config()
				if cfg.get('tactic') != 'smart':
					return
				
				lfs = LuxonisFileSystem("gs://" + gcs["bucket"])
				saved_detections_buffer = lfs.read_to_byte_buffer(detections_file)
				saved_detections_bytes = saved_detections_buffer.read()
				saved_detections_str = saved_detections_bytes.decode('utf-8')
				saved_detections = json.loads(saved_detections_str)
				
				# connect Weaviate
				client = weaviate.connect_to_local()
				w_api = WeaviateAPI(client, self.config_name)

				# get all emb and ids
				embeddings, ids = w_api.get_all_embeddings_and_ids()

				# calculate similarity matrix
				similarity_matrix = calculate_similarity_matrix(embeddings)

				# find representative images
				desired_size = int(len(embeddings)*0.05)
				selected_image_indices = find_representative_kmedoids(similarity_matrix, desired_size)

				# rep_file_names = [img+'.png' for img in ids[rep_ixs]]
				represent_imgs = [ids[i]+'.png' for i in selected_image_indices]
				print(f"Number of representative images: {len(represent_imgs)}")

				# delete all images not representative in dist_dir
				for img in lfs.walk_dir(gcs["prefix"] + '/images'):
					if img not in represent_imgs:
						lfs.delete_file(img)
				
				# update detections.json 
				detections_json = 'detections.json'
				lfs.get_file(gcs["prefix"] + '/' + detections_json, detections_json)
				with open(detections_json, 'r') as f:
					detections = json.load(f)
				detections = [d for d in detections if d['framePath'].split('/')[-1] in represent_imgs]
				with open(detections_json, 'w') as f:
					json.dump(detections, f)
				lfs.put_file(detections_json, gcs["prefix"] + '/' + detections_json)

			# Task to convert images to LDF
			@task
			def convert_to_ldf(config_data, gcs):
				rh_config = RHConfig.from_dict(config_data)
				dest_dir = "gs://" + gcs["bucket"] + "/" + gcs["prefix"] 
				# ldf_converter = LDF_Converter(rh_config, dest_dir)
				# ldf_converter.detections_to_ldf()
			
			@task
			def clear_tmp(gcs):
				lfs = LuxonisFileSystem("gs://" + gcs["bucket"])
				lfs.delete_dir(gcs["prefix"])
			
			# Variables
			rh_token = Variable.get("RH_TOKEN")
			mongo_conn_id = Variable.get("MONGO_CONN_ID")
			max_img_limit = int(Variable.get("MAX_IMG_LIMIT"))
			gcs = {
				"bucket": "luxonis-test-bucket",
				"prefix": "airflow/tmp/"+self.config_name+"_{{ run_id }}",
				"model_path": "airflow/models/resnet50-1.onnx",
			}
			# gcs = Variable.get("GCS", deserialize_json=True)
			# gcs["prefix"] = gcs["prefix"] + DagRun.run_id

			# Tasks
			fetched_config = fetch_config_from_mongo(mongo_conn_id, self.config_name)
			detections_file = get_detections(fetched_config, rh_token, gcs)
			num_batches_list = get_num_batches(detections_file, max_img_limit, gcs)

			download_tasks = image_embedding_tasks.partial(
				detections_file=detections_file,
				config_data=fetched_config,
				rh_token=rh_token,
				max_img_limit=max_img_limit,
				gcs=gcs
			).expand(batch_num=num_batches_list)

			smart = smart_select(fetched_config, detections_file, gcs)

			converted_ldf = convert_to_ldf(fetched_config, gcs)

			rm_tmp = clear_tmp(gcs)

			# Dependencies
			fetched_config >> detections_file >> num_batches_list >> download_tasks >> smart >> converted_ldf >> rm_tmp

		return robothub_dag()
