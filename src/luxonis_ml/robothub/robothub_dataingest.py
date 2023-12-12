import asyncio
from datetime import datetime, timedelta

from airflow import DAG
from airflow.models import Variable
from airflow.decorators import dag, task_group, task
from airflow.providers.mongo.hooks.mongo import MongoHook

from luxonis_ml.robothub.config_rh import RHConfig
from luxonis_ml.robothub.filter_rh import RH_Downloader
from luxonis_ml.robothub.convert_ldf import LDF_Converter

# Instantiate the DAG
class RobotHubIngest():
	def __init__(self, config_name):
		self.config_name = config_name

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
			 schedule_interval=timedelta(days=1),
			 start_date=datetime(2023, 9, 11),
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
			def get_detections(serialized_data, rh_token):
				rh_config = RHConfig.from_dict(serialized_data)
				dest_dir = './tmp/' + self.config_name

				rh_downloader = RH_Downloader(rh_token, rh_config, dest_dir)
				
				detections = rh_downloader.get_all_detections()
				filtered_detections = rh_downloader.filter_detections(detections)
				rh_downloader.save_detections_info(filtered_detections)
				
				return filtered_detections

			@task
			def get_num_batches(filtered_detections, max_img_limit):
				num_batches = (len(filtered_detections) + max_img_limit - 1) // max_img_limit
				return list(range(num_batches))
			
			@task_group
			def image_embedding_tasks(batch_num, filtered_detections, serialized_data, rh_token, max_img_limit):
				# Task to download images
				@task
				def download_images(batch_num, filtered_detections, serialized_data, rh_token, max_img_limit):
					rh_config = RHConfig.from_dict(serialized_data)
					dest_dir = './tmp/' + self.config_name

					rh_downloader = RH_Downloader(rh_token, rh_config, dest_dir + '/images')

					start = batch_num * max_img_limit
					end = (batch_num + 1) * max_img_limit
					if start >= len(filtered_detections) or start < 0 or end < 0:
						return
					subset_detections = filtered_detections[start:end]

					asyncio.run(rh_downloader.download_images(subset_detections))

					return dest_dir
				
				@task
				def get_embeddings(images_path):
					import numpy as np
					from torchvision import datasets, transforms
					import torch
					import onnxruntime
					from luxonis_ml.embeddings.utils import extract_embeddings_onnx

					# Define the transform
					transform = transforms.Compose([
						transforms.Resize((224, 224)),  # Adjust the size according to your model
						transforms.ToTensor(),
						# Add other necessary transforms here
					])

					# Create a dataset using ImageFolder
					dataset = datasets.ImageFolder(root=images_path, transform=transform)

					# Create a DataLoader
					data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

					# Create an ONNX Runtime session
					provider = ['CUDAExecutionProvider'] if torch.cuda.is_available() and 'CUDAExecutionProvider' in onnxruntime.get_available_providers() else None
					ort_session = onnxruntime.InferenceSession("resnet50-1.onnx", providers=provider)

					# Extract embeddings from the dataset
					embeddings, labels = extract_embeddings_onnx(ort_session, data_loader, "/Flatten_output_0")

					# export to file
					emb_file = images_path + '/embeddings.txt'
					np.savetxt(emb_file, embeddings.numpy())
					img_paths = [p[0] for p in dataset.imgs]
					impath_savefile = images_path + '/img_paths.txt'
					np.savetxt(impath_savefile, img_paths, fmt='%s')
					return {'embeddings_path': emb_file, 'img_paths_file': img_paths}

				@task
				def emb_to_qdrant(embeddings_out):
					import numpy as np
					from luxonis_ml.embeddings.utils.qdrant import QdrantAPI, QdrantManager
					from qdrant_client.models import Distance

					embeddings_file_path = embeddings_out['embeddings_path']
					img_paths_file = embeddings_out['img_paths_file']

					# Start Qdrant docker container
					QdrantManager("qdrant/qdrant", "qdrant_container2").start_docker_qdrant()

					# Connect to Qdrant
					qdrant_api = QdrantAPI("localhost", 6333, self.config_name)

					# Create a collection
					embeddings = np.loadtxt(embeddings_file_path)
					vector_size = embeddings.shape[1]
					qdrant_api.create_collection(vector_size=vector_size, distance=Distance.COSINE)

					# Insert the embeddings into the collection
					# all zero labels
					labels = [0] * len(embeddings)
					labels = np.array(labels)
					img_paths = np.loadtxt(img_paths_file, dtype=str)
					qdrant_api.batch_insert_embeddings_nooverwrite(embeddings, labels, img_paths, batch_size=50)
				
				img_path = download_images(batch_num, filtered_detections, serialized_data, rh_token, max_img_limit)
				emb_out = get_embeddings(img_path)
				eq = emb_to_qdrant(emb_out)
				img_path >> emb_out >> eq
			
			@task
			def smart_select(serialized_data, filtered_detections):
				import os
				import json
				import numpy as np
				from luxonis_ml.embeddings.utils.qdrant import QdrantAPI
				from luxonis_ml.embeddings.methods.representative import calculate_similarity_matrix, find_representative_kmedoids

				rh_config = RHConfig.from_dict(serialized_data)
				cfg = rh_config.get_config()
				if cfg.get('tactic') != 'smart':
					return

				# Connect to Qdrant
				qdrant_api = QdrantAPI("localhost", 6333, self.config_name)
				ids = qdrant_api.get_all_ids()
				ids, embeddings = qdrant_api.get_all_embeddings()
				similarity_matrix = calculate_similarity_matrix(embeddings)

				desired_size = int(len(embeddings)*0.05)
				# desired_size = 10
				selected_image_indices = find_representative_kmedoids(similarity_matrix, desired_size)

				ids_sel = np.array(ids)[selected_image_indices].tolist()
				payloads = qdrant_api.get_payloads_from_ids(ids_sel)
				print("Retrieved {len(payloads)} representative images from Qdrant")

				represent_imgs = [p['image_path'] for p in payloads]
				represent_imgs = [img.split('/')[-1] for img in represent_imgs]
				
				# get representative images that are in this sample batch
				filtered_detections = [d['frames'][0]['path'] for d in filtered_detections]
				# Join represent_imgs into a single string
				represent_string = ' '.join(represent_imgs)
				# Initialize an empty list to hold the filtered representations
				represent_filtered = []
				# Loop through filtered_detections and check if each is in represent_string
				for filt_d in filtered_detections:
					if filt_d in represent_string:
						# Extract the full path from represent_imgs that contains filt_d
						matching_paths = [img for img in represent_imgs if filt_d in img]
						represent_filtered.extend(matching_paths)
				represent_imgs = represent_filtered
				print(f"Representative images in this batch: {len(represent_imgs)}")

				# delete all images not representative in dist_dir
				dist_dir = './tmp/' + self.config_name 
				for img in os.listdir(dist_dir + '/images'):
					if img not in represent_imgs and img != 'detections.json':
						os.remove(os.path.join(dist_dir+'/images', img))
				
				# update detections.json 
				detections_json = os.path.join(dist_dir, 'detections.json')
				with open(detections_json, 'r') as f:
					detections = json.load(f)
				detections = [d for d in detections if d['framePath'].split('/')[-1] in represent_imgs]
				with open(detections_json, 'w') as f:
					json.dump(detections, f)

			# Task to convert images to LDF
			@task
			def convert_to_ldf(serialized_data):
				rh_config = RHConfig.from_dict(serialized_data)
				dest_dir = './tmp/' + self.config_name
				ldf_converter = LDF_Converter(rh_config, dest_dir)
				ldf_converter.detections_to_ldf()
			
			@task
			def clear_tmp():
				import shutil
				shutil.rmtree('./tmp/' + self.config_name)

			# Variables
			rh_token = Variable.get("RH_TOKEN")
			mongo_conn_id = Variable.get("MONGO_CONN_ID")
			max_img_limit = int(Variable.get("MAX_IMG_LIMIT"))

			# Tasks
			fetched_config = fetch_config_from_mongo(mongo_conn_id, self.config_name)
			detections_result = get_detections(fetched_config, rh_token)
			num_batches_list = get_num_batches(detections_result, max_img_limit)

			download_tasks = image_embedding_tasks.partial(
				filtered_detections=detections_result,
				serialized_data=fetched_config,
				rh_token=rh_token,
				max_img_limit=max_img_limit
			).expand(batch_num=num_batches_list)

			smart = smart_select(fetched_config, detections_result)

			converted_ldf = convert_to_ldf(fetched_config)

			rm_tmp = clear_tmp()

			# Dependencies
			fetched_config >> detections_result >> num_batches_list >> download_tasks >> smart >> converted_ldf >> rm_tmp

		return robothub_dag()
