import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import subprocess
from copy import deepcopy

import torch.onnx
import onnx
import onnxruntime

import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import Grayscale, Lambda

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, SearchRequest
from qdrant_client.http import models

from luxonis_ml.data.dataset import HType, IType, LDFComponent, LuxonisDataset, BucketStorage

from luxonis_ml.embeddings.data_utils import *
from luxonis_ml.embeddings.model_utils import *
from luxonis_ml.embeddings.embedding_utils import *
from luxonis_ml.embeddings.qdrant_utils import *


def _get_sample_payloads_coco(luxonis_dataset):
    # Iterate over the samples in the LuxonisDataset to get all img_paths and payloads
    all_payloads = []

    for sample in luxonis_dataset.fo_dataset:
        sample_id = sample['id']
        instance_id = sample['instance_id']
        # filepath = sample['filepath']
        class_name = sample['class']
        split = sample['split']

        img_path = luxonis_dataset.path + sample.filepath.split(luxonis_dataset.path.split('/')[-1])[-1]

        all_payloads.append({
            "sample_id": sample_id,
            "instance_id": instance_id,
            "image_path": img_path,
            "class": class_name['classifications'][0]['label'],
            "split": split
        })
    
    return all_payloads

def _get_sample_payloads(luxonis_dataset):
    # Iterate over the samples in the LuxonisDataset to get all img_paths and payloads
    all_payloads = []

    for sample in luxonis_dataset.fo_dataset:
        
        instance_id = sample['instance_id']
        img_path = luxonis_dataset.path + sample.filepath.split(luxonis_dataset.path.split('/')[-1])[-1]
        sample_id = sample['id']
        # class_name = sample['class']['classifications'][0]['label']
        # split = sample['split']

        all_payloads.append({
            "instance_id": instance_id,
            "image_path": img_path,
            "sample_id": sample_id,
            # "class": class_name,
            # "split": split
        })
    
    return all_payloads

def _filter_new_samples(qdrant_client, qdrant_collection_name="mnist", vector_size=2048, all_payloads=[]):
    # Filter out samples that are already in the Qdrant database
    search_queries = [SearchRequest(
        vector=[0] * vector_size,  # Dummy vector
        filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="sample_id",
                    match=models.MatchText(text=payload["sample_id"])
                )
            ]
        ),
        limit=1
    ) for payload in all_payloads]


    search_results = qdrant_client.search_batch(
        collection_name=qdrant_collection_name,
        requests=search_queries
    )

    new_payloads = [all_payloads[i] for i, res in enumerate(search_results) if not res]

    return new_payloads

def _filter_new_samples_by_id(qdrant_client, qdrant_collection_name="mnist", all_payloads=[]):
    # Filter out samples that are already in the Qdrant database
    if len(all_payloads) == 0:
        print("Payloads list is empty!")
        return all_payloads

    ids = [payload["instance_id"] for payload in all_payloads]
    search_results = qdrant_client.retrieve(
        collection_name=qdrant_collection_name,
        ids=ids,
        with_payload=False,
        with_vectors=False
    )

    retrieved_ids = [res.id for res in search_results]

    new_payloads = [payload for payload in all_payloads if payload["instance_id"] not in retrieved_ids]

    return new_payloads

def _generate_new_embeddings(ort_session, output_layer_name="/Flatten_output_0", emb_batch_size=64, new_payloads=[], transform=None):
    # Generate embeddings for the new images using batching
    new_embeddings = []

    if transform is None:
        # Define a transformation for resizing and normalizing the images
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),  # Resize images to (224, 224) or any desired size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
        ])

    for i in range(0, len(new_payloads), emb_batch_size):
        batch= new_payloads[i:i+emb_batch_size]
        batch_img_paths = [payload["image_path"] for payload in batch]
        
        # Load, preprocess, and resize a batch of images
        batch_images = [cv2.imread(img_path) for img_path in batch_img_paths]
        batch_tensors = [transform(img) for img in batch_images]
        batch_tensor = torch.stack(batch_tensors).cuda()

        # Run the ONNX model on the batch
        ort_inputs = {ort_session.get_inputs()[0].name: batch_tensor.cpu().numpy()}
        ort_outputs = ort_session.run([output_layer_name], ort_inputs)
        
        # Append the embeddings from the batch to the new_embeddings list
        batch_embeddings = ort_outputs[0].squeeze()
        new_embeddings.extend(batch_embeddings.tolist())
    
    return new_embeddings

def _batch_upsert(qdrant_client, new_embeddings, new_payloads, qdrant_batch_size = 64, qdrant_collection_name="mnist"):
    # Perform batch upserts to Qdrant
    for i in range(0, len(new_embeddings), qdrant_batch_size):
        batch_embeddings = new_embeddings[i:i+qdrant_batch_size]
        batch_payloads = new_payloads[i:i+qdrant_batch_size]

        points = [models.PointStruct(
            id=payload["instance_id"],
            vector=embedding,
            payload=payload
        ) for j, (embedding, payload) in enumerate(zip(batch_embeddings, batch_payloads))]

        try:
            qdrant_client.upsert(collection_name=qdrant_collection_name, points=points)
            print(f"Upserted batch {i // qdrant_batch_size + 1} / {len(new_embeddings) // qdrant_batch_size + 1} of size {len(points)} to Qdrant.")
        except Exception as e:
            print(e)
            print(f"Failed to upsert batch {i // qdrant_batch_size + 1} to Qdrant.")

def generate_embeddings(luxonis_dataset, 
                         ort_session, 
                         qdrant_client, 
                         qdrant_collection_name="mnist", 
                         output_layer_name="/Flatten_output_0",
                         emb_batch_size=64, 
                         qdrant_batch_size=64):
    
    all_payloads = _get_sample_payloads_coco(luxonis_dataset)
    # all_payloads = _get_sample_payloads(luxonis_dataset)

    new_payloads = _filter_new_samples_by_id(qdrant_client, 
                                        qdrant_collection_name, 
                                        all_payloads)
    
    new_embeddings = _generate_new_embeddings(ort_session,
                                                output_layer_name,
                                                emb_batch_size,
                                                new_payloads)
    
    _batch_upsert(qdrant_client,
                    new_embeddings,
                    new_payloads,
                    qdrant_batch_size,
                    qdrant_collection_name)
    
    # make a sample_id : embedding dictionary
    sample_id_to_embedding = {payload["sample_id"]: embedding for payload, embedding in zip(new_payloads, new_embeddings)}
    print("Embeddings generation and insertion completed!")

    return sample_id_to_embedding