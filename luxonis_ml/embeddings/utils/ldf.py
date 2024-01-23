"""This script provides utilities for generating embeddings from images, filtering new
samples, and inserting them into a Qdrant database.

Modules Used:
    - cv2: For reading and processing images.
    - numpy: For numerical operations.
    - torch: PyTorch library for deep learning.
    - torch.onnx: PyTorch's ONNX utilities.
    - onnx: Open Neural Network Exchange.
    - onnxruntime: Runtime for ONNX models.
    - torchvision: PyTorch's computer vision library.
    - qdrant_client: Client for interacting with Qdrant.

Main Functions:
    - _get_sample_payloads_coco: Extracts payloads from the LuxonisDataset for the COCO dataset format.
    - _get_sample_payloads: Extracts payloads from the LuxonisDataset.
    - _filter_new_samples: Filters out samples that are already in the Qdrant database based on their sample ID.
    - _filter_new_samples_by_id: Filters out samples that are already in the Qdrant database based on their instance ID.
    - _generate_new_embeddings: Generates embeddings for new images using a given ONNX runtime session.
    - _batch_upsert: Performs batch upserts of embeddings to Qdrant.
    - generate_embeddings: Main function that generates embeddings for a given dataset and inserts them into Qdrant.

Note:
Ensure that the Qdrant server is running and accessible before using these utilities.
"""

from typing import Any, Dict, List
import uuid

import cv2
import numpy as np
import torch
import torch.onnx
import torchvision.transforms as transforms
from qdrant_client.http import models

from luxonis_ml.data import LuxonisDataset

def _get_sample_payloads_coco(dataset: LuxonisDataset) -> List[Dict[str, Any]]:
    """
    Extract payloads using LuxonisLoader for the COCO dataset format for all views (train, val, test).
    The function assumes that the loader provides images and annotations 
    for each sample in the dataset.

    @type dataset: LuxonisDataset
    @param dataset: An instance of LuxonisDataset.
    @rtype: List[Dict[str, Any]]
    @return: List of payloads.
    """

    all_payloads = []
    df = dataset._load_df_offline()
    file_index = dataset._get_file_index()

    df = df.merge(file_index, on="instance_id")

    for row in df.iterrows():
        if row[1]["type"] == "classification":
            try:
                instance_id = row[1]["original_filepath"].split("/")[-1].split(".")[0]
                # throw if not uuid
                uuid.UUID(instance_id)
            except:
                print(f"Skipping {row[1]['original_filepath']}, not a valid uuid")
                continue
            #instance_id = row[1]["instance_id"]
            img_path = row[1]["original_filepath"]
            class_name = row[1]["class"]

            all_payloads.append({
                "instance_id": instance_id,
                "image_path": img_path,
                "class": class_name,
            })

    return all_payloads


def _filter_new_samples_by_id(qdrant_client, collection_name, all_payloads=None):
    """Filter out samples that are already in the Qdrant database based on their
    instance ID.

    @param qdrant_client: Qdrant client instance.
    @param collection_name: Name of the Qdrant collection.
    @param all_payloads: List of all payloads.
    @return: List of new payloads that are not in the Qdrant database.
    """
    all_payloads = all_payloads or []
    # Filter out samples that are already in the Qdrant database
    if len(all_payloads) == 0:
        print("Payloads list is empty!")
        return all_payloads

    ids = [payload["instance_id"] for payload in all_payloads]
    search_results = qdrant_client.retrieve(
        collection_name=collection_name, ids=ids, with_payload=False, with_vectors=False
    )

    retrieved_ids = [res.id for res in search_results]

    new_payloads = [
        payload
        for payload in all_payloads
        if payload["instance_id"] not in retrieved_ids
    ]

    return new_payloads

def _filter_new_samples_by_id_weaviate(weaviate_api, all_payloads=None):
    """Filter out samples that are already in the Weaviate database based on their
    instance ID.

    @param weaviate_api: WeaviateAPI instance.
    @return: List of new payloads that are not in the Weaviate database.
    """
    all_payloads = all_payloads or []
    # Filter out samples that are already in the Qdrant database
    if len(all_payloads) == 0:
        print("Payloads list is empty!")
        return all_payloads

    retrieved_ids = weaviate_api.get_all_ids()

    new_payloads = [
        payload
        for payload in all_payloads
        if payload["instance_id"] not in retrieved_ids
    ]

    return new_payloads


def _generate_new_embeddings(
    ort_session,
    output_layer_name="/Flatten_output_0",
    emb_batch_size=64,
    new_payloads=None,
    transform=None,
):
    """Generate embeddings for new images using a given ONNX runtime session.

    @type ort_session: L{InferenceSession}
    @param ort_session: ONNX runtime session.
    @type output_layer_name: str
    @param output_layer_name: Name of the output layer in the ONNX model.
    @type emb_batch_size: int
    @param emb_batch_size: Batch size for generating embeddings.
    @type new_payloads: List[Dict[str, Any]]
    @param new_payloads: List of new payloads.
    @type transform: torchvision.transforms
    @param transform: Optional torchvision transform for preprocessing images.
    @rtype: List[Dict[str, Any]]
    @return: List of generated embeddings.
    """
    # Generate embeddings for the new images using batching
    new_embeddings = []
    new_payloads = new_payloads or []

    if transform is None:
        # Define a transformation for resizing and normalizing the images
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(
                    (224, 224)
                ),  # Resize images to (224, 224) or any desired size
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # Normalize images
            ]
        )

    for i in range(0, len(new_payloads), emb_batch_size):
        batch = new_payloads[i : i + emb_batch_size]
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


def _batch_upsert(
    qdrant_client, collection_name, new_embeddings, new_payloads, qdrant_batch_size=64
):
    """Perform batch upserts of embeddings to Qdrant.

    @type qdrant_client: L{QdrantClient}
    @param qdrant_client: Qdrant client instance.
    @type collection_name: str
    @param collection_name: Name of the Qdrant collection.
    @type new_embeddings: List[Dict[str, Any]]
    @param new_embeddings: List of new embeddings.
    @type new_payloads: List[Dict[str, Any]]
    @param new_payloads: List of new payloads.
    @type qdrant_batch_size: int
    @param qdrant_batch_size: Batch size for inserting into Qdrant.
    """
    # Perform batch upserts to Qdrant
    for i in range(0, len(new_embeddings), qdrant_batch_size):
        batch_embeddings = new_embeddings[i : i + qdrant_batch_size]
        batch_payloads = new_payloads[i : i + qdrant_batch_size]

        points = [
            models.PointStruct(
                id=payload["instance_id"], vector=embedding, payload=payload
            )
            for j, (embedding, payload) in enumerate(
                zip(batch_embeddings, batch_payloads)
            )
        ]

        try:
            qdrant_client.upsert(collection_name=collection_name, points=points)
            print(
                f"Upserted batch {i // qdrant_batch_size + 1} / {len(new_embeddings) // qdrant_batch_size + 1} of size {len(points)} to Qdrant.",
                flush=True
            )
        except Exception as e:
            print(e)
            print(f"Failed to upsert batch {i // qdrant_batch_size + 1} to Qdrant.")

def _batch_upsert_weaviate(
    weaviate_api, new_embeddings, new_payloads, weaviate_batch_size=64
):
    """Perform batch upserts of embeddings to Weaviate.

    @type weaviate_api: L{WeaviateAPI}
    @param weaviate_api: WeaviateAPI instance.
    @type new_embeddings: List[Dict[str, Any]]
    @param new_embeddings: List of new embeddings.
    @type new_payloads: List[Dict[str, Any]]
    @param new_payloads: List of new payloads.
    @type weaviate_batch_size: int
    @param weaviate_batch_size: Batch size for inserting into Weaviate.
    """
    uuids = [payload["instance_id"] for payload in new_payloads]
    embeddings = new_embeddings
    labels = [payload["class"] for payload in new_payloads]

    try:
        weaviate_api.insert_embeddings(uuids, embeddings, labels, weaviate_batch_size)
        print(f"Upserted {len(uuids)} of embeddings to Weaviate.")
        
    except Exception as e:
        print(e)
        print(f"Failed to upsert embeddings to Weaviate.")


def generate_embeddings(
    luxonis_dataset,
    ort_session,
    qdrant_api,
    output_layer_name,
    transform=None,
    emb_batch_size=64,
    qdrant_batch_size=64,
):
    """Generate embeddings for a given dataset and insert them into Qdrant.

    @type luxonis_dataset: L{LuxonisDataset}
    @param luxonis_dataset: The dataset object.
    @type ort_session: L{InferenceSession}
    @param ort_session: ONNX runtime session.
    @type qdrant_api: L{QdrantAPI}
    @param qdrant_api: Qdrant client API instance.
    @type output_layer_name: str
    @param output_layer_name: Name of the output layer in the ONNX model.
    @type emb_batch_size: int
    @param emb_batch_size: Batch size for generating embeddings.
    @type qdrant_batch_size: int
    @param qdrant_batch_size: Batch size for inserting into Qdrant.
    @type: Dict[str, List[float]]
    @return: Dictionary of instance ID to embedding.
    """

    all_payloads = _get_sample_payloads_coco(luxonis_dataset)
    # all_payloads = _get_sample_payloads(luxonis_dataset)

    qdrant_client, collection_name = qdrant_api.client, qdrant_api.collection_name

    new_payloads = _filter_new_samples_by_id(
        qdrant_client, collection_name, all_payloads
    )

    new_embeddings = _generate_new_embeddings(
        ort_session, output_layer_name, emb_batch_size, new_payloads, transform=transform
    )

    _batch_upsert(
        qdrant_client, collection_name, new_embeddings, new_payloads, qdrant_batch_size
    )

    # make a instance_id : embedding dictionary
    instance_id_to_embedding = {
        payload["instance_id"]: embedding
        for payload, embedding in zip(new_payloads, new_embeddings)
    }
    print("Embeddings generation and insertion completed!")

    # returns only the new embeddings
    return instance_id_to_embedding



def generate_embeddings_weaviate(
    luxonis_dataset,
    ort_session,
    weaviate_api,
    output_layer_name,
    transform=None,
    emb_batch_size=64,
    weaviate_batch_size=64,
):
    """Generate embeddings for a given dataset and insert them into Qdrant.

    @type luxonis_dataset: L{LuxonisDataset}
    @param luxonis_dataset: The dataset object.
    @type ort_session: L{InferenceSession}
    @param ort_session: ONNX runtime session.
    @type qdrant_api: L{QdrantAPI}
    @param qdrant_api: Qdrant client API instance.
    @type output_layer_name: str
    @param output_layer_name: Name of the output layer in the ONNX model.
    @type emb_batch_size: int
    @param emb_batch_size: Batch size for generating embeddings.
    @type qdrant_batch_size: int
    @param qdrant_batch_size: Batch size for inserting into Qdrant.
    @type: Dict[str, List[float]]
    @return: Dictionary of instance ID to embedding.
    """

    all_payloads = _get_sample_payloads_coco(luxonis_dataset)
    # all_payloads = _get_sample_payloads(luxonis_dataset)

    new_payloads = _filter_new_samples_by_id_weaviate(
        weaviate_api, all_payloads
    )

    new_embeddings = _generate_new_embeddings(
        ort_session, output_layer_name, emb_batch_size, new_payloads, transform=transform
    )

    _batch_upsert_weaviate(
        weaviate_api, new_embeddings, new_payloads, weaviate_batch_size
    )

    # make a instance_id : embedding dictionary
    instance_id_to_embedding = {
        payload["instance_id"]: embedding
        for payload, embedding in zip(new_payloads, new_embeddings)
    }
    print("Embeddings generation and insertion completed!")

    # returns only the new embeddings
    return instance_id_to_embedding
