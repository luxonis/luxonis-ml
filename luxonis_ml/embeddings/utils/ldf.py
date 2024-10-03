"""Utilities for generating image embeddings and inserting them into a
VectorDB database.

This script provides functions for:

    - Extracting payloads from LuxonisDatasets, specifically for classification datasets.
    - Filtering new samples based on their instance IDs to avoid duplicates in the database.
    - Generating embeddings for new images using an ONNX runtime session.
    - Performing batch upserts of embeddings into a VectorDB database.

Key modules used:

    - I{luxonis_ml.data}: For loading and working with LuxonisDatasets.
    - I{luxonis_ml.embeddings.utils.embedding}: For extracting embeddings from images.
    - I{luxonis_ml.embeddings.utils.vectordb}: For interacting with VectorDB databases.

Main functions:

    - I{_get_sample_payloads_LDF}: Extracts payloads from a LuxonisDataset for classification datasets.
    - I{_filter_new_samples_by_id}: Filters out samples already in the database based on instance IDs.
    - I{_batch_upsert}: Performs batch upserts of embeddings into the database.
    - I{generate_embeddings}: Main function to generate embeddings for a dataset and insert them into the database.

Important note:

Ensure that a VectorDB server is running and accessible before using these utilities.
"""

from typing import Any, Callable, Dict, List

import numpy as np
import onnxruntime

from luxonis_ml.data import LuxonisDataset
from luxonis_ml.embeddings.utils.embedding import extract_embeddings
from luxonis_ml.embeddings.utils.vectordb import VectorDBAPI


def _get_sample_payloads_LDF(dataset: LuxonisDataset) -> List[Dict[str, Any]]:
    """Extract payloads from the LuxonisDataset. Currently supports
    classification datasets.

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
            instance_id = row[1]["instance_id"]
            img_path = row[1]["original_filepath"]
            class_name = row[1]["class"]

            all_payloads.append(
                {
                    "instance_id": instance_id,
                    "image_path": img_path,
                    "class": class_name,
                }
            )

    return all_payloads


def _filter_new_samples_by_id(
    vectordb_api: VectorDBAPI, all_payloads: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Filter out samples that are already in the Vector database based
    on their instance ID.

    @type vectordb_api: L{VectorDBAPI}
    @param vectordb_api: Vector database API instance.
    @type all_payloads: List[Dict[str, Any]]
    @param all_payloads: List of all payloads.
    @rtype: List[Dict[str, Any]]
    @return: List of new payloads.
    """
    all_payloads = all_payloads or []
    # Filter out samples that are already in the Qdrant database
    if len(all_payloads) == 0:
        print("Payloads list is empty!")
        return all_payloads

    retrieved_ids = vectordb_api.retrieve_all_ids()

    new_payloads = [
        payload
        for payload in all_payloads
        if payload["instance_id"] not in retrieved_ids
    ]

    return new_payloads


def _batch_upsert(
    vectordb_api: VectorDBAPI,
    new_embeddings: List[List[float]],
    new_payloads: List[Dict[str, Any]],
    vectordb_batch_size: int = 64,
) -> None:
    """Perform batch upserts of embeddings to VectorDB.

    @type vectordb_api: L{VectorDBAPI}
    @param vectordb_api: VectorDBAPI instance.
    @type new_embeddings: List[List[float]]
    @param new_embeddings: List of new embeddings.
    @type new_payloads: List[Dict[str, Any]]
    @param new_payloads: List of new payloads.
    @type vectordb_batch_size: int
    @param vectordb_batch_size: Batch size for inserting into VectorDB.
    """
    uuids = []
    qdrant_payloads = []
    for payload in new_payloads:
        uuids.append(payload["instance_id"])
        qdrant_payloads.append(
            {"label": payload["class"], "image_path": payload["image_path"]}
        )

    try:
        vectordb_api.insert_embeddings(
            uuids, new_embeddings, qdrant_payloads, vectordb_batch_size
        )
        print(f"Upserted {len(uuids)} of embeddings to VectorDB.")

    except Exception as e:
        print(e)
        print("Failed to upsert embeddings to VectorDB.")


def generate_embeddings(
    luxonis_dataset: LuxonisDataset,
    ort_session: onnxruntime.InferenceSession,
    vectordb_api: VectorDBAPI,
    output_layer_name: str = "/Flatten_output_0",
    transform: Callable[[np.ndarray], np.ndarray] = None,
    emb_batch_size: int = 64,
    vectordb_batch_size: int = 64,
) -> Dict[str, List[float]]:
    """Generate embeddings for a given dataset and insert them into a
    VectorDB.

    @type luxonis_dataset: L{LuxonisDataset}
    @param luxonis_dataset: The dataset object.
    @type ort_session: L{InferenceSession}
    @param ort_session: ONNX runtime session.
    @type vectordb_api: L{VectorDBAPI}
    @param vectordb_api: VectorDBAPI instance.
    @type output_layer_name: str
    @param output_layer_name: Name of the output layer in the ONNX
        model.
    @type transform: Callable[[np.ndarray], np.ndarray]
    @param transform: Preprocessing function for images. If None,
        default preprocessing is used.
    @type emb_batch_size: int
    @param emb_batch_size: Batch size for generating embeddings.
    @type vectordb_batch_size: int
    @param vectordb_batch_size: Batch size for inserting into a vector
        DB.
    @type: Dict[str, List[float]]
    @return: Dictionary of instance ID to embedding.
    """

    all_payloads = _get_sample_payloads_LDF(luxonis_dataset)
    # all_payloads = _get_sample_payloads(luxonis_dataset)

    new_payloads = _filter_new_samples_by_id(vectordb_api, all_payloads)

    new_img_paths = [payload["image_path"] for payload in new_payloads]
    new_embeddings, succ_ix = extract_embeddings(
        new_img_paths,
        ort_session,
        luxonis_dataset.fs,
        transform,
        output_layer_name,
        emb_batch_size,
    )
    new_payloads = [new_payloads[ix] for ix in succ_ix]

    _batch_upsert(
        vectordb_api, new_embeddings, new_payloads, vectordb_batch_size
    )

    # make a instance_id : embedding dictionary
    instance_id_to_embedding = {
        payload["instance_id"]: embedding
        for payload, embedding in zip(new_payloads, new_embeddings)
    }
    print("Embeddings generation and insertion completed!")

    # returns only the new embeddings
    return instance_id_to_embedding
