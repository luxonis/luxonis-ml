"""
Qdrant Docker Management and Embedding Operations

This script provides a set of utility functions to manage Qdrant using Docker and perform various operations related to embeddings.

Features:
- Docker management: Check if Docker is installed, running, and if specific images or containers exist.
- Qdrant management: Start a Qdrant container, connect to Qdrant, and create a collection.
- Embedding operations: Insert, batch insert, search, and retrieve embeddings from a Qdrant collection.

Dependencies:
- docker: For Docker-related operations.
- numpy: For numerical operations on embeddings.
- qdrant_client: For interacting with Qdrant.

Usage:
1. Ensure Docker is installed and running.
2. Start the Qdrant Docker container using `start_docker_qdrant()`.
3. Connect to Qdrant using `connect_to_qdrant()`.
4. Perform various embedding operations like inserting, searching, and retrieving.

Note:
- The default collection name is set to "mnist".
- Ensure the user has the appropriate permissions to run Docker commands without sudo.
  See https://docs.docker.com/engine/install/linux-postinstall/ for more details.
"""

import docker
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, SearchRequest
from qdrant_client.http import models

DEFAULT_COLLECTION_NAME = "mnist"

def is_docker_installed(client_docker):
    """Check if Docker is installed."""
    try:
        client_docker.version()
        return True
    except docker.errors.DockerException:
        return False

def is_docker_running(client_docker):
    """Check if Docker daemon is running."""
    try:
        client_docker.ping()
        return True
    except docker.errors.DockerException:
        return False

def does_image_exist(client_docker, image_name):
    """Check if a Docker image exists."""
    try:
        client_docker.images.get(image_name)
        return True
    except docker.errors.ImageNotFound:
        return False

def does_container_exist(client_docker, container_name):
    """Check if a Docker container exists."""
    try:
        client_docker.containers.get(container_name)
        return True
    except docker.errors.NotFound:
        return False
def is_container_running(client_docker, container_name):
    """Check if a Docker container is running."""
    try:
        container = client_docker.containers.get(container_name)
        return container.status == 'running'
    except docker.errors.NotFound:
        return False

def start_docker_qdrant(image_name = "qdrant/qdrant", container_name = "qdrant_container"):
    """
    Start the Qdrant Docker container.

    NOTE: Make sure the user has the appropriate permissions to run Docker commands
         without sudo. Otherwise, the client_docker.images.pull() command will fail.
         See https://docs.docker.com/engine/install/linux-postinstall/ for more details.
    """
    client_docker = docker.from_env()

    if not is_docker_installed(client_docker):
        print("Docker is not installed. Please install Docker to proceed.")
        return

    if not is_docker_running(client_docker):
        print("Docker daemon is not running. Please start Docker manually and try again.")
        return

    if not does_image_exist(client_docker, image_name):
        print("Image does not exist. Pulling image...")
        client_docker.images.pull(image_name)

    if not does_container_exist(client_docker, container_name):
        print("Container does not exist. Creating container...")
        client_docker.containers.run(
            image_name, 
            detach=True, 
            name=container_name, 
            ports={'6333/tcp': 6333}, 
            volumes={'~/.cache/qdrant_data': {'bind': '/app/data', 'mode': 'rw'}}
        )
    elif not is_container_running(client_docker, container_name):
        print("Container is not running. Starting container...")
        container = client_docker.containers.get(container_name)
        container.start()
    else:
        print("Container is already running.")

def connect_to_qdrant(host="localhost", port=6333):
    """Connect to Qdrant."""
    return QdrantClient(host=host, port=port)

def create_collection(client, collection_name=DEFAULT_COLLECTION_NAME, vector_size=512, distance=Distance.COSINE):
    """Create a collection in Qdrant."""
    try:
        client.get_collection(collection_name=collection_name)
        print("Collection already exists")
    except models.QdrantError:
        client.recreate_collection(collection_name=collection_name,
                                   vectors_config=VectorParams(size=vector_size, distance=distance))
        print("Created new collection")

def insert_embeddings(client, embeddings, labels, collection_name=DEFAULT_COLLECTION_NAME):
    """Insert embeddings and their labels into a Qdrant collection."""
    client.upsert(collection_name=collection_name, 
                    points= [PointStruct(
                            id=i,
                            vector=embeddings[i].tolist(),
                            payload={"label": labels[i].item()}
                        ) for i in range(len(embeddings))])

def insert_embeddings_nooverwrite(client, embeddings, labels, collection_name=DEFAULT_COLLECTION_NAME):
    """Insert embeddings and labels into a Qdrant collection only if they don't already exist (independent of the id)."""
    # Create a list of search requests
    search_queries = [SearchRequest(
        vector=embeddings[i].tolist(),
        limit=1,
        score_threshold=0.9999) for i in range(len(embeddings))]
    
    # Search for the nearest neighbors
    batch_search_results = client.search_batch(
        collection_name=collection_name,
        requests=search_queries
    )
    
    # Get the indices of the embeddings that are not in the collection
    new_ix = [i for i, res in enumerate(batch_search_results) if len(res) == 0]
    if len(new_ix) == 0:
        print("No new embeddings to insert")
        return
    new_embeddings = embeddings[new_ix]
    new_labels = labels[new_ix]

    # Get the collection size - the number of points in the collection - start from this index
    collection_info = client.get_collection(collection_name=collection_name)
    collection_size = collection_info.vectors_count

    # Insert the embeddings into the collection
    client.upsert(collection_name=collection_name,
                    points= [PointStruct(
                            id=collection_size + i,
                            vector=new_embeddings[i].tolist(),
                            payload={"label": new_labels[i].item()}
                        ) for i in range(len(new_embeddings))])
    
    print("Inserted {} new embeddings".format(len(new_embeddings)))

def batch_insert_embeddings(client, embeddings, labels, img_paths, batch_size = 50, collection_name=DEFAULT_COLLECTION_NAME):
    """Batch insert embeddings, labels, and image paths into a Qdrant collection."""
    total_len = len(embeddings)

    for i in range(0, total_len, batch_size):
        start = i
        end = min(i + batch_size, total_len)

        # Ensure IDs start from 1, not 0
        batch_ids = list(range(start + 1, end + 1))
        batch_vectors = embeddings[start:end].tolist()
        batch_payloads = [{'label': labels[j].item(), 'image_path': img_paths[j]} for j in range(start, end)]

        batch = models.Batch(
            ids=batch_ids, 
            vectors=batch_vectors, 
            payloads=batch_payloads
        )

        # Upsert the batch of points to the Qdrant collection
        client.upsert(collection_name=collection_name, points=batch)

def batch_insert_embeddings_nooverwrite(client, embeddings, labels, batch_size=50, collection_name=DEFAULT_COLLECTION_NAME):
    """Batch insert embeddings and labels into a Qdrant collection, avoiding overwriting existing embeddings."""
    total_len = len(embeddings)

    for i in range(0, total_len, batch_size):
        start = i
        end = min(i + batch_size, total_len)

        batch_embeddings = embeddings[start:end]
        batch_labels = labels[start:end]
        # batch_img_paths = img_paths[start:end]

        # Create a list of search requests for the current batch
        search_queries = [SearchRequest(
            vector=vec.tolist(),
            limit=1,
            score_threshold=0.9999) for vec in batch_embeddings]

        # Search for the nearest neighbors
        batch_search_results = client.search_batch(
            collection_name=collection_name,
            requests=search_queries
        )

        # Get the indices of the embeddings that are not in the collection
        new_ix = [j for j, res in enumerate(batch_search_results) if len(res) == 0]
        if len(new_ix) == 0:
            print(f"No new embeddings to insert for batch {start}-{end}")
            continue
        new_embeddings = batch_embeddings[new_ix]
        new_labels = batch_labels[new_ix]
        # new_img_paths = [batch_img_paths[j] for j in new_ix]

        # Get the collection size to determine the starting index for new embeddings
        collection_info = client.get_collection(collection_name=collection_name)
        collection_size = collection_info.vectors_count

        # Insert the new embeddings into the collection
        client.upsert(collection_name=collection_name,
                      points=[PointStruct(
                          id=collection_size + k + 1,  # Ensure IDs start from 1, not 0
                          vector=new_embeddings[k].tolist(),
                          payload={"label": new_labels[k].item()}#, "image_path": new_img_paths[k]}
                      ) for k in range(len(new_embeddings))])
        
        # client.upload_collection(collection_name=collection_name,
        #                          vectors=new_embeddings.tolist(),
        #                          payload=[{"label": new_labels[k].item()} for k in range(len(new_embeddings))],
        #                          ids=None,
        #                          parallel=2)

        print(f"Inserted {len(new_embeddings)} new embeddings for batch {start}-{end}")


def search_embeddings(client, embedding:np.ndarray, collection_name=DEFAULT_COLLECTION_NAME, top=5):
    """Search for the top similar embeddings in a Qdrant collection."""
    # Search for the nearest neighbors
    search_results = client.search(collection_name=collection_name, 
                                    query_vector=embedding.tolist(), 
                                    limit=top)
    return search_results

def search_embeddings_by_imagepath(client, embedding, image_path_part, collection_name=DEFAULT_COLLECTION_NAME, top=5):
    """Search for top similar embeddings in a Qdrant collection based on a partial image path."""
    hits = client.search(
        collection_name=collection_name,
        query_vector=embedding.tolist(),
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="image_path",
                    match=models.MatchText(text=image_path_part)
                )
            ] 
        ),
        limit= top
    )
    return hits

def get_similarities(qdrant_client, reference_id, other_ids, qdrant_collection_name=DEFAULT_COLLECTION_NAME):
    """
    Get a list of similarity scores between the reference embedding and other embeddings.
    NOTE: The initial order of the other_ids list is NOT preserved.

    Parameters
    ----------
    qdrant_client : QdrantClient
        The Qdrant client instance to use for searches.
    reference_id : int
        The instance_id of the reference embedding.
    other_ids : list[int]
        The list of instance_ids of other embeddings to compare with the reference.
    qdrant_collection_name : str
        The name of the Qdrant collection. Default is DEFAULT_COLLECTION_NAME.

    Returns
    -------
    list[int]
        The list of instance_ids of the other embeddings.
    list[float]
        The list of similarity scores.
    """
    # Retrieve the embedding vector for the reference_id
    reference_embedding = get_embeddings_from_ids(qdrant_client, [reference_id], qdrant_collection_name)[0]

    # Search for similar embeddings using the reference embedding
    hits = qdrant_client.search(
        collection_name=qdrant_collection_name,
        query_vector=reference_embedding,
        query_filter=models.Filter(
            should=[
                models.FieldCondition(
                    key="instance_id",
                    match=models.MatchText(text=str(id))
                ) for id in other_ids
            ] 
        )
    )

    ids = [hit.id for hit in hits]
    scores = [hit.score for hit in hits]

    return ids, scores

def get_full_similarity_matrix(client, collection_name=DEFAULT_COLLECTION_NAME, batch_size=100):
    """
    Compute a full similarity matrix for all embeddings in a Qdrant collection.

    NOTE: This method is not recommended for large collections.
          It is better to use the get_all_embeddings() method and compute the similarity matrix yourself.
    """
    # Get all embeddings
    ids, embeddings = get_all_embeddings(client, collection_name)
    print("Retrieved {} embeddings".format(len(embeddings)))

    # Create a list of search requests
    search_queries = [SearchRequest(
        vector=emb,
        with_payload=False,
        with_vector=False,
        limit=len(embeddings)
        ) for emb in embeddings]
    print("Created {} search queries".format(len(search_queries)))
    
    # Search for the nearest neighbors
    batch_search_results = []
    for patch in range(0, len(search_queries), batch_size):
        batch_search_results_i = client.search_batch(
            collection_name=collection_name,
            requests=search_queries[patch:patch+batch_size]
        )
        batch_search_results.extend(batch_search_results_i)
        print("Completed search for batch {}-{}".format(patch, patch+batch_size))
    
    # Create a dictionary for O(1) lookup of ids
    id_to_index = {id: index for index, id in enumerate(ids)}

    # Get the similarity matrix
    sim_matrix = [[0 for _ in range(len(embeddings))] for _ in range(len(embeddings))]
    for i, res in enumerate(batch_search_results):
        for hit in res:
            j = id_to_index[hit.id]
            sim_matrix[i][j] = hit.score
    
    print("Created similarity matrix")
    return ids, sim_matrix

def get_payloads_from_ids(client, ids, collection_name=DEFAULT_COLLECTION_NAME):
    """Retrieve payloads associated with a list of IDs from a Qdrant collection.
    (The order of the payloads IS preserved.)"""
    # Retrieve the payloads for the given ids
    hits = client.retrieve(collection_name=collection_name, 
                            ids=ids, 
                            with_payload=True, 
                            with_vectors=False)
    
    # Convert the payloads to a list and order them by the original instance_id
    id_to_index = {id: index for index, id in enumerate(ids)}
    payloads = [None for _ in range(len(ids))]
    for hit in hits:
        i = id_to_index[hit.id]
        payloads[i] = hit.payload

    return payloads

def get_embeddings_from_ids(client, ids, collection_name=DEFAULT_COLLECTION_NAME):
    """Retrieve embeddings associated with a list of IDs from a Qdrant collection.
    (The order of the embeddings IS preserved.)"""
    # Retrieve the embeddings for the given ids
    hits = client.retrieve(collection_name=collection_name, 
                            ids=ids, 
                            with_payload=False, 
                            with_vectors=True)
    # embeddings = [hit.vector for hit in hits]
    id_to_index = {id: index for index, id in enumerate(ids)}
    embeddings = [None for _ in range(len(ids))]
    for hit in hits:
        i = id_to_index[hit.id]
        embeddings[i] = hit.vector
    return embeddings

def get_all_ids(client, collection_name=DEFAULT_COLLECTION_NAME):
    """Retrieve all IDs from a Qdrant collection."""
    # Get the number of points in the collection
    collection_info = client.get_collection(collection_name=collection_name)
    collection_size = collection_info.vectors_count

    # Use qdrant scroll method
    hits, _offset = client.scroll(collection_name=collection_name, 
                                  limit=collection_size)
    ids = [hit.id for hit in hits]
    return ids

def get_all_instance_and_sample_ids(client, collection_name=DEFAULT_COLLECTION_NAME):
    """Retrieve all instance and sample IDs from a Qdrant collection."""
    # Get the number of points in the collection
    collection_info = client.get_collection(collection_name=collection_name)
    collection_size = collection_info.vectors_count

    # Use qdrant scroll method
    hits, _offset = client.scroll(collection_name=collection_name, 
                                  limit=collection_size)
    
    instance_ids = [hit.payload['instance_id'] for hit in hits]
    sample_ids = [hit.payload['sample_id'] for hit in hits]
    return instance_ids, sample_ids

def get_all_embeddings(client, collection_name=DEFAULT_COLLECTION_NAME):
    """Retrieve all embeddings and their IDs from a Qdrant collection."""
    # Get the number of points in the collection
    collection_info = client.get_collection(collection_name=collection_name)
    collection_size = collection_info.vectors_count

    # Use qdrant scroll method
    hits, _offset = client.scroll(collection_name=collection_name, 
                                  limit=collection_size,
                                  with_vectors=True)
    
    ids = [hit.id for hit in hits]
    embeddings = [hit.vector for hit in hits]
    return ids, embeddings