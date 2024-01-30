"""Qdrant Docker Management and Embedding Operations.

This script provides a set of utility functions to manage Qdrant using Docker and perform various operations related to embeddings.

Features:
    - Docker management: Check if Docker is installed, running, and if specific images or containers exist.
    - Qdrant management: Start a Qdrant container, connect to Qdrant, and create a collection.
    - Embedding operations: Insert, batch insert, search, and retrieve embeddings from a Qdrant collection.

Dependencies:
    - os: For file operations.
    - docker: For Docker-related operations.
    - numpy: For numerical operations on embeddings.
    - qdrant_client: For interacting with Qdrant.

Usage:
    1. Ensure Docker is installed and running.
    2. Start a Qdrant container using the QdrantManager class and the start_docker_qdrant() method.
    3. Connect to Qdrant or create a new collection using the QdrantAPI class.
    4. Perform various operations on the Qdrant collection using the QdrantAPI class.

Note:
    - The default collection name is set to "mnist".
    - Ensure the user has the appropriate permissions to run Docker commands without sudo.
      See U{https://docs.docker.com/engine/install/linux-postinstall/} for more details.
"""

import os

import docker
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import Distance, PointStruct, SearchRequest, VectorParams

from luxonis_ml.embeddings.utils.vectordb import VectorDBAPI

convert_distance_metric = {
    "cosine": Distance.COSINE,
    "dot": Distance.DOT,
    "euclidean": Distance.EUCLID,
}

class QdrantManager:
    """Class to manage Qdrant Docker container and perform various operations related to
    embeddings."""

    def __init__(self, image_name="qdrant/qdrant", container_name="qdrant_container"):
        """Initialize the QdrantManager."""
        self.image_name = image_name
        self.container_name = container_name
        self.client_docker = docker.from_env()

    def is_docker_installed(self):
        """Check if Docker is installed."""
        try:
            self.client_docker.version()
            return True
        except docker.errors.APIError:
            return False

    def is_docker_running(self):
        """Check if Docker daemon is running."""
        try:
            self.client_docker.ping()
            return True
        except docker.errors.APIError:
            return False

    def does_image_exist(self):
        """Check if a Docker image exists."""
        try:
            self.client_docker.images.get(self.image_name)
            return True
        except docker.errors.ImageNotFound:
            return False

    def does_container_exist(self):
        """Check if a Docker container exists."""
        try:
            self.client_docker.containers.get(self.container_name)
            return True
        except docker.errors.NotFound:
            return False

    def is_container_running(self):
        """Check if a Docker container is running."""
        try:
            container = self.client_docker.containers.get(self.container_name)
            return container.status == "running"
        except docker.errors.NotFound:
            return False

    def start_docker_qdrant(self):
        """Start the Qdrant Docker container.

        @note: Make sure the user has the appropriate permissions to run Docker commands
            without sudo. Otherwise, the client_docker.images.pull() command will fail.
            See U{https://docs.docker.com/engine/install/linux-postinstall/} for more details.
        """
        if not self.is_docker_installed():
            print("Docker is not installed. Please install Docker to proceed.")
            return

        if not self.is_docker_running():
            print(
                "Docker daemon is not running. Please start Docker manually and try again."
            )
            return

        if not self.does_image_exist():
            print("Image does not exist. Pulling image...")
            self.client_docker.images.pull(self.image_name)

        if not self.does_container_exist():
            print("Container does not exist. Creating container...")
            # Expand the user's home directory
            volume_path = os.path.expanduser(
                "~/.cache/" + self.container_name + "/data"
            )

            # Create the directory if it doesn't exist
            if not os.path.exists(volume_path):
                os.makedirs(volume_path)

            self.client_docker.containers.run(
                self.image_name,
                detach=True,
                name=self.container_name,
                ports={"6333/tcp": 6333},
                volumes={volume_path: {"bind": "/app/data", "mode": "rw"}},
            )
        elif not self.is_container_running():
            print("Container is not running. Starting container...")
            container = self.client_docker.containers.get(self.container_name)
            container.start()
        else:
            print("Container is already running.")

class QdrantAPI(VectorDBAPI):
    """Class to perform various Qdrant operations related to embeddings."""

    def __init__(
        self, host="localhost", port=6333, collection_name="mnist"
    ):
        """Initialize the QdrantAPI."""
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name

    def create_collection(self, label=None, vector_size=512, distance="cosine"):
        """Create a collection in Qdrant."""
        try:
            self.client.get_collection(collection_name=self.collection_name)
            print("Collection already exists")
        except Exception:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size, 
                    distance=convert_distance_metric[distance]
                ),
            )
            print("Created new collection")
    
    def delete_collection(self):
        """Delete a collection in Qdrant."""
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            print("Deleted collection")
        except Exception:
            print("Collection does not exist")
    
    def insert_embeddings(self, ids, embeddings, labels, batch_size=50):
        """Batch insert embeddings, labels, and image paths into a Qdrant collection."""
        total_len = len(embeddings)

        for i in range(0, total_len, batch_size):
            start = i
            end = min(i + batch_size, total_len)

            batch_ids = ids[start:end]
            batch_vectors = embeddings[start:end]
            batch_payloads = [
                {"label": labels[j]}
                for j in range(start, end)
            ]

            batch = models.Batch(
                ids=batch_ids, vectors=batch_vectors, payloads=batch_payloads
            )
            
            # Upsert the batch of points to the Qdrant collection
            self.client.upsert(collection_name=self.collection_name, points=batch)

    def search_similar_embeddings(self, embedding, top_k=5):
        """Search for the top similar embeddings in a Qdrant collection."""
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=embedding,
            limit=top_k,
        )

        ids, scores = [], []
        for hit in search_results:
            ids.append(hit.id)
            scores.append(hit.score)
        
        return ids, scores

    def get_similarity_scores(self, reference_id, other_ids, sort_distances=True):
        """Get a list of similarity scores between the reference embedding and other
        embeddings.

        @type reference_id: int
        @param reference_id: The instance_id of the reference embedding.
        @type other_ids: List[int]
        @param other_ids: The list of instance_ids of other embeddings to compare with
            the reference.
        @type sort_distances: bool
        @param sort_distances: Whether to sort the results by distance or keep the
            original order.
        @rtype: Tuple[List[int], List[float]
        @return: The list of instance_ids of the other embeddings and the list of
            similarity scores.
        """
        # Retrieve the embedding vector for the reference_id
        reference_embedding = self.get_embeddings_from_ids([reference_id])[0]

        # Search for similar embeddings using the reference embedding
        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=reference_embedding,
            query_filter=models.Filter(
                should=[
                    models.FieldCondition(
                        key="instance_id", match=models.MatchText(text=str(id))
                    )
                    for id in other_ids
                ]
            ),
        )

        if sort_distances:
            ids = []; scores = []
            for hit in hits:
                ids.append(hit.id)
                scores.append(hit.score)
        else:
            ids = other_ids
            scores = [0 for _ in range(len(other_ids))]
            for hit in hits:
                i = other_ids.index(hit.id)
                scores[i] = hit.score

        return ids, scores

    def compute_similarity_matrix(self):
        """Compute a full similarity matrix for all embeddings in a Qdrant collection.

        @rtype: Tuple[List[str], List[List[float]]]
        @return: The list of instance_ids of the embeddings and the similarity matrix.

        @note: This method is not recommended for large collections. It is better to use
            the L{get_all_embeddings} method and compute the similarity matrix yourself.
        """
        # Get all embeddings
        ids, embeddings = self.retrieve_all_embeddings()
        print("Retrieved {} embeddings".format(len(embeddings)))

        # Create a list of search requests
        search_queries = [
            SearchRequest(
                vector=emb, 
                with_payload=False, 
                with_vector=False, 
                limit=len(embeddings)
            )
            for emb in embeddings
        ]
        print("Created {} search queries".format(len(search_queries)))

        # Perform batch search
        batch_search_results = []
        for patch in range(0, len(search_queries), 100):
            batch_search_results_i = self.client.search_batch(
                collection_name=self.collection_name,
                requests=search_queries[patch : patch + 100],
            )
            batch_search_results.extend(batch_search_results_i)
            print("Completed search for batch {}-{}".format(patch, patch + 100))

        # Create a dictionary for O(1) lookup of ids
        id_to_index = {id: index for index, id in enumerate(ids)}

        # Get the similarity matrix
        sim_matrix = [
            [0 for _ in range(len(embeddings))] for _ in range(len(embeddings))
        ]
        for i, res in enumerate(batch_search_results):
            for hit in res:
                j = id_to_index[hit.id]
                sim_matrix[i][j] = hit.score

        print("Created similarity matrix")
        return sim_matrix

    def retrieve_embeddings_by_ids(self, ids):
        """Retrieve embeddings associated with a list of IDs from a Qdrant collection.

        The order of the embeddings IS preserved.
        """
        # Retrieve the embeddings for the given ids
        hits = self.client.retrieve(
            collection_name=self.collection_name,
            ids=ids,
            with_payload=False,
            with_vectors=True,
        )
        
        # Order the embeddings according to the given ids
        id_to_index = {id: index for index, id in enumerate(ids)}
        embeddings = [None for _ in range(len(ids))]
        for hit in hits:
            i = id_to_index[hit.id]
            embeddings[i] = hit.vector

        return embeddings

    def retrieve_all_ids(self):
        """Retrieve all IDs from a Qdrant collection."""
        # Get the number of points in the collection
        collection_info = self.client.get_collection(
            collection_name=self.collection_name
        )
        collection_size = collection_info.vectors_count

        if collection_size == 0:
            return []

        # Use qdrant scroll method
        hits, _offset = self.client.scroll(
            collection_name=self.collection_name, limit=collection_size
        )
        ids = [hit.id for hit in hits]
        return ids

    def retrieve_all_embeddings(self):
        """Retrieve all embeddings and their IDs from a Qdrant collection."""
        # Get the number of points in the collection
        collection_info = self.client.get_collection(
            collection_name=self.collection_name
        )
        collection_size = collection_info.vectors_count

        # Use qdrant scroll method
        hits, _offset = self.client.scroll(
            collection_name=self.collection_name,
            limit=collection_size,
            with_vectors=True,
        )

        ids, embeddings = [], []
        for hit in hits:
            ids.append(hit.id)
            embeddings.append(hit.vector)
            
        return ids, embeddings

    def retrieve_labels_by_ids(self, ids):
        """Retrieve labels associated with a list of IDs from a Qdrant collection.

        The order of the labels IS preserved.
        """
        # Retrieve the payloads for the given ids
        hits = self.client.retrieve(
            collection_name=self.collection_name,
            ids=ids,
            with_payload=True,
            with_vectors=False,
        )

        # Order the labels according to the given ids
        id_to_index = {id: index for index, id in enumerate(ids)}
        labels = [None for _ in range(len(ids))]
        for hit in hits:
            i = id_to_index[hit.id]
            labels[i] = hit.payload["label"]

        return labels