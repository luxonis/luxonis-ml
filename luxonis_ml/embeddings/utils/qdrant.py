"""Qdrant Docker Management and Embedding Operations.

This script provides utility functions for managing Qdrant via Docker and performing operations related to embeddings.

Features include:
    - Docker management: Checks for Docker installation, running status, and existence of specific images or containers.
    - Qdrant management: Facilitates starting a Qdrant container, connecting to Qdrant, and creating collections.
    - Embedding operations: Supports inserting, batch inserting, searching, and retrieving embeddings within a Qdrant collection.

Dependencies:
    - os: Standard library module for interacting with the operating system.
    - docker: Library to manage Docker containers.
    - qdrant_client: Client library for interacting with the Qdrant service.

Usage steps:
    1. Ensure Docker is installed and running on your system.
    2. Utilize the QdrantManager class and its start_docker_qdrant() method to initiate a Qdrant container.
    3. Employ the QdrantAPI class to either connect to an existing Qdrant service or create a new collection.
    4. Use the QdrantAPI class to perform various embedding-related operations within the specified Qdrant collection.

Note:
    - The default collection name is 'mnist'.
    - It is essential that the user has appropriate permissions to execute Docker commands without sudo.
For guidance on setting this up, refer to: https://docs.docker.com/engine/install/linux-postinstall/
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import docker
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import Distance, SearchRequest, VectorParams

from luxonis_ml.embeddings.utils.vectordb import VectorDBAPI


class QdrantManager:
    """Class to manage Qdrant Docker container and perform various
    operations related to embeddings."""

    def __init__(
        self, image_name="qdrant/qdrant", container_name="qdrant_container"
    ):
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

    def stop_docker_qdrant(self):
        """Stop the Qdrant Docker container."""
        if self.does_container_exist():
            container = self.client_docker.containers.get(self.container_name)
            container.stop()
            print("Stopped container")
        else:
            print("Container does not exist")


class QdrantAPI(VectorDBAPI):
    """Class to perform various Qdrant operations related to
    embeddings."""

    def __init__(self, host: str = "localhost", port: int = 6333) -> None:
        """Initialize the QdrantAPI without setting a specific
        collection.

        @type host: str
        @param host: The host address of the Qdrant server. Default is
            "localhost".
        @type port: int
        @param port: The port number of the Qdrant server. Default is
            6333.
        """
        self.client = QdrantClient(host=host, port=port)

    def create_collection(
        self,
        collection_name: str,
        properties: List[str],
        vector_size: int = 512,
    ) -> None:
        """Create a collection in Qdrant with specified properties.

        @type collection_name: str
        @param collection_name: The name of the collection.
        @type properties: List[str]
        @param properties: The list of properties for the collection.
        @type vector_size: int
        @param vector_size: The size of the embedding vectors. Default
            is 512.
        """
        self.collection_name = collection_name
        self.properties = properties
        try:
            self.client.get_collection(collection_name=self.collection_name)
            print("Collection already exists")
        except Exception:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size, distance=Distance.COSINE
                ),
            )
            print("Created new collection")

    def delete_collection(self) -> None:
        """Delete a collection in Qdrant."""
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            print("Deleted collection")
        except Exception:
            print("Collection does not exist")

    def insert_embeddings(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        payloads: List[Dict[str, Any]],
        batch_size: int = 50,
    ) -> None:
        """Batch insert embeddings with IDs and additional metadata into
        a collection.

        @type ids: List[str]
        @param ids: The list of instance_ids for the embeddings.
        @type embeddings: List[List[float]]
        @param embeddings: The list of embedding vectors.
        @type payloads: List[Dict[str, Any]]
        @param payloads: The list of additional metadata for the
            embeddings.
        @type batch_size: int
        @param batch_size: The batch size for inserting embeddings.
            Default is 50.
        """
        total_len = len(embeddings)

        # check if payloads key values subset of the self.properties
        for payload in payloads:
            if not set(payload.keys()).issubset(set(self.properties)):
                raise ValueError(
                    "Payload keys should be subset of the properties"
                )

        for i in range(0, total_len, batch_size):
            batch_ids = ids[i : i + batch_size]
            batch_vectors = embeddings[i : i + batch_size]
            batch_payloads = payloads[i : i + batch_size]

            batch = models.Batch(
                ids=batch_ids, vectors=batch_vectors, payloads=batch_payloads
            )

            # Upsert the batch of points to the Qdrant collection
            self.client.upsert(
                collection_name=self.collection_name, points=batch
            )

    def search_similar_embeddings(
        self, embedding: List[float], top_k: int = 5
    ) -> Tuple[List[str], List[float]]:
        """Search for the top similar embeddings in a Qdrant collection.

        @type embedding: List[float]
        @param embedding: The query embedding vector.
        @type top_k: int
        @param top_k: The number of similar embeddings to retrieve.
            Default is 5.
        @rtype: Tuple[List[str], List[float]]
        @return: The list of instance_ids of the similar embeddings and
            the list of similarity scores.
        """
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

    def get_similarity_scores(
        self,
        reference_id: str,
        other_ids: List[str],
        sort_distances: bool = True,
    ) -> Tuple[List[str], List[float]]:
        """Get a list of similarity scores between the reference
        embedding and other embeddings.

        @type reference_id: int
        @param reference_id: The instance_id of the reference embedding.
        @type other_ids: List[int]
        @param other_ids: The list of instance_ids of other embeddings
            to compare with the reference.
        @type sort_distances: bool
        @param sort_distances: Whether to sort the results by distance
            or keep the original order.
        @rtype: Tuple[List[int], List[float]
        @return: The list of instance_ids of the other embeddings and
            the list of similarity scores.
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
            ids = []
            scores = []
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

    def compute_similarity_matrix(self) -> List[List[float]]:
        """Compute a full similarity matrix for all embeddings in a
        Qdrant collection.

        @rtype: Tuple[List[str], List[List[float]]]
        @return: The list of instance_ids of the embeddings and the
            similarity matrix.
        @note: This method is not recommended for large collections. It
            is better to use the L{get_all_embeddings} method and
            compute the similarity matrix yourself.
        """
        # Get all embeddings
        ids, embeddings = self.retrieve_all_embeddings()
        print(f"Retrieved {len(embeddings)} embeddings")

        # Create a list of search requests
        search_queries = [
            SearchRequest(
                vector=emb,
                with_payload=False,
                with_vector=False,
                limit=len(embeddings),
            )
            for emb in embeddings
        ]
        print(f"Created {len(search_queries)} search queries")

        # Perform batch search
        batch_search_results = []
        for patch in range(0, len(search_queries), 100):
            batch_search_results_i = self.client.search_batch(
                collection_name=self.collection_name,
                requests=search_queries[patch : patch + 100],
            )
            batch_search_results.extend(batch_search_results_i)
            print(f"Completed search for batch {patch}-{patch + 100}")

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

    def retrieve_embeddings_by_ids(self, ids: List[str]) -> List[List[float]]:
        """Retrieve embeddings associated with a list of IDs from a
        Qdrant collection. The order of the embeddings IS preserved.

        @type ids: List[str]
        @param ids: The list of instance_ids of the embeddings to
            retrieve.
        @rtype: List[List[float]]
        @return: The list of embedding vectors.
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

    def retrieve_all_ids(self) -> List[str]:
        """Retrieve all IDs from a Qdrant collection.

        @rtype: List[str]
        @return: The list of instance_ids of the embeddings.
        """
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

    def retrieve_all_embeddings(self) -> Tuple[List[str], List[List[float]]]:
        """Retrieve all embeddings and their IDs from a Qdrant
        collection.

        @rtype: Tuple[List[str], List[List[float]]]
        @return: The list of instance_ids of the embeddings and the list
            of embedding vectors.
        """
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

    def retrieve_payloads_by_ids(
        self, ids: List[str], properties: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve specified payload properties for a list of IDs from
        a collection. The order of the labels IS preserved.

        @type ids: List[str]
        @param ids: The list of instance_ids of the embeddings to
            retrieve.
        @type properties: Optional[List[str]]
        @param properties: The list of payload properties to retrieve.
            Default is None.
        @rtype: List[Dict[str, Any]]
        @return: The list of payload dictionaries.
        """
        # Retrieve the payloads for the given ids
        hits = self.client.retrieve(
            collection_name=self.collection_name,
            ids=ids,
            with_payload=True,
            with_vectors=False,
        )

        # Order the payloads according to the given ids
        id_to_index = {id: index for index, id in enumerate(ids)}
        payloads = [None for _ in range(len(ids))]
        for hit in hits:
            i = id_to_index[hit.id]
            if properties is not None:
                payload = {prop: hit.payload[prop] for prop in properties}
            else:
                payload = hit.payload
            payloads[i] = payload

        return payloads
