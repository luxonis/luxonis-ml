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

DEFAULT_COLLECTION_NAME = "mnist"


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


class QdrantAPI:
    """Class to perform various Qdrant operations related to embeddings."""

    def __init__(
        self, host="localhost", port=6333, collection_name=DEFAULT_COLLECTION_NAME
    ):
        """Initialize the QdrantAPI."""
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name

    def create_collection(self, vector_size=512, distance=Distance.COSINE):
        """Create a collection in Qdrant."""
        try:
            self.client.get_collection(collection_name=self.collection_name)
            print("Collection already exists")
        except Exception:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=distance),
            )
            print("Created new collection")

    def insert_embeddings(self, embeddings, labels):
        """Insert embeddings and their labels into a Qdrant collection."""
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=i,
                    vector=embeddings[i].tolist(),
                    payload={"label": labels[i].item()},
                )
                for i in range(len(embeddings))
            ],
        )

    def insert_embeddings_nooverwrite(self, embeddings, labels):
        """Insert embeddings and labels into a Qdrant collection only if they don't
        already exist (independent of the id)."""
        # Create a list of search requests
        search_queries = [
            SearchRequest(
                vector=embeddings[i].tolist(), limit=1, score_threshold=0.9999
            )
            for i in range(len(embeddings))
        ]

        # Search for the nearest neighbors
        batch_search_results = self.client.search_batch(
            collection_name=self.collection_name, requests=search_queries
        )

        # Get the indices of the embeddings that are not in the collection
        new_ix = [i for i, res in enumerate(batch_search_results) if len(res) == 0]
        if len(new_ix) == 0:
            print("No new embeddings to insert")
            return
        new_embeddings = embeddings[new_ix]
        new_labels = labels[new_ix]

        # Get the collection size - the number of points in the collection - start from this index
        collection_info = self.client.get_collection(
            collection_name=self.collection_name
        )
        collection_size = collection_info.vectors_count

        # Insert the embeddings into the collection
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=collection_size + i,
                    vector=new_embeddings[i].tolist(),
                    payload={"label": new_labels[i].item()},
                )
                for i in range(len(new_embeddings))
            ],
        )

        print("Inserted {} new embeddings".format(len(new_embeddings)))

    def batch_insert_embeddings(self, embeddings, labels, img_paths, batch_size=50):
        """Batch insert embeddings, labels, and image paths into a Qdrant collection."""
        total_len = len(embeddings)

        for i in range(0, total_len, batch_size):
            start = i
            end = min(i + batch_size, total_len)

            # Ensure IDs start from 1, not 0
            batch_ids = list(range(start + 1, end + 1))
            batch_vectors = embeddings[start:end].tolist()
            batch_payloads = [
                {"label": labels[j].item(), "image_path": img_paths[j]}
                for j in range(start, end)
            ]

            batch = models.Batch(
                ids=batch_ids, vectors=batch_vectors, payloads=batch_payloads
            )

            # Upsert the batch of points to the Qdrant collection
            self.client.upsert(collection_name=self.collection_name, points=batch)

    def batch_insert_embeddings_nooverwrite(self, embeddings, labels, batch_size=50):
        """Batch insert embeddings and labels into a Qdrant collection, avoiding
        overwriting existing embeddings."""
        total_len = len(embeddings)

        for i in range(0, total_len, batch_size):
            start = i
            end = min(i + batch_size, total_len)

            batch_embeddings = embeddings[start:end]
            batch_labels = labels[start:end]
            # batch_img_paths = img_paths[start:end]

            # Create a list of search requests for the current batch
            search_queries = [
                SearchRequest(vector=vec.tolist(), limit=1, score_threshold=0.9999)
                for vec in batch_embeddings
            ]

            # Search for the nearest neighbors
            batch_search_results = self.client.search_batch(
                collection_name=self.collection_name, requests=search_queries
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
            collection_info = self.client.get_collection(
                collection_name=self.collection_name
            )
            collection_size = collection_info.vectors_count

            # Insert the new embeddings into the collection
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=collection_size + k + 1,  # Ensure IDs start from 1, not 0
                        vector=new_embeddings[k].tolist(),
                        payload={
                            "label": new_labels[k].item()
                        },  # , "image_path": new_img_paths[k]}
                    )
                    for k in range(len(new_embeddings))
                ],
            )

            # self.client.upload_collection(collection_name=self.collection_name,
            #                          vectors=new_embeddings.tolist(),
            #                          payload=[{"label": new_labels[k].item()} for k in range(len(new_embeddings))],
            #                          ids=None,
            #                          parallel=2)

            print(
                f"Inserted {len(new_embeddings)} new embeddings for batch {start}-{end}"
            )

    def search_embeddings(self, embedding: np.ndarray, top=5):
        """Search for the top similar embeddings in a Qdrant collection."""
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=embedding.tolist(),
            limit=top,
        )
        return search_results

    def search_embeddings_by_imagepath(self, embedding, image_path_part, top=5):
        """Search for top similar embeddings in a Qdrant collection based on a partial
        image path."""
        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=embedding.tolist(),
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="image_path", match=models.MatchText(text=image_path_part)
                    )
                ]
            ),
            limit=top,
        )
        return hits

    def get_similarities(self, reference_id, other_ids):
        """Get a list of similarity scores between the reference embedding and other
        embeddings. @note: The initial order of the other_ids list is NOT preserved.

        @type reference_id: int
        @param reference_id: The instance_id of the reference embedding.
        @type other_ids: List[int]
        @param other_ids: The list of instance_ids of other embeddings to compare with
            the reference.
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

        ids = [hit.id for hit in hits]
        scores = [hit.score for hit in hits]

        return ids, scores

    def get_full_similarity_matrix(self, batch_size=100):
        """Compute a full similarity matrix for all embeddings in a Qdrant collection.

        @note: This method is not recommended for large collections. It is better to use
            the L{get_all_embeddings} method and compute the similarity matrix yourself.
        """
        # Get all embeddings
        ids, embeddings = self.get_all_embeddings()
        print("Retrieved {} embeddings".format(len(embeddings)))

        # Create a list of search requests
        search_queries = [
            SearchRequest(
                vector=emb, with_payload=False, with_vector=False, limit=len(embeddings)
            )
            for emb in embeddings
        ]
        print("Created {} search queries".format(len(search_queries)))

        # Search for the nearest neighbors
        batch_search_results = []
        for patch in range(0, len(search_queries), batch_size):
            batch_search_results_i = self.client.search_batch(
                collection_name=self.collection_name,
                requests=search_queries[patch : patch + batch_size],
            )
            batch_search_results.extend(batch_search_results_i)
            print("Completed search for batch {}-{}".format(patch, patch + batch_size))

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
        return ids, sim_matrix

    def get_payloads_from_ids(self, ids):
        """Retrieve payloads associated with a list of IDs from a Qdrant collection.

        The order of the payloads IS preserved.
        """
        # Retrieve the payloads for the given ids
        hits = self.client.retrieve(
            collection_name=self.collection_name,
            ids=ids,
            with_payload=True,
            with_vectors=False,
        )

        # Convert the payloads to a list and order them by the original instance_id
        id_to_index = {id: index for index, id in enumerate(ids)}
        payloads = [None for _ in range(len(ids))]
        for hit in hits:
            i = id_to_index[hit.id]
            payloads[i] = hit.payload

        return payloads

    def get_embeddings_from_ids(self, ids):
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
        # embeddings = [hit.vector for hit in hits]
        id_to_index = {id: index for index, id in enumerate(ids)}
        embeddings = [None for _ in range(len(ids))]
        for hit in hits:
            i = id_to_index[hit.id]
            embeddings[i] = hit.vector
        return embeddings

    def get_all_ids(self):
        """Retrieve all IDs from a Qdrant collection."""
        # Get the number of points in the collection
        collection_info = self.client.get_collection(
            collection_name=self.collection_name
        )
        collection_size = collection_info.vectors_count

        # Use qdrant scroll method
        hits, _offset = self.client.scroll(
            collection_name=self.collection_name, limit=collection_size
        )
        ids = [hit.id for hit in hits]
        return ids

    def get_all_instance_and_sample_ids(self):
        """Retrieve all instance and sample IDs from a Qdrant collection."""
        # Get the number of points in the collection
        collection_info = self.client.get_collection(
            collection_name=self.collection_name
        )
        collection_size = collection_info.vectors_count

        # Use qdrant scroll method
        hits, _offset = self.client.scroll(
            collection_name=self.collection_name, limit=collection_size
        )

        instance_ids = [hit.payload["instance_id"] for hit in hits]
        sample_ids = [hit.payload["sample_id"] for hit in hits]
        return instance_ids, sample_ids

    def get_all_embeddings(self):
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

        ids = [hit.id for hit in hits]
        embeddings = [hit.vector for hit in hits]
        return ids, embeddings
