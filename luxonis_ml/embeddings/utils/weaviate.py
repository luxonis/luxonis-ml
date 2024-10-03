from typing import Any, Dict, List, Optional, Tuple

import weaviate
import weaviate.classes as wvc
from weaviate.collections.classes.filters import FilterMetadata

from luxonis_ml.embeddings.utils.vectordb import VectorDBAPI


class WeaviateAPI(VectorDBAPI):
    """Provides a Python interface for interacting with Weaviate,
    facilitating operations such as creating collections, managing
    embeddings, and querying for similar embeddings.

    It only supports cosine similarity for now.
    """

    def __init__(
        self,
        url: str = "http://localhost:8080",
        grpc_url: str = "http://localhost:50051",
        auth_api_key: str = None,
    ) -> None:
        """Initializes the Weaviate API client with connection details.

        @type url: str
        @param url: URL of the Weaviate instance, defaults to
            U{localhost:8080}.
        @type grpc_url: str
        @param grpc_url: URL of the gRPC Weaviate instance, defaults to
            U{localhost:50051}.
        @type auth_api_key: str
        @param auth_api_key: API key for authentication. Defaults to
            C{None}.
        """
        if auth_api_key is not None:
            auth_api_key = weaviate.AuthApiKey(auth_api_key)

        if "localhost" in url:
            self.client = weaviate.connect_to_local()
        else:
            # make sure the gRPC port is correct
            self.client = weaviate.connect_to_custom(
                http_host=url.split("://")[1].split(":")[0],
                http_port=url.split("://")[1].split(":")[1],
                http_secure=url.split("://")[0] == "https",
                grpc_host=grpc_url.split("://")[1].split(":")[0],
                grpc_port=grpc_url.split("://")[1].split(":")[1],
                grpc_secure=grpc_url.split("://")[0] == "https",
                auth_credentials=auth_api_key,
            )

    def create_collection(
        self, collection_name: str, properties: List[str] = None
    ) -> None:
        """Creates a new collection in the Weaviate database.

        @type collection_name: str
        @param collection_name: Name of the collection to create.
        @type properties: List[str]
        @param properties: List of properties for the collection.
            Defaults to None.
        """
        self.collection_name = collection_name
        self.properties = properties

        if not self.client.collections.exists(self.collection_name):
            # print(f"Collection {self.collection_name} does not exist. Creating...")

            properties = []
            for prop in self.properties:
                properties.append(
                    wvc.Property(
                        name=prop,
                        data_type=wvc.DataType.TEXT,
                        skip_vectorization=True,
                    )
                )

            self.collection = self.client.collections.create(
                name=self.collection_name,
                properties=properties,
                vector_index_config=wvc.Configure.VectorIndex.hnsw(
                    distance_metric=wvc.VectorDistance.COSINE
                ),
            )
            print(f"Collection {self.collection_name} created.")
        else:
            self.collection = self.client.collections.get(self.collection_name)
            print(f"Collection {self.collection_name} already exists.")

    def delete_collection(self) -> None:
        """Deletes a collection from the Weaviate database."""
        try:
            self.client.collections.delete(self.collection_name)
            print(f"Collection {self.collection_name} deleted.")
        except Exception:
            print(f"Collection {self.collection_name} does not exist.")

    def insert_embeddings(
        self,
        uuids: List[str],
        embeddings: List[List[float]],
        payloads: List[Dict[str, Any]],
        batch_size: int = 100,
    ) -> None:
        """Inserts embeddings with associated payloads into a
        collection.

        @type uuids: List[str]
        @param uuids: List of UUIDs for the embeddings.
        @type embeddings: List[List[float]]
        @param embeddings: List of embeddings.
        @type payloads: List[Dict[str, Any]]
        @param payloads: List of payloads.
        @type batch_size: int
        @param batch_size: Batch size for inserting the embeddings.
        """
        data = []
        for i, embedding in enumerate(embeddings):
            data.append(
                wvc.DataObject(
                    properties=payloads[i], uuid=uuids[i], vector=embedding
                )
            )

            if len(data) == batch_size:
                self.collection.data.insert_many(data)
                data = []

        if len(data) > 0:
            self.collection.data.insert_many(data)

    def search_similar_embeddings(
        self, embedding: List[float], top_k: int = 10
    ) -> Tuple[List[str], List[float]]:
        """Searches for embeddings similar to a given vector.

        @type embedding: List[float]
        @param embedding: Embedding to find similar embeddings for.
        @type top_k: int
        @param top_k: Number of similar embeddings to find.
        @rtype uuids: List[str]
        @return uuids: List of UUIDs of the similar embeddings.
        @rtype scores: List[float]
        @return scores: List of similarity scores.
        """
        response = self.collection.query.near_vector(
            near_vector=embedding,
            limit=top_k,
            return_metadata=wvc.query.MetadataQuery(distance=True),
        )

        uuids = []
        scores = []
        for result in response.objects:
            uuids.append(str(result.uuid))
            scores.append(1 - result.metadata.distance)

        return uuids, scores

    def get_similarity_scores(
        self,
        reference_id: str,
        other_ids: List[str],
        sort_distances: bool = False,
    ) -> Tuple[List[str], List[float]]:
        """Calculates the similarity score between the reference
        embedding and the specified embeddings.

        @type reference_id: str
        @param reference_id: UUID of the reference embedding.
        @type other_ids: List[str]
        @param other_ids: List of UUIDs of the embeddings to compare to.
        @type sort_distances: bool
        @param sort_distances: Whether to sort the results by distance
            or keep order of the UUIDs. Defaults to False.
        @rtype ids: List[str]
        @return ids: List of UUIDs of the embeddings.
        @rtype scores: List[float]
        @return scores: List of similarity scores.
        """
        response = self.collection.query.near_object(
            near_object=reference_id,
            limit=len(other_ids),
            filters=FilterMetadata.ById.contains_any(other_ids),
            return_metadata=wvc.query.MetadataQuery(distance=True),
        )
        # near object gives order by distance result

        if sort_distances:
            ids, scores = [], []
            for result in response.objects:
                ids.append(str(result.uuid))
                scores.append(1 - result.metadata.distance)
        else:
            ids = other_ids
            scores = [0] * len(other_ids)
            for result in response.objects:
                scores[other_ids.index(str(result.uuid))] = (
                    1 - result.metadata.distance
                )

        return ids, scores

    def compute_similarity_matrix(self) -> List[List[float]]:
        """Calculates the similarity matrix for all the embeddings in
        the collection. @note: This is a very inefficient
        implementation. For large numbers of embeddings, calculate the
        similarity matrix by hand
        (sklearn.metrics.pairwise.cosine_similarity).

        @rtype sim_matrix: List[List[float]]
        @return sim_matrix: Similarity matrix for all the embeddings in
            the collection.
        """
        uuids = self.retrieve_all_ids()

        sim_matrix = []
        for uuid in uuids:
            ids, scores = self.get_similarity_scores(
                uuid, uuids, sort_distances=False
            )
            sim_matrix.append(scores)

        return sim_matrix

    def retrieve_embeddings_by_ids(
        self, uuids: List[str]
    ) -> List[List[float]]:
        """Gets the embeddings for the specified UUIDs, up to a maximum
        of 10000.

        @type uuids: List[str]
        @param uuids: List of UUIDs of the embeddings to get.
        @rtype embeddings: List[List[float]]
        @return embeddings: List of embeddings.
        """
        embeddings = []

        response = self.collection.query.fetch_objects(
            limit=10000,
            filters=FilterMetadata.ById.contains_any(uuids),
            include_vector=True,
        )

        # Create a dictionary mapping UUIDs to embeddings
        uuid_embedding_map = {
            str(result.uuid): result.vector for result in response.objects
        }
        # Retrieve embeddings in the order of the provided UUIDs
        embeddings = [
            uuid_embedding_map[uuid]
            for uuid in uuids
            if uuid in uuid_embedding_map
        ]

        return embeddings

    def retrieve_payloads_by_ids(
        self, uuids: List[str], properties: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Gets the payloads for the specified UUIDs, up to a maximum of
        10000.

        @type uuids: List[str]
        @param uuids: List of UUIDs of the embeddings to get.
        @type properties: List[str]
        @param properties: List of properties to retrieve.
        @rtype payloads: List[Dict[str, Any]]
        @return payloads: List of payloads.
        """
        payloads = []

        response = self.collection.query.fetch_objects(
            limit=10000,
            filters=FilterMetadata.ById.contains_any(uuids),
            return_properties=properties,
        )

        # Create a dictionary mapping UUIDs to payloads
        uuid_payload_map = {
            str(result.uuid): result.properties for result in response.objects
        }
        # Retrieve payloads in the order of the provided UUIDs
        payloads = [
            uuid_payload_map[uuid]
            for uuid in uuids
            if uuid in uuid_payload_map
        ]

        return payloads

    def retrieve_all_ids(self) -> List[str]:
        """Gets all the UUIDs in the Weaviate collection.

        @rtype uuids: List[str]
        @return uuids: List of UUIDs.
        """
        uuids = []
        for item in self.collection.iterator():
            uuids.append(str(item.uuid))

        return uuids

    def retrieve_all_embeddings(self) -> Tuple[List[str], List[List[float]]]:
        """Gets all the embeddings and UUIDs in the Weaviate collection.

        @rtype uuids: List[str]
        @return uuids: List of UUIDs.
        @rtype embeddings: List[List[float]]
        @return embeddings: List of embeddings.
        """
        uuids = []
        embeddings = []

        for item in self.collection.iterator(include_vector=True):
            uuids.append(str(item.uuid))
            embeddings.append(item.vector)

        return uuids, embeddings
