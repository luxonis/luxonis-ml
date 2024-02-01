import weaviate
import weaviate.classes as wvc
from weaviate.collections.classes.filters import FilterMetadata

from luxonis_ml.embeddings.utils.vectordb import VectorDBAPI

convert_distance_metric = {
    "cosine": wvc.VectorDistance.COSINE,
    "dot": wvc.VectorDistance.DOT,
    "euclidean": wvc.VectorDistance.L2_SQUARED
}

class WeaviateAPI(VectorDBAPI):
    def __init__(self, url="http://localhost:8080", grpc_url="http://localhost:50051", auth_api_key=None, collection_name="mnist"):
        """
        Initializes the Weaviate API.

        @type url: str
        @param url: URL of the Weaviate instance, defaults to http://localhost:8080.
        @type grpc_url: str
        @param grpc_url: URL of the gRPC Weaviate instance, defaults to http://localhost:50051.
        @type auth_api_key: str
        @param auth_api_key: API key for authentication.
        @type collection_name: str
        @param collection_name: Name of the collection to use.
        """
        if auth_api_key is not None:
            auth_api_key = weaviate.AuthApiKey(auth_api_key)
        
        if "localhost" in url:
            self.client = weaviate.connect_to_local()
        else:
            # make sure the gRPC port is correct
            self.client = weaviate.connect_to_custom(
                http_host = url.split("://")[1].split(":")[0],
                http_port = url.split("://")[1].split(":")[1],
                http_secure = url.split("://")[0] == "https",
                grpc_host = grpc_url.split("://")[1].split(":")[0],
                grpc_port = grpc_url.split("://")[1].split(":")[1],
                grpc_secure = grpc_url.split("://")[0] == "https",
                auth_credentials=auth_api_key
            )
        
        self.collection_name = collection_name
        
    def create_collection(self, label=False, vector_size: int = 512, distance_metric: str = "cosine"):
        """
        Creates a Weaviate collection.

        @type label: bool
        @param label: Whether to add a label property to the collection. Defaults to False.
        @type vector_size: int
        @param vector_size: Size of the embeddings. Not used in Weaviate. Defaults to 512.
        @type distance_metric: str
        @param distance_metric: Distance metric to use for the vector index. Can be "cosine", "dot" or "euclidean".
        """
        if not self.client.collections.exists(self.collection_name):
            print(f"Collection {self.collection_name} does not exist. Creating...")
            
            properties = None

            if label:
                properties = [
                    wvc.Property(
                        name="label",
                        data_type=wvc.DataType.TEXT,
                        skip_vectorization=True
                    )
                ]

            self.collection = self.client.collections.create(
                name=self.collection_name,
                properties=properties,
                vector_index_config=wvc.Configure.VectorIndex.hnsw(
                    distance_metric=convert_distance_metric[distance_metric]
                ),
            )
        else:
            self.collection = self.client.collections.get(self.collection_name)
        
    def delete_collection(self):
        """
        Deletes the Weaviate collection.
        """
        self.client.collections.delete(self.collection_name)

    def insert_embeddings(self, uuids, embeddings, labels=None, batch_size=100):
        """
        Inserts embeddings into the Weaviate collection.

        @type uuids: List[str]
        @param uuids: List of UUIDs for the embeddings.
        @type embeddings: List[List[float]]
        @param embeddings: List of embeddings.
        @type labels: List[str]
        @param labels: List of labels for the embeddings. Defaults to None.
        @type batch_size: int
        @param batch_size: Batch size for inserting the embeddings.
        """
        data = []
        if labels is not None:
            properties = [{"label": label} for label in labels]
        else:
            properties = [{}] * len(uuids)

        for i, embedding in enumerate(embeddings):
            data.append(
                wvc.DataObject(
                    properties=properties[i],
                    uuid=uuids[i],
                    vector=embedding
                )
            )

            if len(data) == batch_size:
                self.collection.data.insert_many(data)
                data = []

        if len(data) > 0:
            self.collection.data.insert_many(data)

    def search_similar_embeddings(self, embedding, top_k=10):
        """
        Finds similar embeddings to the specified embedding.

        @type embedding: List[float]
        @param embedding: Embedding to search for.
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
            return_metadata=wvc.query.MetadataQuery(distance=True)
        )

        uuids = []
        scores = []
        for result in response.objects:
            uuids.append(result.uuid)
            scores.append(result.metadata.distance)

        return uuids, scores
    
    def get_similarity_scores(self, reference_id, other_ids, sort_distances=False):
        """
        Calculates the similarity score between the reference embedding and the specified embeddings.

        @type reference_id: str
        @param reference_id: UUID of the reference embedding.
        @type other_ids: List[str]
        @param other_ids: List of UUIDs of the embeddings to compare to.
        @type sort_distances: bool
        @param sort_distances: Whether to sort the results by distance or keep order of the UUIDs. Defaults to False.

        @rtype ids: List[str]
        @return ids: List of UUIDs of the embeddings.
        @rtype scores: List[float]
        @return scores: List of similarity scores.
        """
        response = self.collection.query.near_object(
            near_object=reference_id,
            limit=len(other_ids),
            filters=FilterMetadata.ById.contains_any(other_ids),
            return_metadata=wvc.query.MetadataQuery(distance=True)
        )
        # near object gives order by distance result

        if sort_distances:
            ids = []
            scores = []
            for result in response.objects:
                ids.append(result.uuid)
                scores.append(result.metadata.distance)
        else:
            ids = other_ids
            scores = [0] * len(other_ids)
            for result in response.objects:
                scores[other_ids.index(result.uuid)] = result.metadata.distance

        return ids, scores
    
    def compute_similarity_matrix(self):
        """
        Calculates the similarity matrix for all the embeddings in the collection.
        @note: This is a very inefficient implementation. 
              For large numbers of embeddings, calculate the similarity matrix 
              by hand (sklearn.metrics.pairwise.cosine_similarity).
        
        @type uuids: List[str]
        @param uuids: List of UUIDs of the embeddings to compare.

        @rtype scores: List[List[float]]
        @return scores: List of similarity scores.
        """
        uuids = self.retrieve_all_ids()

        sim_matrix = []
        for uuid in uuids:
            sim_matrix.append(self.get_similarity_scores(uuid, uuids))

        return sim_matrix
    
    def retrieve_embeddings_by_ids(self, uuids):
        """
        Gets the embeddings for the specified UUIDs.

        @type uuids: List[str]
        @param uuids: List of UUIDs of the embeddings to get.

        @rtype embeddings: List[List[float]]
        @return embeddings: List of embeddings.
        """
        embeddings = []

        response = self.collection.query.fetch_objects(
            limit=10000,
            filters=FilterMetadata.ById.contains_any(uuids),
            include_vector=True
        )

        # Create a dictionary mapping UUIDs to embeddings
        uuid_embedding_map = {str(result.uuid): result.vector for result in response.objects}
        # Retrieve embeddings in the order of the provided UUIDs
        embeddings = [uuid_embedding_map[uuid] for uuid in uuids if uuid in uuid_embedding_map]

        return embeddings

    def retrieve_labels_by_ids(self, uuids):
        """
        Gets the labels for the specified UUIDs.

        @type uuids: List[str]
        @param uuids: List of UUIDs of the embeddings to get.

        @rtype labels: List[str]
        @return labels: List of labels.
        """
        labels = []

        response = self.collection.query.fetch_objects(
            limit=10000,
            filters=FilterMetadata.ById.contains_any(uuids),
            return_properties=["label"]
        )
        
        # Create a dictionary mapping UUIDs to labels
        uuid_label_map = {str(result.uuid): result.properties["label"] for result in response.objects}
        # Retrieve labels in the order of the provided UUIDs
        labels = [uuid_label_map[uuid] for uuid in uuids if uuid in uuid_label_map]

        return labels
    
    def retrieve_all_ids(self):
        """
        Gets all the UUIDs in the Weaviate collection, up to a maximum of 10000.

        @rtype uuids: List[str]
        @return uuids: List of UUIDs.
        """
        uuids = []
        for item in self.collection.iterator():
            uuids.append(str(item.uuid))
        
        return uuids
        
    def retrieve_all_embeddings(self):
        """
        Gets all the embeddings and UUIDs in the Weaviate collection, up to a maximum of 10000.

        @rtype uuids: List[str]
        @return uuids: List of UUIDs.
        @rtype embeddings: List[List[float]]
        @return embeddings: List of embeddings.
        """
        uuids = []
        embeddings = []

        for item in self.collection.iterator(
            include_vector=True
        ):
            uuids.append(str(item.uuid))
            embeddings.append(item.vector)
        
        return uuids, embeddings