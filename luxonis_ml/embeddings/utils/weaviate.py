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

        Parameters:
        - url (str): URL of the Weaviate instance. Defaults to http://localhost:8080.
        - grpc_url (str): URL of the Weaviate gRPC instance. Defaults to http://localhost:50051.
        - auth_key (str): Authentication key for the Weaviate instance.
        - collection_name (str): Name of the collection. Defaults to "mnist".
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

        Parameters:
        - label (bool): Whether to add a label property to the collection. Defaults to False.
        - vector_size (int): Size of the embeddings. Not used in Weaviate. Defaults to 512.
        - distance_metric (str): Distance metric to use for the vector index. Can be "cosine", "dot" or "euclidean".
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

        Parameters:
        - uuids (List[str]): List of UUIDs for the embeddings.
        - embeddings (List[List[float]]): List of embeddings.
        - labels (List[str]): List of labels for the embeddings. Defaults to None.
        - batch_size (int): Batch size for inserting the embeddings.
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

        Parameters:
        - embedding (List[float]): Embedding to search for.
        - k (int): Number of similar embeddings to find.

        Returns:
        - uuids (List[str]): List of UUIDs of the similar embeddings.
        - scores (List[float]): List of similarity scores.
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

        Parameters:
        - ref_uuid (str): UUID of the reference embedding.
        - uuids (List[str]): List of UUIDs of the embeddings to compare to.
        - sort_distance (bool): Whether to sort the results by distance or keep order of the UUIDs.
                                Defaults to False.

        Returns:
        - ids (List[str]): List of UUIDs of the embeddings.
        - scores (List[float]): List of similarity scores.
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
        Calculates the similarity matrix between the specified embeddings.
        NOTE: This is a very inefficient implementation. 
              For large numbers of embeddings, calculate the similarity matrix by hand (sklearn.metrics.pairwise.cosine_similarity).

        Parameters:
        - uuids (List[str]): List of UUIDs of the embeddings to compare.

        Returns:
        - scores (List[List[float]]): List of similarity scores.
        """
        uuids = self.retrieve_all_ids()

        sim_matrix = []
        for uuid in uuids:
            sim_matrix.append(self.get_similarity_scores(uuid, uuids))

        return sim_matrix
    
    def retrieve_embeddings_by_ids(self, uuids):
        """
        Gets the embeddings for the specified UUIDs.

        Parameters:
        - uuids (List[str]): List of UUIDs of the embeddings to get.

        Returns:
        - embeddings (List[List[float]]): List of embeddings.
        """
        embeddings = []

        response = self.collection.query.fetch_objects(
            limit=10000,
            filters=FilterMetadata.ById.contains_any(uuids),
            include_vector=True
        )

        # Create a dictionary mapping UUIDs to embeddings
        uuid_embedding_map = {result.uuid: result.vector for result in response.objects}
        # Retrieve embeddings in the order of the provided UUIDs
        embeddings = [uuid_embedding_map[uuid] for uuid in uuids if uuid in uuid_embedding_map]

        return embeddings

    def retrieve_labels_by_ids(self, uuids):
        """
        Gets the labels for the specified UUIDs.

        Parameters:
        - uuids (List[str]): List of UUIDs of the embeddings to get.

        Returns:
        - labels (List[str]): List of labels.
        """
        labels = []

        response = self.collection.query.fetch_objects(
            limit=10000,
            filters=FilterMetadata.ById.contains_any(uuids),
            return_properties=["label"]
        )
        
        # Create a dictionary mapping UUIDs to labels
        uuid_label_map = {result.uuid: result.properties["label"] for result in response.objects}
        # Retrieve labels in the order of the provided UUIDs
        labels = [uuid_label_map[uuid] for uuid in uuids if uuid in uuid_label_map]

        return labels
    
    def retrieve_all_ids(self):
        """
        Gets all the UUIDs in the Weaviate collection, up to a maximum of 10000.

        Returns:
        - uuids (List[str]): List of UUIDs.
        """
        uuids = []
        for item in self.collection.iterator():
            uuids.append(item.uuid)
        
        return uuids
        
    def retrieve_all_embeddings(self):
        """
        Gets all the embeddings and UUIDs in the Weaviate collection, up to a maximum of 10000.

        Returns:
        - embeddings (List[List[float]]): List of embeddings.
        - uuids (List[str]): List of UUIDs.
        """

        uuids = []; embeddings = []

        for item in self.collection.iterator(
            include_vector=True
        ):
            uuids.append(item.uuid)
            embeddings.append(item.vector)
        
        return uuids, embeddings
            