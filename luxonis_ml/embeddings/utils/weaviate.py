import weaviate
import weaviate.classes as wvc
from weaviate.collections.classes.filters import FilterMetadata

class WeaviateAPI:
    def __init__(self, url="http://localhost:8080", auth_api_key=None):
        """
        Initializes the Weaviate API.

        Parameters:
        - url (str): URL of the Weaviate instance. Defaults to http://localhost:8080.
        - grpc_url (str): URL of the Weaviate gRPC instance. Defaults to http://localhost:50052.
        - auth_key (str): Authentication key for the Weaviate instance.
        """
        if auth_api_key is not None:
            auth_api_key = weaviate.AuthApiKey(auth_api_key)
        
        # # old v3 client
        # self.client = weaviate.Client(
        #     url=url,
        #     auth_client_secret=auth_api_key
        # )

        # # works only with gRPC enabled
        # self.client = weaviate.WeaviateClient(
        #     connection_params=weaviate.ConnectionParams.from_params(
        #         http_host = url.split("://")[1].split(":")[0],
        #         http_port = url.split("://")[1].split(":")[1],
        #         http_secure = url.split("://")[0] == "https",
        #         grpc_host = grpc_url.split("://")[1].split(":")[0],
        #         grpc_port = grpc_url.split("://")[1].split(":")[1],
        #         grpc_secure = grpc_url.split("://")[0] == "https"
        #     ),
        #     auth_client_secret=auth_key
        # )
        
        if "localhost" in url:
            self.client = weaviate.connect_to_local()
        else:
            # Not implemented yet
            raise NotImplementedError("Weaviate API only supports local connections at the moment.")

    def create_collection(self, collection_name: str, label=False):
        """
        Creates a Weaviate collection.

        Parameters:
        - collection_name (str): Name of the collection.
        - label (bool): Whether to add a label property to the collection. Defaults to False.
        """
        self.collection_name = collection_name
        if not self.client.collections.exists(collection_name):
            print(f"Collection {collection_name} does not exist. Creating...")
            
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
                name=collection_name,
                properties=properties,
                vector_index_config=wvc.Configure.VectorIndex.hnsw(
                    distance_metric=wvc.VectorDistance.COSINE
                ),
            )
        else:
            self.collection = self.client.collections.get(collection_name)
        
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

    def search_embeddings(self, uuids):
        """
        Searches for embeddings in the Weaviate collection.

        Parameters:
        - uuids (List[str]): List of UUIDs for the embeddings.

        Returns:
        - embeddings (List[List[float]]): List of embeddings found.
        """
        embeddings = []
        for uuid in uuids:
            embedding = self.collection.query.fetch_object_by_id(uuid, include_vector=True).vector
            embeddings.append(embedding)

        return embeddings

    def find_similar_embeddings(self, embedding, k=10):
        """
        Finds similar embeddings to the specified embedding.

        Parameters:
        - embedding (List[float]): Embedding to search for.
        - k (int): Number of similar embeddings to find.

        Returns:
        - uuids (List[str]): List of UUIDs of the similar embeddings.
        """
        response = self.collection.query.near_vector(
            near_vector=embedding,
            limit=k
        )

        uuids = []
        for result in response.objects:
            uuids.append(result.uuid)

        return uuids

    def find_similar_embeddings_by_id(self, uuid, k=10):
        """
        Finds similar embeddings to the specified embedding.

        Parameters:
        - uuid (str): UUID of the embedding to search for.
        - k (int): Number of similar embeddings to
                     find.  Defaults to 10.
                    
        Returns:
        - uuids (List[str]): List of UUIDs of the similar embeddings.
        """
        response = self.collection.query.near_object(
            near_object=uuid,
            limit=k
        )

        uuids = []
        for result in response.objects:
            uuids.append(result.uuid)

        return uuids
    
    def get_similarity_score(self, ref_uuid, uuids, sort_distance=False):
        """
        Calculates the similarity score between the reference embedding and the specified embeddings.

        Parameters:
        - ref_uuid (str): UUID of the reference embedding.
        - uuids (List[str]): List of UUIDs of the embeddings to compare to.
        - sort_distance (bool): Whether to sort the results by distance or keep order of the UUIDs.
                                Defaults to False.

        Returns:
        - scores (List[float]): List of similarity scores.
        """
        response = self.collection.query.near_object(
            near_object=ref_uuid,
            limit=len(uuids),
            filters=FilterMetadata.ById.contains_any(uuids),
            return_metadata=wvc.query.MetadataQuery(distance=True)
        )
        # near object gives order by distance result

        if sort_distance:
            scores = []
            for result in response.objects:
                scores.append(result.metadata.distance)
        else:
            scores = [0] * len(uuids)
            for result in response.objects:
                scores[uuids.index(result.uuid)] = result.metadata.distance

        return scores
    
    def get_similarity_matrix(self, uuids):
        """
        Calculates the similarity matrix between the specified embeddings.
        NOTE: This is a very inefficient implementation. 
              For large numbers of embeddings, calculate the similarity matrix by hand (sklearn.metrics.pairwise.cosine_similarity).

        Parameters:
        - uuids (List[str]): List of UUIDs of the embeddings to compare.

        Returns:
        - scores (List[List[float]]): List of similarity scores.
        """
        sim_matrix = []
        for uuid in uuids:
            sim_matrix.append(self.get_similarity_score(uuid, uuids))

        return sim_matrix
    
    def get_embeddings(self, uuids):
        """
        Gets the embeddings for the specified UUIDs.

        Parameters:
        - uuids (List[str]): List of UUIDs of the embeddings to get.

        Returns:
        - embeddings (List[List[float]]): List of embeddings.
        """
        embeddings = []
        # for uuid in uuids:
        #     embedding = self.collection.query.fetch_object_by_id(uuid, include_vector=True).vector
        #     embeddings.append(embedding)

        response = self.collection.query.fetch_objects(
            limit=10000,
            filters=FilterMetadata.ById.contains_any(uuids),
            include_vector=True
        )

        for result in response.objects:
            embeddings.append(result.vector)

        return embeddings

    def get_labels(self, uuids):
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

        for result in response.objects:
            labels.append(result.properties["label"])

        return labels
    
    def get_all_ids(self):
        """
        Gets all the UUIDs in the Weaviate collection, up to a maximum of 10000.

        Returns:
        - uuids (List[str]): List of UUIDs.
        """
        # uuids = []
        # limit = 10000
        # while(True):
        #     # Fetch the objects without their vectors
        #     response_i = self.collection.query.fetch_objects(
        #         limit=limit
        #     )

        #     # Append the UUIDs to the list
        #     uuids.extend([r.uuid for r in response_i.objects])

        #     # Break if the number of objects returned is less than the limit (i.e. we've reached the end)
        #     if len(response_i.objects) < limit:
        #         break
        
        uuids = []
        for item in self.collection.iterator():
            uuids.append(item.uuid)
        
        return uuids
        
    def get_all_embeddings(self):
        """
        Gets all the embeddings in the Weaviate collection, up to a maximum of 10000.

        Returns:
        - embeddings (List[List[float]]): List of embeddings.
        """
        # embeddings = []
        # limit = 10000
        # while(True):
        #     # Fetch the objects with their vectors
        #     response_i = self.collection.query.fetch_objects(
        #         limit=limit,
        #         include_vector=True
        #     )

        #     # Append the vectors to the list
        #     emb_i = [r.vector for r in response_i.objects]
        #     embeddings.extend(emb_i)

        #     # Break if the number of objects returned is less than the limit (i.e. we've reached the end)
        #     if len(response_i.objects) < limit:
        #         break

        embeddings = []
        for item in self.collection.iterator(
            include_vector=True
        ):
            embeddings.append(item.vector)
        
        return embeddings
    
    def get_all_embeddings_and_ids(self):
        """
        Gets all the embeddings and UUIDs in the Weaviate collection, up to a maximum of 10000.

        Returns:
        - embeddings (List[List[float]]): List of embeddings.
        - uuids (List[str]): List of UUIDs.
        """

        embeddings = []
        uuids = []
        for item in self.collection.iterator(
            include_vector=True
        ):
            embeddings.append(item.vector)
            uuids.append(item.uuid)
        
        return embeddings, uuids
            