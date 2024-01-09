import weaviate
import weaviate.classes as wvc
from weaviate.collections.classes.filters import FilterMetadata

class WeaviateAPI:
    def __init__(self, client: weaviate.client.WeaviateClient, collection_name: str):
        self.client = client
        self.collection_name = collection_name

        if not client.collections.exists(collection_name):
            print(f"Collection {collection_name} does not exist. Creating...")
            self.create_collection(collection_name)

        self.collection = client.collections.get(collection_name)
    
    def create_collection(self, name: str):
        """
        Creates a Weaviate collection.

        Parameters:
        - name (str): Name of the collection.
        """
        self.client.collections.create(
            name=name,
            vector_index_config=wvc.Configure.VectorIndex.hnsw(
                distance_metric=wvc.VectorDistance.COSINE
            ),
        )

    def insert_embeddings(self, uuids, embeddings, batch_size=100):
        """
        Inserts embeddings into the Weaviate collection.

        Parameters:
        - uuids (List[str]): List of UUIDs for the embeddings.
        - embeddings (List[List[float]]): List of embeddings.
        - batch_size (int): Batch size for inserting the embeddings.
        """
        data = []
        for i, embedding in enumerate(embeddings):
            data.append(
                wvc.DataObject(
                    properties={}, # no properties
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
            vector=embedding,
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
            