from abc import ABC, abstractmethod

class VectorDBAPI(ABC):
    """
    Abstract class for Vector Database APIs.

    This class defines a common interface for vector database operations for different implementations like Qdrant and Weaviate.
    """

    @abstractmethod
    def create_collection(self, label: bool, vector_size: int = 512, distance_metric: str = "cosine"):
        """
        Create a collection in the vector database.
        Distance metric can be "cosine", "dot" or "euclidean".
        """
        pass

    @abstractmethod
    def delete_collection(self):
        """
        Delete the collection from the vector database.
        """
        pass

    @abstractmethod
    def insert_embeddings(self, ids, embeddings, labels=None, batch_size: int = 100):
        """
        Insert embeddings into the collection.
        """
        pass

    @abstractmethod
    def search_similar_embeddings(self, embedding, top_k: int = 10):
        """
        Search for similar embeddings in the collection.
        """
        pass

    @abstractmethod
    def get_similarity_scores(self, reference_id, other_ids, sort_distances: bool = True):
        """
        Get similarity scores between a reference embedding and other embeddings.
        """
        pass

    @abstractmethod
    def compute_similarity_matrix(self):
        """
        Compute a similarity matrix for all the embeddings in the collection.
        """
        pass

    @abstractmethod
    def retrieve_embeddings_by_ids(self, ids):
        """
        Retrieve embeddings associated with a list of IDs.
        """
        pass

    @abstractmethod
    def retrieve_all_embeddings(self):
        """
        Retrieve all embeddings from the collection.
        """
        pass

    @abstractmethod
    def retrieve_all_ids(self):
        """
        Retrieve all IDs from the collection.
        """
        pass

    @abstractmethod
    def retrieve_labels_by_ids(self, ids):
        """
        Retrieve labels associated with a list of IDs.
        """
        pass

    # @abstractmethod
    # def filter_properties(self, property, value):
    #     """
    #     Filter properties from the collection.
    #     """
    #     pass

    # @abstractmethod
    # def update_property(self, id, property, value):
    #     """
    #     Update a property for an embedding.
    #     """
    #     pass