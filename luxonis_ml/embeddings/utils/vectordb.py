from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class VectorDBAPI(ABC):
    """Abstract class for Vector Database APIs.

    This class defines a common interface for vector database operations
    for different implementations like Qdrant and Weaviate.
    """

    @abstractmethod
    def create_collection(
        self, collection_name: str, properties: List[str]
    ) -> None:
        """Create a collection in the vector database."""

    @abstractmethod
    def delete_collection(self) -> None:
        """Delete the collection from the vector database."""

    @abstractmethod
    def insert_embeddings(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        payloads: List[Dict[str, Any]],
        batch_size: int = 100,
    ) -> None:
        """Insert embeddings into the collection."""

    @abstractmethod
    def search_similar_embeddings(
        self, embedding: List[float], top_k: int = 10
    ) -> Tuple[List[str], List[float]]:
        """Search for similar embeddings in the collection."""

    @abstractmethod
    def get_similarity_scores(
        self,
        reference_id: str,
        other_ids: List[str],
        sort_distances: bool = True,
    ) -> Tuple[List[str], List[float]]:
        """Get similarity scores between a reference embedding and other
        embeddings."""

    @abstractmethod
    def compute_similarity_matrix(self) -> List[List[float]]:
        """Compute a similarity matrix for all the embeddings in the
        collection."""

    @abstractmethod
    def retrieve_embeddings_by_ids(self, ids: List[str]) -> List[List[float]]:
        """Retrieve embeddings associated with a list of IDs."""

    @abstractmethod
    def retrieve_all_embeddings(self) -> Tuple[List[str], List[List[float]]]:
        """Retrieve all embeddings from the collection."""

    @abstractmethod
    def retrieve_all_ids(self) -> List[str]:
        """Retrieve all IDs from the collection."""

    @abstractmethod
    def retrieve_payloads_by_ids(
        self, ids: List[str], properties: Optional[List[str]]
    ) -> List[Dict[str, Any]]:
        """Retrieve payloads associated with a list of IDs."""
