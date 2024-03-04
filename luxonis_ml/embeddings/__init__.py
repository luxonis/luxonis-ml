from ..guard_extras import guard_missing_extra

with guard_missing_extra("embedd"):
    from .methods import (
        calculate_similarity_matrix,
        find_mismatches_centroids,
        find_mismatches_knn,
        find_representative_greedy,
        find_representative_kmedoids,
        find_similar,
        isolation_forest_OOD,
        leverage_OOD,
    )
    from .utils import (
        QdrantAPI,
        QdrantManager,
        VectorDBAPI,
        WeaviateAPI,
        extend_output_onnx,
        extract_embeddings,
        generate_embeddings,
        load_model_onnx,
        save_model_onnx,
    )

__all__ = [
    "find_similar",
    "find_mismatches_centroids",
    "find_mismatches_knn",
    "isolation_forest_OOD",
    "leverage_OOD",
    "calculate_similarity_matrix",
    "find_representative_greedy",
    "find_representative_kmedoids",
    "load_model_onnx",
    "save_model_onnx",
    "extend_output_onnx",
    "QdrantManager",
    "QdrantAPI",
    "WeaviateAPI",
    "VectorDBAPI",
    "extract_embeddings",
    "generate_embeddings",
]
