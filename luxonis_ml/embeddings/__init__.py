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
    "QdrantAPI",
    "QdrantManager",
    "VectorDBAPI",
    "WeaviateAPI",
    "calculate_similarity_matrix",
    "extend_output_onnx",
    "extract_embeddings",
    "find_mismatches_centroids",
    "find_mismatches_knn",
    "find_representative_greedy",
    "find_representative_kmedoids",
    "find_similar",
    "generate_embeddings",
    "isolation_forest_OOD",
    "leverage_OOD",
    "load_model_onnx",
    "save_model_onnx",
]
