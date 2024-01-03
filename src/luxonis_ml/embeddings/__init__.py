from ..guard_extras import guard_missing_extra

with guard_missing_extra("embedd"):
    from .methods import (
        find_similar_qdrant,
        find_mismatches_centroids,
        find_mismatches_knn,
        isolation_forest_OOD,
        leverage_OOD,
        calculate_similarity_matrix,
        find_representative_greedy,
        find_representative_kmedoids,
    )
    from .utils import (
        load_model_resnet50_minuslastlayer,
        load_model,
        export_model_onnx,
        load_model_onnx,
        extend_output_onnx,
        extend_output_onnx_overwrite,
        QdrantManager,
        QdrantAPI,
        extract_embeddings,
        extract_embeddings_onnx,
        save_embeddings,
        load_embeddings,
        generate_embeddings,
    )

__all__ = [
    "find_similar_qdrant",
    "find_mismatches_centroids",
    "find_mismatches_knn",
    "isolation_forest_OOD",
    "leverage_OOD",
    "calculate_similarity_matrix",
    "find_representative_greedy",
    "find_representative_kmedoids",
    "load_model_resnet50_minuslastlayer",
    "load_model",
    "export_model_onnx",
    "load_model_onnx",
    "extend_output_onnx",
    "extend_output_onnx_overwrite",
    "QdrantManager",
    "QdrantAPI",
    "extract_embeddings",
    "extract_embeddings_onnx",
    "save_embeddings",
    "load_embeddings",
    "generate_embeddings",
]
