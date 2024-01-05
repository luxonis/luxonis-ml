from ..guard_extras import guard_missing_extra

with guard_missing_extra("embedd"):
    from .methods import (
        calculate_similarity_matrix,
        find_mismatches_centroids,
        find_mismatches_knn,
        find_representative_greedy,
        find_representative_kmedoids,
        find_similar_qdrant,
        isolation_forest_OOD,
        leverage_OOD,
    )
    from .utils import (
        QdrantAPI,
        QdrantManager,
        export_model_onnx,
        extend_output_onnx,
        extend_output_onnx_overwrite,
        extract_embeddings,
        extract_embeddings_onnx,
        generate_embeddings,
        load_embeddings,
        load_model,
        load_model_onnx,
        load_model_resnet50_minuslastlayer,
        save_embeddings,
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
