"""
Find representative images

Available methods:
    - greedy search
    - k-medoids

Useful for:
    - dataset reduction
    - validation set creation
"""

from sklearn.metrics.pairwise import cosine_similarity
from kmedoids import KMedoids

from luxonis_ml.embeddings.qdrant_utils import *


def calculate_similarity_matrix(embeddings):
    return cosine_similarity(embeddings)

def find_representative_greedy(distance_matrix, desired_size=1000, seed=0):
    """
    Find the most representative images using a greedy algorithm.
    Gready search of maximally unique embeddings

    Parameters
    ----------
    distance_matrix : np.array
        The distance matrix to use.
    desired_size : int
        The desired size of the representative set. Default is 1000.
    seed : int
        The index of the seed image. Default is 0. 
        Must be in the range [0, num_images-1].

    Returns
    -------
    np.array
        The indices of the representative images.
    """
    num_images = distance_matrix.shape[0]
    selected_images = set()
    selected_images.add(seed)  # If seed==0: start with the first image as a seed.

    while len(selected_images) < desired_size:
        max_distance = -1
        best_image = None

        for i in range(num_images):
            if i not in selected_images:
                # Calculate the minimum similarity to all previously selected images
                min_distance = min([distance_matrix[i, j] for j in selected_images])
                if min_distance > max_distance:
                    max_distance = min_distance
                    best_image = i

        if best_image is not None:
            selected_images.add(best_image)

    return list(selected_images)

# # Example usage:
# # 'embeddings' is a numpy array of shape (num_images, embedding_dim)
# similarity_matrix = calculate_similarity_matrix(embeddings)
# desired_size = int(len(embeddings)*0.1)
# selected_image_indices = find_representative_greedy(1-similarity_matrix, desired_size)

def find_representative_greedy_qdrant(qdrant_client, desired_size=1000, qdrant_collection_name="mnist", seed=0):
    """
    Find the most representative embeddings using a greedy algorithm with Qdrant.
    NOTE: Due to many Qdrant requests, this function is very slow. Use get_all_embeddings() and find_representative_greedy() instead.

    Parameters
    ----------
    qdrant_client : QdrantClient
        The Qdrant client instance to use for searches.
    desired_size : int
        The desired size of the representative set. Default is 1000.
    qdrant_collection_name : str
        The name of the Qdrant collection. Default is "mnist".
    seed : int
        The ID of the seed embedding. Default is 0.

    Returns
    -------
    list
        The IDs of the representative embeddings.
    """
    all_ids = get_all_ids(qdrant_client, qdrant_collection_name)
    selected_embeddings = set()
    selected_embeddings.add(all_ids[seed])

    while len(selected_embeddings) < desired_size:
        max_similarity = -1
        best_embedding = None

        for embedding_id in all_ids:
            if embedding_id not in selected_embeddings:
                # Get similarities of the current embedding with the already selected embeddings
                _, scores = get_similarities(qdrant_client, embedding_id, list(selected_embeddings), qdrant_collection_name)
                
                # Calculate the minimum similarity to all previously selected embeddings
                min_similarity = max(scores) if scores else -1
                
                if 1-min_similarity > max_similarity:
                    max_similarity = min_similarity
                    best_embedding = embedding_id

        if best_embedding is not None:
            selected_embeddings.add(best_embedding)

    return list(selected_embeddings)


def find_representative_kmedoids(similarity_matrix, desired_size=1000, max_iter=100, seed=42):
    """
    Find the most representative images using k-medoids.
    K-medoids clustering of embeddings

    Parameters
    ----------
    similarity_matrix : np.array
        The similarity matrix to use.
    desired_size : int
        The desired size of the representative set. Default is 1000.
    max_iter : int
        The maximum number of iterations to use. Default is 100.
    seed : int
        The random seed to use. Default is 42.

    Returns
    -------
    np.array
        The indices of the representative images.    
    """
    num_images = similarity_matrix.shape[0]
    k = min(desired_size, num_images)  # Choose 'k' as the desired size or the number of images, whichever is smaller.

    # Use k-medoids to cluster the images based on the similarity matrix.
    kmedoids_instance = KMedoids(n_clusters=k, metric='precomputed', init='random', max_iter=max_iter, random_state=seed)
    medoid_indices = kmedoids_instance.fit(similarity_matrix).medoid_indices_

    selected_images = set(medoid_indices)
    return list(selected_images)


# # Example usage:
# # to get all embeddings from qdrant:
# ids, embeddings = get_all_embeddings(qdrant_client, collection_name="mnist")
# # Assuming you have 'embeddings' as a numpy array of shape (num_images, embedding_dim)
# similarity_matrix = calculate_similarity_matrix(embeddings)
# desired_size = int(len(embeddings)*0.1)
# selected_image_indices = find_representative_kmedoids(similarity_matrix, desired_size)
