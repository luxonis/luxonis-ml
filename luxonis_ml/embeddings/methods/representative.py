"""Find Representative Images from Embeddings.

This module offers techniques to identify representative images or embeddings within a dataset.
This aids in achieving a condensed yet expressive view of your data.

Methods
=======
    - Greedy Search: Aims to find a diverse subset of images by maximizing the minimum similarity to any image outside the set.

    - K-Medoids: An adaptation of the k-means clustering algorithm, it partitions data into k clusters, each associated with a medoid.

Main Applications
=================
    - Dataset Reduction: Helps in representing large datasets with a minimal subset while retaining the essence.

    - Validation Set Creation: Identifies diverse samples for a robust validation set.

Dependencies
============
    - numpy
    - scikit-learn
    - kmedoids
    - luxonis_ml

Example
=======

    1. Greedy Search:

        # Assuming you have 'embeddings' as a numpy array of shape (num_images, embedding_dim)
        similarity_matrix = calculate_similarity_matrix(embeddings)
        desired_size = int(len(embeddings) * 0.1)
        selected_image_indices = find_representative_greedy(1-similarity_matrix, desired_size)

    2. K-Medoids:

        # to get all embeddings from qdrant:
        ids, embeddings = get_all_embeddings(qdrant_client, collection_name="mnist")
        # Assuming you have 'embeddings' as a numpy array of shape (num_images, embedding_dim)
        similarity_matrix = calculate_similarity_matrix(embeddings)
        desired_size = int(len(embeddings) * 0.1)
        selected_image_indices = find_representative_kmedoids(similarity_matrix, desired_size)
"""

from typing import List

import numpy as np
from kmedoids import KMedoids
from sklearn.metrics.pairwise import cosine_similarity

from luxonis_ml.embeddings.utils.vectordb import VectorDBAPI


def calculate_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    return cosine_similarity(embeddings)


def find_representative_greedy(
    distance_matrix: np.ndarray, desired_size: int = 1000, seed: int = 0
) -> List[int]:
    """Find the most representative images using a greedy algorithm.
    Gready search of maximally unique embeddings.

    @type distance_matrix: np.array
    @param distance_matrix: The distance matrix to use.
    @type desired_size: int
    @param desired_size: The desired size of the representative set.
        Default is 1000.
    @type seed: int
    @param seed: The index of the seed image. Default is 0. Must be in
        the range [0, num_images-1].
    @rtype: List[int]
    @return: The indices of the representative images.
    """
    num_images = distance_matrix.shape[0]
    selected_images = set()
    selected_images.add(
        seed
    )  # If seed==0: start with the first image as a seed.

    while len(selected_images) < desired_size:
        max_distance = -1
        best_image = None

        for i in range(num_images):
            if i not in selected_images:
                # Calculate the minimum similarity to all previously selected images
                min_distance = min(
                    [distance_matrix[i, j] for j in selected_images]
                )
                if min_distance > max_distance:
                    max_distance = min_distance
                    best_image = i

        if best_image is not None:
            selected_images.add(best_image)

    return list(selected_images)


def find_representative_greedy_vectordb(
    vectordb_api: VectorDBAPI, desired_size: int = 1000, seed: int = None
) -> List[int]:
    """Find the most representative embeddings using a greedy algorithm
    with VectorDB.

    @note: Due to many requests, this function is very slow. Use
        vectordb_api.retrieve_all_embeddings() and
        find_representative_greedy() instead.
    @type vectordb_api: VectorDBAPI
    @param vectordb_api: The Vector database client instance to use for
        searches.
    @type desired_size: int
    @param desired_size: The desired size of the representative set.
        Default is 1000.
    @type seed: int
    @param seed: The ID of the seed embedding. Default is None, which
        means a random seed is chosen.
    @rtype: List[int]
    @return: The IDs of the representative embeddings.
    """
    all_ids = vectordb_api.retrieve_all_ids()

    if seed is None:
        seed = np.random.choice(all_ids)
    elif seed > len(all_ids):
        raise ValueError(f"Seed must be in the range [0, {len(all_ids) - 1}].")

    selected_embeddings = set()
    selected_embeddings.add(all_ids[seed])

    while len(selected_embeddings) < desired_size:
        max_similarity = -1
        best_embedding = None

        for embedding_id in all_ids:
            if embedding_id not in selected_embeddings:
                # Get similarities of the current embedding with the already selected embeddings
                _, scores = vectordb_api.get_similarity_scores(
                    embedding_id, list(selected_embeddings)
                )

                # Calculate the minimum similarity to all previously selected embeddings
                min_similarity = max(scores) if scores else -1

                if 1 - min_similarity > max_similarity:
                    max_similarity = min_similarity
                    best_embedding = embedding_id

        if best_embedding is not None:
            selected_embeddings.add(best_embedding)

    return list(selected_embeddings)


def find_representative_kmedoids(
    similarity_matrix: np.ndarray,
    desired_size: int = 1000,
    max_iter: int = 100,
    seed: int = None,
) -> List[int]:
    """Find the most representative images using k-medoids. K-medoids
    clustering of embeddings.

    @type similarity_matrix: np.array
    @param similarity_matrix: The similarity matrix to use.
    @type desired_size: int
    @param desired_size: The desired size of the representative set.
        Default is 1000.
    @type max_iter: int
    @param max_iter: The maximum number of iterations to use. Default is
        100.
    @type seed: int
    @param seed: The random seed to use. Default is None.
    @rtype: list
    @return: The indices of the representative images.
    """
    num_images = similarity_matrix.shape[0]
    k = min(
        desired_size, num_images
    )  # Choose 'k' as the desired size or the number of images, whichever is smaller.

    # Use k-medoids to cluster the images based on the similarity matrix.
    kmedoids_instance = KMedoids(
        n_clusters=k,
        metric="precomputed",
        init="random",
        max_iter=max_iter,
        random_state=seed,
    )
    medoid_indices = kmedoids_instance.fit(similarity_matrix).medoid_indices_

    selected_images = set(medoid_indices)
    return list(selected_images)
