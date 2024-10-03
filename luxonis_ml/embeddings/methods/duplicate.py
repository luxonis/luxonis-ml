"""Near-duplicate Search with Qdrant and Weaviate.

Overview:
This module provides utilities to detect and remove near-duplicate data points within a given set of embeddings.
It leverages vector databases (Qdrant or Weaviate) for efficient search and retrieval,
and employs Kernel Density Estimation (KDE) for optimal split based on embeddings' cosine similarity.
This approach is particularly well-suited for handling high-dimensional embeddings.

Key Features:
    - Vector Database Integration: Supports both Qdrant and Weaviate for flexible deployment options.
    - KDE-Based Near-Duplicate Detection: Uses KDE to identify clusters of near-duplicates, ensuring accuracy in high-dimensional spaces.
    - Visualization: Allows plotting KDE results using matplotlib for intuitive understanding.
    - Dynamic KDE Peak Selection: Automatically determines the best candidates for removal based on KDE peaks, minimizing manual thresholding.

Dependencies:
    - KDEpy
    - Qdrant (optional, for Qdrant-specific features)
    - Weaviate (optional, for Weaviate-specific features)

Functions:
    - search_vectordb(vectordb_api, query_vector, property_name, top_k): Searches for similar embeddings within the specified vector database.
    - _plot_kde(xs, s, density, maxima, minima): Plots a KDE distribution.
    - kde_peaks(data, bandwidth="scott", plot=False): Identifies peaks in a KDE distribution.
    - find_similar(reference_embeddings, vectordb_api, k=100, n=1000, method="first", k_method=None, kde_bw="scott", plot=False): Finds the most similar embeddings to the given reference embeddings.

Examples (using Qdrant):
    1. Initialize a Qdrant client and retrieve embeddings:
        from luxonis_ml.embeddings.utils.qdrant import QdrantAPI
        qdrant_api = QdrantAPI(host="localhost", port=6333)
        qdrant_api.create_collection("images", ["image_path", "embedding"])
        id_X, X = qdrant_api.get_all_embeddings()

    2. Find similar embeddings using various methods:
        # By instance ID:
        ix, paths = find_similar_qdrant(id_X[i], qdrant_api, "image_path", 5, 100, "first")

        # Using KDE Peaks method:
        ix, paths = find_similar_qdrant(X[i], qdrant_api, "image_path", 5, 100, "first", "kde_peaks", "silverman", plot=False)

        # Based on average of multiple embeddings:
        dark_ix = np.array([10, 123, 333, 405])
        emb_dark = X[dark_ix]
        remove_dark_ix, paths = find_similar_qdrant(emb_dark, qdrant_api, "image_path", 25, 5000, "average", "kde_basic", "scott", plot=True)

Additional Notes:
    - For Weaviate-specific examples, refer to the provided code examples.
    - The search_vectordb function can be used with either Qdrant or Weaviate, depending on the provided vectordb_api object.
    - Adjust parameters like k, n, and kde_bw based on your dataset and requirements.
"""

from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from KDEpy import FFTKDE
from scipy.signal import argrelextrema

from luxonis_ml.embeddings.utils.vectordb import VectorDBAPI


def _plot_kde(
    xs: np.ndarray,
    s: np.ndarray,
    density: np.ndarray,
    maxima: np.ndarray,
    minima: np.ndarray,
) -> None:
    """Plot a KDE distribution.

    @type xs: np.ndarray
    @param xs: The x-axis values.
    @type s: np.ndarray
    @param s: The y-axis values.
    @type density: np.ndarray
    @param density: The density values.
    @type maxima: np.ndarray
    @param maxima: The indices of the local maxima.
    @type minima: np.ndarray
    @param minima: The indices of the local minima.
    """
    plt.plot(xs, density, label="KDE")
    plt.plot(xs[maxima], s[maxima], "ro", label="local maxima")
    plt.plot(xs[minima], s[minima], "bo", label="local minima")
    plt.plot(xs[np.argmax(s)], np.max(s), "go", label="global maxima")
    plt.legend()
    plt.show()


def kde_peaks(
    data: np.ndarray,
    bandwidth: Union[str, float] = "scott",
    plot: bool = False,
) -> Tuple[np.ndarray, np.ndarray, int, float]:
    """Find peaks in a KDE distribution using scipy's argrelextrema
    function.

    @type data: np.ndarray
    @param data: The data to fit the KDE.
    @type bandwidth: Union[str, float]
    @param bandwidth: The bandwidth to use for the KDE. Default is
        'scott'.
    @type plot: bool
    @param plot: Whether to plot the KDE.
    @rtype: Tuple[np.ndarray, np.ndarray, int, float]
    @return: The indices of the KDE maxima, the indices of the KDE
        minima, the index of the global maxima, and the standard
        deviation of the data.
    """
    # fit density
    kde = FFTKDE(kernel="gaussian", bw=bandwidth)
    xs = np.linspace(np.min(data) - 0.01, np.max(data) + 0.01, 1000)
    density = kde.fit(data).evaluate(xs)

    # find local maxima
    s = np.array(density)
    maxima = argrelextrema(s, np.greater)
    minima = argrelextrema(s, np.less)

    # of the local maxima, find the global maxima index
    global_max_ix = np.argmax(s[maxima])

    # find variance of distribution
    std = np.std(data)

    # plot
    if plot:
        _plot_kde(xs, s, density, maxima, minima)

    return xs[maxima], xs[minima], global_max_ix, std


def find_similar(
    reference_embeddings: Union[str, List[str], List[List[float]], np.ndarray],
    vectordb_api: VectorDBAPI,
    k: int = 100,
    n: int = 1000,
    method: str = "first",
    k_method: Union[str, None] = None,
    kde_bw: Union[str, float] = "scott",
    plot: bool = False,
) -> np.ndarray:
    """Find the most similar embeddings to the reference embeddings.

    @type reference_embeddings: Union[str, List[str], List[List[float]],
        np.ndarray]
    @param reference_embeddings: The embeddings to compare against. Or a
        list of of embedding instance_ids that reside in VectorDB.
    @type vectordb_api: VectorDBAPI
    @param vectordb_api: The VectorDBAPI instance to use.
    @type k: int
    @param k: The number of embeddings to return. Default is 100.
    @type n: int
    @param n: The number of embeddings to compare against. Default is
        1000. (This is the number of embeddings that are returned by the
        VectorDB search. It matters for the KDE, as it can be slow for
        large n. Your choice of n depends on the amount of duplicates in
        your dataset, the more duplicates, the larger n should be. If
        you have 2-10 duplicates per image, n=100 should be ok. If you
        have 50-300 duplicates per image, n=1000 should work good
        enough.
    @type method: str
    @param method: The method to use to find the most similar
        embeddings. If 'first' use the first of the reference
        embeddings. If 'average', use the average of the reference
        embeddings.
    @type k_method: str
    @param k_method: The method to select the best k. If None, use k as
        is. If 'kde_basic', use the minimum of the KDE. If 'kde_peaks',
        use the minimum of the KDE peaks, according to a specific
        hardcoded hevristics/thresholds.
    @type kde_bw: Union[str, float]
    @param kde_bw: The bandwidth to use for the KDE. Default is 'scott'.
    @type plot: bool
    @param plot: Whether to plot the KDE.
    @rtype: np.array
    @return: The instance_ids of the most similar embeddings.
    """

    # Get the reference embeddings
    if isinstance(
        reference_embeddings, str
    ):  # if it is a single instance_id: string uuid
        reference_embeddings = [reference_embeddings]
    if isinstance(
        reference_embeddings[0], str
    ):  # if it is a list of instance_ids: list of strings
        reference_embeddings = vectordb_api.retrieve_embeddings_by_ids(
            reference_embeddings
        )

    # Select the reference embedding
    if method == "first":
        if isinstance(
            reference_embeddings, list
        ):  # if it is a list of embeddings: list of lists of floats
            reference_embeddings = np.array(reference_embeddings)
        if len(reference_embeddings.shape) > 1:
            reference_embeddings = reference_embeddings[0]
    elif method == "average":
        # Calculate the average of the reference embeddings
        avg_embedding = np.mean(reference_embeddings, axis=0)
        reference_embeddings = avg_embedding
    else:
        raise ValueError(f"Unknown method: {method}")

    ix, similarities = vectordb_api.search_similar_embeddings(
        reference_embeddings, top_k=n
    )
    ix, similarities = np.array(ix), np.array(similarities)

    # Select the best k embeddings
    if k_method is None:
        best_embeddings_ix = np.argsort(similarities)[-k:]

    elif k_method == "kde_basic":
        _, new_min, _, _ = kde_peaks(similarities, bandwidth=kde_bw, plot=plot)

        if len(new_min) > 0:
            k = len(np.where(similarities > new_min[-1])[0])
            if k < 2 and len(new_min) > 1:
                k = len(np.where(similarities > new_min[-2])[0])
            print(k, new_min)

        best_embeddings_ix = np.argsort(similarities)[-k:]

    elif k_method == "kde_peaks":
        if len(similarities) > 50000:
            print("Too many embeddings, using 97 percentile")
            # take top 97 procentile of closest points
            p_97 = np.percentile(similarities, 97)
            ix_sim = np.where(similarities > p_97)[0]
            _, new_min, _, _ = kde_peaks(
                similarities[ix_sim], bandwidth=kde_bw, plot=plot
            )

            if len(new_min) > 0:
                k = len(np.where(similarities > new_min[-1])[0])
                print(k, new_min)

        else:
            # get maxima and minima of the KDE on the distances
            _, minima, _, _ = kde_peaks(
                similarities, bandwidth=kde_bw, plot=plot
            )
            if len(minima) > 0:
                minima = minima[-1]
                if minima < 0.94:
                    minima = 0.97
                else:
                    minima = max(minima, 0.96)

            else:
                minima = 0.97

            # select k closest embeddings
            k = len(np.where(similarities > minima)[0])
            if minima > 0.97 and k < 2:
                k = len(np.where(similarities > 0.97)[0])
            print(k, minima)

        # select the best k embeddings
        best_embeddings_ix = np.argsort(similarities)[-k:]

    else:
        raise ValueError(f"Unknown k_method: {k_method}")

    return ix[best_embeddings_ix]
