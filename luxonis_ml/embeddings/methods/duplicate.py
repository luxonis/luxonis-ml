"""Near-duplicate Search with Qdrant.

This module provides utilities to detect and remove near-duplicate data points
within a given set of embeddings. The removal process uses Kernel Density
Estimation (KDE) on embeddings cosine similarity for optimal split, making
this approach particularly suited for embeddings in high dimensional spaces.

Functionality Includes:
    - Using Qdrant for efficient search and retrieval of embeddings.
    - Applying Kernel Density Estimation (KDE) to detect similarity peaks.
    - Visualizing KDE results with matplotlib.
    - Dynamic selection of best candidates for removal based on KDE peaks.

Dependencies:
    - Requires the KDEpy library for KDE.
    - Utilizes Qdrant (an open-source vector database) for embedding storage and search.

Functions:
    - search_qdrant: Search embeddings within Qdrant based on query vectors.
    - _plot_kde: Utility function to plot a KDE distribution.
    - kde_peaks: Determine peaks in a KDE distribution.
    - find_similar_qdrant: Find the most similar embeddings to the given reference embeddings.

Examples:
    1. Initialize a Qdrant client and retrieve all embeddings from a specific collection:

        from luxonis_ml.embeddings.utils.qdrant import QdrantAPI
        qdrant_api = QdrantAPI(host="localhost", port=6333, collection_name="webscraped_real_all")
        id_X, X = qdrant_api.get_all_embeddings()
        i = 111

    2. Search for similar embeddings in Qdrant for a specific embedding:

        ids, similarites, image_paths = search_qdrant(qdrant_api, X[i], "real", 1000)
        vals = np.array(similarites)
        k = len(np.where(vals > 0.961)[0])  # manually selected threshold
        imgk = image_paths[:k]

    3. Find similar embeddings by providing an instance ID:

        ix, paths = find_similar_qdrant(id_X[i], qdrant_api, "real", 5, 100, "first")

    4. Find similar embeddings using the KDE Peaks method:

        ix, paths = find_similar_qdrant(X[i], qdrant_api, "real", 5, 100, "first", "kde_peaks", "silverman", plot=False)

    5. Calculate the average of multiple embeddings and then find similar embeddings:

        dark_ix = np.array([10,123,333,405])
        emb_dark = X[dark_ix]
        remove_dark_ix, paths = find_similar_qdrant(emb_dark, qdrant_api, "real", 25, 5000, "average", "kde_basic", "scott", plot=True)
"""


import matplotlib.pyplot as plt
import numpy as np
from KDEpy import FFTKDE

# Near-duplicate search
from scipy.signal import argrelextrema

# Qdrant


def search_qdrant(qdrant_api, query_vector, data_name, limit=5000):
    """Search embeddings in Qdrant."""
    hits = qdrant_api.search_embeddings_by_imagepath(query_vector, data_name, top=limit)

    ix = [h.id for h in hits]
    vals = [h.score for h in hits]
    res = [h.payload["image_path"] for h in hits]

    return ix, vals, res


def _plot_kde(xs, s, density, maxima, minima):
    """Plot a KDE distribution."""
    plt.plot(xs, density, label="KDE")
    plt.plot(xs[maxima], s[maxima], "ro", label="local maxima")
    plt.plot(xs[minima], s[minima], "bo", label="local minima")
    plt.plot(xs[np.argmax(s)], np.max(s), "go", label="global maxima")
    plt.legend()
    plt.show()


def kde_peaks(data, bandwidth="scott", plot=False):
    """Find peaks in a KDE distribution using scipy's argrelextrema function."""
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


def find_similar_qdrant(
    reference_embeddings,
    qdrant_api,
    dataset,
    k=100,
    n=1000,
    method="first",
    k_method=None,
    kde_bw="scott",
    plot=False,
):
    """Find the most similar embeddings to the reference embeddings.

    @type reference_embeddings: Union[np.array, list]
    @param reference_embeddings: The embeddings to compare against. Or a list of of
        embedding instance_ids that reside in Qdrant.
    @type qdrant_api: QdrantAPI
    @param qdrant_api: The Qdrant client API instance to use for searches.
    @type dataset: str
    @param dataset: The dataset to use. (It actually filters on the image_path field, so
        it can be any string. It can be helpful if you have different datasets in
        subfolders for the same collection, like 'real/img1.jpg' and 'synth/img1.jpg'.)
    @type k: int
    @param k: The number of embeddings to return. Default is 100.
    @type n: int
    @param n: The number of embeddings to compare against. Default is 1000. (This is the
        number of embeddings that are returned by the Qdrant search. It matters for the
        KDE, as it can be slow for large n. Your choice of n depends on the amount of
        duplicates in your dataset, the more duplicates, the larger n should be. If you
        have 2-10 duplicates per image, n=100 should be ok. If you have 50-300
        duplicates per image, n=1000 should work good enough.
    @type method: str
    @param method: The method to use to find the most similar embeddings. If 'first' use
        the first of the reference embeddings. If 'average', use the average of the
        reference embeddings.
    @type k_method: str
    @param k_method: The method to select the best k. If None, use k as is. If
        'kde_basic', use the minimum of the KDE. If 'kde_peaks', use the minimum of the
        KDE peaks, according to a specific hardcoded hevristics/thresholds.
    @type kde_bw: Union[str, float]
    @param kde_bw: The bandwidth to use for the KDE. Default is 'scott'.
    @type plot: bool
    @param plot: Whether to plot the KDE.
    @rtype: np.array
    @return: The instance_ids of the most similar embeddings.
    """
    # Get the reference embeddings
    # check if reference_embeddings is a list of instance_ids
    if isinstance(reference_embeddings, str):
        reference_embeddings = [reference_embeddings]
    if isinstance(reference_embeddings[0], str):
        reference_embeddings = qdrant_api.get_embeddings_from_ids(reference_embeddings)

    # Select the reference embedding
    if method == "first":
        if isinstance(reference_embeddings, list):
            reference_embeddings = np.array(reference_embeddings)
        if len(reference_embeddings.shape) > 1:
            reference_embeddings = reference_embeddings[0]
    elif method == "average":
        # Calculate the average of the reference embeddings
        avg_embedding = np.mean(reference_embeddings, axis=0)
        reference_embeddings = avg_embedding
    else:
        raise ValueError(f"Unknown method: {method}")

    # Search for similar embeddings in Qdrant
    ix, vals, res = search_qdrant(
        qdrant_api, reference_embeddings, data_name=dataset, limit=n
    )
    ix, similarities, res = np.array(ix), np.array(vals), np.array(res)

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
            _, minima, _, _ = kde_peaks(similarities, bandwidth=kde_bw, plot=plot)
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

    return ix[best_embeddings_ix], res[best_embeddings_ix]
