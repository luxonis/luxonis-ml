"""
Near-duplicate search 

Dynamically remove duplicates (Near-duplicate search) :
    - using KDE (Kernel Density Estimation) on embeddings cosine similarity and finding the minimum for optimal split
"""

import matplotlib.pyplot as plt
import numpy as np

# Near-duplicate search
from scipy.signal import argrelextrema
from scipy.stats import gaussian_kde
import scipy.spatial.distance as distance
from KDEpy import FFTKDE

# Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models

from luxonis_ml.embeddings.qdrant_utils import *


# client = QdrantClient(host="localhost", port=6333)

def search_qdrant(qdrant_client, query_vector, collection_name="webscraped_real_all", data_name="flash", limit=5000):
    
    hits = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_vector.tolist(),
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="image_path",
                    match=models.MatchText(text=data_name)
                )
            ] 
        ),
        limit= limit
    )

    ix = [h.id for h in hits]
    vals = [h.score for h in hits]
    res = [h.payload['image_path'] for h in hits]
    
    return ix, vals, res


def kde_peaks(data, bandwidth="scott", plot=False):
    """
    Find peaks in a KDE distribution using scipy's argrelextrema function.
    """
    # fit density
    kde = FFTKDE(kernel='gaussian', bw=bandwidth)
    xs = np.linspace(np.min(data)-0.01, np.max(data)+0.01, 1000)
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
        plt.plot(xs,density, label='KDE')
        plt.plot(xs[maxima], s[maxima], 'ro', label='local maxima')
        plt.plot(xs[minima], s[minima], 'bo', label='local minima')
        plt.plot(xs[np.argmax(s)], np.max(s), 'go', label='global maxima')
        plt.legend()
        plt.show()

    return xs[maxima], xs[minima], global_max_ix, std


def find_similar_qdrant(
        reference_embeddings,
        qdrant_client,
        k=100,         
        collection_name="webscraped_real_all",
        dataset="flash",
        n=1000,
        method='first', 
        k_method=None, 
        kde_bw="scott", 
        plot=False
    ):
    """
    Find the most similar embeddings to the reference embeddings.

    Parameters
    ----------
    reference_embeddings : np.array / list
        The embeddings to compare against.
        Or a list of of embedding instance_ids that reside in Qdrant.
    qdrant_client : QdrantClient
        The Qdrant client instance to use for searches.
    k : int
        The number of similar embeddings to return.
    collection_name : str
        The name of the Qdrant collection. Default is 'webscraped_real_all'.
    dataset : str
        The dataset to use. Default is 'flash'.
    n : int
        The number of embeddings to compare against. Default is 1000.
    method : str
        The method to use to find the most similar embeddings. 
        If 'first' use the first of the reference embeddings. If 'average', use the average of the reference embeddings. 
    k_method : str
        The method to select the best k. 
        If None, use k as is. 
        If 'kde_basic', use the minimum of the KDE. 
        If 'kde_peaks', use the minimum of the KDE peaks, according to some hevristics/thresholds.
    kde_bw : str/float
        The bandwidth to use for the KDE. Default is 'scott'. See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html.
    plot : bool
        Whether to plot the KDE.
    
    Returns
    -------
    np.array
        The indices of the most similar embeddings.
    np.array
        The paths of the most similar embeddings.

    """
    # Get the reference embeddings
    # check if reference_embeddings is a list of instance_ids
    if isinstance(reference_embeddings, str):
        reference_embeddings = [reference_embeddings]
    if isinstance(reference_embeddings[0], str):
        reference_embeddings = get_embeddings_from_ids(qdrant_client, reference_embeddings, collection_name=collection_name)

    # Select the reference embedding
    if method == 'first':
        if isinstance(reference_embeddings, list):
            reference_embeddings = np.array(reference_embeddings)
        if len(reference_embeddings.shape) > 1:
            reference_embeddings = reference_embeddings[0]
    elif method == 'average':
        # Calculate the average of the reference embeddings
        avg_embedding = np.mean(reference_embeddings, axis=0)
        reference_embeddings = avg_embedding
    else:
        raise ValueError(f'Unknown method: {method}')
    
    ix, vals, res = search_qdrant(qdrant_client, reference_embeddings, collection_name=collection_name, data_name=dataset, limit=n)
    ix, similarities, res = np.array(ix), np.array(vals), np.array(res)
    
    # Select the best k embeddings
    if k_method is None:
        best_embeddings_ix = np.argsort(similarities)[-k:]
        
    elif k_method == 'kde_basic':
        _, new_min, _, _ = kde_peaks(similarities, bandwidth=kde_bw, plot=plot)
        
        if len(new_min) > 0:
            k = len(np.where(similarities > new_min[-1])[0])
            if k < 2 and len(new_min) > 1:
                k = len(np.where(similarities > new_min[-2])[0])
            print(k, new_min)
        else:
            pass

        best_embeddings_ix = np.argsort(similarities)[-k:]

    elif k_method == 'kde_peaks':

        if len(similarities) > 50000:
            print('Too many embeddings, using 97 percentile')
            # take top 97 procentile of closest points
            p_97 = np.percentile(similarities, 97)
            ix = np.where(similarities > p_97)[0]
            _, new_min, _, _ = kde_peaks(similarities[ix], bandwidth=kde_bw, plot=plot)
            
            if len(new_min) > 0:
                k = len(np.where(similarities > new_min[-1])[0])
                print(k, new_min)
            else:
                pass

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
    
    return ix[best_embeddings_ix], res[best_embeddings_ix]


# ixs, vals, res = search_qdrant(X_real[i], "webscraped_real_all", "real")
# vals = np.array(vals)
# k = len(np.where(vals > 0.961)[0])
# imgk = res[:k]

# ix, paths = find_similar_qdrant(X_real[i], 5, "real", 100, "first", "kde_peaks", "silverman", plot=False)

# remove_dark_ix, paths = find_similar_qdrant(embedding, 5, "real", 50000, "first", "kde_basic", "silverman", plot=True)
# remove_dark_ix, paths = find_similar_qdrant(emb_dark, 25, "real", 5000, "average", "kde_basic", "scott", plot=True)
