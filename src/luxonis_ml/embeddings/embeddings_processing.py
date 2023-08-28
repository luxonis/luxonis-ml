"""
Embeddings based methods for Luxonis Dataset Format (https://github.com/luxonis/luxonis-ml)

Features:
- Extracing embeddings from second to last layer of ONNX Model
- Uploading them to Qdrant (a vector similarity search engine / database)

- Finding similar images
    - Dynamically remove duplicates (Near-duplicate search) - using KDE on embeddings cosine similarity and finding the minimum for optimal split
- Finding representative images - greedy search, k-medoids
    - dataset reduction
    - validation set creation
- Out-of-distribution detection: using leverage and linear regression, using isolation forests
    - anomaly detection / dataset reduction
    - finding useful samples (using additional images outside of the dataset)
- Mistakes detection (prediction of wrong labels) (predictions needed) - centroids, knn, dbscan (underperforming - not used)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Near-duplicate search
from scipy.signal import argrelextrema
from scipy.stats import gaussian_kde
import scipy.spatial.distance as distance
from KDEpy import FFTKDE

# Representative images
from sklearn.metrics.pairwise import cosine_similarity
from kmedoids import KMedoids

# OOD detection
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

# Mistakes detection
import scipy.spatial.distance as distance
from sklearn.neighbors import KNeighborsClassifier

# Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models

# from sklearn.preprocessing import LabelEncoder
# from car_loader import CarLoaderColor

import os
import json
import shutil

import cv2
import torch
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from luxonis_ml.embeddings.qdrant_utils import *


# **********************************************************************************************************************
# ******************************************Near-duplicate search*******************************************************

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


# **********************************************************************************************************************
# *****************************************Find representative images***************************************************

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


# **********************************************************************************************************************
# **************************************Out-of-distribution detection***************************************************


def isolation_forest_OOD(X,  contamination='auto', n_jobs=-1, verbose=1, random_state=42):
    """
    Out-of-distribution detection using Isolation Forests.

    Parameters
    ----------
    X : np.array
        The embeddings to use.
    contamination : float
        The contamination parameter for Isolation Forests. Default is 'auto'.
    n_jobs : int
        The number of jobs to use. Default is -1, which means all available CPUs.
    verbose : int
        The verbosity level. Default is 1.
    random_state : int
        The random state to use. Default is 42.
    
    Returns
    -------
    np.array
        The indices of the embeddings that are in-distribution.

    """
    # Initialize the Isolation Forest model
    isolation_forest = IsolationForest(contamination=contamination, 
                                       n_jobs=n_jobs, 
                                       verbose=verbose, 
                                       random_state=random_state)

    # Fit the model on all embeddings
    isolation_forest.fit(X)

    # Predict the outliers
    predicted_labels = isolation_forest.predict(X)

    # Get the indices of the outliers (where the predicted label is -1)
    outlier_indices_forest = np.where(predicted_labels == -1)[0]

    return outlier_indices_forest


def leverage_OOD(X):
    """
    Out-of-distribution detection using leverage and linear regression.

    Parameters
    ----------
    X : np.array
        The embeddings to use.
    
    Returns
    -------
    np.array
        The indices of the embeddings that are out-of-distribution.

    """
    # # Fit a linear regression model to the embeddings
    # regression_model = LinearRegression()
    # regression_model.fit(X, y)

    # # Predict targets for all embeddings
    # predicted_targets = regression_model.predict(X)

    # Calculate the hat matrix (projection matrix) to get the leverage for each point
    hat_matrix = np.matmul(np.matmul(X, np.linalg.inv(np.matmul(X.T, X))), X.T)
    leverage = np.diagonal(hat_matrix)

    # Calculate the leverage threshold using 3 standard deviations
    mean, std = np.mean(leverage), np.std(leverage)
    upper_threshold, lower_threshold = mean + 3 * std, mean - 3 * std
    outlier_indices_lev_3std = np.where((leverage > upper_threshold) | (leverage < lower_threshold))[0]

    return outlier_indices_lev_3std


# **********************************************************************************************************************
# ********************************Mistakes detection (prediction of wrong labels)***************************************

def find_mismatches_centroids(X, y):
    """
    Find mismatches in the dataset. 
    A mismatch is defined as a sample that is closer to another centroid than to its own centroid.

    Parameters
    ----------
    X : np.array
        The embeddings to use.
    y : np.array
        The targets to use.

    Returns
    -------
    np.array
        The indices of the mismatches.
    np.array
        The new predicted labels.
    """
    unique_labels = np.unique(y)
    # Create a mapping from string labels to integer indices
    label_to_index = {label: index for index, label in enumerate(unique_labels)}

    # calculate centroids of each class
    centroids = []
    for label in unique_labels:
        # Get the indices of the samples in the train set that belong to the specified class
        indices = np.where(y == label)[0]

        # Get the embeddings of the samples in the specified class
        embeddings_class = X[indices]

        # Calculate the centroid of the specified class
        centroid = np.mean(embeddings_class, axis=0)
        centroids.append(centroid)

    # append the centroids to the embeddings
    Centroids = np.array(centroids)

    # make the distance matrix
    dist_matrix = distance.cdist(X, Centroids, metric='cosine')

    # find mismatches if the distance to other centroids is 1.5 times larger than the distance to their own centroid
    mismatches = []
    predicted_labels = []
    for i in range(len(X)):
        # find the distance to the closest centroid
        closest_centroid = np.argmin(dist_matrix[i,:])
        closest_distance = dist_matrix[i, closest_centroid]

        # find the distance to their own centroid
        own_centroid = dist_matrix[i, label_to_index[y[i]]]

        # if the distance to the closest centroid is 1.5 times larger than the distance to their own centroid, then it is a mismatch
        if closest_distance * 1.5 < own_centroid:
            mismatches.append(i)
            predicted_labels.append(unique_labels[closest_centroid])
        # else:
        #     predicted_labels.append(y[i])

    mismatches = np.array(mismatches)
    predicted_labels = np.array(predicted_labels)
    new_labels = predicted_labels

    return mismatches, new_labels


def find_mismatches_knn(X, y, n_neighbors=5):
    """
    Find mismatches in the dataset. 
    Single Algorithm Filter (see Figure 1 in Brodley, Carla E., and Mark A. Friedl. "Identifying mislabeled training data.").
    Idea: if the vast majority of the data is correctly labeled and you do knn prediction, the minority of mislabeled data will be engulfed (corrected) by the correct neighbors.

    Parameters
    ----------
    X : np.array
        The embeddings to use.
    y : np.array
        The targets to use.
    n_neighbors : int
        The number of neighbors to use for KNN. Default is 5.

    Returns
    -------
    np.array
        The indices of the mismatches.
    np.array
        The new predicted labels.
    """
    
    # Step 1: Fit KNN on the train set with corrupted labels
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X, y)

    # Step 2: Predict labels on the train set using the trained KNN model
    y_pred = knn.predict(X)

    # Step 3: Find the mismatched labels
    mismatches = np.where(y_pred != y)[0]

    new_labels = y_pred

    return mismatches, new_labels