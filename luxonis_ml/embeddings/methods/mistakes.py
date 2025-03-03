"""Mismatch Detection in Labelled Data.

This module provides functionalities to detect mismatches or potential mislabelling
in a dataset based on various strategies. This is crucial in supervised machine
learning tasks where the quality of labels significantly affects model performance.

Methods implemented
===================
    - Centroids: This method identifies mismatches by comparing the distance of data points
                 to the centroid of their own class against the distances to centroids of
                 other classes.
    - KNN (k-Nearest Neighbors): This approach leverages the idea that if the majority of data
                                 is correctly labelled, then mislabelled data will be corrected
                                 by its nearest neighbors.
    - [Note: DBSCAN was considered but not implemented due to underperformance.]

Usage
=====
    To use this module, import the desired methods and provide the embeddings and labels:

        >>> from mismatch_detection import find_mismatches_centroids, find_mismatches_knn
        >>> # Detect mismatches using centroids
        >>> mismatches, new_labels = find_mismatches_centroids(X_train, y_train)
        >>> # Detect mismatches using KNN
        >>> mismatches, new_labels = find_mismatches_knn(X_train, y_train)
"""

from typing import Tuple

import numpy as np
from scipy.spatial import distance
from sklearn.neighbors import KNeighborsClassifier


def find_mismatches_centroids(
    X: np.array, y: np.array
) -> Tuple[np.array, np.array]:
    """Find mismatches in the dataset. A mismatch is defined as a sample
    that is closer to another centroid than to its own centroid.

    @type X: np.array
    @param X: The embeddings to use.
    @type y: np.array
    @param y: The targets to use.
    @rtype: Tuple[np.array, np.array]
    @return: The indices of the mismatches and the new labels.
    """
    unique_labels = np.unique(y)
    # Create a mapping from string labels to integer indices
    label_to_index = {
        label: index for index, label in enumerate(unique_labels)
    }

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
    centroids = np.array(centroids)

    # make the distance matrix
    dist_matrix = distance.cdist(X, centroids, metric="cosine")

    # find mismatches if the distance to other centroids is 1.5 times larger than the distance to their own centroid
    mismatches = []
    predicted_labels = []
    for i in range(len(X)):
        # find the distance to the closest centroid
        closest_centroid = np.argmin(dist_matrix[i, :])
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


def find_mismatches_knn(
    X: np.array, y: np.array, n_neighbors: int = 5
) -> Tuple[np.array, np.array]:
    """
    Find mismatches in the dataset.
    Single Algorithm Filter (see Figure 1 in Brodley, Carla E., and Mark A. Friedl. "Identifying mislabeled training data.").
    Idea: if the vast majority of the data is correctly labeled and you do knn prediction, the minority of mislabeled data will be engulfed (corrected) by the correct neighbors.

    @type X: np.array
    @param X: The embeddings to use.

    @type y: np.array
    @param y: The targets to use.

    @type n_neighbors: int
    @param n_neighbors: The number of neighbors to use for KNN. Default is 5.

    @rtype: Tuple[np.array, np.array]
    @return: The indices of the mismatches and the new labels.
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
