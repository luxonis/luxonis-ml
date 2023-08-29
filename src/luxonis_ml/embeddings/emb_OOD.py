"""
Out-of-distribution detection

- OOD detection algorithms:
    - Isolation Forests
    - Leverage (and linear regression)

- Useful for:
    - anomaly detection / dataset reduction
    - finding useful samples (using additional images outside of the dataset)
"""

import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
import scipy.spatial.distance as distance

from luxonis_ml.embeddings.qdrant_utils import *


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

