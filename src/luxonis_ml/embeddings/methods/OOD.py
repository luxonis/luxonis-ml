"""
Out-of-Distribution Detection for Embeddings

This module provides two primary methods for detecting out-of-distribution (OOD) samples 
in embeddings. OOD samples can be crucial to identify as they represent anomalies or novel 
patterns that don't conform to the expected distribution of the dataset.

Methods available:
- Isolation Forests: A tree-based model that partitions the space in such a manner that 
  anomalies are isolated from the rest.
  
- Leverage with Linear Regression: Leverages (or hat values) represent the distance between 
  the predicted values and the true values. Higher leverages indicate potential OOD points.

Typical use cases include:
- Anomaly Detection: Identifying rare patterns or outliers.
  
- Dataset Reduction: By removing or studying OOD samples, we can have a more homogeneous dataset.
  
- Expanding Datasets: Recognizing valuable data points that are distinct from the current distribution can be helpful 
  when we're looking to diversify the dataset, especially in iterative learning scenarios.

Dependencies:
- numpy
- scikit-learn

"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

def isolation_forest_OOD(X, contamination='auto', n_jobs=-1, verbose=1, random_state=None):
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
        The random state to use. Default is None.
    
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


def leverage_OOD(X, std_threshold=3):
    """
    Out-of-distribution detection using leverage and linear regression.

    Parameters
    ----------
    X : np.array
        The embeddings to use.
    std_threshold : int
        The number of standard deviations to use for the leverage threshold. Default is 3.
    
    Returns
    -------
    np.array
        The indices of the embeddings that are out-of-distribution.

    """
    # Calculate the hat matrix (projection matrix) to get the leverage for each point
    hat_matrix = np.matmul(np.matmul(X, np.linalg.inv(np.matmul(X.T, X))), X.T)
    leverage = np.diagonal(hat_matrix)

    # Calculate the leverage threshold using 3 standard deviations
    mean, std = np.mean(leverage), np.std(leverage)
    upper_threshold, lower_threshold = mean + std_threshold * std, mean - std_threshold * std
    outlier_indices_lev_3std = np.where((leverage > upper_threshold) | (leverage < lower_threshold))[0]

    return outlier_indices_lev_3std

