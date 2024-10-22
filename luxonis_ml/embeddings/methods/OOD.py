"""Out-of-Distribution Detection for Embeddings.

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

from typing import Optional, Union

import numpy as np
from sklearn.ensemble import IsolationForest


def isolation_forest_OOD(
    X: np.array,
    contamination: Union[float, str] = "auto",
    n_jobs: int = -1,
    verbose: int = 1,
    random_state: Optional[int] = None,
) -> np.array:
    """Out-of-distribution detection using Isolation Forests.

    @type X: np.array
    @param X: The embeddings to use.
    @type contamination: Union[float, str]
    @param contamination: The contamination parameter for Isolation
        Forests. Default is 'auto'.
    @type n_jobs: int
    @param n_jobs: The number of jobs to use. Default is -1, which means
        all available CPUs.
    @type verbose: int
    @param verbose: The verbosity level. Default is 1.
    @type random_state: Optional[int]
    @param random_state: The random state to use. Default is None.
    @rtype: np.array
    @return: The indices of the embeddings that are in-distribution.
    """
    # Initialize the Isolation Forest model
    isolation_forest = IsolationForest(
        contamination=contamination,
        n_jobs=n_jobs,
        verbose=verbose,
        random_state=random_state,
    )

    # Fit the model on all embeddings
    isolation_forest.fit(X)

    # Predict the outliers
    predicted_labels = isolation_forest.predict(X)

    # Get the indices of the outliers (where the predicted label is -1)
    outlier_indices_forest = np.where(predicted_labels == -1)[0]

    return outlier_indices_forest


def leverage_OOD(X: np.array, std_threshold: int = 3) -> np.array:
    """Out-of-distribution detection using leverage and linear
    regression.

    @type X: np.array
    @param X: The embeddings to use.
    @type std_threshold: int
    @param std_threshold: The number of standard deviations to use for
        the leverage threshold. Default is 3.
    @rtype: np.array
    @return: The indices of the embeddings that are out-of-distribution.
    """
    # Calculate the hat matrix (projection matrix) to get the leverage for each point
    hat_matrix = np.matmul(np.matmul(X, np.linalg.inv(np.matmul(X.T, X))), X.T)
    leverage = np.diagonal(hat_matrix)

    # Calculate the leverage threshold using 3 standard deviations
    mean, std = np.mean(leverage), np.std(leverage)
    upper_threshold, lower_threshold = (
        mean + std_threshold * std,
        mean - std_threshold * std,
    )
    outlier_indices_lev_3std = np.where(
        (leverage > upper_threshold) | (leverage < lower_threshold)
    )[0]

    return outlier_indices_lev_3std
