import numpy as np
from abc import ABC, abstractmethod
from utils import DendrogramNode


class Clustering(ABC):
    """
    Abstract class for clustering algorithms.

    Attributes:
        k: int K hyperparameter for the clustering algorithm.
        centroids: list[np.ndarray] | None Centroids of the clusters.
        clusters: list[DendrogramNode] | None Discovered clusters of the data.
    """
    k: int
    centroids: list[np.ndarray] | None
    clusters: list[DendrogramNode] | None

    def __init__(self, k: int) -> None:
        """
        Initialize the clustering algorithm.

        :param k: K hyperparameter for the clustering algorithm.
        """
        self.k = k
        self.centroids = None
        self.clusters = None

    @abstractmethod
    def fit(self, data: list[np.ndarray]) -> None:
        """
        Fit the clustering algorithm to the data.

        :param data: Data to fit the clustering algorithm to.
        :return:
        """
        pass

    @abstractmethod
    def predict(self, data_point: np.ndarray) -> DendrogramNode | None:
        """
        Predict the cluster of a data point.

        :param data_point: Data point to predict the cluster of.
        :return: Cluster of the data point.
        """
        pass

    def get_centroids(self) -> list[np.ndarray] | None:
        """
        Get the centroids of the clusters.

        :return: List of centroids of the clusters.
        """
        return self.centroids

    def get_clusters(self) -> list[DendrogramNode] | None:
        """
        Get the clusters.

        :return: List of clusters.
        """
        return self.clusters
