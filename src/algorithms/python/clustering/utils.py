from __future__ import annotations

import numpy as np


class DendrogramNode:
    """
    Node in a dendrogram.

    Attributes:
        name: str Name of the node.
        centroid: np.ndarray | None Centroid of the cluster.
        children: list[DendrogramNode] Children of the node.
        distance: float Distance between the left and right children.
    """
    name: str
    centroid: np.ndarray | None
    children: list[DendrogramNode]
    distance: float

    def __init__(self, name: str, centroid: np.ndarray | None = None):
        """
        Initialize a dendrogram node.

        :param name: Name of the node.
        :param centroid: Centroid of the cluster.
        """
        self.name = name
        self.centroid = centroid
        self.children = []
        self.distance = 0.0

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"DendrogramNode(node={self.name}, children={self.children}, distance={self.distance})"

    def __eq__(self, other: DendrogramNode) -> bool:
        return other is not None and self.name == other.name


def calculate_jaccard_coeff(left: np.ndarray, right: np.ndarray) -> float:
    """
    Calculate the Jaccard coefficient between two sets

    :param left: NumPy array of shape (n,) containing the indices of the items in the left set
    :param right: NumPy array of shape (m,) containing the indices of the items in the right set
    :return: Jaccard coefficient between the two sets
    """
    return len(np.intersect1d(left, right)) / len(np.union1d(left, right))


def calculate_jaccard_distance(left: np.ndarray, right: np.ndarray) -> float:
    """
    Calculate the Jaccard distance between two sets

    :param left: NumPy array of shape (n,) containing the indices of the items in the left set
    :param right: NumPy array of shape (m,) containing the indices of the items in the right set
    :return: Jaccard distance between the two sets
    """
    return 1 - calculate_jaccard_coeff(left, right)


def calculate_single_linkage_distance(left_distance: float, right_distance: float) -> float:
    """
    Calculate the single linkage distance between two clusters

    :param left_distance: Distance between the left cluster and another cluster
    :param right_distance: Distance between the right cluster and another cluster
    :return: Single linkage distance between the two clusters
    """
    return min(left_distance, right_distance)


def calculate_complete_linkage_distance(left_distance: float, right_distance: float) -> float:
    """
    Calculate the complete linkage distance between two clusters

    :param left_distance: Distance between the left cluster and another cluster
    :param right_distance: Distance between the right cluster and another cluster
    :return: Complete linkage distance between the two clusters
    """
    return max(left_distance, right_distance)


def calculate_euclidean_distance(left: np.ndarray, right: np.ndarray) -> float:
    """
    Calculate the Euclidean distance between two vectors

    :param left: NumPy array of shape (n,) containing the left vector
    :param right: NumPy array of shape (m,) containing the right vector
    :return: Euclidean distance between the two vectors
    """
    return np.linalg.norm(left - right)


def calculate_sse(data: list[np.ndarray], centroids: list[np.ndarray]) -> list[float]:
    """
    Calculate the sum of squared errors (SSE) of the clusters

    :param data: Data points
    :param centroids: Centroids of the clusters
    :return: Sum of squared errors (SSE) of the clusters
    """
    sse = [0.0 for _ in range(len(centroids))]
    for data_point in data:
        min_idx = np.argmin([calculate_euclidean_distance(data_point, centroid) for centroid in centroids])
        centroid = centroids[min_idx]
        sse[min_idx] += calculate_euclidean_distance(data_point, centroid) ** 2
    return sse
