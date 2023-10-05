import numpy as np
import logging

from utils import calculate_euclidean_distance, DendrogramNode, calculate_sse
from clustering import Clustering

np.random.seed(42)
logging.basicConfig(level=logging.INFO)


class KMeans(Clustering):
    """
    K-Means clustering algorithm. It is a centroid-based clustering algorithm where the number of clusters is specified
    by the K hyperparameter. The algorithm is implemented using a dendrogram.
    """

    def __init__(self, k: int, max_iter: int) -> None:
        """
        Initialize the K-Means clustering algorithm.

        :param k: K hyperparameter for the clustering algorithm.
        :param max_iter: Maximum number of iterations to run the algorithm for.
        """
        super().__init__(k)
        self.max_iter = max_iter

    def fit(self, data: list[np.ndarray]) -> None:
        """
        Fit the K-Means clustering algorithm to the data.

        :param data: Data to fit the K-Means clustering algorithm to.
        :return:
        """
        logging.info(f"Fitting K-Means clustering algorithm with {self.k} clusters.")

        min_point, max_point = np.min(data), np.max(data)
        self.centroids = [np.random.uniform(min_point, max_point, size=data[0].shape) for _ in range(self.k)]

        logging.info(f"Initial centroids:\n {self.centroids}")

        iteration = 0
        prev_centroids = None
        clusters = [[] for _ in range(self.k)]

        while iteration < self.max_iter and np.not_equal(prev_centroids, self.centroids).any():
            logging.info(f"Iteration {iteration + 1} of {self.max_iter}.")

            clusters = self._calculate_clusters(data)
            prev_centroids = self.centroids
            self.centroids = self._calculate_centroids(clusters, prev_centroids)
            iteration += 1

            logging.info(f"Centroids:\n {self.centroids}")
            logging.info(f"Clusters:\n {clusters}")

        self.clusters = self._create_dendrogram(clusters)

    def predict(self, data_point: np.ndarray) -> DendrogramNode | None:
        distances = [calculate_euclidean_distance(data_point, centroid) for centroid in self.centroids]
        min_index = np.argmin(distances)
        return self.clusters[min_index]

    def _create_dendrogram(self, clusters: list[list[np.ndarray]]) -> list[DendrogramNode]:
        """
        Create a dendrogram from the clusters.

        :param clusters: Clusters to create the dendrogram from.
        :return: Dendrogram created from the clusters.
        """
        node_clusters = []

        for cluster, centroid in zip(clusters, self.centroids):
            cluster_node = DendrogramNode(cluster, centroid)
            for data_point in cluster:
                cluster_node.children.append(DendrogramNode(data_point, data_point))
            node_clusters.append(DendrogramNode(cluster, centroid))

        return node_clusters

    def _calculate_clusters(self, data: list[np.ndarray]) -> list[list[np.ndarray]]:
        """
        Calculate the clusters of the data.

        :param data: Data to calculate the clusters of.
        :return: Clusters of the data.
        """
        clusters = [[] for _ in range(self.k)]
        for data_point in data:
            distances = [calculate_euclidean_distance(data_point, centroid) for centroid in self.centroids]
            clusters[np.argmin(distances)].append(data_point)
        return clusters

    @staticmethod
    def _calculate_centroids(clusters: list[list[np.ndarray]], prev_centroids: list[np.ndarray]) \
            -> list[np.ndarray]:
        """
        Calculate the centroids of the clusters.

        :param clusters: Clusters to calculate the centroids of.
        :param prev_centroids: Previous centroids of the clusters.
        :return: Centroids of the clusters.
        """
        new_centroids = [np.mean(cluster, axis=0) for cluster in clusters]
        return [new_centroid if not np.isnan(new_centroid) else prev_centroid for new_centroid, prev_centroid in
                zip(new_centroids, prev_centroids)]


def test_kmeans():
    test_array = [
        np.array([6]),
        np.array([12]),
        np.array([18]),
        np.array([22]),
        np.array([30]),
        np.array([42]),
        np.array([48]),
    ]

    kmeans = KMeans(k=2, max_iter=8)
    kmeans.fit(test_array)
    centroids = kmeans.get_centroids()
    clusters = kmeans.get_clusters()
    print(calculate_sse(test_array, [np.array([15]), np.array([40])]))


if __name__ == '__main__':
    test_kmeans()
