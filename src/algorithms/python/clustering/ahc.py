import numpy as np
import utils
import clustering
import logging
from src.algorithms.python.clustering import DendrogramNode
from enum import Enum

logging.basicConfig(level=logging.INFO)


class LinkageType(Enum):
    SINGLE = "single"
    COMPLETE = "complete"


class DistanceType(Enum):
    JACCARD = "jaccard"
    EUCLIDEAN = "euclidean"


class AgglomerativeHierarchicalClustering(clustering.Clustering):
    """
    Agglomerative hierarchical clustering algorithm. It is a bottom-up approach where each data point starts as its own
    cluster and clusters are merged together until the desired number of clusters is reached. The distance between
    clusters is calculated using the Jaccard or Euclidean distance. The algorithm is implemented using a dendrogram.

    Attributes:
        k: int K hyperparameter for the clustering algorithm.
        centroids: list[np.ndarray] | None Centroids of the clusters.
        clusters: list[DendrogramNode] | None Discovered clusters of the data.
        linkage: LinkageType Linkage type for the clustering algorithm.
        distance_type: DistanceType Distance type for the clustering algorithm.
    """
    linkage: LinkageType

    def __init__(self, k: int, linkage: LinkageType = LinkageType.SINGLE,
                 distance_type: DistanceType = DistanceType.JACCARD) -> None:
        """
        Initialize the agglomerative hierarchical clustering algorithm.

        :param k: K hyperparameter for the clustering algorithm.
        :param linkage: Linkage type for the clustering algorithm.
        :param distance_type: Distance type for the clustering algorithm.
        """
        super().__init__(k)
        self.linkage = linkage
        self.distance_type = distance_type

    def fit(self, data: list[np.ndarray]) -> None:
        dist_mat: np.ndarray = np.full((len(data), len(data)), np.inf)
        nodes: list[str] = list(map(str, range(1, len(data) + 1)))
        self.clusters = [DendrogramNode(node, centroid) for node, centroid in zip(nodes, data)]

        logging.info(f'Fitting agglomerative hierarchical clustering algorithm with {self.linkage.value} linkage '
                     f'and {self.k} clusters.')

        logging.info('Generating distance matrix.')
        for i in range(len(data)):
            for j in range(i + 1, len(data)):
                if self.distance_type == DistanceType.EUCLIDEAN:
                    dist_mat[i, j] = utils.calculate_euclidean_distance(data[i], data[j])
                elif self.distance_type == DistanceType.JACCARD:
                    dist_mat[i, j] = utils.calculate_jaccard_distance(data[i], data[j])

        logging.info(f'Distance matrix:\n {dist_mat}')

        logging.info('Generating dendrogram.')
        while len(self.clusters) > self.k and np.min(dist_mat) != np.inf:
            min_indices: np.ndarray = np.where(dist_mat == np.min(dist_mat))
            min_i, min_j = min_indices[0][0], min_indices[1][0]

            logging.info(f'Merging clusters {self.clusters[min_i].name} and {self.clusters[min_j].name}.')

            new_cluster: DendrogramNode = self._create_new_cluster(min_i, min_j, dist_mat)

            self.clusters.insert(min_i, new_cluster)
            for child in new_cluster.children:
                self.clusters.remove(child)

            logging.info(f'New cluster: {new_cluster.name}.')
            logging.info(f'New cluster centroid: {new_cluster.centroid}.')

            logging.info(f'Generating new distance matrix.')
            dist_mat = self._generate_new_distance_matrix(min_i, min_j, dist_mat)
            logging.info(f'New distance matrix:\n {dist_mat}.')

        self.centroids = [cluster.centroid for cluster in self.clusters]

        logging.info(f'Final clusters:\n {self.clusters}.')

    def predict(self, data_point: np.ndarray) -> DendrogramNode | None:
        distances = []
        if self.distance_type == DistanceType.EUCLIDEAN:
            distances = [utils.calculate_euclidean_distance(data_point, centroid)
                         for centroid in self.centroids]
        elif self.distance_type == DistanceType.JACCARD:
            distances = [utils.calculate_jaccard_distance(data_point, centroid)
                         for centroid in self.centroids]
        return self.clusters[np.argmin(distances)]

    def _create_new_cluster(self, i: int, j: int, dist_mat: np.ndarray) \
            -> DendrogramNode:
        """
        Create a new cluster from two clusters.

        :param i: Index of the first cluster.
        :param j: Index of the second cluster.
        :param dist_mat: Distance matrix of the clusters.
        :return: New cluster.
        """
        centroid: np.ndarray = np.intersect1d(self.clusters[i].centroid, self.clusters[j].centroid,
                                              return_indices=True)[0] \
            if self.distance_type == DistanceType.JACCARD\
            else np.mean([self.clusters[i].centroid, self.clusters[j].centroid], axis=0)

        new_cluster: DendrogramNode = DendrogramNode(f'({self.clusters[i]}, {self.clusters[j]})', centroid)
        new_cluster.children = [self.clusters[i], self.clusters[j]]
        new_cluster.distance = dist_mat[i, j]

        return new_cluster

    def _generate_new_distance_matrix(self, cluster_i: int, cluster_j: int, dist_mat: np.ndarray) -> np.ndarray:
        """
        Generate a new distance matrix after merging two clusters.

        :param cluster_i: Index of the first cluster.
        :param cluster_j:  Index of the second cluster.
        :param dist_mat: Distance matrix of the clusters.
        :return: New distance matrix.
        """
        new_dist_mat = np.full((dist_mat.shape[0] - 1, dist_mat.shape[1] - 1), np.inf)

        for i in range(dist_mat.shape[0]):
            for j in range(i + 1, dist_mat.shape[0]):
                if i == cluster_i and j == cluster_j:
                    continue
                if i == cluster_i:
                    if self.linkage == LinkageType.SINGLE:
                        new_dist_mat[i, j - 1] = \
                            utils.calculate_single_linkage_distance(dist_mat[cluster_i, j],
                                                                    dist_mat[cluster_j, j])
                    elif self.linkage == LinkageType.COMPLETE:
                        new_dist_mat[i, j - 1] = \
                            utils.calculate_complete_linkage_distance(dist_mat[cluster_i, j],
                                                                      dist_mat[cluster_j, j])
                    continue
                if i == cluster_j:
                    continue
                new_i = i if i < cluster_j else i - 1
                new_j = j if j < cluster_j else j - 1
                new_dist_mat[new_i, new_j] = dist_mat[i, j]

        return new_dist_mat


def test_ahc():
    test_data = [
        np.array([1, 2, 3, 4, 5, 6]),
        np.array([1, 2, 3, 4, 5]),
        np.array([2, 3, 4, 5]),
        np.array([6, 7]),
        np.array([1, 6, 7]),
        np.array([8, 9])
    ]
    test_array = [
        np.array([6]),
        np.array([12]),
        np.array([18]),
        np.array([22]),
        np.array([30]),
        np.array([42]),
        np.array([48]),
    ]

    ahc = AgglomerativeHierarchicalClustering(2, LinkageType.SINGLE, DistanceType.EUCLIDEAN)
    ahc.fit(test_array)
    clusters = ahc.get_clusters()
    print(clusters)
    # prediction = ahc.predict(np.array([2, 3, 4, 5, 6, 7]))
    # print(prediction)


if __name__ == '__main__':
    test_ahc()
