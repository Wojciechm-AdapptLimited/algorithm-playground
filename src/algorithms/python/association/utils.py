import numpy as np


def get_support(candidate_set: np.ndarray, basket: np.ndarray) -> np.ndarray:
    """
    Get support of each candidate in the candidate set

    :param candidate_set: NumPy array of shape (n, m), where n is the number of candidates and m is the current level,
    and each row is a candidate containing the indices of the items in the item set
    :param basket: NumPy array of shape (o, p) where o is the number of transactions and p is the number of possible
    items
    :return: NumPy array of shape (n, 1) containing the support of each candidate
    """
    support = np.ones((basket.shape[0], candidate_set.shape[0]))

    for i in range(candidate_set.shape[1]):
        support *= basket[:, candidate_set[:, i]]

    return np.sum(support, axis=0)/basket.shape[0]
