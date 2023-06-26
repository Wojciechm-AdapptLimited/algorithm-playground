from __future__ import annotations

import numpy as np


class Node:
    value: str
    children: dict[str, Node]

    def __init__(self, value: str) -> None:
        self.value = value
        self.children = {}

    def __repr__(self) -> str:
        return self.value

    def __eq__(self, other: Node) -> bool:
        return self.value == other.value


def calculate_entropy(target: np.ndarray) -> np.ndarray:
    """
    Calculates the entropy of a given target array.

    :param target: NumPy array of target values
    :return: Entropy of the target array
    """
    counts: np.ndarray = np.unique(target, return_counts=True)[1]

    entropy: np.ndarray = np.sum(
        -counts / target.size * np.log2(counts / target.size, out=np.zeros_like(counts, dtype=float), where=counts != 0)
    )

    return entropy


def calculate_information_gain(features: np.ndarray, target: np.ndarray) -> float:
    """
    Calculates the information gain of a given feature array.

    :param features: NumPy array of feature values
    :param target: NumPy array of target values
    :return: Information gain of the feature array
    """
    counts: np.ndarray
    unique_features: np.ndarray
    unique_features, counts = np.unique(features, return_counts=True)

    total_entropy: np.ndarray = calculate_entropy(target)
    weighted_entropy: np.ndarray = np.sum([
        count / features.size * calculate_entropy(target[features == feature])
        for (feature, count) in zip(unique_features, counts)
    ])

    return total_entropy - weighted_entropy


def calculate_information_gain_ratio(features: np.ndarray, target: np.ndarray) -> float:
    """
    Calculates the information gain ratio of a given feature array.

    :param features: NumPy array of feature values
    :param target: NumPy array of target values
    :return: Information gain ratio of the feature array
    """
    counts: np.ndarray
    unique_features: np.ndarray
    unique_features, counts = np.unique(features, return_counts=True)

    intrinsic_value: np.ndarray = np.sum([
        -count / features.size
        * np.log2(count / features.size, out=np.zeros_like(counts, dtype=float), where=counts != 0)
        for count in counts
    ])

    return calculate_information_gain(features, target) / intrinsic_value
