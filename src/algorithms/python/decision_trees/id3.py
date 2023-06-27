import numpy as np
from utils import Node, calculate_information_gain_ratio


class ID3Classifier:
    """
    Decision Tree Classifier using ID3 Algorithm (Iterative Dichotomiser 3) for classification. The ID3 algorithm builds
    decision trees using a top-down greedy search approach through the space of possible branches with no backtracking.
    ID3 uses entropy to calculate the homogeneity of a sample. Sample is split into subsets to reduce the entropy. The
    resulting entropy is subtracted from the entropy before the split to calculate the information gain. ID3 algorithm
    finishes when all data is classified or there the max depth is reached.
    """
    max_depth: int

    def __init__(self, max_depth: int):
        """
        Initialize the ID3 algorithm

        :param max_depth: Max depth of the decision tree
        """
        self.max_depth = max_depth

    def fit(self, attributes: np.ndarray, features: np.ndarray, target: np.ndarray) -> Node:
        """
        Fit the decision tree classifier according to the given training data.

        :param attributes: NumPy array of attribute names
        :param features: NumPy array of feature values
        :param target: NumPy array of target values
        :return: Root node of the decision tree
        """
        return self._fit(attributes, features, target, 0)

    def _fit(self, attributes: np.ndarray, features: np.ndarray, target: np.ndarray, depth: int) -> Node:
        """
        Fit the decision tree classifier recursively according to the given training data.

        :param features: NumPy array of feature values
        :param target: NumPy array of target values
        :param depth: Current depth of the decision tree
        :return: Node of the decision tree
        """
        if depth == self.max_depth:
            return Node(np.unique(target)[np.argmax(np.unique(target, return_counts=True)[1])])

        if np.unique(target).size == 1:
            return Node(target[0])

        information_gains = np.array([
            calculate_information_gain_ratio(features[:, i], target)
            for i in range(features.shape[1])
        ])

        best_feature = np.argmax(information_gains)
        unique_values = np.unique(features[:, best_feature])

        node = Node(f'{attributes[best_feature]}')

        for value in unique_values:
            node.children[value] = self._fit(
                attributes,
                features[features[:, best_feature] == value],
                target[features[:, best_feature] == value],
                depth + 1
            )

        return node


def test_id3():
    attributes = np.array(['Outlook', 'Temperature', 'Humidity', 'Wind'])
    data = np.array([
        ['Sunny', 'Hot', 'High', 'Weak', 'No'],
        ['Sunny', 'Hot', 'High', 'Strong', 'No'],
        ['Overcast', 'Hot', 'High', 'Weak', 'Yes'],
        ['Rain', 'Mild', 'High', 'Weak', 'Yes'],
        ['Rain', 'Cool', 'Normal', 'Weak', 'Yes'],
        ['Rain', 'Cool', 'Normal', 'Strong', 'No'],
        ['Overcast', 'Cool', 'Normal', 'Strong', 'Yes'],
        ['Sunny', 'Mild', 'High', 'Weak', 'No'],
        ['Sunny', 'Cool', 'Normal', 'Weak', 'Yes'],
        ['Rain', 'Mild', 'Normal', 'Weak', 'Yes'],
        ['Sunny', 'Mild', 'Normal', 'Strong', 'Yes'],
        ['Overcast', 'Mild', 'High', 'Strong', 'Yes'],
        ['Overcast', 'Hot', 'Normal', 'Weak', 'Yes'],
        ['Rain', 'Mild', 'High', 'Strong', 'No']
    ])

    id3 = ID3Classifier(max_depth=5)
    tree = id3.fit(attributes, data[:, :-1], data[:, -1])
    print(tree)


test_id3()
