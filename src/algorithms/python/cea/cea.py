import logging
import numpy as np

logging.basicConfig(format='%(process)d-%(levelname)s-%(message)s', level=logging.INFO)


class CEAClassifier:
    """
    Implementation of the Candidate Elimination Algorithm.

    The Candidate Elimination Algorithm (CEA) is a method for learning a hypothesis space H from training examples.
    The hypothesis space H is represented by two sets: the set of strict hypotheses S and the set of general hypotheses
    G (S ⊆ G). The algorithm starts with the most general hypothesis in H and the most specific hypothesis in H. Then,
    it iterates over the training examples and removes inconsistent hypotheses from the hypothesis space. If the example
    is positive, the algorithm generalizes the strict hypothesis set and removes inconsistent hypotheses from the
    general hypothesis set. If the example is negative, the algorithm specializes the general hypothesis set and removes
    inconsistent hypotheses from the strict hypothesis set. The algorithm stops when the hypothesis space is empty or
    when all training examples have been processed.

    References:
        - Mitchell, T. M. (1997). Machine learning. McGraw Hill.
    """
    positive_label: str

    def __init__(self, positive_label: str = 'yes') -> None:
        """
        Initialize the CEA algorithm.

        :param positive_label: String representing the positive label.
        """
        self.positive_label = positive_label

    def fit(self, concepts: np.ndarray, targets: np.ndarray) -> \
            tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Train the CEA algorithm.

        :param concepts: NumPy array of training examples.
        :param targets: NumPy array of training labels.
        :return: Tuple containing the strict and general hypothesis sets.
        :raises ValueError: if the number of concepts and targets is different.
        """
        if len(concepts) != len(targets):
            raise ValueError('Number of concepts and targets must be equal.')

        logging.info('Starting CEA algorithm.')
        logging.info(f'Concepts: {concepts}')
        logging.info(f'Targets: {targets}')

        # Initialize the strict and general hypothesis sets.
        strict_hypothesis_set: list[np.ndarray] = [np.array(concepts[0])]
        general_hypothesis_set: list[np.ndarray] = [np.full_like(concepts[0], '?')]

        # Iterate over the training examples.
        for example_idx, (concept, target) in enumerate(zip(concepts, targets)):
            logging.info(f'Example {example_idx + 1}: {concept} -> {target}')
            if target == self.positive_label:
                # If example is positive, remove inconsistent hypotheses from the hypothesis set and generalize the
                # strict hypothesis set.
                general_hypothesis_set = self._remove_inconsistent(general_hypothesis_set, concept, True)
                strict_hypothesis_set = self._generalize(strict_hypothesis_set, concept)
            else:
                # If example is negative, remove inconsistent hypotheses from the hypothesis set and specialize the
                # general hypothesis set
                strict_hypothesis_set = self._remove_inconsistent(strict_hypothesis_set, concept, False)
                general_hypothesis_set = self._specialize(strict_hypothesis_set, general_hypothesis_set, concept)

            if not strict_hypothesis_set or not general_hypothesis_set:
                # If the hypothesis space is empty, stop the algorithm.
                logging.info('Hypothesis space is empty.')
                break

        logging.info('Finished CEA algorithm.')
        logging.info(f'Strict hypothesis set: {strict_hypothesis_set}')
        logging.info(f'General hypothesis set: {general_hypothesis_set}')
        return strict_hypothesis_set, general_hypothesis_set

    @staticmethod
    def _remove_inconsistent(hypothesis_set: list[np.ndarray], example: np.ndarray, positive: bool) \
            -> list[np.ndarray]:
        """
        Remove inconsistent hypotheses from the hypothesis set.

        :param hypothesis_set: List of NumPy arrays representing the hypothesis set.
        :param example: NumPy array representing a training example.
        :param positive: Boolean representing whether the example is positive or negative.
        :return: List of NumPy arrays representing consistent hypotheses.
        """
        # Initialize the consistent hypothesis set.
        consistent_hypothesis_set: list[np.ndarray] = []

        logging.info(f'Starting to remove inconsistent hypotheses from hypothesis set: {hypothesis_set}')

        # Iterate over the hypotheses.
        for hypothesis in hypothesis_set:
            # If example is positive and any attribute of the hypothesis is not '?'
            # and different from the corresponding example attribute, this hypothesis is inconsistent.
            if positive and np.any((hypothesis != '?') & (hypothesis != example)):
                logging.info(f'Hypothesis {hypothesis} is inconsistent.')
                continue

            # If example is negative and all attributes of the hypothesis are equal to the corresponding example
            # attributes, this hypothesis is inconsistent.
            if not positive and np.all(hypothesis == example):
                logging.info(f'Hypothesis {hypothesis} is inconsistent.')
                continue

            # If the hypothesis is consistent, add it to the consistent hypothesis set.
            logging.info(f'Hypothesis {hypothesis} is consistent.')
            consistent_hypothesis_set.append(hypothesis)

        logging.info(f'Consistent hypothesis set: {consistent_hypothesis_set}')
        return consistent_hypothesis_set

    @staticmethod
    def _generalize(hypothesis_set: list[np.ndarray], example: np.ndarray) -> list[np.ndarray]:
        """
        Generalize the hypothesis set.

        :param hypothesis_set: List of NumPy arrays representing hypotheses.
        :param example: NumPy array representing a training example.
        :return: List of NumPy arrays representing generalized hypotheses.
        """
        # Initialize the generalized hypothesis set.
        generalized_hypothesis_set: list[np.ndarray] = []
        empty_hypothesis: np.ndarray = np.full_like(example, '?')

        logging.info(f'Starting to generalize hypothesis set: {hypothesis_set}')

        # Iterate over the hypotheses.
        for hypothesis in hypothesis_set:
            # For each attribute of the hypothesis, if the attribute is not '?' and different from the corresponding
            # example attribute, set it to '?'.
            generalized_hypothesis = np.where(hypothesis != example, empty_hypothesis, hypothesis)

            # Add the generalized hypothesis to the generalized hypothesis set.
            logging.info(f'Hypothesis: {hypothesis}')
            logging.info(f'Generalized hypothesis: {generalized_hypothesis}')
            generalized_hypothesis_set.append(generalized_hypothesis)

        logging.info(f'Generalized hypothesis set: {generalized_hypothesis_set}')
        return generalized_hypothesis_set

    @staticmethod
    def _specialize(strict_hypothesis_set: list[np.ndarray], general_hypothesis_set: list[np.ndarray],
                    example: np.ndarray) -> list[np.ndarray]:
        """
        Specialize the general hypothesis set.

        :param strict_hypothesis_set: List of NumPy arrays representing strict hypotheses.
        :param general_hypothesis_set: List of NumPy arrays representing general hypotheses.
        :param example: NumPy array representing a training example.
        :return: List of NumPy arrays representing specialized hypotheses.
        """
        # Initialize the specialized hypothesis set.
        specialized_hypothesis_set: list[np.ndarray] = []

        logging.info(f'Starting to specialize hypothesis set: {general_hypothesis_set}')

        # Iterate over the general hypotheses.
        for hypothesis in general_hypothesis_set:
            # Iterate over the attributes of the hypothesis.
            logging.info(f'Hypothesis: {hypothesis}')
            for strict_hypothesis in strict_hypothesis_set:
                for idx, (strict_hypothesis_attribute, example_attribute) in enumerate(zip(strict_hypothesis, example)):
                    # If the attribute of the strict hypothesis is '?' or to the corresponding example
                    # attribute, skip it.
                    if strict_hypothesis_attribute == example_attribute or strict_hypothesis_attribute == '?':
                        continue
                    # Copy the hypothesis and set the attribute to the corresponding strict hypothesis attribute.
                    specialized_hypothesis = np.array(hypothesis)
                    specialized_hypothesis[idx] = strict_hypothesis_attribute

                    # Add the specialized hypothesis to the specialized hypothesis set.
                    logging.info(f'Specialized hypothesis: {specialized_hypothesis}')
                    specialized_hypothesis_set.append(specialized_hypothesis)

        logging.info(f'Specialized hypothesis set: {specialized_hypothesis_set}')
        return specialized_hypothesis_set


def test_cea() -> None:
    """
    Test the CEA algorithm.
    """
    # Define the training examples and labels.
    concepts = np.array([
        ['sunny', 'warm', 'normal', 'strong', 'warm', 'same'],
        ['sunny', 'warm', 'high', 'strong', 'warm', 'same'],
        ['rainy', 'cold', 'high', 'strong', 'warm', 'change'],
        ['sunny', 'warm', 'high', 'strong', 'cool', 'change']
    ])
    targets = np.array(['yes', 'yes', 'no', 'yes'])
    # concepts = np.array([
    #     ['big', 'red', 'circle'],
    #     ['small', 'red', 'square'],
    #     ['small', 'red', 'circle'],
    #     ['big', 'blue', 'circle']
    # ])
    # targets = np.array(['yes', 'no', 'yes', 'no'])

    # Run the CEA algorithm.
    cea = CEA()
    _ = cea.fit(concepts, targets)


test_cea()
