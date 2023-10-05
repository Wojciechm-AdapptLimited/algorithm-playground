import numpy as np
import logging

from utils import get_support

logging.basicConfig(level=logging.INFO)


class Apriori:
    """
    Apriori algorithm for association rule mining in data mining. Apriori is an algorithm for frequent item set mining
    and association rule learning over transactional databases. It proceeds by identifying the frequent individual items
    in the database and extending them to larger and larger item sets as long as those item sets appear sufficiently
    often in the database. The frequent item sets determined by Apriori can be used to determine association rules which
    highlight general trends in the database: this has applications in domains such as market basket analysis.

    Apriori uses a "bottom up" approach, where frequent subsets are extended one item at a time (a step known as
    candidate generation), and groups of candidates are tested against the data. The algorithm terminates when
    no further successful extensions are found.
    """
    min_support: float
    min_confidence: float

    def __init__(self, min_support: float, min_confidence: float):
        """
        Initialize Apriori algorithm

        :param min_support: Minimum support for a candidate to be considered frequent
        :param min_confidence: Minimum confidence for a rule to be considered strong
        """
        self.min_support = min_support
        self.min_confidence = min_confidence

    def find_frequent_item_sets(self, basket: np.ndarray) -> list[np.ndarray]:
        """
        Find frequent item sets from a basket

        :param basket: NumPy array of shape (n, m) where n is the number of transactions and m is the number of possible
            items
        :return: List of frequent item sets (NumPy arrays)
        """
        logging.info('Finding frequent item sets')

        candidates: np.ndarray = np.arange(basket.shape[1]).reshape(-1, 1)
        frequent_item_sets: list[np.ndarray] = []
        pruned_candidates: np.ndarray = self._prune_candidates(candidates, basket)

        logging.info('LEVEL 1')
        logging.info(f'Candidates:\n {candidates}')
        logging.info(f'Pruned candidates:\n {pruned_candidates}')
        logging.info(f'{pruned_candidates.shape[0]} frequent item sets')

        level: int = 1
        while pruned_candidates.size > 0:
            level += 1

            logging.info(f'LEVEL {level}')
            frequent_item_sets.extend([*pruned_candidates])
            candidates = self._generate_candidates(pruned_candidates)
            logging.info(f'Candidates:\n {candidates}')
            pruned_candidates = self._prune_candidates(candidates, basket)
            logging.info(f'Pruned candidates:\n {pruned_candidates}')

        logging.info(f'Found {len(frequent_item_sets)} frequent item sets')
        return frequent_item_sets

    def find_closed_frequent_item_sets(self, basket: np.ndarray) -> list[np.ndarray]:
        """
        Find closed frequent item sets from a basket

        :param basket: NumPy array of shape (n, m) where n is the number of transactions and m is the number of possible
            items
        :return: List of closed frequent item sets (NumPy arrays)
        """
        frequent_item_sets: list[np.ndarray] = self.find_frequent_item_sets(basket)
        closed_frequent_item_sets: list[np.ndarray] = []
        supports: list[np.ndarray] = [get_support(frequent_set[np.newaxis, :], basket)[0]
                                      for frequent_set in frequent_item_sets]

        logging.info('Finding closed frequent item sets')
        for idx, frequent_set in enumerate(frequent_item_sets):
            logging.info(f'Checking if {frequent_set} is closed')
            logging.info(f'Support: {supports[idx]}')

            is_closed: bool = True
            for other_idx, other_frequent_set in enumerate(frequent_item_sets):
                if np.array_equal(frequent_set, other_frequent_set) \
                        or not (np.isin(frequent_set, other_frequent_set).all()):
                    continue
                if not np.isin(frequent_set, other_frequent_set).all():
                    continue

                logging.info(f'Checking if {frequent_set} is closed with {other_frequent_set}')

                if supports[idx] <= supports[other_idx]:
                    logging.info(f'{frequent_set} is not closed')
                    is_closed = False
                    break
            if is_closed:
                logging.info(f'{frequent_set} is closed')
                closed_frequent_item_sets.append(frequent_set)

        logging.info(f'Found {len(closed_frequent_item_sets)} closed frequent item sets')
        return closed_frequent_item_sets

    def find_maximal_frequent_item_sets(self, basket: np.ndarray) -> list[np.ndarray]:
        """
        Find maximal frequent item sets from a basket

        :param basket: NumPy array of shape (n, m) where n is the number of transactions and m is the number of possible
            items
        :return: List of maximal frequent item sets (NumPy arrays)
        """
        frequent_item_sets: list[np.ndarray] = self.find_frequent_item_sets(basket)
        maximal_frequent_item_sets: list[np.ndarray] = []

        logging.info('Finding maximal frequent item sets')
        for frequent_set in frequent_item_sets:
            is_maximal: bool = True
            for other_frequent_set in frequent_item_sets:
                if frequent_set.size >= other_frequent_set.size:
                    continue
                if np.array_equal(frequent_set, other_frequent_set):
                    continue

                logging.info(f'Checking if {frequent_set} is maximal with {other_frequent_set}')

                if np.isin(frequent_set, other_frequent_set).all():
                    logging.info(f'{frequent_set} is not maximal')
                    is_maximal = False
                    break

            if is_maximal:
                logging.info(f'{frequent_set} is maximal')
                maximal_frequent_item_sets.append(frequent_set)

        logging.info(f'Found {len(maximal_frequent_item_sets)} maximal frequent item sets')
        return maximal_frequent_item_sets

    def find_association_rules(self, basket: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Find association rules from a basket

        :param basket: NumPy array of shape (n, m) where n is the number of transactions and m is the number of possible
            items
        :return: List of association rules where each rule is a tuple of antecedent and consequent
        """
        frequent_item_sets: list[np.ndarray] = self.find_frequent_item_sets(basket)
        association_rules: list[tuple[np.ndarray, np.ndarray]] = []
        supports: list[np.ndarray] = [get_support(frequent_set[np.newaxis, :], basket)[0]
                                      for frequent_set in frequent_item_sets]

        logging.info('Finding association rules')
        for idx, frequent_set in enumerate(frequent_item_sets):
            if frequent_set.size == 1:
                continue
            for other_idx, other_frequent_set in enumerate(frequent_item_sets):
                if other_frequent_set.size >= frequent_set.size:
                    continue
                if np.array_equal(frequent_set, other_frequent_set):
                    continue
                if not np.isin(other_frequent_set, frequent_set).all():
                    continue

                confidence = supports[idx] / supports[other_idx]

                if confidence < self.min_confidence:
                    continue

                logging.info(f'Found association rule {other_frequent_set} -> '
                             f'{np.setdiff1d(frequent_set, other_frequent_set)} with confidence {confidence}')
                association_rules.append((other_frequent_set, np.setdiff1d(frequent_set, other_frequent_set)))

        logging.info(f'Found {len(association_rules)} association rules')
        return association_rules

    def _prune_candidates(self, candidates: np.ndarray, basket: np.ndarray) -> np.ndarray:
        """
        Prune candidates that are not frequent

        :param candidates: NumPy array of shape (n, m) where n is the number of candidates and m is the current level,
            and each row is a candidate containing the indices of the items in the item set
        :param basket: NumPy array of shape (o, p) where o is the number of transactions and p is the number of possible
            items
        :return: NumPy array of shape (r, m) where n is the number of pruned candidates and m is the current level
        """
        support: np.ndarray = get_support(candidates, basket)
        logging.info(f'Support:\n {support[:, np.newaxis]}')
        return candidates[support >= self.min_support]

    @staticmethod
    def _generate_candidates(pruned_candidates: np.ndarray) -> np.ndarray:
        """
        Generate candidates from pruned candidates

        :param pruned_candidates: NumPy array of shape (n, m) where n is the number of pruned candidates and m is the
            current level
        :return:
        """
        candidates: np.ndarray = np.empty((0, pruned_candidates.shape[1] + 1), dtype=np.int64)
        for i in range(pruned_candidates.shape[0]):
            for j in range(i + 1, pruned_candidates.shape[0]):
                prefix_1: np.ndarray = pruned_candidates[i, :-1]
                prefix_2: np.ndarray = pruned_candidates[j, :-1]
                if np.array_equal(prefix_1, prefix_2):
                    new_candidate: np.ndarray = np.append(pruned_candidates[i, :], pruned_candidates[j, -1])
                    candidates: np.ndarray = np.concatenate((candidates, new_candidate[np.newaxis, :]), axis=0)
        return candidates


def test_apriori():
    test_basket = np.array([
        [0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0]

    ])
    apriori = Apriori(0.3, 0.75)

    frequent_item_sets = apriori.find_frequent_item_sets(test_basket)
    closed_frequent_item_sets = apriori.find_closed_frequent_item_sets(test_basket)
    maximal_frequent_item_sets = apriori.find_maximal_frequent_item_sets(test_basket)
    association_rules = apriori.find_association_rules(test_basket)

    print('Frequent item sets')
    for frequent_item_set in frequent_item_sets:
        print(frequent_item_set)

    print()
    print('Closed frequent item sets')
    for closed_frequent_item_set in closed_frequent_item_sets:
        print(closed_frequent_item_set)

    print()
    print('Maximal frequent item sets')
    for maximal_frequent_item_set in maximal_frequent_item_sets:
        print(maximal_frequent_item_set)

    print()
    print('Association rules')
    for association_rule in association_rules:
        print(f'{association_rule[0]} -> {association_rule[1]}')


if __name__ == '__main__':
    test_apriori()
