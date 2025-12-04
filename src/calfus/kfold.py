"""
    K-Fold Cross-Validation module.
    
    This module defines the KFold class for splitting datasets into k folds
    for cross-validation tasks.
"""

import numpy as np


class KFold:
    """
    Class for k-fold cross-validation splitting of a dataset.
    
    Attributes:
        dataset (np.ndarray): The dataset to be split.
        labels (np.ndarray): The true labels corresponding to the dataset.
        K (int): The number of folds.
        N (int): The number of samples in the dataset.
    """
    def __init__(self, dataset, labels, K):
        self.dataset = dataset
        self.labels = labels
        self.K = K
        self.N = dataset.shape[1]

        self._shuffle()

    def _shuffle(self):
        """Shuffle the dataset and labels in unison."""
        np.random.seed(0)
        idx = np.random.permutation(range(0, self.N))
        self.dataset, self.labels = self.dataset[:, idx], self.labels[idx]

    def _training_folds(self, index):
        """Get the training folds excluding the fold at the given index."""
        d_tr = np.hstack([self.dataset[:, i::self.K] for i in range(self.K) if i != index])
        l_tr = np.hstack([self.labels[i::self.K] for i in range(self.K) if i != index])
        return d_tr, l_tr

    def _validation_fold(self, index):
        """Get the validation fold at the given index."""
        assert 0 <= index < self.K

        d_val = self.dataset[:, index::self.K]
        l_val = self.labels[index::self.K]
        return d_val, l_val

    def split(self, index):
        """Split the dataset into training and validation folds for the given index."""
        return (
            self._training_folds(index),
            self._validation_fold(index)
        )