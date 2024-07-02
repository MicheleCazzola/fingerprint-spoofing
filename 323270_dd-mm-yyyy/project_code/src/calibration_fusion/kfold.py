import numpy as np


class KFold:
    def __init__(self, dataset, labels, K):
        self.dataset = dataset
        self.labels = labels
        self.K = K
        self.N = dataset.shape[1]
        self.result = None, None
        self.unfolded_LVAL = []

        self._shuffle()

    def _shuffle(self):
        np.random.seed(0)
        idx = np.random.permutation(range(0, self.N))
        self.dataset, self.labels = self.dataset[:, idx], self.labels[idx]

    def _training_folds(self, index):
        d_tr = np.hstack([self.dataset[:, i::self.K] for i in range(self.K) if i != index])
        l_tr = np.hstack([self.labels[i::self.K] for i in range(self.K) if i != index])
        return d_tr, l_tr

    def _validation_fold(self, index):
        assert 0 <= index < self.K

        d_val = self.dataset[:, index::self.K]
        l_val = self.labels[index::self.K]
        return d_val, l_val

    def split(self, index):
        return (self._training_folds(index),
                self._validation_fold(index))