import numpy as np


class KFold:
    def __init__(self, dataset, labels, K):
        self.dataset = dataset
        self.labels = labels
        self.K = K
        self.N = dataset.shape[1]
        self.result = None, None
        self.unfolded_LVAL = []

    def _shuffle(self):
        idx = np.random.permutation(range(0, self.N))
        return self.dataset[:, idx], self.labels[idx]

    def _split_K_fold(self, D, L):
        D_folds = []
        L_folds = []
        for i in range(self.K):
            D_folds.append(D[:, i::self.K])
            L_folds.append(L[i::self.K])

        self.unfolded_LVAL = np.concatenate(L_folds, axis=0)

        return D_folds, L_folds

    def _training_folds(self, d_folds, l_folds, index):
        d_tr = np.concatenate([d_folds[i] for i in range(self.K) if i != index], axis=1)
        l_tr = np.concatenate([l_folds[i] for i in range(self.K) if i != index], axis=0)
        return d_tr, l_tr

    def _validation_fold(self, d_folds, l_folds, index):
        assert 0 <= index < self.K

        d_val = d_folds[index]
        l_val = l_folds[index]
        return d_val, l_val

    def create_folds(self):
        return self._split_K_fold(self.dataset, self.labels)

    def split(self, d_folds, l_folds, index):
        return (self._training_folds(d_folds, l_folds, index),
                self._validation_fold(d_folds, l_folds, index))

    def pool(self, scores, labels):
        if self.result[0] is None:
            self.result = scores, self.result[1]
        else:
            self.result = np.concatenate([self.result[0], scores], axis=1), self.result[1]

        if self.result[1] is None:
            self.result = self.result[0], labels
        else:
            self.result = self.result[0], np.concatenate([self.result[1], labels], axis=1)

    def get_results(self):
        res = self.result
        self.result = None, None
        return res

    def get_real_labels(self):
        labels = self.unfolded_LVAL
        self.unfolded_LVAL = None
        return labels