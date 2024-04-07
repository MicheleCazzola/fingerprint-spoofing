import numpy as np
from lab2_loading_plots import print_hist
from numpy.linalg import linalg
import scipy.linalg as scalg


PLOT_PATH_PCA = "output\\plots\\PCA_features\\"
PLOT_PATH_LDA = "output\\plots\\LDA_features\\"

def vcol(array):
    """
    Converts a 1D-ndarray into a column 2D-ndarray

    :param array: 1D-ndarray
    :return: column 2D-ndarray
    """
    return array.reshape(array.size, 1)


def rcol(array):
    """
    Converts a 1D-ndarray into a row 2D-ndarray

    :param array: 1D-ndarray
    :return: row 2D-ndarray
    """
    return array.reshape(1, array.size)


def PCA(D, m=2):
    """
    Computes PCA with specified dimensions (default is 2)

    :param D: dataset to compute PCA on
    :param m: number of principal components to select
    :return: dataset projected over selected PCA components
    """
    mu = D.mean(axis=1)
    DC = D - vcol(mu)
    C = DC @ DC.T / DC.shape[1]

    s, U = linalg.eigh(C)
    P = U[:, ::-1][:, 0:m]

    DP = P.T @ D

    return DP


def LDA_matrices(D, L):
    mu = D.mean(axis=1)
    Sb = np.zeros((D.shape[0], D.shape[0]))

    for c in set(L.tolist()):
        nc = D[:, L == c].shape[1]
        mu_class = D[:, L == c].mean(axis=1)
        mu_centered = vcol(mu_class) - vcol(mu)
        SBc = nc * mu_centered @ mu_centered.T
        Sb += SBc
    Sb /= D.shape[1]

    Sw = np.zeros((D.shape[0], D.shape[0]))
    for c in set(L.tolist()):
        mu_class = D[:, L == c].mean(axis=1)
        DC = D[:, L == c] - vcol(mu_class)
        SWc = DC @ DC.T
        Sw += SWc
    Sw /= D.shape[1]

    return Sb, Sw


class LDA:
    def __init__(self, D, L, m):
        self.D = D
        self.L = L
        self.m = m
        self.W = self.apply(self.D, self.L)

    def apply(self, D, L):
        SB, SW = LDA_matrices(D, L)
        # w1 proportional to w2 = w3
        #w1 = LDA_special_case(D, L, self.SW)
        #w2 = LDA_generalized_eigenvalue(self.SB, self.SW, self.m)
        w = w3 = LDA_joint_diag(SB, SW, self.m)

        return w

    def projection(self, D, W):
        return W.T @ D

    def classification(self):
        (DTR, LTR), (DVAL, LVAL)  = split_db_2to1(self.D, self.L)

        W = -self.apply(DTR, LTR)

        DTR_lda = W.T @ DTR
        DVAL_lda = W.T @ DVAL

        mu0 = DTR_lda[0, LTR == 0].mean()
        mu1 = DTR_lda[0, LTR == 1].mean()

        threshold = (mu0 + mu1) / 2
        PVAL = np.zeros(LVAL.shape, dtype=np.int32)
        PVAL[DVAL_lda[0] >= threshold] = 1
        PVAL[DVAL_lda[0] < threshold] = 0

        error_rate = np.sum(LVAL != PVAL) / LVAL.shape
        return PVAL, LVAL, error_rate, threshold

    def classification_best_threshold(self):
        (DTR, LTR), (DVAL, LVAL) = split_db_2to1(self.D, self.L)

        W = -self.apply(DTR, LTR)

        DTR_lda = W.T @ DTR
        DVAL_lda = W.T @ DVAL

        mu0 = DTR_lda[0, LTR == 0].mean()
        mu1 = DTR_lda[0, LTR == 1].mean()

        error_rate_trend_th = error_rate_trend(DVAL_lda, LVAL, mu0, mu1)

        return error_rate_trend_th


def error_rate_trend(DVAL_lda, LVAL, mu0, mu1):

    th, er = [], []
    min_value = -0.25
    max_value = 0.25
    num_samples = int(1e5)
    values = np.linspace(min_value, max_value, num_samples)
    for t in values.tolist():
        threshold = t
        PVAL = np.zeros(LVAL.shape, dtype=np.int32)
        PVAL[DVAL_lda[0] >= threshold] = 1
        PVAL[DVAL_lda[0] < threshold] = 0

        error_rate = np.sum(LVAL != PVAL) / LVAL.shape
        th.append(threshold)
        er.append(error_rate)

    return th, er


def LDA_projection(D, L, SW, SB, m):
    # w1 proportional to w2 = w3
    w1 = LDA_special_case(D, L, SW)
    w2 = LDA_generalized_eigenvalue(SB, SW, m)
    w = w3 = LDA_joint_diag(SB, SW, m)

    DP = w.T @ D

    return DP


def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]

    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]

    return (DTR, LTR), (DVAL, LVAL)


def LDA_special_case(D, L, SW):
    mu1 = D[:, L == 0].mean(axis=1)
    mu2 = D[:, L == 1].mean(axis=1)

    w = linalg.inv(SW) @ vcol(mu2 - mu1)

    return -w


def LDA_generalized_eigenvalue(SB, SW: np.ndarray, m):
    s, U = scalg.eigh(SB, SW)
    W = U[:, ::-1][:,0:m]

    return W


def LDA_joint_diag(SB, SW, m):
    U1, s1, _ = linalg.svd(SW)
    # s1, U1 = npalg.eigh(SW) same as above maybe
    P1 = U1 @ np.diag(1 / (s1 ** 0.5)) @ U1.T
    Sbt = P1 @ SB @ P1.T

    s2, U2 = linalg.eigh(Sbt)
    P2 = U2[:, ::-1][:, 0:m]
    W = -P1.T @ P2

    return W