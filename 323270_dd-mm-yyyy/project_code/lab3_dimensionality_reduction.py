import numpy as np
from numpy.linalg import linalg
import scipy.linalg as scalg


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
    P = PCA_reduce(D, m)
    DP = project(D, P)

    return DP


def PCA_reduce(D, m=2):
    """
        Computes PCA with specified dimensions (default is 2)

        :param D: dataset to compute PCA on
        :param m: number of principal components to select
        :return: PCA projection matrix
        """
    mu = D.mean(axis=1)
    DC = D - vcol(mu)
    C = DC @ DC.T / DC.shape[1]

    s, U = linalg.eigh(C)
    P = U[:, ::-1][:, 0:m]

    return P


def covariances(D, L):
    """
    Computes between-class covariance and within-class covariance matrices

    :param D: dataset
    :param L: labels
    :return: computed matrices
    """
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


def reduce(SB, SW):
    """
    Computes LDA transformation matrix W, using joint diagonalization between
    covariance matrices SB and SW

    :param SB: between-class covariance matrix
    :param SW: within-class covariance matrix
    :return: LDA transformation matrix
    """
    # w1 proportional to w2 = w3
    # w1 = LDA_special_case(D, L, self.SW)
    # w2 = LDA_generalized_eigenvalue(self.SB, self.SW, self.m)
    W = W3 = LDA_joint_diag(SB, SW, 1)

    return W


def project(D, M):
    """
    Project data over basis spanned by columns of matrix M

    :param D: dataset
    :param M: transformation matrix
    :return: projected dataset
    """
    return M.T @ D


def split_db_2to1(D, L, seed=0):
    """
    Splits dataset and labels into two subsets:
    - training set and labels
    - validation set and labels
    Split is computed randomly, using an optional seed parameter

    :param D: dataset
    :param L: labels
    :param seed: random seed (default 0)
    :return: training set and labels, validation set and labels
    """
    nTrain = int(D.shape[1] * 2.0 / 3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]

    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]

    return (DTR, LTR), (DVAL, LVAL)


def thres_compute(DTR, LTR):
    """
    Computes threshold for LDA binary classification, using training set and its labels

    :param DTR: training set
    :param LTR: labels
    :return: training set class means, computed threshold
    """
    mu0 = DTR[0, LTR == 0].mean()
    mu1 = DTR[0, LTR == 1].mean()

    threshold = (mu0 + mu1) / 2

    return mu0, mu1, threshold


def assign_1_above(PVAL, DVAL, threshold):
    """
    Assigns labels to data, using validation set and threshold:
    label 1 is given to records above threshold

    :param PVAL: predicted validation set labels
    :param DVAL: validation set data
    :param threshold: discriminant threshold
    :return: None
    """
    PVAL[DVAL[0] >= threshold] = 1
    PVAL[DVAL[0] < threshold] = 0


def assign_1_below(PVAL, DVAL, threshold):
    """
    Assigns labels to data, using validation set and threshold:
    label 1 is given to records below -1 * threshold

    :param PVAL: predicted validation set labels
    :param DVAL: validation set data
    :param threshold: discriminant threshold
    :return: None
    """
    PVAL[DVAL[0] >= -threshold] = 0
    PVAL[DVAL[0] < -threshold] = 1


def predict(DVAL, LVAL, assign_function, threshold):
    """
    Computes predicted labels for validation set

    :param DVAL: validation set data
    :param LVAL: labels of validation set
    :param assign_function: assignment function
    :param threshold: threshold to use
    :return: predicted labels
    """
    PVAL = np.zeros(LVAL.shape, dtype=np.int32)
    assign_function(PVAL, DVAL, threshold)

    return PVAL


def error_rate(PVAL, LVAL):
    """
    Computes error rate

    :param PVAL: predicted labels
    :param LVAL: validation set labels
    :return: error rate
    """
    return np.sum(LVAL != PVAL) / LVAL.shape[0]


def LDA_estimate(D, L):
    """
    Estimates LDA transformation matrix, given dataset and labels

    :param D: dataset
    :param L: labels
    :return: LDA transformation matrix
    """
    SB, SW = covariances(D, L)
    W = reduce(SB, SW)

    return W


def LDA_apply(D, L):
    """
    Applies LDA transformation, given dataset and labels

    :param D: dataset
    :param L: labels
    :return: projected dataset
    """
    W = LDA_estimate(D, L)
    DP = project(D, W)

    return DP


def LDA_classify(D, L, m=None, PCA_enabled=False):
    """
    Performs LDA classification, given dataset and labels

    :param D: dataset
    :param L: labels
    :param m: PCA dimensions (default None): must be valid if PCA_enabled is True
    :param PCA_enabled: PCA used flag (default False): uses PCA before LDA if True
    :return: predicted labels, error rate, threshold used
    """
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    if PCA_enabled:
        P = PCA_reduce(DTR, m)

        DTR = project(DTR, P)
        DVAL = project(DVAL, P)

    W = LDA_estimate(DTR, LTR)
    DTR_lda = project(DTR, W)
    DVAL_lda = project(DVAL, W)

    mu0, mu1, threshold = thres_compute(DTR_lda, LTR)
    PVAL = predict(DVAL_lda, LVAL, assign_1_above if mu1 > mu0 else assign_1_below, threshold)

    err_rate = error_rate(PVAL, LVAL)

    return PVAL, err_rate, threshold


def LDA_classification_best_threshold(D, L):
    """
    Performs LDA classification without PCA, tracking error rate for each threshold value used

    :param D: dataset
    :param L: labels
    :return: error rate trend, computed both on all dataset domain and on reduced one
    """
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    W = LDA_estimate(DTR, LTR)
    DTR_lda = project(DTR, W)
    DVAL_lda = project(DVAL, W)

    mu0 = DTR_lda[0, LTR == 0].mean()
    mu1 = DTR_lda[0, LTR == 1].mean()
    err_rate_trend, err_rate_trend_reduced = error_rate_trend(DVAL_lda, LVAL,
                                                              assign_1_above if mu1 > mu0 else assign_1_below)

    return err_rate_trend, err_rate_trend_reduced


def LDA_classification_PCA(D, L):
    """
    Compute the LDA classification with PCA preprocessing, using different dimensions

    :param D: dataset
    :param L: labels
    :return: error rates, depending on the dimensionality of the PCA
    """

    dimensions = list(range(5, 1, -1))
    error_rates = []
    for m in dimensions:
        _, error_rate, _ = LDA_classify(D, L, m, True)
        error_rates.append(error_rate)

    return dimensions, error_rates


def error_rate_trend(DVAL_lda, LVAL, assign_function):
    """
    Computes error rate, given validation set, validation set labels and label assignment function

    :param DVAL_lda: dataset (after LDA processing)
    :param LVAL: labels of validation set
    :param assign_function: label assignment function
    :return: (x, y) pairs for error rate, (x, y) pairs for error rate on reduced domain
    """
    num_samples = int(1e5)
    th, er = np.zeros(num_samples), np.zeros(num_samples)
    min_value = DVAL_lda[0].min()
    max_value = DVAL_lda[0].max()
    values = np.linspace(min_value, max_value, num_samples)
    c = 0
    for t in values:
        threshold = t
        PVAL = predict(DVAL_lda, LVAL, assign_function, t)
        err_rate = error_rate(PVAL, LVAL)

        th[c] = threshold
        er[c] = err_rate

        c = c + 1

    mask = np.logical_and(th >= -0.3, th <= 0.3)
    red_th = th[mask]
    red_er = er[mask]

    return (th, er), (red_th, red_er)


def LDA_special_case(D, L, SW):
    """
    Computes LDA transformation matrix using binary LDA property

    :param D: dataset
    :param L: labels
    :param SW: within-class covariance matrix
    :return: transformation matrix (unit vector), with positive direction
    """
    mu0 = D[:, L == 0].mean(axis=1)
    mu1 = D[:, L == 1].mean(axis=1)

    w = linalg.inv(SW) @ vcol(mu1 - mu0)

    return w / linalg.norm(w, 2) * (1 if mu1 > mu0 else -1)


def LDA_generalized_eigenvalue(SB, SW, m):
    """
    Computes LDA transformation matrix by solving generalized eigenvalue problem

    :param SB: between-class covariance matrix
    :param SW: within-class covariance matrix
    :param m: LDA dimensions
    :return: LDA transformation matrix
    """
    s, U = scalg.eigh(SB, SW)
    W = U[:, ::-1][:, 0:m]

    return W


def LDA_joint_diag(SB, SW, m):
    """
    Computes LDA transformation matrix by using joint diagonalization

    :param SB: between-class covariance matrix
    :param SW: within-class covariance matrix
    :param m: LDA dimensions
    :return: LDA transformation matrix
    """
    U1, s1, _ = linalg.svd(SW)
    # s1, U1 = npalg.eigh(SW) same as above maybe
    P1 = U1 @ np.diag(1 / (s1 ** 0.5)) @ U1.T
    Sbt = P1 @ SB @ P1.T

    s2, U2 = linalg.eigh(Sbt)
    P2 = U2[:, ::-1][:, 0:m]
    W = P1.T @ P2

    return W
