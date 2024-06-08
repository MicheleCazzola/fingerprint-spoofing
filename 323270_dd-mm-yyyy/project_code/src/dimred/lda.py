import numpy as np
from numpy.linalg import linalg
from . import pca_old
from src.utilities.utilities import vcol, split_db_2to1, project


def covariances(D, L):
    """
    Computes between-class covariance and within-class covariance matrices

    :param D: dataset
    :param L: labels
    :return: computed matrices
    """
    mu = D.mean(axis=1)
    SB = np.zeros((D.shape[0], D.shape[0]))
    SW = np.zeros((D.shape[0], D.shape[0]))
    for label in np.unique(L):
        D_label = D[:, L == label]
        mu_label = D_label.mean(axis=1)
        nc_label = D_label.shape[1]
        SB += nc_label * vcol(mu_label - mu) @ vcol(mu_label - mu).T
        SW += (D_label - vcol(mu_label)) @ (D_label - vcol(mu_label)).T

    return SB / D.shape[1], SW / D.shape[1]


def transform(SB, SW):
    """
    Computes LDA transformation matrix W, using joint diagonalization between
    covariance matrices SB and SW

    :param SB: between-class covariance matrix
    :param SW: within-class covariance matrix
    :return: LDA transformation matrix
    """
    W = joint_diag(SB, SW, 1)

    return W


def threshold_compute(DTR, LTR):
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
    PVAL[DVAL[0] >= threshold] = 0
    PVAL[DVAL[0] < threshold] = 1


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


def estimate(D, L):
    """
    Estimates LDA transformation matrix, given dataset and labels

    :param D: dataset
    :param L: labels
    :return: LDA transformation matrix
    """
    SB, SW = covariances(D, L)
    W = transform(SB, SW)

    return W


def apply(D, L):
    """
    Applies LDA transformation, given dataset and labels

    :param D: dataset
    :param L: labels
    :return: projected dataset
    """
    W = estimate(D, L)
    #W = -estimate(D, L)
    DP = project(D, W)

    return DP


def classify(D, L, m=None, PCA_enabled=False):
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
        P = pca.reduce(DTR, m)

        DTR = project(DTR, P)
        DVAL = project(DVAL, P)

    W = estimate(DTR, LTR)
    DTR_lda = project(DTR, W)
    DVAL_lda = project(DVAL, W)

    mu0, mu1, threshold = threshold_compute(DTR_lda, LTR)
    PVAL = predict(DVAL_lda, LVAL, assign_1_above if mu1 > mu0 else assign_1_below, threshold)

    err_rate = error_rate(PVAL, LVAL)

    return PVAL, err_rate, threshold


def classify_best_threshold(D, L):
    """
    Performs LDA classification without PCA, tracking error rate for each threshold value used

    :param D: dataset
    :param L: labels
    :return: error rate trend, computed both on all dataset domain and on reduced one
    """
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    W = estimate(DTR, LTR)
    DTR_lda = project(DTR, W)
    DVAL_lda = project(DVAL, W)

    mu0 = DTR_lda[0, LTR == 0].mean()
    mu1 = DTR_lda[0, LTR == 1].mean()
    err_rate_trend, err_rate_trend_reduced = error_rate_trend(DVAL_lda, LVAL,
                                                              assign_1_above if mu1 > mu0 else assign_1_below)

    return err_rate_trend, err_rate_trend_reduced


def classify_PCA_preprocess(D, L):
    """
    Compute the LDA classification with PCA preprocessing, using different dimensions

    :param D: dataset
    :param L: labels
    :return: error rates, depending on the dimensionality of the PCA
    """

    dimensions = list(range(5, 1, -1))
    error_rates = []
    for m in dimensions:
        _, error_rate, _ = classify(D, L, m, True)
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


def joint_diag(SB, SW, m):
    """
    Computes LDA transformation matrix by using joint diagonalization, such as
    SB becomes diagonal and SW becomes the identity matrix

    :param SB: between-class covariance matrix
    :param SW: within-class covariance matrix
    :param m: LDA dimensions
    :return: LDA transformation matrix
    """
    # Whitening transformation
    U1, s1, _ = linalg.svd(SW)
    P1 = U1 @ np.diag(1 / (s1 ** 0.5)) @ U1.T
    Sbt = P1 @ SB @ P1.T

    # Diagonalization
    s2, U2 = linalg.eigh(Sbt)
    P2 = U2[:, ::-1][:, 0:m]
    W = P1.T @ P2

    return W
