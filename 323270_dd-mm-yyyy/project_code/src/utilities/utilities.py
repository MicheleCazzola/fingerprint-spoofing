import numpy as np


def compute_statistics(features, labels, **functions):
    """
    Computes some statistics about features and labels and store them in a dictionary

    :param features: features to compute statistics for
    :param labels: labels to compute statistics for
    :param functions: dictionary of functions with statistics to compute
    :return: a dictionary with statistics about features and labels
    """
    r = {}
    for (name, func) in functions.items():
        result = func(features, 1, labels)
        r[name] = result

    return r


def vcol(array):
    """
    Converts a 1D-ndarray into a column 2D-ndarray

    :param array: 1D-ndarray
    :return: column 2D-ndarray
    """
    return array.reshape(array.size, 1)


def vrow(array):
    """
    Converts a 1D-ndarray into a row 2D-ndarray

    :param array: 1D-ndarray
    :return: row 2D-ndarray
    """
    return array.reshape(1, array.size)


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


def project(D, M):
    """
    Project data over basis spanned by columns of matrix M

    :param D: dataset
    :param M: transformation matrix
    :return: projected dataset
    """
    return M.T @ D


def effective_prior(application):
    """
    Computes the effective prior of an application

    :param application: application triplet (prior, false negative cost, false positive cost)
    :return: effective prior
    """
    return application[0] * application[1] / (application[0] * application[1] + (1 - application[0]) * application[2])