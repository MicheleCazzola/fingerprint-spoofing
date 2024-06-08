from numpy.linalg import linalg
from src.utilities.utilities import vcol, project


def apply(D, m=2):
    """
    Computes PCA with specified dimensions (default is 2)

    :param D: dataset to compute PCA on
    :param m: number of principal components to select
    :return: dataset projected over selected PCA components
    """
    P = reduce(D, m)
    DP = project(D, P)

    return DP


def reduce(D, m=2):
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
