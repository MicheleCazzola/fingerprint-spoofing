from numpy.linalg import linalg

PLOT_PATH = "output\\plots\\PCA_features\\"


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