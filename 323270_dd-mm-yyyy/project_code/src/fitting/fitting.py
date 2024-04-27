import numpy as np
from utilities.utilities import vcol, vrow
from numpy.linalg import linalg


def logpdf_GAU_ND(X, mu, C):
    """
    Computes the log-density Gaussian distribution for the dataset **X**, by direct computing

    :param X: dataset to compute the Gaussian distribution on
    :param mu: mean vector of the Gaussian distribution
    :param C: covariance matrix of the Gaussian distribution
    :return: values of the Gaussian distribution for the dataset
    """
    M = X.shape[0]
    sign, det_val = linalg.slogdet(C)  # sign = 1 since C is semi-definite positive (supposed not singular)
    return -M * np.log(2 * np.pi) / 2 - det_val / 2 - ((X - vcol(mu)) * (linalg.inv(C) @ (X - vcol(mu)))).sum(0) / 2


def compute_estimators(X, mu):
    """
    Computes the estimates of the Gaussian-distributed dataset **X**, using the maximum likelihood method

    :param X: source dataset, distributed as Multivariate Gaussian (MVG)
    :param mu: dataset mean
    :return: estimates of mean vector (1D-array) and covariance matrix
    """
    N = X.shape[1]
    mu_ML = np.sum(X, axis=1) / N
    cov_ML = (X - vcol(mu)) @ (X - vcol(mu)).T / N

    return mu_ML, cov_ML


def gaussian_estimation(D, L):
    """
    Computes the estimates of the Gaussian-distributed dataset **D**

    :param D: dataset, distributed as Multivariate Gaussian
    :param L: corresponding labels
    :return: a triplet with features domain, Gaussian estimated plot and dataset samples, divided by class
    """

    features = []
    Yplots = []
    XPlot = np.linspace(-5, 5, 1000)
    for i in range(D.shape[0]):
        D0 = D[i:i + 1, L == 0]
        D1 = D[i:i + 1, L == 1]

        m_ML0, C_ML0 = compute_estimators(D0, D0.mean(axis=1))
        m_ML1, C_ML1 = compute_estimators(D1, D1.mean(axis=1))

        YPlot0 = np.exp(logpdf_GAU_ND(vrow(XPlot), m_ML0, C_ML0))
        YPlot1 = np.exp(logpdf_GAU_ND(vrow(XPlot), m_ML1, C_ML1))

        features.append((D0, D1))
        Yplots.append((YPlot0, YPlot1))

    return XPlot, Yplots, features
