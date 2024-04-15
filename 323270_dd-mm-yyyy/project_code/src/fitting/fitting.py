import numpy as np
from utilities.utilities import vcol, vrow
from numpy.linalg import linalg


def logpdf_GAU_ND(X, mu, C):
    Y = []
    for i in range(X.shape[1]):
        Y.append(logpdf_GAU_ND_sample(X[:, i:i + 1], mu, C)[0, 0])

    return np.array(Y)


def logpdf_GAU_ND_sample(x, mu, C):
    """
    Log-gaussian estimation for a sample
    """
    M = vcol(x).shape[0]
    sign, det_val = linalg.slogdet(C)  # sign = 1 since C is semi-definite positive (supposed not singular)
    return -M * np.log(2 * np.pi) / 2 - det_val / 2 - (x - vcol(mu)).T @ linalg.inv(C) @ (x - vcol(mu)) / 2


def compute_estimators(X, mu):
    N = X.shape[1]
    mu_ML = np.sum(X, axis=1) / N
    cov_ML = (X - vcol(mu)) @ (X - vcol(mu)).T / N

    return mu_ML, cov_ML


def gaussian_estimation(D, L):

    features = []
    Yplots = []
    XPlot = np.linspace(-5, 5, 1000)
    for i in range(D.shape[0]):
        D0 = D[i:i + 1, L == 0]
        D1 = D[i:i + 1, L == 1]
        m_ML0, C_ML0 = compute_estimators(D0, D0.mean(axis=1))
        m_ML1, C_ML1 = compute_estimators(D1, D1.mean(axis=1))

        llPlot0 = logpdf_GAU_ND(vrow(XPlot), m_ML0, C_ML0)
        llPlot1 = logpdf_GAU_ND(vrow(XPlot), m_ML1, C_ML1)
        YPlot0 = np.exp(llPlot0)
        YPlot1 = np.exp(llPlot1)

        features.append((D0, D1))
        Yplots.append((YPlot0, YPlot1))

    return XPlot, Yplots, features
