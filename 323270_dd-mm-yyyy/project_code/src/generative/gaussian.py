from sys import argv

import numpy as np

from src.io.fio import load_csv
from utilities.utilities import vcol, split_db_2to1, vrow
from fitting.fitting import logpdf_GAU_ND, compute_estimators


def estimate(D, L):
    mu_c, cov_c = [], []
    for c in np.unique(L):
        Dc = D[:, L == c]
        mu, cov = compute_estimators(Dc, np.mean(Dc, axis=1))
        mu_c.append(vcol(mu))
        cov_c.append(cov)

    return np.array(mu_c), np.array(cov_c)


def class_conditional_log(D, L, mu_c, cov_c):
    S = []
    for c in np.unique(L):
        yc = logpdf_GAU_ND(D, mu_c[c], cov_c[c])
        S.append(yc)

    return np.array(S)


def within_class_covariance(cov_c, DTR, LTR):
    cov = np.zeros((cov_c.shape[1], cov_c.shape[2]))
    for c in np.unique(LTR):
        Nc = DTR[:, LTR == c].shape[1]
        cov += Nc * cov_c[c]
    return cov / DTR.shape[1]


def compute_llr(DTE, LTE, mu_c, cov_c):
    S = class_conditional_log(DTE, LTE, mu_c, cov_c)
    return S[1] - S[0]


def predict(LTE, llr, threshold):
    LPR = np.zeros(LTE.shape, dtype=np.int32)
    LPR[llr >= threshold] = 1
    LPR[llr < threshold] = 0

    return LPR


def compute_error_rate(LPR, LTE):
    return np.sum(LPR != LTE) / LTE.shape[0]


def MVG(DTR, LTR, DTE, LTE):
    mu_c, cov_c = estimate(DTR, LTR)
    prior = vcol(np.array([1 / 2, 1 / 2]))
    llr = compute_llr(DTE, LTE, mu_c, cov_c)

    threshold = -np.log(prior[1] / prior[0])
    LPR = predict(LTE, llr, threshold)

    err_rate = compute_error_rate(LPR, LTE)

    print(f"{err_rate:.4f}")


def TiedMVG(DTR, LTR, DTE, LTE):
    mu_c, cov_c = estimate(DTR, LTR)
    cov = within_class_covariance(cov_c, DTR, LTR)
    prior = vcol(np.array([1 / 2, 1 / 2]))

    llr = compute_llr(DTE, LTE, mu_c, np.array([cov] * 2))

    threshold = -np.log(prior[1] / prior[0])
    LPR = predict(LTE, llr, threshold)

    err_rate = compute_error_rate(LPR, LTE)

    print(f"{err_rate:.4f}")


def Naive_BayesMVG(DTR, LTR, DTE, LTE):
    mu_c, cov_c = estimate(DTR, LTR)
    cov_c = cov_c * np.eye(cov_c[0].shape[1])
    prior = vcol(np.array([1 / 2, 1 / 2]))

    llr = compute_llr(DTE, LTE, mu_c, cov_c)

    threshold = -np.log(prior[1] / prior[0])
    LPR = predict(LTE, llr, threshold)

    err_rate = compute_error_rate(LPR, LTE)

    print(f"{err_rate:.4f}")


def compute_correlations(DTR, LTR):
    _, cov_c = estimate(DTR, LTR)

    corr_matrices = [C / (vcol(C.diagonal() ** 0.5) * vrow(C.diagonal() ** 0.5)) for C in cov_c]

    for corr_matrix in corr_matrices:
        for line in corr_matrix:
            for element in line:
                print(f"{element:.2f}", end="\t")
            print()
        print()


if __name__ == '__main__':
    D, L = load_csv(argv[1])

    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

    # 1: error rate = 7 %
    MVG(DTR, LTR, DTE, LTE)

    # 2: error rate = 9.3 % (same as LDA) -> TIED = LDA (no preprocessing)
    TiedMVG(DTR, LTR, DTE, LTE)

    # 3: error rate = 7.2 %
    Naive_BayesMVG(DTR, LTR, DTE, LTE)

    # 4: low correlation, but not null -> Indeed Naive is good, but little worse than MVG
    compute_correlations(DTR, LTR)