from pprint import pprint
from sys import argv

import numpy as np
import scipy.optimize as scopt
import scipy.linalg as alg
from matplotlib import pyplot as plt

from evaluation.evaluation import Evaluator
from fio import load_csv
from utilities.utilities import split_db_2to1


def vrow(array):
    return array.reshape(1, array.size)


def vcol(array):
    return array.reshape(array.size, 1)


class SupportVectorMachine:
    def __init__(self, K=None, C=None, ker="poly"):
        self.w = None
        self.alpha = None
        self.K = K
        self.C = C
        self.kernel_type = ker
        self.kernel_args = None
        self.dual_loss = None
        self.primal_loss = None
        self.duality_gap = None
        self.DTR = None
        self.ZTR = None

    def expand(self, D, K=None):
        if K is not None:
            self.setParams(K=K)
        D_exp = np.vstack((D, self.K * np.ones((1, D.shape[1]))))

        return D_exp

    def setParams(self, **kwargs):
        self.K = kwargs.get("K", self.K)
        self.C = kwargs.get("C", self.C)
        self.kernel_type = kwargs.get("ker_type", self.kernel_type)

    def _kernel_poly(self, D1, D2, degree, offset):
        return ((D1.T @ D2) + offset) ** degree

    def _kernel_rbf(self, D1, D2, scale):
        n1 = alg.norm(D1, ord=2, axis=0)
        n2 = alg.norm(D2, ord=2, axis=0)
        norm = vcol(n1) ** 2 + vrow(n2) ** 2 - 2 * D1.T @ D2
        exponent = -scale * norm
        return np.exp(exponent)

    def _kernel_fun(self, D1, D2, kernel_args):
        reg_bias = self.K ** 2
        if self.kernel_type == "poly":
            return self._kernel_poly(D1, D2, kernel_args["degree"], kernel_args["offset"]) + reg_bias
        elif self.kernel_type == "rbf":
            return self._kernel_rbf(D1, D2, kernel_args["scale"]) + reg_bias
        else:
            # Should not arrive here
            raise ValueError(f"Unknown kernel type {self.kernel_type}")

    def fit(self, DTR, LTR, C, primal=False, **kernel_args):
        self.setParams(C=C)
        n = DTR.shape[1]
        D = DTR.shape[0]

        self.kernel_args = kernel_args
        G = self._kernel_fun(DTR, DTR, kernel_args)
        ZTR = vcol(2 * LTR - 1)
        H = (ZTR @ ZTR.T) * G

        def opt(alpha):
            l_min = 0.5 * vrow(alpha) @ H @ vcol(alpha) - np.sum(alpha)
            grad = H @ vcol(alpha) - 1

            return l_min, grad.ravel()

        def primal_fun():
            v = 1 - ZTR * vcol(self.w.T @ self.expand(self.DTR, self.K))
            m = np.max(np.hstack((v, np.zeros((v.shape[0], 1)))), axis=1)
            return 0.5 * alg.norm(self.w) ** 2 + self.C * np.sum(m)

        x, f, d = scopt.fmin_l_bfgs_b(func=opt,
                                      x0=np.zeros(n),
                                      bounds=[(0, C)] * n,
                                      factr=1.0)

        self.alpha = x
        self.DTR = DTR
        self.ZTR = ZTR
        self.dual_loss = -f

        if primal:
            self.w = vcol(np.sum(vrow(vcol(x) * ZTR) * self.expand(self.DTR, self.K), axis=1))
            self.primal_loss = primal_fun()
            self.duality_gap = self.primal_loss - self.dual_loss

    def score(self, DVAL):
        k = self._kernel_fun(self.DTR, DVAL, self.kernel_args)
        g = vcol(vcol(self.alpha) * self.ZTR)
        return vrow(np.sum(g * k, axis=0))

    def predict(self, DVAL, eff_prior):
        s = self.score(DVAL)
        threshold = -np.log(eff_prior / (1 - eff_prior))

        LPR = np.zeros((1, DVAL.shape[1]), dtype=np.int32)
        LPR[s >= threshold] = 1
        LPR[s < threshold] = 0

        return LPR


def linear_svm(DTR, LTR, DVAL, LVAL, app_prior):
    results_min_dcf, results_dcf = [], []
    svm = SupportVectorMachine(K=1.0)
    evaluator = Evaluator("SVM_linear")
    cv = np.logspace(-5, 0, 11)

    print("--LINEAR KERNEL--")
    print("K,C,Primal loss,Dual loss,Duality gap,Error rate (%),Minimum DCF,DCF")
    for c in cv:
        svm.fit(DTR, LTR, c, primal=True, degree=1, offset=0)

        llr = svm.score(DVAL)
        LPR = svm.predict(DVAL, app_prior)

        error_rate = 100 * np.sum(LPR != LVAL) / LVAL.shape[0]
        min_dcf, dcf = map(evaluator.evaluate2(llr, LPR, LVAL, eff_prior=app_prior).get("results").get, ["min_dcf", "dcf"])
        p_loss, d_loss, d_gap = svm.primal_loss, svm.dual_loss, svm.duality_gap

        print(f"1,{c},{p_loss:.6e},{d_loss:.6e},{d_gap:.6e},{error_rate:.1f},{min_dcf:.4f},{dcf:.4f}")
        results_min_dcf.append(min_dcf)
        results_dcf.append(dcf)

    return cv, results_min_dcf, results_dcf


def poly_svm(DTR, LTR, DVAL, LVAL, app_prior):
    results_min_dcf, results_dcf = [], []
    svm = SupportVectorMachine(K=0)
    evaluator = Evaluator("SVM_linear")
    cv = np.logspace(-5, 0, 11)

    print("--Polynomial KERNEL (degree=2, offset=1)--")
    print("K,C,Dual loss,Error rate (%),Minimum DCF,DCF")
    for c in cv:
        svm.fit(DTR, LTR, c, primal=False, degree=2, offset=1)

        llr = svm.score(DVAL)
        LPR = svm.predict(DVAL, app_prior)

        error_rate = 100 * np.sum(LPR != LVAL) / LVAL.shape[0]
        min_dcf, dcf = map(evaluator.evaluate2(llr, LPR, LVAL, eff_prior=app_prior).get("results").get,
                           ["min_dcf", "dcf"])
        _, d_loss, _ = svm.primal_loss, svm.dual_loss, svm.duality_gap

        print(f"1,{c},{d_loss:.6e},{error_rate:.1f},{min_dcf:.4f},{dcf:.4f}")
        results_min_dcf.append(min_dcf)
        results_dcf.append(dcf)

    return cv, results_min_dcf, results_dcf


def rbf_svm(DTR, LTR, DVAL, LVAL, app_prior):
    results_min_dcf, results_dcf = {}, {}
    svm = SupportVectorMachine(K=1)
    evaluator = Evaluator("SVM_linear")
    cv = np.logspace(-5, 0, 11)
    scale_values = np.exp(np.array(range(-4,0)))

    print("--RBF KERNEL (bias = 1)--")
    print("K,C,Dual loss,Error rate (%),Minimum DCF,DCF")
    for scale in scale_values:
        res_min, res_act = [], []
        for c in cv:
            svm.setParams(ker_type='rbf')
            svm.fit(DTR, LTR, c, primal=False, scale=scale)

            llr = svm.score(DVAL)
            LPR = svm.predict(DVAL, app_prior)

            error_rate = 100 * np.sum(LPR != LVAL) / LVAL.shape[0]
            min_dcf, dcf = map(evaluator.evaluate2(llr, LPR, LVAL, eff_prior=app_prior).get("results").get,
                               ["min_dcf", "dcf"])
            _, d_loss, _ = svm.primal_loss, svm.dual_loss, svm.duality_gap

            print(f"1,{c},{d_loss:.6e},{error_rate:.1f},{min_dcf:.4f},{dcf:.4f}")
            res_min.append(min_dcf)
            res_act.append(dcf)

        results_min_dcf[scale] = res_min
        results_dcf[scale] = res_act

    return cv, results_min_dcf, results_dcf


def plot_log_double_line(x, y1, y2, title, x_label, y_label, legend1, legend2):
    plt.figure(title)
    plt.xscale('log', base=10)
    plt.plot(x, y1, label=legend1)
    plt.plot(x, y2, label=legend2)
    plt.grid()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.title(title)


def plot_log_N_double_lines(x, ys1, ys2, title, x_label, y_label, legends1, legends2):
    plt.figure(title)
    plt.xscale('log', base=10)
    for (y1, y2, legend1, legend2) in zip(ys1, ys2, legends1, legends2):
        plt.plot(x, ys1[y1], label=legend1)
        plt.plot(x, ys2[y2], label=legend2)
    plt.grid()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.title(title)


def main():
    D, L = load_csv(argv[1])
    np.random.seed(0)
    idx = np.random.permutation(D.shape[1])[0:D.shape[1] // 3]
    D = D[:, idx]
    L = L[idx]
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    app_prior = 0.1

    # Linear SVM, no preprocessing
    cv, eval_results_min_dcf, eval_results_dcf = linear_svm(DTR, LTR, DVAL, LVAL, app_prior)

    plot_log_double_line(cv, eval_results_min_dcf, eval_results_dcf,
                         "SVM linear kernel no preprocess",
                         "Regularization values",
                         "DCF values",
                         "Min. DCF",
                         "DCF")

    best_linear = (cv[np.argmin(eval_results_min_dcf)], np.min(eval_results_min_dcf))

    # Linear SVM, data centering
    DTR_mean = vcol(np.sum(DTR, axis=1)) / DTR.shape[1]
    DTR_preprocess, DVAL_preprocess = DTR - DTR_mean, DVAL - DTR_mean
    cv, eval_results_min_dcf, eval_results_dcf = linear_svm(DTR_preprocess, LTR, DVAL_preprocess, LVAL, app_prior)

    plot_log_double_line(cv, eval_results_min_dcf, eval_results_dcf,
                         "SVM linear kernel with data centering",
                         "Regularization values",
                         "DCF values",
                         "Min. DCF",
                         "DCF")

    best_linear_preprocess = (cv[np.argmin(eval_results_min_dcf)], np.min(eval_results_min_dcf))

    # Polynomial SVM (degree=2, offset=1)
    cv, eval_results_min_dcf, eval_results_dcf = poly_svm(DTR, LTR, DVAL, LVAL, app_prior)

    plot_log_double_line(cv, eval_results_min_dcf, eval_results_dcf,
                         "SVM polynomial kernel (degree=2, offset=1)",
                         "Regularization values",
                         "DCF values",
                         "Min. DCF",
                         "DCF")

    best_poly = (cv[np.argmin(eval_results_min_dcf)], np.min(eval_results_min_dcf))

    # RBF SVM (bias = 1), scale = [e-4, e-3, e-2, e-1]
    cv, eval_results_min_dcf, eval_results_dcf = rbf_svm(DTR, LTR, DVAL, LVAL, app_prior)

    plot_log_N_double_lines(cv, eval_results_min_dcf, eval_results_dcf,
                            "SVM RBF kernel (bias = 1)",
                            "Regularization values",
                            "DCF values",
                            ["Min. DCF (g=e-4)", "Min. DCF (g=e-3)", "Min. DCF (g=e-2)", "Min. DCF (g=e-1)"],
                            ["DCF (g=e-4)", "DCF (g=e-3)", "DCF (g=e-2)", "DCF (g=e-1)"])

    best_rbf = {scale: (cv[np.argmin(eval_results_min_dcf[scale])], np.min(eval_results_min_dcf[scale])) for scale in eval_results_min_dcf}

    pprint(best_linear)
    pprint(best_linear_preprocess)
    pprint(best_poly)
    pprint(best_rbf)

    plt.show()


if __name__ == '__main__':
    main()