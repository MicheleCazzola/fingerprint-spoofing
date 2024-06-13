import numpy as np
import scipy.optimize as scopt
import scipy.linalg as alg

from constants import PLOT_PATH_SVM
from evaluation.evaluation import Evaluator
from plot import plot_log_double_line, plot_log_N_double_lines
from utilities.utilities import split_db_2to1, vcol, vrow


def optimal_bayes(svm, evaluator, DVAL, LVAL, app_prior):
    llr = svm.score(DVAL)
    LPR = svm.predict(DVAL, app_prior)

    min_dcf, dcf = map(evaluator.evaluate2(llr, LPR, LVAL, eff_prior=app_prior).get("results").get,
                       ["min_dcf", "dcf"])
    return min_dcf, dcf


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


def linear_svm(DTR, LTR, DVAL, LVAL, app_prior, svm, evaluator, c_values):
    results_min_dcf, results_dcf = [], []

    for c in c_values:
        svm.fit(DTR, LTR, c, primal=True, degree=1, offset=0)

        min_dcf, dcf = optimal_bayes(svm, evaluator, DVAL, LVAL, app_prior)

        results_min_dcf.append(min_dcf)
        results_dcf.append(dcf)

    return results_min_dcf, results_dcf


def poly_svm(DTR, LTR, DVAL, LVAL, app_prior, svm, evaluator, c_values):
    results_min_dcf, results_dcf = [], []

    for c in c_values:
        svm.fit(DTR, LTR, c, primal=False, degree=2, offset=1)

        min_dcf, dcf = optimal_bayes(svm, evaluator, DVAL, LVAL, app_prior)

        results_min_dcf.append(min_dcf)
        results_dcf.append(dcf)

    return results_min_dcf, results_dcf


def rbf_svm(DTR, LTR, DVAL, LVAL, app_prior, svm, evaluator, c_values, scale_values):
    results_min_dcf, results_dcf = {}, {}

    for scale in scale_values:
        res_min, res_act = [], []
        for c in c_values:
            svm.setParams(ker_type='rbf')
            svm.fit(DTR, LTR, c, primal=False, scale=scale)

            min_dcf, dcf = optimal_bayes(svm, evaluator, DVAL, LVAL, app_prior)

            res_min.append(min_dcf)
            res_act.append(dcf)

        results_min_dcf[scale] = res_min
        results_dcf[scale] = res_act

    return results_min_dcf, results_dcf


def svm(D, L):
    """
    # Dataset random filtering (1/6)
    np.random.seed(0)
    idx = np.random.permutation(D.shape[1])[0:D.shape[1] // 6]
    D = D[:, idx]
    L = L[idx]
    """

    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)
    print(f"SVM dataset: {DTR.shape}, {DVAL.shape}")

    app_prior = 0.1
    c_values = np.logspace(-5, 0, 11)
    k_values = [1, 1, 0, 1]
    scale_values_rbf = np.exp(np.array(range(-4, 0)))
    svm = SupportVectorMachine()
    evaluator = Evaluator("SVM")

    eval_results = [{"min_dcf": [], "dcf": []} for _ in range(0, 4)]
    best_results = [()] * 3 + [{}]
    task_names = [
        "SVM with linear kernel",
        "SVM with linear kernel and data centering",
        "SVM with polynomial kernel",
        "SVM with RBF kernel"
    ]

    titles = [
        "SVM linear kernel no preprocess",
        "SVM linear kernel with data centering",
        "SVM polynomial kernel (degree=2, offset=1)",
        "SVM RBF kernel (bias = 1)"
    ]

    names = [
        "linear_no_preprocessing",
        "linear_data_centering",
        "quadratic_no_preprocessing",
        "RBF_kernel_bias1",
    ]

    print("SVM: linear kernel")
    # Linear SVM, no preprocessing
    svm.setParams(K=k_values[0])
    eval_results[0]["min_dcf"], eval_results[0]["dcf"] = linear_svm(DTR, LTR, DVAL, LVAL, app_prior, svm, evaluator,
                                                                    c_values)

    print("SVM: linear kernel with preprocessing")
    # Linear SVM, data centering
    DTR_mean = vcol(np.sum(DTR, axis=1)) / DTR.shape[1]
    DTR_preprocess, DVAL_preprocess = DTR - DTR_mean, DVAL - DTR_mean
    svm.setParams(K=k_values[1])
    eval_results[1]["min_dcf"], eval_results[1]["dcf"] = linear_svm(DTR_preprocess, LTR, DVAL_preprocess, LVAL,
                                                                    app_prior, svm,
                                                                    evaluator, c_values)

    print("SVM: polynomial kernel")
    # Polynomial SVM (degree=2, offset=1)
    svm.setParams(K=k_values[2])
    eval_results[2]["min_dcf"], eval_results[2]["dcf"] = poly_svm(DTR, LTR, DVAL, LVAL, app_prior, svm, evaluator,
                                                                  c_values)

    print("SVM: RBF kernel")
    # RBF SVM (bias = 1), scale = [e-4, e-3, e-2, e-1]
    svm.setParams(K=k_values[3], ker_type="rbf")
    eval_results[3]["min_dcf"], eval_results[3]["dcf"] = rbf_svm(DTR, LTR, DVAL, LVAL, app_prior, svm, evaluator,
                                                                 c_values, scale_values_rbf)

    for i in range(len(best_results[:-1])):
        eval_result = eval_results[i]
        best_results[i] = (c_values[np.argmin(eval_result["min_dcf"])], np.min(eval_result["dcf"]), k_values[i])

    best_results[-1] = {
        scale: (c_values[np.argmin(eval_results[-1]["min_dcf"][scale])], np.min(eval_results[-1]["min_dcf"][scale]), k_values[-1])
        for scale in eval_results[-1]["min_dcf"]}

    for (eval_result, title, name) in zip(eval_results[:-1], titles[:-1], names[:-1]):
        plot_log_double_line(c_values, eval_result["min_dcf"], eval_result["dcf"],
                             title, "Regularization values", "DCF values", "Min. DCF", "DCF",
                             PLOT_PATH_SVM, name, "pdf")

    plot_log_N_double_lines(c_values, eval_results[-1]["min_dcf"], eval_results[-1]["dcf"],
                            titles[-1], "Regularization values", "DCF values",
                            ["Min. DCF (g=e-4)", "Min. DCF (g=e-3)", "Min. DCF (g=e-2)", "Min. DCF (g=e-1)"],
                            ["DCF (g=e-4)", "DCF (g=e-3)", "DCF (g=e-2)", "DCF (g=e-1)"],
                            PLOT_PATH_SVM, names[-1], "pdf")

    return {
        "tasks": task_names,
        "results": best_results
    }
