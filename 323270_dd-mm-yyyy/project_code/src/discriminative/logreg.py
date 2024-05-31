import sys
from pprint import pprint

import numpy as np
from scipy import optimize as opt, linalg as alg
import matplotlib.pyplot as plt

from evaluation.evaluation import Evaluator, refactor_evaluation_results
from fio import load_csv
from utilities.utilities import vrow, vcol, split_db_2to1
from src.io.constants import LR_STANDARD, PRIOR_WEIGHTED_LR


class LogisticRegression:
    def __init__(self, variant=LR_STANDARD):
        self.variant = variant
        self.w = None
        self.b = None
        self.j_min = None
        self.opt_info = None
        self.target_prior = None

    def fit(self, DTR, LTR, reg_coeff, app_prior=None):
        D = DTR.shape[0]
        n = DTR.shape[1]

        if self.variant == PRIOR_WEIGHTED_LR and app_prior is None:
            raise ValueError("Application prior must be defined if variant is prior-weighted")

        self.target_prior = app_prior if app_prior is not None else np.sum(LTR == 1) / n

        def logreg_obj_lr(v):
            w, b = v[0:-1], v[-1]
            S = (vcol(w).T @ DTR + b).ravel()
            ZTR = 2 * LTR - 1
            J_min = reg_coeff * alg.norm(w, 2) ** 2 / 2 + np.sum(np.logaddexp(0, -ZTR * S)) / n

            G = -ZTR / (1 + np.exp(ZTR * S))
            grad_b = np.array([np.sum(G) / n])
            grad_w = reg_coeff * w + np.sum(vrow(G) * DTR, axis=1) / n
            return J_min, np.concatenate((grad_w, grad_b))

        def logreg_obj_pwlr(v):
            w, b = v[0:-1], v[-1]
            S = (vcol(w).T @ DTR + b).ravel()
            ZTR = 2 * LTR - 1
            mask_t, mask_f = ZTR == 1, ZTR == -1
            psi = (app_prior / np.sum(mask_t)) * mask_t + ((1 - app_prior) / np.sum(mask_f)) * mask_f
            J_min = reg_coeff * alg.norm(w, 2) ** 2 / 2 + np.sum(psi * np.logaddexp(0, -ZTR * S))

            G = -ZTR / (1 + np.exp(ZTR * S))
            grad_b = np.array([np.sum(psi * G)])
            grad_w = reg_coeff * w + np.sum(psi * vrow(G) * DTR, axis=1)
            return J_min, np.concatenate((grad_w, grad_b))

        loss_function = logreg_obj_lr if self.variant == LR_STANDARD else logreg_obj_pwlr

        x, f_min, d = opt.fmin_l_bfgs_b(func=loss_function,
                                        approx_grad=False,
                                        x0=np.zeros(D + 1))
        self.w = x[0:-1]
        self.b = x[-1]
        self.j_min = f_min
        self.opt_info = d

    def scores(self, features):
        return vrow(self.w) @ features + self.b - np.log(self.target_prior / (1 - self.target_prior))

    def predict(self, features, app_prior=None):

        if self.w is None or self.b is None or self.target_prior is None:
            raise ValueError("No model defined")

        S = self.scores(features)
        target_prior = app_prior if app_prior is not None else self.target_prior
        threshold = -np.log(target_prior / (1 - target_prior))

        LPR = np.zeros((1, features.shape[1]), dtype=np.int32)
        LPR[S >= threshold] = 1
        LPR[S < threshold] = 0

        return LPR

    def print_model(self):
        print(f"w_min: {self.w}")
        print(f"b_min: {self.b}")
        print(f"J (w_min, b_min): {self.j_min}")
        print(f"Optimization information:")
        for (info_key, info_value) in self.opt_info.items():
            print(f"\t{info_key}: {info_value}")


# class PriorWeightedLogisticRegression(LogisticRegression):
#     def __init__(self, DTR, LTR, reg_coeff, prior):
#         super().__init__(DTR, LTR, reg_coeff)
#         self.prior = None
# 
#     def fit(self, DTR, LTR, reg_coeff, prior):
#         x, f_min, d = opt.fmin_l_bfgs_b(func=self.logreg_obj,
#                                         approx_grad=False,
#                                         x0=np.zeros(self.DTR.shape[0] + 1))
#         self.w = x[0:-1]
#         self.b = x[-1]
#         self.j_min = f_min
#         self.opt_info = d
# 
#     def scores(self, features):
#         return vrow(self.w) @ features + self.b - np.log(self.prior / (1 - self.prior))
# 
#     def predict(self, features):
#         S = self.scores(features)
#         threshold = -np.log(self.prior / (1 - self.prior))
# 
#         LPR = np.zeros((1, features.shape[1]), dtype=np.int32)
#         LPR[S >= threshold] = 1
#         LPR[S < threshold] = 0
# 
#         return LPR
# 
#     def logreg_obj(self, v):
#         w, b = v[0:-1], v[-1]
#         S = (vcol(w).T @ self.DTR + b).ravel()
#         ZTR = 2 * self.LTR - 1
#         mask_t, mask_f = ZTR == 1, ZTR == -1
#         psi = (self.prior / np.sum(mask_t)) * mask_t + ((1 - self.prior) / np.sum(mask_f)) * mask_f
#         J_min = self.reg_coeff * alg.norm(w, 2) ** 2 / 2 + np.sum(psi * np.logaddexp(0, -ZTR * S))
# 
#         G = -ZTR / (1 + np.exp(ZTR * S))
#         grad_b = np.array([np.sum(psi * G)])
#         grad_w = self.reg_coeff * w + np.sum(psi * vrow(G) * self.DTR, axis=1)
#         return J_min, np.concatenate((grad_w, grad_b))


def expand(DTR):
    expanded = []
    for i in range(DTR.shape[1]):
        vxxt = (DTR[:, i:i + 1] @ DTR[:, i:i + 1].T).ravel()
        x = DTR[:, i]
        arr = np.concatenate([vxxt, x])
        expanded.append(arr)
    return np.array(expanded).T


def logistic_regression(DTR, LTR, DVAL, LVAL, app_prior, reg_coefficients, variant, name):
    dcf, min_dcf = [], []
    eval_results = []
    preprocess = None
    lr = LogisticRegression(variant)
    evaluator = Evaluator(name)

    print(f"--{name}--")
    print(f"Reg. coefficients J(w_min, b_min) Minimum dcf Actual dcf")
    for l in reg_coefficients:

        lr_prior = app_prior if variant == PRIOR_WEIGHTED_LR else None
        lr.fit(DTR, LTR, l, app_prior=lr_prior)

        llr = lr.scores(DVAL)
        LPR = lr.predict(DVAL, app_prior)
        eval_result = evaluator.evaluate2(llr, LPR, LVAL, eff_prior=app_prior, preprocess=None, reg_coeff=l)

        preprocess = eval_result["params"]["preprocess"]
        eval_result["results"]["reg_coeff"] = eval_result["params"]["reg_coeff"]
        eval_results.append((eval_result["results"]["dcf"], eval_result["results"]["min_dcf"]))

        # eval_results = evaluator.get_results()
        # print(f"{l:^17.2e} "
        #       f"{lr.j_min:^15.6f} "
        #       f"{eval_results[0][app_prior]['min_dcf']:^11.4f} "
        #       f"{eval_results[0][app_prior]['dcf']:^10.4f}")
        # dcf.append(eval_results[0][app_prior]['dcf'])
        # min_dcf.append(eval_results[0][app_prior]['min_dcf'])
    # print()

    return {
        "preprocess": preprocess,
        "results": list(zip(*eval_results))
    }


# def prior_weighted_logistic_regression(DTR, LTR, DVAL, LVAL, app_prior, reg_coefficients, name):
#     dcf, min_dcf = [], []
#     print(f"--{name}--")
#     print(f"Reg. coefficients J(w_min, b_min) Minimum dcf Actual dcf")
#     for l in reg_coefficients:
#         # lr = PriorWeightedLogisticRegression(DTR, LTR, l, app_prior)
#         lr = LogisticRegression()
#         evaluator = Evaluator(name)
#         lr.fit(DTR, LTR, l, variant=PRIOR_WEIGHTED_LR, app_prior=app_prior)
#
#         llr = lr.scores(DVAL)
#         LPR = lr.predict(DVAL)
#         evaluator.evaluate(llr, LPR, LVAL, app_prior, "Not applied")
#
#         eval_results = evaluator.get_results()
#         print(f"{l:^17.2e} "
#               f"{lr.j_min:^15.6f} "
#               f"{eval_results[0][app_prior]['min_dcf']:^11.4f} "
#               f"{eval_results[0][app_prior]['dcf']:^10.4f}")
#         dcf.append(eval_results[0][app_prior]['dcf'])
#         min_dcf.append(eval_results[0][app_prior]['min_dcf'])
#     print()
#
#     return dcf, min_dcf


def plot_dcfs(reg_coefficients, results, preprocess, title):
    plt.figure(title)
    plt.xscale('log', base=10)
    plt.plot(reg_coefficients, results[0], label="DCF")
    plt.plot(reg_coefficients, results[1], label="Min. DCF")
    plt.grid()
    plt.xlabel("Regularization coefficient")
    plt.ylabel("DCF value")
    plt.title(title)
    plt.suptitle(preprocess if not None else "No preprocess", fontsize=2)
    plt.legend()


def main():
    D, L = load_csv(sys.argv[1])
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    app_prior = 0.1
    reg_coefficients = np.logspace(-4, 2, 13)

    # 1: standard non-weighted LR
    result = logistic_regression(DTR, LTR, DVAL, LVAL, app_prior, reg_coefficients,
                                       LR_STANDARD, LR_STANDARD)
    plot_dcfs(reg_coefficients, result["results"], result["preprocess"], "Logistic Regression DCFs for standard non-weighted model")

    # 2: reduced dataset
    result = logistic_regression(DTR[:, ::50], LTR[::50], DVAL, LVAL, app_prior, reg_coefficients,
                                       LR_STANDARD, LR_STANDARD)
    plot_dcfs(reg_coefficients, result["results"], result["preprocess"], "Logistic Regression DCFs for filtered non-weighted model")

    # 3: prior-weighted LR
    result = logistic_regression(DTR, LTR, DVAL, LVAL, app_prior, reg_coefficients,
                                       PRIOR_WEIGHTED_LR, PRIOR_WEIGHTED_LR)
    plot_dcfs(reg_coefficients, result["results"], result["preprocess"], "Prior-weighted Logistic Regression DCFs")

    # 4: quadratic LR
    DTR_expanded = expand(DTR)
    DVAL_expanded = expand(DVAL)
    result = logistic_regression(DTR_expanded, LTR, DVAL_expanded, LVAL, app_prior, reg_coefficients,
                                       LR_STANDARD, LR_STANDARD)
    plot_dcfs(reg_coefficients, result["results"], result["preprocess"], "Prior-weighted Logistic Regression DCFs with expanded feature space")

    # 5: preprocess data and apply regularized model
    DTR_mean = vcol(np.sum(DTR, axis=1)) / DTR.shape[1]
    DTR_preprocess, DVAL_preprocess = DTR - DTR_mean, DVAL - DTR_mean
    result = logistic_regression(DTR_preprocess, LTR, DVAL_preprocess, LVAL, app_prior, reg_coefficients,
                                       PRIOR_WEIGHTED_LR, "Prior-weighted LR with data centering")
    plot_dcfs(reg_coefficients, result["results"], result["preprocess"], "Prior-weighted Logistic Regression DCFs with data centering")

    # 6: comparison with other models for app_prior = 0.1, basing on minimum DCF

    plt.show()


if __name__ == '__main__':
    main()
