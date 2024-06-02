import numpy as np
from scipy import optimize as opt, linalg as alg

from evaluation.evaluation import Evaluator
from plot import plot_log_double_line
from utilities.utilities import vrow, vcol, split_db_2to1
from src.io.constants import LR_STANDARD, PRIOR_WEIGHTED_LR, PLOT_PATH_LOGISTIC_REGRESSION


class LogisticRegression:
    def __init__(self, variant=LR_STANDARD):
        self.variant = variant
        self.w = None
        self.b = None
        self.j_min = None
        self.opt_info = None
        self.target_prior = None

    def setParams(self, **kwargs):
        self.variant = kwargs.get("variant", self.variant)

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


def expand(DTR):
    expanded = []
    for i in range(DTR.shape[1]):
        arr = np.concatenate([(DTR[:, i:i + 1] @ DTR[:, i:i + 1].T).ravel(), DTR[:, i]])
        expanded.append(arr)
    return np.array(expanded).T


def logistic_regression(DTR, LTR, DVAL, LVAL, app_prior, reg_coefficients, variant, name, preprocess=None):
    eval_results = []
    lr = LogisticRegression(variant)
    evaluator = Evaluator(name)

    for reg_coeff in reg_coefficients:
        lr_prior = app_prior if variant == PRIOR_WEIGHTED_LR else None
        lr.fit(DTR, LTR, reg_coeff, app_prior=lr_prior)

        llr = lr.scores(DVAL)
        LPR = lr.predict(DVAL, app_prior)
        eval_result = evaluator.evaluate2(llr, LPR, LVAL,
                                          eff_prior=app_prior, preprocess=preprocess, reg_coeff=reg_coeff)

        preprocess = eval_result["params"]["preprocess"]
        eval_result["results"]["reg_coeff"] = eval_result["params"]["reg_coeff"]
        eval_results.append((eval_result["results"]["dcf"],
                             eval_result["results"]["min_dcf"],
                             eval_result["results"]["reg_coeff"]))

    return {
        "preprocess": preprocess,
        "results": list(zip(*eval_results))
    }


def LR_classification(D, L):
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    app_prior = 0.1
    reg_coefficients = np.logspace(-4, 2, 13)

    titles = [
        "Logistic Regression DCFs for standard non-weighted model",
        "Logistic Regression DCFs for filtered non-weighted model",
        "Prior-weighted Logistic Regression DCFs",
        "Prior-weighted Logistic Regression DCFs with expanded feature space",
        "Prior-weighted Logistic Regression DCFs with preprocessing"
    ]

    file_names = [
        "LR_standard_non_weighted",
        "LR_standard_non_weighted_filtered_dataset",
        "Prior_weighted_LR",
        "Prior_weighted_LR_expanded_feature_space",
        "Prior_weighted_LR_data_centering"
    ]

    results = [{}] * 5

    # 1: standard non-weighted LR
    results[0] = logistic_regression(DTR, LTR, DVAL, LVAL, app_prior, reg_coefficients,
                                     LR_STANDARD, LR_STANDARD)

    # 2: reduced dataset LR
    results[1] = logistic_regression(DTR[:, ::50], LTR[::50], DVAL, LVAL, app_prior, reg_coefficients,
                                     LR_STANDARD, LR_STANDARD)

    # 3: prior-weighted LR
    results[2] = logistic_regression(DTR, LTR, DVAL, LVAL, app_prior, reg_coefficients,
                                     PRIOR_WEIGHTED_LR, PRIOR_WEIGHTED_LR)

    # 4: quadratic LR
    DTR_expanded = expand(DTR)
    DVAL_expanded = expand(DVAL)
    results[3] = logistic_regression(DTR_expanded, LTR, DVAL_expanded, LVAL, app_prior, reg_coefficients,
                                     LR_STANDARD, LR_STANDARD)

    # 5: preprocess data and apply regularized model
    DTR_mean = vcol(np.sum(DTR, axis=1)) / DTR.shape[1]
    DTR_preprocess, DVAL_preprocess = DTR - DTR_mean, DVAL - DTR_mean
    results[4] = logistic_regression(DTR_preprocess, LTR, DVAL_preprocess, LVAL, app_prior, reg_coefficients,
                                     PRIOR_WEIGHTED_LR, PRIOR_WEIGHTED_LR, "Data centering")

    eval_results_best = []
    for (result, title, file_name) in zip(results, titles, file_names):
        [dcf, min_dcf, reg_coeff] = result["results"]
        eval_results_best.append((np.min(min_dcf), reg_coeff[np.argmin(min_dcf)], title.replace(" DCFs", "") + " data centering" ))
        plot_log_double_line(reg_coefficients, dcf, min_dcf,
                             title,
                             result["preprocess"] if result["preprocess"] is not None else "",
                             "Regularization coefficient",
                             "DCF value",
                             "DCF",
                             "Min. DCF",
                             PLOT_PATH_LOGISTIC_REGRESSION,
                             file_name,
                             "pdf")

    return eval_results_best
