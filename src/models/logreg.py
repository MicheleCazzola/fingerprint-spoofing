import numpy as np
from scipy import optimize as opt, linalg as alg


from src.config.config import LR_STANDARD, PRIOR_WEIGHTED_LR
from utils.utils import vrow, vcol


class LogReg:
    def __init__(self, variant=LR_STANDARD, training_prior=None, app_prior=None):
        self.variant = variant
        self.w = None
        self.b = None
        self.j_min = None
        self.opt_info = None
        self.training_prior = training_prior
        self.app_prior = app_prior

    def setParams(self, **kwargs):
        self.variant = kwargs.get("variant", self.variant)
        self.training_prior = kwargs.get("training_prior", self.training_prior)
        self.app_prior = kwargs.get("app_prior", self.app_prior)

    def fit(self, DTR, LTR, reg_coeff=0, training_prior=None, app_prior=None):
        D = DTR.shape[0]
        n = DTR.shape[1]

        if self.variant == PRIOR_WEIGHTED_LR and app_prior is None:
            raise ValueError("Application prior must be defined if variant is prior-weighted")

        self.app_prior = app_prior if app_prior is not None else np.sum(LTR == 1) / n
        self.training_prior = training_prior if training_prior is not None else self.app_prior

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

            assert self.training_prior is not None, "Training prior must be defined for prior-weighted logistic regression."
            
            w, b = v[0:-1], v[-1]
            S = (vcol(w).T @ DTR + b).ravel()
            ZTR = 2 * LTR - 1
            mask_t, mask_f = ZTR == 1, ZTR == -1
            psi = ((self.training_prior / np.sum(mask_t)) * mask_t +
                   ((1 - self.training_prior) / np.sum(mask_f)) * mask_f)
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
        assert self.training_prior is not None, "Training prior must be defined for scoring."
        assert self.w is not None and self.b is not None, "No model defined"
        return vrow(self.w) @ features + self.b - np.log(self.training_prior / (1 - self.training_prior))

    def predict(self, features, app_prior=None):

        if self.w is None or self.b is None or self.app_prior is None:
            raise ValueError("No model defined")

        S = self.scores(features)
        target_prior = app_prior if app_prior is not None else self.app_prior
        threshold = -np.log(target_prior / (1 - target_prior))

        LPR = np.zeros((1, features.shape[1]), dtype=np.int32)
        LPR[S >= threshold] = 1
        LPR[S < threshold] = 0

        return LPR