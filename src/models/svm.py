from datetime import datetime
import os
import joblib
import numpy as np
import scipy.linalg as alg
import scipy.optimize as scopt

from src.config.config import MODEL_PATH_SVM
from src.utils.utils import vcol, vrow

class SupportVectorMachine:
    def __init__(self, K=None, C=None, kernel="poly"):
        self.w = None
        self.alpha = None
        self.K = K
        self.C = C
        self.kernel_type = kernel
        self.kernel_args = None
        self.dual_loss = None
        self.primal_loss = None
        self.duality_gap = None
        self.opt_info = None
        self.DTR = None
        self.ZTR = None
        
    def load_state_dict(self, path, id):
        
        filepath = f"{path}/svm_{id}.pkl"
        
        if not os.path.exists(filepath):
            print(f"Warning: SVM model file ('{filepath}') not found.")
            return False
        
        state_dict = joblib.load(filepath)
        
        self.w = state_dict.get("w", self.w)
        self.alpha = state_dict.get("alpha", self.alpha)
        self.K = state_dict.get("K", self.K)
        self.C = state_dict.get("C", self.C)
        self.kernel_type = state_dict.get("kernel_type", self.kernel_type)
        self.kernel_args = state_dict.get("kernel_args", self.kernel_args)
        self.dual_loss = state_dict.get("dual_loss", self.dual_loss)
        self.primal_loss = state_dict.get("primal_loss", self.primal_loss)
        self.duality_gap = state_dict.get("duality_gap", self.duality_gap)
        self.opt_info = state_dict.get("opt_info", self.opt_info)
        self.DTR = state_dict.get("DTR", self.DTR)
        self.ZTR = state_dict.get("ZTR", self.ZTR)
        
        return True
        
    def save_state_dict(self, filepath):
        d = {
            "w": self.w,
            "alpha": self.alpha,
            "K": self.K,
            "C": self.C,
            "kernel_type": self.kernel_type,
            "kernel_args": self.kernel_args,
            "dual_loss": self.dual_loss,
            "primal_loss": self.primal_loss,
            "duality_gap": self.duality_gap,
            "opt_info": self.opt_info,
            "DTR": self.DTR,
            "ZTR": self.ZTR
        }
        
        id = datetime.now().strftime("%Y%m%d-%H%M%S%f")
        joblib.dump(d, f"{filepath}/svm_{id}.pkl")
        return id

    def expand(self, D, K=None):
        if K is not None:
            self.setParams(K=K)
        assert self.K is not None, "K parameter must be set for expansion."
        D_exp = np.vstack((D, self.K * np.ones((1, D.shape[1]))))

        return D_exp

    def setParams(self, **kwargs):
        self.K = kwargs.get("K", self.K)
        self.C = kwargs.get("C", self.C)
        self.kernel_type = kwargs.get("kernel", self.kernel_type)

    @staticmethod
    def _kernel_poly(D1, D2, degree, offset):
        return ((D1.T @ D2) + offset) ** degree

    @staticmethod
    def _kernel_rbf(D1, D2, scale):
        n1 = alg.norm(D1, ord=2, axis=0)
        n2 = alg.norm(D2, ord=2, axis=0)
        norm = vcol(n1) ** 2 + vrow(n2) ** 2 - 2 * D1.T @ D2 # type: ignore
        exponent = -scale * norm
        return np.exp(exponent)

    def _kernel_fun(self, D1, D2):
        
        assert self.K is not None, "K parameter must be set for kernel computation."
        assert self.kernel_args is not None, "Kernel arguments must be set for kernel computation."
        
        reg_bias = self.K ** 2
        if self.kernel_type == "poly":
            return SupportVectorMachine._kernel_poly(D1, D2, self.kernel_args["degree"], self.kernel_args["offset"]) + reg_bias
        elif self.kernel_type == "rbf":
            return SupportVectorMachine._kernel_rbf(D1, D2, self.kernel_args["scale"]) + reg_bias
        else:
            # Should not arrive here
            raise ValueError(f"Unknown kernel type {self.kernel_type}")

    def fit(self, DTR, LTR, C=None, primal=False, **kernel_args):
        if C is not None:
            self.setParams(C=C)
        n = DTR.shape[1]

        self.kernel_args = kernel_args
        G = self._kernel_fun(DTR, DTR)
        ZTR = vcol(2 * LTR - 1)
        H = (ZTR @ ZTR.T) * G

        def opt(alpha):
            l_min = 0.5 * vrow(alpha) @ H @ vcol(alpha) - np.sum(alpha)
            grad = H @ vcol(alpha) - 1

            return l_min, grad.ravel()

        def primal_fun():
            v = 1 - ZTR * vcol(self.w.T @ self.expand(self.DTR, self.K)) # type: ignore
            m = np.max(np.hstack((v, np.zeros((v.shape[0], 1)))), axis=1)
            return 0.5 * alg.norm(self.w) ** 2 + self.C * np.sum(m)

        x, f, d = scopt.fmin_l_bfgs_b(func=opt,
                                      x0=np.zeros(n),
                                      bounds=[(0, self.C) for _ in range(n)],
                                      factr=1.0)

        self.alpha = x
        self.DTR = DTR
        self.ZTR = ZTR
        self.dual_loss = -f
        self.opt_info = d

        if primal:
            self.w = vcol(np.sum(vrow(vcol(x) * ZTR) * self.expand(self.DTR, self.K), axis=1))
            self.primal_loss = primal_fun()
            self.duality_gap = self.primal_loss - self.dual_loss

    def scores(self, DVAL):
        assert self.alpha is not None, "Model not trained. Fit the model before scoring."
        
        k = self._kernel_fun(self.DTR, DVAL)
        g = vcol(vcol(self.alpha) * self.ZTR)
        return vrow(np.sum(g * k, axis=0))

    def predict(self, DVAL, app_prior=0.5):
        s = self.scores(DVAL)
        threshold = -np.log(app_prior / (1 - app_prior))

        LPR = np.zeros((1, DVAL.shape[1]), dtype=np.int32)
        LPR[s >= threshold] = 1
        LPR[s < threshold] = 0

        return LPR

    def __str__(self):
        return (f"alpha: {self.alpha}, K: {self.K}, C: {self.C}, ker: {self.kernel_type},"
                f"ker_args: {self.kernel_args}, opt_info: {self.opt_info}")