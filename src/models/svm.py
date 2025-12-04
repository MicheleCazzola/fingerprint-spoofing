"""
    Support Vector Machine (SVM) model implementation.
    
    This module defines the SupportVectorMachine class, which provides methods for training,
    scoring, and predicting using SVMs with different kernel functions. It supports polynomial
    and RBF kernels.
"""

from datetime import datetime
import os
import joblib
import numpy as np
import scipy.linalg as alg
import scipy.optimize as scopt

from src.config.config import MODEL_PATH_SVM
from src.utils.utils import vcol, vrow

class SupportVectorMachine:
    """
    Class implementing Support Vector Machine (SVM) model for binary classification.
    Provides methods for fitting the model, scoring, and predicting class labels.
    
    Attributes:
        w (np.ndarray): Weight vector of the SVM model.
        alpha (np.ndarray): Dual coefficients of the SVM model.
        K (float): Kernel parameter.
        C (float): Regularization parameter.
        kernel_type (str): Type of kernel used in the SVM model.
        kernel_args (dict): Arguments for the kernel function.
        dual_loss (float): Dual loss value after training.
        primal_loss (float): Primal loss value after training.
        duality_gap (float): Duality gap after training.
        opt_info (dict): Optimization information from the training process.
        DTR (np.ndarray): Training data matrix.
        ZTR (np.ndarray): Transformed training data matrix.
    """
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
        """
        Load the state dictionary of the SVM model from a file.
        
        :param path: Directory path where the model file is located.
        :param id: Identifier for the model file.
        :return: True if the model was loaded successfully, False otherwise.
        """
        
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
        """
        Save the state dictionary of the SVM model to a file.
        
        :param filepath: Directory path where the model file will be saved.
        :return id: Identifier for the saved model file.
        """
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
        """
        Expand the data matrix D by adding an additional row with constant value K.
        
        :param D: Data matrix of shape (D, N).
        :param K: Constant value to be added as an additional row. If None, uses the instance's K attribute.
        :return D_exp: Expanded data matrix of shape (D+1, N).
        """
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
        """
        Compute the polynomial kernel between two data matrices.
        
        :param D1: First data matrix of shape (D, N1).
        :param D2: Second data matrix of shape (D, N2).
        :param degree: Degree of the polynomial kernel.
        :param offset: Offset term for the polynomial kernel.
        :return: Kernel matrix of shape (N1, N2).
        """
        return ((D1.T @ D2) + offset) ** degree

    @staticmethod
    def _kernel_rbf(D1, D2, scale):
        """
        Compute the RBF kernel between two data matrices.
        
        :param D1: First data matrix of shape (D, N1).
        :param D2: Second data matrix of shape (D, N2).
        :param scale: Scale parameter for the RBF kernel.
        :return: Kernel matrix of shape (N1, N2).
        """
        n1 = alg.norm(D1, ord=2, axis=0)
        n2 = alg.norm(D2, ord=2, axis=0)
        norm = vcol(n1) ** 2 + vrow(n2) ** 2 - 2 * D1.T @ D2 # type: ignore
        exponent = -scale * norm
        return np.exp(exponent)

    def _kernel_fun(self, D1, D2):
        """
        Compute the kernel matrix between two data matrices using the specified kernel type.
        
        :param D1: First data matrix of shape (D, N1).
        :param D2: Second data matrix of shape (D, N2).
        :raise ValueError: If kernel type is unknown or K/kernel_args are not set.
        :return: Kernel matrix of shape (N1, N2).
        """
        
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
        """
        Fit the SVM model to the training data using L-BFGS-B optimization.
        
        :param DTR: Training data matrix of shape (D, N).
        :param LTR: Training labels vector of shape (N,).
        :param C: Regularization parameter for the SVM model.
        :param primal: If True, compute the primal loss and duality gap after training.
        :param kernel_args: Arguments for the kernel function.
        """
        if C is not None:
            self.setParams(C=C)
        n = DTR.shape[1]

        self.kernel_args = kernel_args
        G = self._kernel_fun(DTR, DTR)
        ZTR = vcol(2 * LTR - 1)
        H = (ZTR @ ZTR.T) * G

        # (Dual) Objective function for SVM
        def opt(alpha):
            l_min = 0.5 * vrow(alpha) @ H @ vcol(alpha) - np.sum(alpha)
            grad = H @ vcol(alpha) - 1

            return l_min, grad.ravel()

        # Primal loss function for SVM
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
        """
        Compute the scores for the validation data using the trained SVM model.
        
        :param DVAL: Validation data matrix of shape (D, N).
        :return: Scores vector of shape (N,).
        """
        assert self.alpha is not None, "Model not trained. Fit the model before scoring."
        
        k = self._kernel_fun(self.DTR, DVAL)
        g = vcol(vcol(self.alpha) * self.ZTR)
        return vrow(np.sum(g * k, axis=0))

    def predict(self, DVAL, app_prior=0.5):
        """
        Predict class labels for the validation data using the trained SVM model.
        
        :param DVAL: Validation data matrix of shape (D, N).
        :param app_prior: Prior probability of the positive class in the application data.
        :return LPR: Predicted labels vector of shape (1, N).
        """
        s = self.scores(DVAL)
        threshold = -np.log(app_prior / (1 - app_prior))

        LPR = np.zeros((1, DVAL.shape[1]), dtype=np.int32)
        LPR[s >= threshold] = 1
        LPR[s < threshold] = 0

        return LPR

    def __str__(self):
        return (f"alpha: {self.alpha}, K: {self.K}, C: {self.C}, ker: {self.kernel_type},"
                f"ker_args: {self.kernel_args}, opt_info: {self.opt_info}")