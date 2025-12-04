"""
    Logistic Regression model implementation.
    
    This module defines the LogReg class, which provides methods for training,
    scoring, and predicting using logistic regression. It supports standard,
    prior-weighted, and quadratic variants of logistic regression.
"""

from datetime import datetime
import os
import joblib
import numpy as np
from scipy import optimize as opt, linalg as alg


from src.config.config import LR_STANDARD, PRIOR_WEIGHTED_LR, QUADRATIC_LR
from src.utils.utils import vrow, vcol


class LogReg:
    """
    Logistic Regression model supporting standard, prior-weighted, and quadratic variants.
    Provides methods for fitting the model, scoring, and predicting class labels.
    
    Attributes:
        variant (int): Variant of logistic regression to use.
        w (np.ndarray): Weight vector of the logistic regression model.
        b (float): Bias term of the logistic regression model.
        j_min (float): Minimum value of the objective function after training.
        opt_info (dict): Optimization information from the training process.
        training_prior (float): Prior probability of the positive class in the training data.
        app_prior (float): Prior probability of the positive class in the application data.
    """
    def __init__(self, variant=LR_STANDARD, training_prior=None, app_prior=None):
        self.variant = variant
        self.w = None
        self.b = None
        self.j_min = None
        self.opt_info = None
        self.training_prior = training_prior
        self.app_prior = app_prior
        
    def load_state_dict(self, path, id):
        """
        Load the state dictionary of the LR model from a file.
        
        :param path: Directory path where the model file is located.
        :param id: Identifier for the model file.
        :return: True if the model was loaded successfully, False otherwise.
        """
        filepath = f"{path}/logreg_{id}.pkl"
        
        if not os.path.exists(filepath):
            print(f"Warning: LR model file ('{filepath}') not found.")
            return False
        
        state_dict = joblib.load(filepath)
        
        self.variant = state_dict.get("variant", self.variant)
        self.w = state_dict.get("w", self.w)
        self.b = state_dict.get("b", self.b)
        self.j_min = state_dict.get("j_min", self.j_min)
        self.opt_info = state_dict.get("opt_info", self.opt_info)
        self.training_prior = state_dict.get("training_prior", self.training_prior)
        self.app_prior = state_dict.get("app_prior", self.app_prior)
        
        return True
        
    def save_state_dict(self, filepath):
        """
        Save the state dictionary of the LR model to a file.
        
        :param filepath: Directory path where the model file will be saved.
        :return id: Identifier for the saved model file.
        """
        d = {
            "variant": self.variant,
            "w": self.w,
            "b": self.b,
            "j_min": self.j_min,
            "opt_info": self.opt_info,
            "training_prior": self.training_prior,
            "app_prior": self.app_prior
        }
        
        id = datetime.now().strftime("%Y%m%d-%H%M%S%f")
        joblib.dump(d, f"{filepath}/logreg_{id}.pkl")
        return id

    def setParams(self, **kwargs):
        self.variant = kwargs.get("variant", self.variant)
        self.training_prior = kwargs.get("training_prior", self.training_prior)
        self.app_prior = kwargs.get("app_prior", self.app_prior)

    def fit(self, DTR, LTR, reg_coeff=0, training_prior=None, app_prior=None):
        """
        Fit the logistic regression model to the training data. Uses L-BFGS-B optimization.
        
        :param DTR: Training data matrix of shape (D, N).
        :param LTR: Training labels vector of shape (N,).
        :param reg_coeff: Regularization coefficient for L2 regularization.
        :param training_prior: Prior probability of the positive class in the training data.
        :param app_prior: Prior probability of the positive class in the application data.
        :raise ValueError: If prior-weighted logistic regression is selected without an application prior.
        """
        D = DTR.shape[0]
        n = DTR.shape[1]

        if self.variant == PRIOR_WEIGHTED_LR and app_prior is None:
            raise ValueError("Application prior must be defined if variant is prior-weighted")

        self.app_prior = app_prior if app_prior is not None else np.sum(LTR == 1) / n
        self.training_prior = training_prior if training_prior is not None else self.app_prior

        # Objective function for standard logistic regression
        def logreg_obj_lr(v):
            w, b = v[0:-1], v[-1]
            S = (vcol(w).T @ DTR + b).ravel()
            ZTR = 2 * LTR - 1
            J_min = reg_coeff * alg.norm(w, 2) ** 2 / 2 + np.sum(np.logaddexp(0, -ZTR * S)) / n

            G = -ZTR / (1 + np.exp(ZTR * S))
            grad_b = np.array([np.sum(G) / n])
            grad_w = reg_coeff * w + np.sum(vrow(G) * DTR, axis=1) / n
            return J_min, np.concatenate((grad_w, grad_b))

        # Objective function for prior-weighted logistic regression
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

        loss_function = logreg_obj_lr if self.variant in [LR_STANDARD, QUADRATIC_LR] else logreg_obj_pwlr

        x, f_min, d = opt.fmin_l_bfgs_b(func=loss_function,
                                        approx_grad=False,
                                        x0=np.zeros(D + 1))
        self.w = x[0:-1]
        self.b = x[-1]
        self.j_min = f_min
        self.opt_info = d

    def scores(self, features):
        """
        Compute the scores for the given features using the trained logistic regression model.
        
        :param features: Data matrix of shape (D, N).
        :return: Scores vector of shape (N,).
        """
        assert self.training_prior is not None, "Training prior must be defined for scoring."
        assert self.w is not None and self.b is not None, "No model defined"
        return vrow(self.w) @ features + self.b - np.log(self.training_prior / (1 - self.training_prior))

    def predict(self, features, app_prior=None):
        """
        Predict class labels for the given features using the trained logistic regression model.
        
        :param features: Data matrix of shape (D, N).
        :param app_prior: Prior probability of the positive class in the application data.
        :return LPR: Predicted labels vector of shape (1, N).
        """

        if self.w is None or self.b is None or self.app_prior is None:
            raise ValueError("No model defined")

        S = self.scores(features)
        target_prior = app_prior if app_prior is not None else self.app_prior
        threshold = -np.log(target_prior / (1 - target_prior))

        LPR = np.zeros((1, features.shape[1]), dtype=np.int32)
        LPR[S >= threshold] = 1
        LPR[S < threshold] = 0

        return LPR