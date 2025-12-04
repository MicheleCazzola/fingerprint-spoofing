"""
    Gaussian Mixture Model (GMM) implementation with EM and LBG algorithms.

    This module defines the GaussianMixtureModel class, which provides methods for fitting a GMM to data
    using the Expectation-Maximization (EM) algorithm and the LBG algorithm for initializing GMMs with multiple components.
"""

from datetime import datetime
import os
import joblib
import numpy as np
import scipy.special as scspec

from src.utils.fitting import logpdf_GAU_ND
from src.utils.utils import vcol, vrow


class GaussianMixtureModel:
    """
    Class for Gaussian Mixture Model (GMM) implementation with EM and LBG algorithms.
    
    Attributes:
        variant (str): Covariance type ('full', 'diag', 'tied').
        alpha (float): LBG splitting parameter.
        delta (float): Convergence threshold for EM.
        num_components (tuple): Number of components for each class.
        psi (float): Minimum eigenvalue for covariance bounding.
        gmm (list): List of GMM parameters for each class.
    """
    
    def __init__(self, variant="full", alpha=0.1, delta=1e-6, components=(1, 1), psi=0.01):
        self.variant = variant
        self.alpha = alpha
        self.delta = delta
        self.num_components = components
        self.psi = psi
        self.gmm = None
        
    def load_state_dict(self, path, id):
        """
        Load the state dictionary of the GMM model from a file.
        
        :param path: Directory path where the model file is located.
        :param id: Identifier for the model file.
        :return: True if the model was loaded successfully, False otherwise.
        """
        
        filepath = f"{path}/gmm_{id}.pkl"
        
        if not os.path.exists(filepath):
            print(f"Warning: GMM model file ('{filepath}') not found.")
            return False
        
        state_dict = joblib.load(filepath)
        
        self.variant = state_dict.get("variant", self.variant)
        self.alpha = state_dict.get("alpha", self.alpha)
        self.delta = state_dict.get("delta", self.delta)
        self.num_components = state_dict.get("num_components", self.num_components)
        self.psi = state_dict.get("psi", self.psi)
        self.gmm = state_dict.get("gmm", self.gmm)
        
        return True
    
    def save_state_dict(self, filepath):
        """
        Save the state dictionary of the GMM model to a file.
        
        :param filepath: Directory path where the model file will be saved.
        :return id: Identifier for the saved model file.
        """
        d = {
            "variant": self.variant,
            "alpha": self.alpha,
            "delta": self.delta,
            "num_components": self.num_components,
            "psi": self.psi,
            "gmm": self.gmm
        }
        
        id = datetime.now().strftime("%Y%m%d-%H%M%S%f")
        joblib.dump(d, f"{filepath}/gmm_{id}.pkl")
        return id

    def set_params(self, **kwargs):
        self.variant = kwargs.get("variant", self.variant)
        self.alpha = kwargs.get("alpha", self.alpha)
        self.delta = kwargs.get("threshold", self.delta)
        self.num_components = kwargs.get("components", self.num_components)
        self.psi = kwargs.get("psi", self.psi)

    @staticmethod
    def _log_joint(X, gmm):
        """
        Fit the GMM model to the data X and compute the log-joint probabilities.
        
        :param X: Data matrix of shape (D, N).
        :param gmm: GMM parameters (weights, means, covariances).
        :return S: Log-joint probability matrix of shape (M, N).
        """
        M = len(gmm)
        S = np.zeros((M, X.shape[1]))
        for g in range(len(gmm)):
            weight, mu, cov = gmm[g]
            S[g, :] = np.log(weight) + logpdf_GAU_ND(X, mu, cov)
        return S

    def _logpdf_GMM(self, X, c=None, gmm=None):
        """
        Generate log-probability density function for GMM.
        
        :param X: Data matrix of shape (D, N).
        :param c: Class label (optional).
        :param gmm: GMM parameters (optional).
        :return: Log-probability density vector of shape (1, N).
        """
        if gmm is None:
            assert self.gmm is not None, "GMM parameters not initialized. Fit the model first."
            assert c is not None, "Class label 'c' must be provided if GMM parameters are not given."
            gmm = self.gmm
            S = self._log_joint(X, gmm[c])
        else:
            S = self._log_joint(X, gmm)
        return self._log_marginal(S)

    @staticmethod
    def _log_marginal(log_joint):
        """
        Compute the log-marginal probabilities from log-joint probabilities.
        
        :param log_joint: Log-joint probability matrix of shape (M, N).
        :return r: Log-marginal probability vector of shape (1, N).
        """
        r = scspec.logsumexp(log_joint, axis=0, return_sign=False)
        return vrow(r) # type: ignore

    def _expectation(self, X, gmm):
        """
        Perform the expectation step of the EM algorithm. 
        
        :param X: Data matrix of shape (D, N).
        :param gmm: GMM parameters (weights, means, covariances).
        :return responsibilities: Responsibilities matrix of shape (M, N).
        """
        Slog_joint = self._log_joint(X, gmm)
        Slog_marginal = self._log_marginal(Slog_joint)
        responsibilities = np.exp(Slog_joint - Slog_marginal)

        return responsibilities

    def _check(self, X, gmm_current, previous_ll):
        """
        Check for convergence based on the change in log-likelihood.
        
        :param X: Data matrix of shape (D, N).
        :param gmm_current: Current GMM parameters.
        :param previous_ll: Previous log-likelihood value.
        :return converged: Boolean indicating if convergence criteria met.
        :return new_ll: New log-likelihood value.
        """
        new_logdens = self._logpdf_GMM(X, gmm=gmm_current)
        new_ll = np.mean(new_logdens)
        gain = new_ll - previous_ll
        #assert gain >= 0
        return gain <= self.delta, new_ll

    def _bound_cov(self, cov):
        """
        Apply covariance bounding to ensure numerical stability.
        
        :param cov: Covariance matrix.
        :return newCov: Bounded covariance matrix.
        """
        U, s, _ = np.linalg.svd(cov)
        s[s < self.psi] = self.psi
        newCov = U @ (vcol(s) * U.T)

        return newCov

    def _cov_transform(self, w, cov):
        """
        Transform covariance matrices based on the specified variant.
        
        :param w: List of weights for each Gaussian component.
        :param cov: List of covariance matrices for each Gaussian component.
        :return cov: Transformed list of covariance matrices.
        """
        if self.variant == "diag":
            cov = [covg * np.eye(covg.shape[0]) for covg in cov]
        elif self.variant == "tied":
            cov = [np.sum(np.array(w)[:, np.newaxis, np.newaxis] * np.array(cov), axis=0)] * len(cov)
        return cov

    def _maximization(self, X, r):
        """
        Perform the maximization step of the EM algorithm.
        
        :param X: Data matrix of shape (D, N).
        :param r: Responsibilities matrix of shape (M, N).
        :return: Updated GMM parameters (weights , means, covariances).
        """
        Z = np.sum(r, axis=1)
        F = [vcol(np.sum(r[g] * X, axis=1)) for g in range(len(r))]
        S = [np.sum(((r[g] * X)[:, :, np.newaxis] * X[:, :, np.newaxis].T).T, axis=1) for g in range(len(r))]

        mu = [F[g] / Z[g] for g in range(len(r))]
        cov = [S[g] / Z[g] - mu[g] @ mu[g].T for g in range(len(r))]
        w = [Z[g] / X.shape[1] for g in range(len(r))]

        cov = self._cov_transform(w, cov)
        cov = [self._bound_cov(covg) for covg in cov]

        return list(zip(w, mu, cov))

    def _EM(self, X, gmm):
        """
        Fit the GMM model to the data X using the EM algorithm.
        
        :param X: Data matrix of shape (D, N).
        :param gmm: Initial GMM parameters (weights, means, covariances).
        :return gmm: Fitted GMM parameters.
        :return avg_ll: Average log-likelihood of the fitted GMM.
        """
        avg_ll = np.mean(self._logpdf_GMM(X, gmm=gmm))

        stop = False
        while not stop:
            responsibilities = self._expectation(X, gmm)
            gmm = self._maximization(X, responsibilities)

            stop, avg_ll = self._check(X, gmm, avg_ll)

        return gmm, avg_ll

    def _LBG(self, gmm):
        """
        Perform the LBG algorithm to split Gaussian components. Used for initializing GMMs with more components.
        
        :param gmm: Current GMM parameters (weights, means, covariances).
        :return new_gmm: New GMM parameters after splitting.
        """
        new_gmm = []
        for g in gmm:
            w, mu, cov = g

            U, s, Vh = np.linalg.svd(cov)
            d = U[:, 0:1] * s[0] ** 0.5 * self.alpha

            new_gmm.append((w / 2, mu - d, cov))
            new_gmm.append((w / 2, mu + d, cov))

        return new_gmm

    def _EM_LBG(self, XTR, label):
        """
        Fit the GMM model to the data XTR using the LBG algorithm followed by EM.
        
        :param XTR: Data matrix of shape (D, N) for a specific class.
        :param label: Class label.
        :return gmm: Fitted GMM parameters.
        :return avg_ll: Average log-likelihood of the fitted GMM.
        """
        mu = vcol(np.sum(XTR, axis=1)) / XTR.shape[1]
        cov = (XTR - mu) @ (XTR - mu).T / XTR.shape[1]
        cov = self._cov_transform([1.0], [cov])
        cov = self._bound_cov(cov[0])
        gmm = [(1.0, mu, cov)]

        # 1-GMM no LBG
        if self.num_components[label] == 1:
            return gmm, None
            # return self._EM(XTR, gmm)

        avg_ll = None
        while len(gmm) < self.num_components[label]:
            starting_gmm = self._LBG(gmm)
            gmm, avg_ll = self._EM(XTR, starting_gmm)

        return gmm, avg_ll

    def _scores(self, DVAL, LVAL):
        """Compute the log-probability density scores for each class.
        
        :param DVAL: Data matrix of shape (D, N).
        :param LVAL: Labels vector of shape (N,).
        :return S: Score matrix of shape (C, N) where C is the number of classes.
        """
        S = np.zeros((len(np.unique(LVAL)), DVAL.shape[1]))
        for label in sorted(np.unique(LVAL)):
            S[label, :] = self._logpdf_GMM(DVAL, c=label)

        return S

    def scores(self, DVAL, LVAL):
        """Compute the log-likelihood ratio scores for binary classification.
        
        :param DVAL: Data matrix of shape (D, N).
        :param LVAL: Labels vector of shape (N,).
        :return: Log-likelihood ratio scores.
        """
        S = self._scores(DVAL, LVAL)
        return vrow(S[1, :] - S[0, :])

    def fit(self, DTR, LTR, **kwargs):
        """
        Fit the GMM model to the training data DTR with labels LTR.
        
        :param DTR: Training data matrix of shape (D, N).
        :param LTR: Training labels vector of shape (N,).
        :param kwargs: Additional parameters for model configuration.
        """
        self.set_params(**kwargs)
        self.gmm = None

        gmm_list = []
        for label in sorted(np.unique(LTR)):
            gmm, ll = self._EM_LBG(DTR[:, LTR == label], label)
            gmm_list.append(gmm)
        self.gmm = gmm_list

    def predict(self, DVAL, LVAL, app_prior=0.5):
        """
        Predict class labels for the validation data DVAL based on the fitted GMM model.
        
        :param DVAL: Validation data matrix of shape (D, N).
        :param LVAL: True labels vector of shape (N,).
        :param app_prior: Application prior for binary classification thresholding.
        :return LPR: Predicted labels vector of shape (N,).
        """
        S = self._scores(DVAL, LVAL)

        if len(np.unique(LVAL)) == 2:
            threshold = -np.log(app_prior / (1 - app_prior))
            LPR = np.zeros((1, LVAL.shape[0]), dtype=np.int32)
            llr = vrow(S[1, :] - S[0, :])
            LPR[llr >= threshold] = 1
            LPR[llr < threshold] = 0
        else:
            S += vcol(np.log(np.ones(3) / 3))
            LPR = np.argmax(S, axis=0)

        return LPR