import numpy as np

from src.config.config import GAUSSIAN_MODELS
from src.dimred.pca import PCA
from src.evaluator.evaluator import Evaluator
from src.utils.fitting import compute_estimators, logpdf_GAU_ND
from src.utils.utils import vcol, vrow


class MVG:
    def __init__(self):
        
        self.mean = None
        self.cov = None
        
    def _set_threshold(self, prior_false: float, prior_true: float) -> float:
        """
        Set the threshold based on prior probabilities.
        
        :param prior_false: prior probability of false class
        :param prior_true: prior probability of true class
        :return: threshold
        """
        return -np.log(prior_true / prior_false)
    
    def _estimate_parameters(self, data, labels):
        """
        Estimate the mean and covariance for each class in the data.
        
        :param data: data to estimate parameters from
        :param labels: labels of the data
        :return: means and covariances for each class
        """
        mu_c, cov_c = [], []
        for c in np.unique(labels):
            Dc = data[:, labels == c]
            mu, cov = compute_estimators(Dc, np.mean(Dc, axis=1))
            mu_c.append(vcol(mu))
            cov_c.append(cov)

        return np.array(mu_c), np.array(cov_c)
    
    def _compute_class_conditional(self, data, mu_c, cov_c):
        """
        Compute the class-conditional probabilities.
        
        :param data: data to compute class-conditional probabilities for each class
        :param mu_c: means for each class
        :param cov_c: covariances for each class
        :return: class-conditional probabilities for each class
        """
        return logpdf_GAU_ND(data, mu_c[0], cov_c[0]), logpdf_GAU_ND(data, mu_c[1], cov_c[1])
    
    def _compute_llr(self, data, mu_c, cov_c):
        """
        Compute the log-likelihood ratio.
        
        :param data: data to compute log-likelihood ratio for each class
        :param mu_c: means for each class
        :param cov_c: covariances for each class
        :return: log-likelihood ratio
        """
        cc_false, cc_true = self._compute_class_conditional(data, mu_c, cov_c)
        return cc_true - cc_false

    def _predict(self, labels, llr, threshold):
        """
        Predict class labels based on log-likelihood ratio and threshold.
        
        :param labels: true labels of the data
        :param llr: log-likelihood ratio for the data
        :param threshold: threshold to decide class labels
        :return: predicted class labels
        """
        preds = np.zeros(labels.shape, dtype=np.int32)
        preds[llr >= threshold] = 1
        preds[llr < threshold] = 0

        return preds

    def _compute_error_rate(self, preds, labels):
        """
        Compute the error rate between predicted and true labels.
        
        :param preds: predicted class labels
        :param labels: true class labels
        :return: error rate
        """
        return np.mean(preds != labels)

    def _compute_predictions(self, data, labels, mu_c, cov_c, thresholds):
        """
        Compute predictions for the data based on log-likelihood ratio and thresholds.
        
        :param data: data to compute predictions for
        :param labels: true labels of the data
        :param mu_c: means for each class
        :param cov_c: covariances for each class
        :param thresholds: thresholds to decide class labels
        :return: log-likelihood ratio and predicted class labels for each threshold
        """
        llr = self._compute_llr(data, mu_c, cov_c)
        preds = [self._predict(labels, llr, t) for t in thresholds]
        return llr, preds
    
    def _classify(self, data: np.ndarray, labels: np.ndarray, mu: np.ndarray, cov: np.ndarray, eff_priors: np.ndarray, pca: PCA | None, evaluate: bool, eval_results: dict | None, model_name: str) -> dict:
        """
        Classify the data based on the estimated parameters and effective priors.
        
        
        :param data: data to classify
        :param labels: true labels of the data
        :param mu: means for each class
        :param cov: covariances for each class
        :param eff_priors: effective priors for each class
        :param pca: PCA object (if PCA is applied)
        :param evaluate: whether to evaluate the results
        :param eval_results: evaluation results dictionary
        :param model_name: name of the model
        :return: error rates
        """
        
        thresholds = [self._set_threshold(1 - eff_prior, eff_prior) for eff_prior in eff_priors]
        llr, preds = self._compute_predictions(data, labels, mu, cov, thresholds)
        
        if evaluate:
            
            assert eval_results is not None, "Evaluation results dictionary must be provided if evaluate is True."
            
            for (eff_prior, pred) in zip(eff_priors, preds):
                result = Evaluator.evaluate(llr, pred, labels, eff_prior=eff_prior, pca=pca.n_components if pca is not None else None)
                params, results = result['params'], result['results']
                pca_params = params["pca"] if params['pca'] is not None else "Not applied"
                entry = eval_results[eff_prior].get(pca_params, {})
                entry[model_name] = results
                eval_results[eff_prior][pca_params] = entry
                
            # Error rate with 0.5 as application prior
            err_rate = self._compute_error_rate(preds[1], labels)
        else:
            
            # Error rate with 0.1 as application prior
            err_rate = self._compute_error_rate(preds[0], labels)

        return err_rate
    
    def _get_cov(self, *_) -> np.ndarray | None:
        """
        Get the covariance matrices for each class.
        
        :return: covariance matrices for each class
        """
        return self.cov
    
    def get_name(self) -> str:
        """
        Get the name of the MVG model.
        
        :return: name of the MVG model
        """
        return GAUSSIAN_MODELS[0]
    
    def compute_correlations(self, train_data, train_labels):
        """
        Compute the correlation matrices for each class.
        
        :param train_data: training data
        :param train_labels: training labels
        :return: correlation matrices for each class
        """
        
        if self.cov is None:
            self.fit(train_data, train_labels)
        
        assert self.cov is not None, "Covariance matrix not computed. Fit the model first."
        
        return [C / (vcol(C.diagonal() ** 0.5) * vrow(C.diagonal() ** 0.5)) for C in self.cov]
    
    def fit(self, train_data, train_labels):
        """
        Fit the MVG model to the training data.
        
        :param train_data: training data
        :param train_labels: training labels
        """
        self.mean, self.cov = self._estimate_parameters(train_data, train_labels)
        
    def predict(self, data, labels, eff_priors, pca=None, evaluate=False, eval_results=None):
        """
        Predict class labels for the data.
        
        :param data: data to predict class labels for
        :param labels: true labels of the data
        :param eff_priors: effective priors for each class
        :param pca: PCA object (if PCA is applied)
        :param evaluate: whether to evaluate the results
        :param eval_results: evaluation results dictionary
        :return: error rates
        """
        
        cov, name = self._get_cov(), self.get_name()
        
        assert self.mean is not None, "Mean vector not computed. Fit the model first."
        assert cov is not None, "Covariance matrix not computed. Fit the model first."
                
        err_rates = self._classify(
            data,
            labels,
            self.mean,
            cov,
            eff_priors,
            pca=pca,
            evaluate=evaluate,
            eval_results=eval_results,
            model_name=name
        )   
        
        return err_rates 
    
class NaiveBayesMVG(MVG):
    def __init__(self):
        super().__init__()
                
    def _compute_cov_naive_approx(self) -> np.ndarray:
        """
        Compute the naive approximation of the covariance matrix.
        
        :return: naive covariance matrix
        """
        assert self.cov is not None, "Covariance matrix not computed. Fit the model first."
        return self.cov * np.eye(self.cov[0].shape[1])
    
    def _get_cov(self, *_) -> np.ndarray:
        return self._compute_cov_naive_approx()
    
    def get_name(self) -> str:
        return GAUSSIAN_MODELS[2]
    
class TiedMVG(MVG):
    def __init__(self):
        super().__init__()
        
    def _compute_within_class_covariance(self, train_data: np.ndarray, train_labels: np.ndarray) -> np.ndarray:
        """
        Compute the within-class covariance matrix.
        
        :param train_data: training data
        :param train_labels: training labels
        :return: within-class covariance matrix
        """
        assert self.cov is not None, "Covariance matrix not computed. Fit the model first."
        return (train_data[:, train_labels == 0].shape[1] * self.cov[0] + train_data[:, train_labels == 1].shape[1] * self.cov[1]) / train_data.shape[1]
                
    def _get_cov(self, train_data, train_labels): # type: ignore
        cov = self._compute_within_class_covariance(train_data, train_labels)
        return np.array([cov] * 2)
    
    def get_name(self) -> str:
        return GAUSSIAN_MODELS[1]
    
    def predict(self, train_data: np.ndarray, train_labels: np.ndarray, test_data: np.ndarray, test_labels: np.ndarray, eff_priors: np.ndarray, pca=None, evaluate=False, eval_results=None) -> dict: # type: ignore
        """
        Predict class labels for the test data using tied covariance matrix.
        
        :param train_data: training data
        :param train_labels: training labels
        :param test_data: test data
        :param test_labels: test labels
        :param eff_priors: effective priors for each class
        :param pca: PCA object (if PCA is applied)
        :param evaluate: whether to evaluate the results
        :param eval_results: evaluation results dictionary
        :return: error rates
        """
        
        cov = self._get_cov(train_data, train_labels)
        name = self.get_name()
        
        assert self.mean is not None, "Mean vector not computed. Fit the model first."
        assert cov is not None, "Covariance matrix not computed. Fit the model first."
                
        err_rates = self._classify(
            test_data,
            test_labels,
            self.mean,
            cov,
            eff_priors,
            pca=pca,
            evaluate=evaluate,
            eval_results=eval_results,
            model_name=name
        )   
        
        return err_rates