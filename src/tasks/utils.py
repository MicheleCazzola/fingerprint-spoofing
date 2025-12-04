import numpy as np
from src.evaluator.evaluator import Evaluator
from src.models.svm import SupportVectorMachine


def optimal_bayes(svm: SupportVectorMachine, DVAL: np.ndarray, LVAL: np.ndarray, app_prior: float):
    """
    Perform SVM evaluation using optimal Bayes decision rule.
    
    :param svm: Trained SVM model.
    :param DVAL: Validation data.
    :param LVAL: Validation labels.
    :param app_prior: Prior probability of the positive class in the application data.
    :return min_dcf: Minimum DCF.
    :return dcf: Actual DCF.
    :return llr: Log-likelihood ratios.
    """
    llr = svm.scores(DVAL)
    LPR = svm.predict(DVAL, app_prior)

    min_dcf, dcf = map(Evaluator.evaluate(llr, LPR, LVAL, eff_prior=app_prior)["results"].get,
                       ["min_dcf", "dcf"])
    return min_dcf, dcf, llr