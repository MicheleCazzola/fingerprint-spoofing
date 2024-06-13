from pprint import pprint

import numpy as np
from src.io.constants import LR_STANDARD, PRIOR_WEIGHTED_LR, GAUSSIAN


def relative_mis_calibration(dcfs):
    return 100 * (dcfs["dcf"] - dcfs["min_dcf"]) / dcfs["min_dcf"]


class Evaluator:
    def __init__(self, model_name):
        self.model_name = model_name
        self.results = []
        self.data_models = []

    def evaluate(self, llr, LPR, LTE, eff_prior, pca_dim, cache=True):
        M = self.compute_confusion_matrix(LPR, LTE, 2)
        dummy_risk = Evaluator.dummy_risk(eff_prior, 1, 1)
        dcf = Evaluator.normalized_DCF(M, eff_prior, dummy_risk)
        min_dcf = Evaluator.minimum_DCF(llr, LTE, eff_prior, dummy_risk)

        self.results.append({
            eff_prior: {
                "pca": pca_dim,
                "dcf": dcf,
                "min_dcf": min_dcf
            }
        })

        if cache:
            self.data_models.append({
                eff_prior: {
                    "pca": pca_dim,
                    "llr": llr,
                    "LTE": LTE
                }
            })

    def evaluate2(self, llr, LPR, LTE, eff_prior, **model_params):
        M = self.compute_confusion_matrix(LPR, LTE, 2)
        dummy_risk = Evaluator.dummy_risk(eff_prior, 1, 1)
        dcf = Evaluator.normalized_DCF(M, eff_prior, dummy_risk)
        min_dcf = Evaluator.minimum_DCF(llr, LTE, eff_prior, dummy_risk)

        model_results = {
            "dcf": dcf,
            "min_dcf": min_dcf,
            "llr": llr,
            "LTE": LTE
        }

        model_params["eff_prior"] = eff_prior

        return {
            "params": model_params,
            "results": model_results
        }

    def get_results(self):
        return self.results

    @staticmethod
    def _best_configuration_gaussian(eval_results_prior):
        best = {"MVG": {}, "Tied MVG": {}, "Naive Bayes MVG": {}}
        for pca in eval_results_prior:
            for model_name in eval_results_prior[pca]:
                config_value = eval_results_prior[pca][model_name]
                if config_value["min_dcf"] < best[model_name].get("min_dcf", 1):
                    best[model_name] = config_value
                    best[model_name]["pca"] = pca if pca != "Not applied" else None

        return best

    @staticmethod
    def best_configuration(eval_results, mode, eff_prior):

        if mode == GAUSSIAN:
            return Evaluator._best_configuration_gaussian(eval_results[eff_prior])

    @staticmethod
    def compute_confusion_matrix(LPR, LTE, n_classes):
        M = np.zeros((n_classes, n_classes), dtype=np.int32)

        # print(LPR.shape, LTE.shape)
        for (p, c) in zip(LPR.ravel(), LTE.ravel()):
            M[p, c] += 1

        return M

    @staticmethod
    def dummy_risk(prior, C_fn=1, C_fp=1):
        return min(prior * C_fn, (1 - prior) * C_fp)

    @staticmethod
    def unnormalized_DCF(M, eff_prior):
        return eff_prior * (M[0, 1] / (M[0, 1] + M[1, 1])) + (1 - eff_prior) * (M[1, 0] / (M[0, 0] + M[1, 0]))

    @staticmethod
    def normalized_DCF(M, eff_prior, dummy_risk):
        return Evaluator.unnormalized_DCF(M, eff_prior) / dummy_risk

    @staticmethod
    def minimum_DCF(llr, LTE, eff_prior, dummy_risk):
        min_DCF = np.inf

        # Initially, confusion matrix has only false and true positives
        # Since threshold value is below min(llr)
        M = np.array([[0, 0], [np.sum(LTE == 0), np.sum(LTE == 1)]])
        for threshold in sorted(np.unique(llr)):
            # From false positive to true negative
            below_threshold = np.sum((llr == threshold) & (LTE == 0))
            # From true positive to false negative
            above_threshold = np.sum((llr == threshold) & (LTE == 1))

            # Update matrix
            M[0, 0] += below_threshold
            M[1, 0] -= below_threshold
            M[0, 1] += above_threshold
            M[1, 1] -= above_threshold

            min_DCF = min(min_DCF, Evaluator.normalized_DCF(M, eff_prior, dummy_risk))

        return min_DCF

    @staticmethod
    def bayes_error(llr, LTE):
        effective_prior_log_odds = np.linspace(-4, 4, 33)

        dcf, min_dcf = [], []
        for threshold in -effective_prior_log_odds:
            LPR = np.array(llr > threshold, dtype=np.int32)
            M = Evaluator.compute_confusion_matrix(LPR, LTE, llr.shape[0])
            effetctive_prior = 1 / (1 + np.exp(threshold))
            dummy_risk = Evaluator.dummy_risk(effetctive_prior)
            dcf.append(Evaluator.normalized_DCF(M, effetctive_prior, dummy_risk))
            min_dcf.append(Evaluator.minimum_DCF(llr, LTE, effetctive_prior, dummy_risk))

        return effective_prior_log_odds, {"min_dcf": min_dcf, "dcf": dcf}
