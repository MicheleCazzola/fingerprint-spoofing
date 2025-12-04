"""
    Evaluation utilities for machine learning models.

    This module provides the Evaluator class, which includes static methods for evaluating
    various machine learning models such as Gaussian models, Logistic Regression, Support Vector Machines, 
    and Gaussian Mixture Models.
"""

import numpy as np

from src.config.config import GAUSSIAN, GAUSSIAN_MODELS, GMM, LR, SVM


class Evaluator:
    """
    Class with evaluation utilities for machine learning models.
    """
    
    @staticmethod
    def evaluate(llr, LPR, LVAL, eff_prior, **model_params):
        """
        Evaluate the performance of a model using log-likelihood ratios (llr), predicted labels (LPR),
        true labels (LVAL), and an effective prior (eff_prior). Additional model parameters can be passed
        as keyword arguments.
        
        :param llr: log-likelihood ratios
        :param LPR: predicted labels
        :param LVAL: true labels
        :param eff_prior: effective prior probability
        :param model_params: additional model parameters
        :return: dictionary containing model parameters and evaluation results
        """
        M = Evaluator.compute_confusion_matrix(LPR, LVAL, 2)
        dummy_risk = Evaluator.dummy_risk(eff_prior, 1, 1)
        dcf = Evaluator.normalized_DCF(M, eff_prior, dummy_risk)
        min_dcf = Evaluator.minimum_DCF(llr, LVAL, eff_prior, dummy_risk)

        model_results = {
            "dcf": dcf,
            "min_dcf": min_dcf,
            "llr": llr,
            "LVAL": LVAL
        }

        model_params["eff_prior"] = eff_prior

        return {
            "params": model_params,
            "results": model_results
        }

    @staticmethod
    def _best_configuration_gaussian(eval_results_prior):
        """
        Selects the best configuration for Gaussian models based on minimum DCF across different PCA settings.
        
        :param eval_results_prior: dictionary containing evaluation results for different PCA settings
        :return best: dictionary with the best configuration for each Gaussian model
        """
        best = dict(zip(GAUSSIAN_MODELS, [dict() for _ in GAUSSIAN_MODELS]))
        for pca in eval_results_prior:
            for model_name in eval_results_prior[pca]:
                config_value = eval_results_prior[pca][model_name]
                if config_value["min_dcf"] < best[model_name].get("min_dcf", 1):
                    best[model_name] = config_value
                    best[model_name]["pca"] = pca if pca != "Not applied" else None

        return best

    @staticmethod
    def _best_configuration_LR(eval_results):
        """
        Selects the best configuration for Logistic Regression based on minimum DCF.
        
        :param eval_results: list of evaluation results for different configurations
        :return: dictionary with the best configuration for Logistic Regression
        """
        min_dcfs = [r[0] for r in eval_results]
        best_conf = np.argmin(min_dcfs)
        return {
            "min_dcf": eval_results[best_conf][0],
            "act_dcf": eval_results[best_conf][4],
            "llr": eval_results[best_conf][2],
            "params": {
                "variant": eval_results[best_conf][5],
                "λ": eval_results[best_conf][1]
            },
            "id": eval_results[best_conf][6]
        }

    @staticmethod
    def _best_configuration_SVM(eval_results):
        """
        Selects the best configuration for Support Vector Machine based on minimum DCF.
        
        :param eval_results: list of evaluation results for different configurations
        :return: dictionary with the best configuration for Support Vector Machine
        """
        min_dcfs = [r[1] for r in eval_results[:-1]]
        best_conf_no_rbf = np.argmin(min_dcfs)
        min_dcfs_rbf = [r[1] for (_, r) in eval_results[-1].items()]
        best_conf_rbf = np.argmin(min_dcfs_rbf)
        best = "RBF" if list(eval_results[-1].values())[best_conf_rbf][1] < eval_results[:-1][best_conf_no_rbf][1] else "NO_RBF"

        if best == "RBF":
            best_conf = list(eval_results[-1].values())[best_conf_rbf]
            min_dcf = best_conf[1]
            dcf = best_conf[4]
            llr = best_conf[3]
            id = best_conf[5]
            params = {
                "kernel": best_conf[6],
                "scale": list(eval_results[-1].keys())[best_conf_rbf],
                "C": best_conf[0],
                "K": best_conf[2]
            }
        else:
            best_conf = eval_results[:-1][best_conf_no_rbf]
            min_dcf = best_conf[1]
            dcf = best_conf[4]
            llr = best_conf[3]
            id = best_conf[5]
            params = {
                "kernel": best_conf[6],
                "C": best_conf[0],
                "K": best_conf[2]
            }

        return {
            "min_dcf": min_dcf,
            "act_dcf": dcf,
            "llr": llr,
            "id": id,
            "params": params
        }

    @staticmethod
    def _best_configuration_GMM(eval_results):
        """
        Selects the best configuration for Gaussian Mixture Models based on minimum DCF.
        
        :param eval_results: dictionary containing evaluation results for different configurations
        :return: dictionary with the best configuration for Gaussian Mixture Models
        """
        all_results = [["full", c, v["min_dcf"], v["dcf"], v["llr"], v["id"]] for (c, v) in eval_results["full"].items()] + \
                      [["diag", c, v["min_dcf"], v["dcf"], v["llr"], v["id"]] for (c, v) in eval_results["diag"].items()]
        min_dcfs = [r[2] for r in all_results]
        best_conf = np.argmin(min_dcfs)

        return {
            "min_dcf": all_results[best_conf][2],
            "act_dcf": all_results[best_conf][3],
            "llr": all_results[best_conf][4],
            "id": all_results[best_conf][5],
            "params": {
                "variant": all_results[best_conf][0],
                "components": all_results[best_conf][1]
            }
        }

    @staticmethod
    def best_configuration(eval_results, mode, eff_prior=0.1) -> dict:
        """
        Selects the best configuration for a given model based on minimum DCF.
        
        :param eval_results: evaluation results for different configurations
        :param mode: model type (GAUSSIAN, LR, SVM, GMM)
        :param eff_prior: effective prior probability (used for MVG models, default is 0.1)
        :return: dictionary with the best configuration for the specified model
        :raise: KeyError if an invalid model name is provided
        """
        if mode == GAUSSIAN:
            return Evaluator._best_configuration_gaussian(eval_results[eff_prior])
        if mode == LR:
            return Evaluator._best_configuration_LR(eval_results)
        if mode == SVM:
            return Evaluator._best_configuration_SVM(eval_results)
        if mode == GMM:
            return Evaluator._best_configuration_GMM(eval_results)
        raise KeyError("Invalid model name")


    @staticmethod
    def best_model(model_results, key):
        """
        Selects the best model based on a specified key.
        
        :param model_results: dictionary containing model results
        :param key: key to use for selecting the best model
        :return: dictionary with the best model
        """
        keys = [model[key] for model in model_results.values()]
        best_model = np.argmin(keys)

        return dict([list(model_results.items())[best_model]])

    @staticmethod
    def compute_confusion_matrix(LPR, LVAL, n_classes):
        """
        Computes the confusion matrix given predicted labels and true labels.
        
        :param LPR: predicted labels
        :param LVAL: true labels
        :param n_classes: number of classes
        :return M: confusion matrix
        """
        M = np.zeros((n_classes, n_classes), dtype=np.int32)

        # print(LPR.shape, LVAL.shape)
        for (p, c) in zip(LPR.ravel(), LVAL.ravel()):
            M[p, c] += 1

        return M

    @staticmethod
    def dummy_risk(prior, C_fn=1, C_fp=1):
        """
        Computes the dummy risk given prior probabilities and costs.
        
        :param prior: prior probability of the positive class
        :param C_fn: cost of false negative (default is 1)
        :param C_fp: cost of false positive (default is 1)
        :return: dummy risk value
        """
        return min(prior * C_fn, (1 - prior) * C_fp)

    @staticmethod
    def unnormalized_DCF(M, eff_prior):
        """
        Computes the unnormalized Detection Cost Function (DCF).
        
        :param M: confusion matrix
        :param eff_prior: effective prior probability
        :return: unnormalized DCF value
        """
        return eff_prior * (M[0, 1] / (M[0, 1] + M[1, 1])) + (1 - eff_prior) * (M[1, 0] / (M[0, 0] + M[1, 0]))

    @staticmethod
    def normalized_DCF(M, eff_prior, dummy_risk):
        """
        Computes the normalized Detection Cost Function (DCF).
        
        :param M: confusion matrix
        :param eff_prior: effective prior probability
        :param dummy_risk: dummy risk value
        :return: normalized DCF value
        """
        return Evaluator.unnormalized_DCF(M, eff_prior) / dummy_risk

    @staticmethod
    def minimum_DCF(llr, LVAL, eff_prior, dummy_risk):
        """
        Computes the minimum Detection Cost Function (DCF) over all possible thresholds.
        
        :param llr: log-likelihood ratios
        :param LVAL: true labels
        :param eff_prior: effective prior probability
        :param dummy_risk: dummy risk value
        :return: minimum normalized DCF value
        """
        min_DCF = np.inf

        # Initially, confusion matrix has only false and true positives
        # Since threshold value is below min(llr)
        M = np.array([[0, 0], [np.sum(LVAL == 0), np.sum(LVAL == 1)]])
        for threshold in sorted(np.unique(llr)):
            # From false positive to true negative
            below_threshold = np.sum((llr == threshold) & (LVAL == 0))
            # From true positive to false negative
            above_threshold = np.sum((llr == threshold) & (LVAL == 1))

            # Update matrix
            M[0, 0] += below_threshold
            M[1, 0] -= below_threshold
            M[0, 1] += above_threshold
            M[1, 1] -= above_threshold

            min_DCF = min(min_DCF, Evaluator.normalized_DCF(M, eff_prior, dummy_risk))

        return min_DCF

    @staticmethod
    def bayes_error(llr, LVAL, effective_prior_log_odds):
        """
        Computes the Bayes error given log-likelihood ratios, true labels, and effective prior log odds.
        
        :param llr: log-likelihood ratios
        :param LVAL: true labels
        :param effective_prior_log_odds: effective prior log odds
        :return: dictionary containing minimum and actual DCF values across thresholds
        """

        dcf, min_dcf = [], []
        for threshold in -effective_prior_log_odds:
            LPR = np.array(llr > threshold, dtype=np.int32)
            M = Evaluator.compute_confusion_matrix(LPR, LVAL, 2)
            effective_prior = 1 / (1 + np.exp(threshold))
            dummy_risk = Evaluator.dummy_risk(effective_prior)
            dcf.append(Evaluator.normalized_DCF(M, effective_prior, dummy_risk))
            min_dcf.append(Evaluator.minimum_DCF(llr, LVAL, effective_prior, dummy_risk))

        return {"min_dcf": min_dcf, "dcf": dcf}