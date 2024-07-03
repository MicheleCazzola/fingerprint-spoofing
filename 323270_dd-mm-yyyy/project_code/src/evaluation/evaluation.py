import numpy as np
from src.io.constants import GAUSSIAN, LR, SVM, GMM


def relative_mis_calibration(dcfs):
    return 100 * (dcfs["dcf"] - dcfs["min_dcf"]) / dcfs["min_dcf"]


class Evaluator:
    def __init__(self, model_name):
        self.model_name = model_name
        self.results = []
        self.data_models = []

    @staticmethod
    def evaluate(llr, LPR, LVAL, eff_prior, **model_params):
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
    def _best_configuration_LR(eval_results):
        min_dcfs = [r[0] for r in eval_results]
        best_conf = np.argmin(min_dcfs)
        return {
            "min_dcf": eval_results[best_conf][0],
            "act_dcf": eval_results[best_conf][4],
            "llr": eval_results[best_conf][2],
            "params": {
                "type": eval_results[best_conf][5],
                "λ": eval_results[best_conf][1]
            }
        }

    @staticmethod
    def _best_configuration_SVM(eval_results):
        min_dcfs = [r[1] for r in eval_results[:-1]]
        best_conf_no_rbf = np.argmin(min_dcfs)
        min_dcfs_rbf = [r[1] for (_, r) in eval_results[-1].items()]
        best_conf_rbf = np.argmin(min_dcfs_rbf)
        best = "RBF" if list(eval_results[-1].values())[best_conf_rbf][1] < eval_results[:-1][best_conf_no_rbf][
            1] else "NO_RBF"

        if best == "RBF":
            best_conf = list(eval_results[-1].values())[best_conf_rbf]
            min_dcf = best_conf[1]
            dcf = best_conf[4]
            llr = best_conf[3]
            params = {
                "type": best_conf[5],
                "γ": list(eval_results[-1].keys())[best_conf_rbf],
                "C": best_conf[0],
                "K": best_conf[2]
            }
        else:
            best_conf = eval_results[:-1][best_conf_no_rbf]
            min_dcf = best_conf[1]
            dcf = best_conf[4]
            llr = best_conf[3]
            params = {
                "type": best_conf[5],
                "C": best_conf[0],
                "K": best_conf[2]
            }

        return {
            "min_dcf": min_dcf,
            "act_dcf": dcf,
            "llr": llr,
            "params": params
        }

    @staticmethod
    def _best_configuration_GMM(eval_results):
        all_results = [["full", c, v["min_dcf"], v["dcf"], v["llr"]] for (c, v) in eval_results["full"].items()] + \
                      [["diag", c, v["min_dcf"], v["dcf"], v["llr"]] for (c, v) in eval_results["diag"].items()]
        min_dcfs = [r[2] for r in all_results]
        best_conf = np.argmin(min_dcfs)

        return {
            "min_dcf": all_results[best_conf][2],
            "act_dcf": all_results[best_conf][3],
            "llr": all_results[best_conf][4],
            "params": {
                "type": all_results[best_conf][0],
                "components": f"(False: {all_results[best_conf][1][0]}, True: {all_results[best_conf][1][1]})"
            }
        }

    @staticmethod
    def best_configuration(eval_results, mode, eff_prior=0.1):
        if mode == GAUSSIAN:
            return Evaluator._best_configuration_gaussian(eval_results[eff_prior])
        if mode == LR:
            return Evaluator._best_configuration_LR(eval_results)
        if mode == SVM:
            return Evaluator._best_configuration_SVM(eval_results)
        if mode == GMM:
            return Evaluator._best_configuration_GMM(eval_results)

    @staticmethod
    def best_model(model_results, key):
        keys = [model[key] for model in model_results.values()]
        best_model = np.argmin(keys)

        return dict([list(model_results.items())[best_model]])

    @staticmethod
    def compute_confusion_matrix(LPR, LVAL, n_classes):
        M = np.zeros((n_classes, n_classes), dtype=np.int32)

        # print(LPR.shape, LVAL.shape)
        for (p, c) in zip(LPR.ravel(), LVAL.ravel()):
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
    def minimum_DCF(llr, LVAL, eff_prior, dummy_risk):
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

        dcf, min_dcf = [], []
        for threshold in -effective_prior_log_odds:
            LPR = np.array(llr > threshold, dtype=np.int32)
            M = Evaluator.compute_confusion_matrix(LPR, LVAL, 2)
            effective_prior = 1 / (1 + np.exp(threshold))
            dummy_risk = Evaluator.dummy_risk(effective_prior)
            dcf.append(Evaluator.normalized_DCF(M, effective_prior, dummy_risk))
            min_dcf.append(Evaluator.minimum_DCF(llr, LVAL, effective_prior, dummy_risk))

        return {"min_dcf": min_dcf, "dcf": dcf}
