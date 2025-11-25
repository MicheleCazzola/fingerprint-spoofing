import numpy as np

from src.calfus.calibration import Calibrator
from src.config.config import LOG
from src.evaluator.evaluator import Evaluator
from src.calfus.kfold import KFold

def write_best_results(results):
    print_string = "-- Comparisons among models --\n"

    print_string += f"{'Type':^8s}{'Minimum DCF':^13s}{'Actual DCF':^11s}{'Parameters':^44s}\n"
    for (model_type, result) in results.items():
        print_string += f"{model_type:^8s}{result['min_dcf']:^13.3f}{result['act_dcf']:^11.3f}"
        params = ", ".join(
            [
                f"{par_name}: {par_value:.3e}" if type(par_value) in [float, np.float64] else f"{par_name}: {par_value}"
                for (par_name, par_value) in result["params"].items()
            ]
        )
        print_string += f"{params:^44s}"
        print_string += "\n"

    return print_string

class Fusion:
    def __init__(self, labels, scores):
        self.labels = labels
        self.scores = np.vstack(scores)

    def add_score(self, score):
        self.scores = np.vstack([self.scores, score.ravel()])

    @staticmethod
    def fuse(tr_prior, app_prior, kf):
        cal_scores, LVAL_kf, LPR = Calibrator.calibrate(tr_prior, app_prior, kf)
        return cal_scores, LVAL_kf, LPR

    def get_scores(self):
        return self.scores

    def get_labels(self):
        return self.labels


def fusion_task(scores, LVAL, app_prior):
    K = 5
    num_tr_priors = 99
    emp_training_priors = np.linspace(0.01, 0.99, num_tr_priors)

    fus = Fusion(LVAL, scores)
    S = fus.get_scores()

    if LOG:
        print("Score fusion")

    kf = KFold(S, LVAL, K)
    LVAL_unfolded = None

    best_act_dcf, best_min_dcf, best_tr_prior, best_fused_scores = np.inf, np.inf, emp_training_priors[0], None
    for emp_training_prior in emp_training_priors:

        if LOG:
            print(f"Empirical training prior: {emp_training_prior}")

        scores_fused, LVAL_kf, LPR = Fusion.fuse(emp_training_prior, app_prior, kf)

        if LVAL_unfolded is None:
            LVAL_unfolded = LVAL_kf

        min_dcf, act_dcf, _ = map(
            Evaluator.evaluate(scores_fused, LPR, LVAL_kf, eff_prior=app_prior)["results"].get,
            ["min_dcf", "dcf", "llr"])

        if act_dcf < best_act_dcf:
            best_act_dcf = act_dcf
            best_min_dcf = min_dcf
            best_tr_prior = emp_training_prior
            best_fused_scores = scores_fused

    return {
        "min_dcf": best_min_dcf,
        "act_dcf": best_act_dcf,
        "llr": best_fused_scores,
        "params": {
            "training_prior": best_tr_prior
        }
    }, LVAL_unfolded
