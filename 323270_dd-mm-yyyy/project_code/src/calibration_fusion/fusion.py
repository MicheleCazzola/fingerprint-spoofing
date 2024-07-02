import numpy as np

from calibration import Calibrator
from evaluation.evaluation import Evaluator
from kfold import KFold


class Fusion:
    def __init__(self, labels, scores):
        self.labels = labels
        self.scores = np.vstack(scores)

    def add_score(self, score):
        self.scores = np.vstack([self.scores, score.ravel()])

    @staticmethod
    def fuse(S, L, tr_prior, app_prior, kf):
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

    print("Fusing...")

    kf = KFold(S, LVAL, K)
    LVAL_unfolded = None

    best_act_dcf, best_min_dcf, best_tr_prior, best_fused_scores = np.inf, np.inf, emp_training_priors[0], None
    for emp_training_prior in emp_training_priors:
        scores_fused, LVAL_kf, LPR = Fusion.fuse(S, LVAL, emp_training_prior, app_prior, kf)

        if LVAL_unfolded is None:
            LVAL_unfolded = LVAL_kf

        min_dcf, act_dcf, _ = map(
            Evaluator.evaluate2(scores_fused, LPR, LVAL_kf, eff_prior=app_prior).get("results").get,
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
