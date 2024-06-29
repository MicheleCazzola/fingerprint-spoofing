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
    def fuse(scores_folds, labels_folds, tr_prior, app_prior, kf):
        cal_scores, LPR = Calibrator.calibrate(scores_folds, labels_folds, tr_prior, app_prior, kf)
        return cal_scores, LPR

    def get_scores(self):
        return self.scores

    def get_labels(self):
        return self.labels


def fusion_task(scores, LVAL, app_prior):
    K = 5
    num_tr_priors = 101
    emp_training_priors = np.linspace(0.01, 0.99, num_tr_priors)

    fus = Fusion(LVAL, scores)
    S = fus.get_scores()

    print("Fusing...")

    kf = KFold(S, LVAL, K)
    # SHUFFLE HERE !!!
    scores_folds, labels_folds = kf.create_folds()
    LVAL_unfolded = kf.get_real_labels()

    best_act_dcf, best_min_dcf, best_tr_prior, best_fused_scores = np.inf, np.inf, emp_training_priors[0], None
    for emp_training_prior in emp_training_priors:
        scores_fused, LPR = Fusion.fuse(scores_folds, labels_folds, emp_training_prior, app_prior, kf)

        min_dcf, act_dcf, _ = map(
            Evaluator.evaluate2(scores_fused, LPR, LVAL_unfolded, eff_prior=app_prior).get("results").get,
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
