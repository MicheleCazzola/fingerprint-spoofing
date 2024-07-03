import numpy as np

from constants import PRIOR_WEIGHTED_LR, LR, SVM, GMM, PLOT_PATH_CAL_FUS, SAVE, LOG
from evaluation.evaluation import Evaluator
from kfold import KFold
from logreg import LogReg
from plot import plot_bayes_errors


class Calibrator:
    def __init__(self, scores, labels):
        self.scores = scores
        self.labels = labels
        self.calibrated = None

    @staticmethod
    def calibrate(training_prior, app_prior, kf):
        lr = LogReg(variant=PRIOR_WEIGHTED_LR)

        cal_scores = []
        act_labels = []
        pred_labels = []

        for i in range(0, kf.K):
            (SCAL, LCAL), (SVAL, LVAL) = kf.split(i)
            lr.fit(SCAL, LCAL, training_prior=training_prior, app_prior=app_prior)
            cal_score = lr.scores(SVAL)
            LPR = lr.predict(SVAL, app_prior)

            cal_scores.append(cal_score)
            act_labels.append(LVAL)
            pred_labels.append(LPR)

        cal_scores = np.hstack(cal_scores)
        act_labels = np.hstack(act_labels)
        pred_labels = np.hstack(pred_labels)

        return cal_scores, act_labels, pred_labels


def model_calibration(S, LVAL, app_prior, act_dcf_raw, min_dcf_raw):
    K = 5
    num_tr_priors = 99
    emp_training_priors = np.linspace(0.01, 0.99, num_tr_priors)

    kf = KFold(S, LVAL, K)
    LVAL_unfolded = None
    best_act_dcf, best_min_dcf, best_tr_prior, best_cal_scores = act_dcf_raw, min_dcf_raw, emp_training_priors[0], None
    for emp_training_prior in emp_training_priors:
        scores_cal, LVAL_kf, LPR = Calibrator.calibrate(emp_training_prior, app_prior, kf)

        # Same shuffle, so LVAL used in KFold is the same across iterations
        if LVAL_unfolded is None:
            LVAL_unfolded = LVAL_kf

        min_dcf, act_dcf, _ = map(
            Evaluator.evaluate(
                scores_cal,
                LPR,
                LVAL_kf,
                eff_prior=app_prior
            ).get("results").get,
            ["min_dcf", "dcf", "llr"]
        )

        if act_dcf < best_act_dcf:
            best_act_dcf = act_dcf
            best_min_dcf = min_dcf
            best_tr_prior = emp_training_prior
            best_cal_scores = scores_cal

    return {
        "min_dcf": best_min_dcf,
        "act_dcf": best_act_dcf,
        "llr": best_cal_scores,
        "params": {
            "training_prior": best_tr_prior
        }
    }, LVAL_unfolded


def calibration_task(model_results, LVAL, app_prior, bayes_errors_raw, effective_prior_log_odds, log_odd_application):
    calibrated_results = {
        LR: {},
        SVM: {},
        GMM: {}
    }

    LVAL_unfolded_models = {
        LR: {},
        SVM: {},
        GMM: {}
    }

    for model in model_results:

        if LOG:
            print(f"Calibrating {model}")

        scores = model_results[model]["llr"]
        act_dcf_raw = model_results[model]["act_dcf"]
        min_dcf_raw = model_results[model]["min_dcf"]

        calibration_result, LVAL_unfolded = model_calibration(scores, LVAL, app_prior, act_dcf_raw, min_dcf_raw)

        calibrated_results[model] = calibration_result
        LVAL_unfolded_models[model] = LVAL_unfolded

        err_min_dcf_cal, err_act_dcf_cal = map(
            Evaluator.bayes_error(
                calibration_result["llr"],
                LVAL_unfolded,
                effective_prior_log_odds).get, ["min_dcf", "dcf"]
        )

        err_min_dcf_raw, err_act_dcf_raw = bayes_errors_raw[model][0], bayes_errors_raw[model][1]

        if SAVE:
            plot_bayes_errors(
                effective_prior_log_odds,
                [err_min_dcf_raw, err_min_dcf_cal],
                [err_act_dcf_raw, err_act_dcf_cal],
                log_odd_application,
                "Comparison between calibrated and uncalibrated model",
                f"Model: {model}",
                "Prior log-odds",
                "DCF value",
                PLOT_PATH_CAL_FUS,
                f"bayes_error_calibration_{model}",
                "pdf",
                ["Raw", "Cal."]
            )

    return calibrated_results, LVAL_unfolded_models
