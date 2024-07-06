import numpy as np

import constants
from cio import print_scores_stats
from evaluation.evaluation import Evaluator
from gmm import GaussianMixtureModel
from logreg import LogReg, expand
from plot import plot_bayes_errors
from svm import SupportVectorMachine
from utilities.utilities import vcol


def app_LR(DTR, LTR, DEVAL, variant, reg_coeff, training_prior, app_prior):

    if constants.LOG:
        print(f"LR {variant}, reg_coeff: {reg_coeff}, pi_tr = {training_prior}, pi_app = {app_prior}")

    lr = LogReg(variant=variant)
    if variant == constants.LR_STANDARD:
        lr.fit(DTR, LTR, reg_coeff=reg_coeff, training_prior=training_prior)
    else:
        if variant == constants.PRIOR_WEIGHTED_LR_PREPROCESS:
            DTR_mean = vcol(np.sum(DTR, axis=1)) / DTR.shape[1]
            DTR, DEVAL = DTR - DTR_mean, DEVAL - DTR_mean
        elif variant == constants.QUADRATIC_LR:
            DTR, DEVAL = map(expand, [DTR, DEVAL])

        if variant in [constants.PRIOR_WEIGHTED_LR, constants.PRIOR_WEIGHTED_LR_PREPROCESS]:
            lr.fit(DTR, LTR, reg_coeff=reg_coeff, training_prior=training_prior, app_prior=app_prior)
        else:
            lr.fit(DTR, LTR, reg_coeff=reg_coeff, training_prior=training_prior)

    scores = lr.scores(DEVAL)
    LPR = lr.predict(DEVAL, app_prior=app_prior)

    return scores, LPR


def app_SVM(DTR, LTR, DEVAL, kernel, K, C, scale, app_prior):
    ker_type = "poly" if kernel != constants.SVM_RBF else "rbf"

    if constants.LOG:
        print(f"SVM {kernel} ({ker_type}), K = {K}, C = {C}, gamma = {scale}, pi_app = {app_prior}")

    svm = SupportVectorMachine(K=K, C=C, kernel=ker_type)

    if kernel == constants.SVM_RBF:

        if constants.LOG:
            print(f"Doing SVM RBF with scale {scale}")

        svm.fit(DTR, LTR, scale=scale)

        if constants.LOG:
            print(f"Model params (alpha): {svm.alpha} ({svm.alpha.shape})")
            print(f"Other params: C = {svm.C}, K = {svm.K}")
            print(f"Kernel type: {svm.kernel_type}")
            print(f"Kernel args: {svm.kernel_args}")
            print_scores_stats([svm.ZTR, svm.DTR], ["ztr", "dtr"])
            print(svm)

    elif kernel == constants.SVM_POLYNOMIAL:
        svm.fit(DTR, LTR, degree=2, offset=1)
    else:
        if kernel == constants.SVM_LINEAR_PREPROCESS:
            DTR_mean = vcol(np.sum(DTR, axis=1)) / DTR.shape[1]
            DTR, DEVAL = DTR - DTR_mean, DEVAL - DTR_mean

        svm.fit(DTR, LTR, degree=1, offset=0)
    scores = svm.scores(DEVAL)
    LPR = svm.predict(DEVAL, app_prior=app_prior)

    return scores, LPR


def app_GMM(DTR, LTR, DEVAL, LEVAL, variant, components, app_prior):

    if constants.LOG:
        print(f"GMM {variant}, components = {components}, pi_app = {app_prior}")

    gmm = GaussianMixtureModel(variant=variant, components=components)
    gmm.fit(DTR, LTR)
    scores = gmm.scores(DEVAL, LEVAL)
    LPR = gmm.predict(DEVAL, LEVAL, app_prior=app_prior)

    return scores, LPR


def app_fusion(DTR, LTR, DVAL, LVAL, LR_params, SVM_params, GMM_params, app_prior):
    scores_LR = app_LR(DTR, LTR, DVAL, LR_params["variant"], LR_params["λ"], None, app_prior)[0]
    scores_SVM = app_SVM(DTR, LTR, DVAL, SVM_params["kernel"], SVM_params["K"], SVM_params["C"], SVM_params.get("scale", None), app_prior)[0]
    scores_GMM = app_GMM(DTR, LTR, DVAL, LVAL, GMM_params["variant"], GMM_params["components"], app_prior)[0]

    return np.vstack([scores_LR, scores_SVM, scores_GMM])


def app_calibration(SVAL, LVAL, SEVAL, tr_prior, app_prior):
    lr = LogReg(variant=constants.PRIOR_WEIGHTED_LR)
    lr.fit(SVAL, LVAL, training_prior=tr_prior, app_prior=app_prior)
    SEVAL_CAL = lr.scores(SEVAL)
    LPR = lr.predict(SEVAL, app_prior=app_prior)

    return SEVAL_CAL, LPR


def app_evaluation(
    model_name,
    model_params,
    DTR,
    LTR,
    SVAL,
    LVAL,
    DEVAL,
    LEVAL,
    app_prior,
    cal_tr_prior,
    eff_prior_log_odds,
    log_odd_app,
    best
):
    if constants.LOG:
        print(f"Evaluation on application dataset: {model_name}{best * ' (best)'}")
        print(f"Training data/labels: {DTR.shape}, {LTR.shape}")
        print(f"Validation score/labels: {SVAL.shape}, {LVAL.shape}")
        print(f"Evaluation data/labels: {DEVAL.shape} {LEVAL.shape}")

    LPR_raw = None
    match model_name:
        case constants.LR:
            SEVAL, LPR_raw = app_LR(DTR, LTR, DEVAL, model_params["variant"], model_params["λ"], None, app_prior)
            plot_name = constants.EVAL_BAYES_ERR_LR
        case constants.SVM:
            SEVAL, LPR_raw = app_SVM(DTR, LTR, DEVAL, model_params["kernel"], model_params["K"], model_params["C"], model_params.get("scale", None), app_prior)
            plot_name = constants.EVAL_BAYES_ERR_SVM
        case constants.GMM:
            SEVAL, LPR_raw = app_GMM(DTR, LTR, DEVAL, LEVAL, model_params["variant"], model_params["components"], app_prior)
            plot_name = constants.EVAL_BAYES_ERR_GMM
        case constants.FUSION:
            SEVAL = app_fusion(
                DTR,
                LTR,
                DEVAL,
                LEVAL,
                model_params[constants.LR],
                model_params[constants.SVM],
                model_params[constants.GMM],
                app_prior
            )
            plot_name = constants.EVAL_BAYES_ERR_FUSION
        case _:
            raise KeyError("Invalid model")

    min_dcf_raw, act_dcf_raw, bayes_err_min_dcf_raw, bayes_err_act_dcf_raw = None, None, None, None
    if model_name != constants.FUSION:
        min_dcf_raw, act_dcf_raw = map(
            Evaluator.evaluate(SEVAL, LPR_raw, LEVAL, app_prior).get("results").get,
            ["min_dcf", "dcf"]
        )

        bayes_err_min_dcf_raw, bayes_err_act_dcf_raw = map(
            Evaluator.bayes_error(SEVAL, LEVAL, eff_prior_log_odds).get,
            ["min_dcf", "dcf"]
        )

    if constants.LOG:
        print(f"Score calibration: {model_name}, pi_tr = {cal_tr_prior}, pi_app = {app_prior}")

    SEVAL_CAL, LPR = app_calibration(SVAL, LVAL, SEVAL, cal_tr_prior, app_prior)

    if constants.LOG:
        print_scores_stats([SVAL, SEVAL, SEVAL_CAL], ["Validation", "Evaluation - Raw", "Evaluation - Cal."])
        print()

    min_dcf_cal, act_dcf_cal = map(
        Evaluator.evaluate(SEVAL_CAL, LPR, LEVAL, app_prior).get("results").get,
        ["min_dcf", "dcf"]
    )

    bayes_err_min_dcf_cal, bayes_err_act_dcf_cal = map(
        Evaluator.bayes_error(SEVAL_CAL, LEVAL, eff_prior_log_odds).get,
        ["min_dcf", "dcf"]
    )

    if constants.SAVE:
        plot_bayes_errors(
            eff_prior_log_odds,
            [bayes_err_min_dcf_cal],
            [bayes_err_act_dcf_cal],
            log_odd_app,
            f"Bayes error plot {model_name}{best * ' (best)'}",
            "Evaluation dataset",
            "Prior log-odds",
            "DCF value",
            constants.PLOT_PATH_EVAL_CAL,
            plot_name,
            "pdf"
        )

        plot_bayes_errors(
            eff_prior_log_odds,
            [bayes_err_min_dcf_raw, bayes_err_min_dcf_cal],
            [bayes_err_act_dcf_raw, bayes_err_act_dcf_cal],
            log_odd_app,
            f"Bayes error plot {model_name}{best * ' (best)'}",
            "Evaluation dataset",
            "Prior log-odds",
            "DCF value",
            constants.PLOT_PATH_EVAL_CMP,
            plot_name,
            "pdf",
            ["raw", "cal."]
        )

    return {
        "cal": {
            "min_dcf": min_dcf_cal,
            "act_dcf": act_dcf_cal,
            "bayes_err_min_dcf": bayes_err_min_dcf_cal,
            "bayes_err_act_dcf": bayes_err_act_dcf_cal
        },
        "raw": {
            "min_dcf": min_dcf_raw,
            "act_dcf": act_dcf_raw,
            "bayes_err_min_dcf": bayes_err_min_dcf_raw,
            "bayes_err_act_dcf": bayes_err_act_dcf_raw
        }
    }
