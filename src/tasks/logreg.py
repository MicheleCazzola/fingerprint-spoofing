"""
    Logistic Regression task module.
    
    Handles training and evaluation of Logistic Regression models with various configurations.
"""
import os
import numpy as np

from src.evaluator.evaluator import Evaluator
from src.utils.plot import plot_log_double_line
from src.utils.utils import delete_all, expand, vcol
from src.models.logreg import LogReg
from src.config.config import LR_STANDARD, MODEL_PATH_LR, PRIOR_WEIGHTED_LR, PLOT_PATH_LR, SAVE, LR_EVALUATION_RESULTS, LOG, \
    LR_RED_DATA, QUADRATIC_LR, PRIOR_WEIGHTED_LR_PREPROCESS, PLOT_PATH_EVAL_LR
    
def write_LR_results(eval_results):
    print_string = "-- Minimum DCFs, Actual DCFs --\n"
    for [min_dcf, reg_coeff, _, task_name, dcf, _, _] in eval_results:
        print_string += f"{task_name:<70s}: {min_dcf:.3f}, {dcf:.3f} (λ = {reg_coeff:.4f})\n"

    return print_string
    
    
def logistic_regression(DTR, LTR, DVAL, LVAL, app_prior, reg_coefficients, variant, preprocess=None):
    """
    Perform logistic regression training and evaluation for a given variant and set of regularization coefficients.
    
    :param DTR: Training data.
    :param LTR: Training labels.
    :param DVAL: Validation data.
    :param LVAL: Validation labels.
    :param app_prior: Prior probability of the positive class in the application data.
    :param reg_coefficients: List of regularization coefficients to try.
    :param variant: Logistic regression variant to use.
    :param preprocess: Preprocessing method applied to the data, if any.
    :return: Dictionary containing preprocessing info and evaluation results.
    """
    
    eval_results = []
    lr = LogReg(variant)

    for reg_coeff in reg_coefficients:
        lr_prior = app_prior if variant in (PRIOR_WEIGHTED_LR, PRIOR_WEIGHTED_LR_PREPROCESS) else None
        lr.fit(DTR, LTR, reg_coeff, app_prior=lr_prior)

        llr = lr.scores(DVAL)
        LPR = lr.predict(DVAL, app_prior)
        eval_result = Evaluator.evaluate(
            llr,
            LPR,
            LVAL,
            eff_prior=app_prior,
            preprocess=preprocess,
            reg_coeff=reg_coeff
        )
        id = lr.save_state_dict(f"{MODEL_PATH_LR}")

        preprocess = eval_result["params"]["preprocess"]
        eval_results.append((
            eval_result["results"]["dcf"],
            eval_result["results"]["min_dcf"],
            reg_coeff,
            llr,
            id
        ))

    return {
        "preprocess": preprocess,
        "results": list(zip(*eval_results))
    }


def LR_task(trainset, validset, app_prior, target="validation"):
    """
    Execute logistic regression training and evaluation for various variants and regularization coefficients.
    
    :param trainset: Training dataset.
    :param validset: Validation dataset.
    :param app_prior: Prior probability of the positive class in the application data.
    :param target: Target dataset for plotting ('validation' or 'evaluation').
    :return eval_results_best: List of best evaluation results for each logistic regression configuration.
    """
    
    DTR, LTR = trainset.get_data(), trainset.get_labels()
    DVAL, LVAL = validset.get_data(), validset.get_labels()

    reg_coefficients = np.logspace(-4, 2, 13)

    titles = [
        "Logistic Regression DCFs for standard non-weighted model",
        "Logistic Regression DCFs for filtered non-weighted model",
        "Prior-weighted Logistic Regression DCFs",
        "Logistic Regression DCFs with expanded feature space",
        "Prior-weighted Logistic Regression DCFs with preprocessing"
    ]

    LR_types = [
        LR_STANDARD,
        LR_RED_DATA,
        PRIOR_WEIGHTED_LR,
        QUADRATIC_LR,
        PRIOR_WEIGHTED_LR_PREPROCESS
    ]

    results = [{}] * 5
    
    # Clean previous models
    delete_all([os.path.join(MODEL_PATH_LR, f) for f in os.listdir(MODEL_PATH_LR)])

    if LOG:
        print("Standard non-weighted LR")

    # 1: standard non-weighted LR
    results[0] = logistic_regression(DTR, LTR, DVAL, LVAL, app_prior, reg_coefficients, LR_STANDARD)

    if LOG:
        print("Standard non-weighted LR with reduced dataset")

    # 2: reduced dataset LR
    results[1] = logistic_regression(DTR[:, ::50], LTR[::50], DVAL, LVAL, app_prior, reg_coefficients, LR_STANDARD)

    if LOG:
        print("Prior-weighted LR")

    # 3: prior-weighted LR
    results[2] = logistic_regression(DTR, LTR, DVAL, LVAL, app_prior, reg_coefficients, PRIOR_WEIGHTED_LR)

    if LOG:
        print("Quadratic LR")

    # 4: quadratic LR
    DTR_expanded = expand(DTR)
    DVAL_expanded = expand(DVAL)
    results[3] = logistic_regression(DTR_expanded, LTR, DVAL_expanded, LVAL, app_prior, reg_coefficients, QUADRATIC_LR)

    if LOG:
        print("Prior-weighted LR with data preprocessing (data centering)")

    # 5: preprocess data and apply regularized model
    DTR_mean = vcol(np.sum(DTR, axis=1)) / DTR.shape[1]
    DTR_preprocess, DVAL_preprocess = DTR - DTR_mean, DVAL - DTR_mean
    results[4] = logistic_regression(
        DTR_preprocess, LTR, DVAL_preprocess, LVAL, app_prior, reg_coefficients, PRIOR_WEIGHTED_LR_PREPROCESS, "Data centering"
    )

    if LOG:
        print("LR collecting")

    eval_results_best = []
    for (result, title, file_name, LR_type) in zip(results, titles, LR_EVALUATION_RESULTS, LR_types):
        [dcf, min_dcf, reg_coeff, llr, id] = result["results"]
        best_conf = np.argmin(min_dcf)
        eval_results_best.append([
            np.min(min_dcf),
            reg_coeff[best_conf],
            llr[best_conf],
            title.replace(" DCFs", ""),
            dcf[best_conf],
            LR_type,
            id[best_conf]
        ])
        
        delete_all([f"{MODEL_PATH_LR}/logreg_{idx}.pkl" for idx in id if idx != id[best_conf]])

        if SAVE:
            plot_log_double_line(
                reg_coefficients, dcf, min_dcf, title, "Regularization coefficient", "DCF value", "DCF", "Min. DCF",
                PLOT_PATH_LR if target == "validation" else PLOT_PATH_EVAL_LR, file_name, "png",
                "Evaluation" + (f' - {result["preprocess"]}' if result["preprocess"] is not None else '')
            )

    eval_results_best[-1][3] += " (data centering)"

    return eval_results_best