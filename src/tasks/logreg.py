import numpy as np

from src.evaluator.evaluator import Evaluator
from src.utils.plot import plot_log_double_line
from src.utils.utils import expand, vcol
from src.models.logreg import LogReg
from src.config.config import LR_STANDARD, PRIOR_WEIGHTED_LR, PLOT_PATH_LR, SAVE, LR_EVALUATION_RESULTS, LOG, \
    LR_RED_DATA, QUADRATIC_LR, PRIOR_WEIGHTED_LR_PREPROCESS, PLOT_PATH_EVAL_LR
    
    
def logistic_regression(DTR, LTR, DVAL, LVAL, app_prior, reg_coefficients, variant, preprocess=None):
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

        preprocess = eval_result["params"]["preprocess"]
        eval_results.append((
            eval_result["results"]["dcf"],
            eval_result["results"]["min_dcf"],
            reg_coeff,
            llr
        ))

    return {
        "preprocess": preprocess,
        "results": list(zip(*eval_results))
    }


def LR_task(DTR, LTR, DVAL, LVAL, app_prior, target="validation"):

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

    if LOG:
        print("Standard non-weighted LR")

    # 1: standard non-weighted LR
    results[0] = logistic_regression(
        DTR,
        LTR,
        DVAL,
        LVAL,
        app_prior,
        reg_coefficients,
        LR_STANDARD
    )

    if LOG:
        print("Standard non-weighted LR with reduced dataset")

    # 2: reduced dataset LR
    results[1] = logistic_regression(
        DTR[:, ::50],
        LTR[::50],
        DVAL,
        LVAL,
        app_prior,
        reg_coefficients,
        LR_STANDARD
    )

    if LOG:
        print("Prior-weighted LR")

    # 3: prior-weighted LR
    results[2] = logistic_regression(
        DTR,
        LTR,
        DVAL,
        LVAL,
        app_prior,
        reg_coefficients,
        PRIOR_WEIGHTED_LR
    )

    if LOG:
        print("Quadratic LR")

    # 4: quadratic LR
    DTR_expanded = expand(DTR)
    DVAL_expanded = expand(DVAL)
    results[3] = logistic_regression(
        DTR_expanded,
        LTR,
        DVAL_expanded,
        LVAL,
        app_prior,
        reg_coefficients,
        LR_STANDARD
    )

    if LOG:
        print("Prior-weighted LR with data preprocessing (data centering)")

    # 5: preprocess data and apply regularized model
    DTR_mean = vcol(np.sum(DTR, axis=1)) / DTR.shape[1]
    DTR_preprocess, DVAL_preprocess = DTR - DTR_mean, DVAL - DTR_mean
    results[4] = logistic_regression(
        DTR_preprocess,
        LTR,
        DVAL_preprocess,
        LVAL,
        app_prior,
        reg_coefficients,
        PRIOR_WEIGHTED_LR_PREPROCESS,
        "Data centering"
    )

    if LOG:
        print("LR collecting")

    eval_results_best = []
    for (result, title, file_name, LR_type) in zip(results, titles, LR_EVALUATION_RESULTS, LR_types):
        [dcf, min_dcf, reg_coeff, llr] = result["results"]
        best_conf = np.argmin(min_dcf)
        eval_results_best.append([
            np.min(min_dcf),
            reg_coeff[best_conf],
            llr[best_conf],
            title.replace(" DCFs", ""),
            dcf[best_conf],
            LR_type
        ])

        if SAVE:
            plot_log_double_line(
                reg_coefficients,
                dcf,
                min_dcf,
                title,
                "Regularization coefficient",
                "DCF value",
                "DCF",
                "Min. DCF",
                PLOT_PATH_LR if target == "validation" else PLOT_PATH_EVAL_LR,
                file_name,
                "pdf",
                "Evaluation" + (f' - {result["preprocess"]}' if result["preprocess"] is not None else '')
            )

    eval_results_best[-1][3] += " (data centering)"

    return eval_results_best