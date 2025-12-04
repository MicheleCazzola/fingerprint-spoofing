"""
    SVM task module.
    
    Handles training and evaluation of Support Vector Machines (SVMs) with different kernel types.
"""
import os
import numpy as np

from src.config.config import LOG, MODEL_PATH_SVM, PLOT_PATH_SVM, SAVE, SVM_EVALUATION_RESULTS, SVM_LINEAR, SVM_LINEAR_PREPROCESS, SVM_POLYNOMIAL, SVM_RBF
from src.models.svm import SupportVectorMachine
from src.tasks.utils import optimal_bayes
from src.utils.plot import plot_log_N_double_lines, plot_log_double_line
from src.utils.utils import delete_all, vcol


def write_SVM_results(results):
    task_names = results["tasks"]
    best_results = results["results"]

    print_string = "-- Minimum DCFs, Actual DCFs --\n"
    for (task_name, best_result) in zip(task_names[:-1], best_results[:-1]):
        print_string += (f"{task_name}: {best_result[1]:.3f}, "
                         f"{best_result[4]:.3f} (C = {best_result[0]:.3f}, K = {best_result[2]:.1f})\n")

    print_string += f"{task_names[-1]}:\n"
    print_string += f"{'Minimum DCF':<12s}{'Actual DCF':^12s}{'γ':^7s}{'C':^7s}{'K':^5s}\n"
    for (rbf, best_rbf) in best_results[-1].items():
        print_string += f"{best_rbf[1]:^12.3f}{best_rbf[4]:^12.3f}{rbf:^7.3f}{best_rbf[0]:^7.3f}{best_rbf[2]:^5.1f}\n"

    return print_string


def linear_svm(DTR, LTR, DVAL, LVAL, app_prior, svm: SupportVectorMachine, c_values):
    """
    Perform linear SVM training and evaluation for a given set of regularization coefficients.
    
    :param DTR: Training data.
    :param LTR: Training labels.
    :param DVAL: Validation data.
    :param LVAL: Validation labels.
    :param app_prior: Prior probability of the positive class in the application data.
    :param svm: SVM model instance.
    :param c_values: List of regularization coefficients to try.
    :return results_min_dcf: List of minimum DCFs for each coefficient.
    :return results_dcf: List of actual DCFs for each coefficient.
    :return llrs: List of log-likelihood ratios for each coefficient.
    :return ids: List of model state dictionary IDs for each coefficient.
    """

    results_min_dcf, results_dcf, llrs, ids = [], [], [], []
    for c in c_values:

        if LOG:
            print(f"SVM linear (c = {c})")

        svm.fit(DTR, LTR, c, primal=True, degree=1, offset=0)
        id = svm.save_state_dict(f"{MODEL_PATH_SVM}")

        min_dcf, dcf, llr = optimal_bayes(svm, DVAL, LVAL, app_prior)

        results_min_dcf.append(min_dcf)
        results_dcf.append(dcf)
        llrs.append(llr)
        ids.append(id)

    return results_min_dcf, results_dcf, llrs, ids


def poly_svm(DTR, LTR, DVAL, LVAL, app_prior, svm, c_values):
    """
    Perform polynomial SVM training and evaluation for a given set of regularization coefficients.
    
    :param DTR: Training data.
    :param LTR: Training labels.
    :param DVAL: Validation data.
    :param LVAL: Validation labels.
    :param app_prior: Prior probability of the positive class in the application data.
    :param svm: SVM model instance.
    :param c_values: List of regularization coefficients to try.
    :return results_min_dcf: List of minimum DCFs for each coefficient.
    :return results_dcf: List of actual DCFs for each coefficient.
    :return llrs: List of log-likelihood ratios for each coefficient.
    :return ids: List of model state dictionary IDs for each coefficient.
    """

    results_min_dcf, results_dcf, llrs, ids = [], [], [], []
    for c in c_values:

        if LOG:
            print(f"SVM polynomial (c = {c})")

        svm.fit(DTR, LTR, c, primal=False, degree=2, offset=1)
        id = svm.save_state_dict(f"{MODEL_PATH_SVM}")

        min_dcf, dcf, llr = optimal_bayes(svm, DVAL, LVAL, app_prior)

        results_min_dcf.append(min_dcf)
        results_dcf.append(dcf)
        llrs.append(llr)
        ids.append(id)

    return results_min_dcf, results_dcf, llrs, ids


def rbf_svm(DTR, LTR, DVAL, LVAL, app_prior, svm, c_values, scale_values):
    """
    Perform RBF SVM training and evaluation for a given set of regularization coefficients and scale values.
    
    :param DTR: Training data.
    :param LTR: Training labels.
    :param DVAL: Validation data.
    :param LVAL: Validation labels.
    :param app_prior: Prior probability of the positive class in the application data.
    :param svm: SVM model instance.
    :param c_values: List of regularization coefficients to try.
    :param scale_values: List of scale values to try.
    :return results_min_dcf: Dictionary of minimum DCFs for each scale value.
    :return results_dcf: Dictionary of actual DCFs for each scale value.
    :return llrs: Dictionary of log-likelihood ratios for each scale value.
    :return ids: Dictionary of model state dictionary IDs for each scale value.
    """

    results_min_dcf, results_dcf, llrs, ids = {}, {}, {}, {}
    for scale in scale_values:
        res_min, res_act, llrs_scale, ids_scale = [], [], [], []
        for c in c_values:

            svm.setParams(kernel='rbf')
            svm.fit(DTR, LTR, c, primal=False, scale=scale)
            id = svm.save_state_dict(f"{MODEL_PATH_SVM}")

            if LOG:
                print(f"SVM RBF (scale = {scale}, c = {c})")
                #print(f"SVM RBF (scale = {scale}, c = {c}), alpha = {svm.alpha} ({svm.alpha.shape})")
                #print_scores_stats([vrow(svm.alpha)], ["alpha"])
                #print(svm)

            min_dcf, dcf, llr = optimal_bayes(svm, DVAL, LVAL, app_prior)

            res_min.append(min_dcf)
            res_act.append(dcf)
            llrs_scale.append(llr)
            ids_scale.append(id)

        results_min_dcf[scale] = res_min
        results_dcf[scale] = res_act
        llrs[scale] = llrs_scale
        ids[scale] = ids_scale
        
    return results_min_dcf, results_dcf, llrs, ids


def svm_task(trainset, validset, app_prior):
    """
    Perform SVM training and evaluation for different kernel types.
    
    :param trainset: Training dataset.
    :param validset: Validation dataset.
    :param app_prior: Prior probability of the positive class in the application data.
    :return: Dictionary containing best results for each SVM configuration.
    """
    
    DTR, LTR = trainset.get_data(), trainset.get_labels()
    DVAL, LVAL = validset.get_data(), validset.get_labels()

    c_values = np.logspace(-5, 0, 11)
    c_values_rbf = np.logspace(-3, 2, 11)
    k_values = [1, 1, 0, 1]
    scale_values_rbf = np.exp(np.array(range(-4, 0)))
    ker_type = [
        SVM_LINEAR,
        SVM_LINEAR_PREPROCESS,
        SVM_POLYNOMIAL,
        SVM_RBF
    ]
    
    # Clean previous models
    delete_all([os.path.join(MODEL_PATH_SVM, f) for f in os.listdir(MODEL_PATH_SVM)])
    
    svm = SupportVectorMachine()

    eval_results = [{"min_dcf": [], "dcf": []} for _ in range(0, 4)]
    best_results = [()] * 3 + [{}]
    task_names = [
        "Linear kernel",
        "Linear kernel - Data centering",
        "Polynomial kernel",
        "RBF kernel"
    ]

    titles = [
        "SVM linear kernel no preprocess",
        "SVM linear kernel with data centering",
        "SVM polynomial kernel (degree=2, offset=1)",
        "SVM RBF kernel (bias = 1)"
    ]

    if LOG:
        print("SVM: linear kernel")

    # Linear SVM, no preprocessing
    svm.setParams(K=k_values[0])
    eval_results[0]["min_dcf"], eval_results[0]["dcf"], eval_results[0]["llr"], eval_results[0]["id"] = linear_svm(
        DTR, LTR, DVAL, LVAL, app_prior, svm, c_values
    )

    if LOG:
        print("SVM: linear kernel with preprocessing")

    # Linear SVM, data centering
    DTR_mean = vcol(np.sum(DTR, axis=1)) / DTR.shape[1]
    DTR_preprocess, DVAL_preprocess = DTR - DTR_mean, DVAL - DTR_mean
    svm.setParams(K=k_values[1])
    eval_results[1]["min_dcf"], eval_results[1]["dcf"], eval_results[1]["llr"], eval_results[1]["id"] = linear_svm(
        DTR_preprocess, LTR, DVAL_preprocess, LVAL, app_prior, svm, c_values
    )

    if LOG:
        print("SVM: polynomial kernel")

    # Polynomial SVM (degree=2, offset=1)
    svm.setParams(K=k_values[2])
    eval_results[2]["min_dcf"], eval_results[2]["dcf"], eval_results[2]["llr"], eval_results[2]["id"] = poly_svm(
        DTR, LTR, DVAL, LVAL, app_prior, svm, c_values
    )

    if LOG:
        print("SVM: RBF kernel")

    # RBF SVM (bias = 1), scale = [e-4, e-3, e-2, e-1]
    svm.setParams(K=k_values[3], ker_type="rbf")
    eval_results[3]["min_dcf"], eval_results[3]["dcf"], eval_results[3]["llr"], eval_results[3]["id"] = rbf_svm(
        DTR, LTR, DVAL, LVAL, app_prior, svm, c_values_rbf, scale_values_rbf
    )   

    if LOG:
        print("SVM collecting")

    for i in range(len(best_results[:-1])):
        eval_result = eval_results[i]
        best_conf = np.argmin(eval_result["min_dcf"])
        best_results[i] = (
            c_values[best_conf],
            np.min(eval_result["min_dcf"]),
            k_values[i],
            eval_result["llr"][best_conf],
            eval_result["dcf"][best_conf],
            eval_result["id"][best_conf],
            ker_type[i]
        )
        
        delete_all([f"{MODEL_PATH_SVM}/svm_{idx}.pkl" for idx in eval_result["id"] if idx != eval_result["id"][best_conf]])

    best_conf = lambda s: np.argmin(eval_results[-1]["min_dcf"][s])
    best_results[-1] = {
        scale: (
            c_values_rbf[best_conf(scale)],
            np.min(eval_results[-1]["min_dcf"][scale]),
            k_values[-1],
            eval_results[-1]["llr"][scale][best_conf(scale)],
            eval_results[-1]["dcf"][scale][best_conf(scale)],
            eval_results[-1]["id"][scale][best_conf(scale)],
            ker_type[-1]
        ) for scale in eval_results[-1]["min_dcf"]
    }
    
    delete_all([
        f"{MODEL_PATH_SVM}/svm_{idx}.pkl"
        for scale in eval_results[-1]["id"]
        for idx in eval_results[-1]["id"][scale] if idx != eval_results[-1]["id"][scale][best_conf(scale)]
    ])

    if SAVE:
        for (eval_result, title, name) in zip(eval_results[:-1], titles[:-1], SVM_EVALUATION_RESULTS[:-1]):
            plot_log_double_line(
                c_values, eval_result["min_dcf"], eval_result["dcf"], title, "Regularization values", "DCF values",
                "Min. DCF", "DCF", PLOT_PATH_SVM, name, "png"
            )

        plot_log_N_double_lines(
            c_values_rbf, eval_results[-1]["min_dcf"], eval_results[-1]["dcf"], titles[-1], "Regularization values",
            "DCF values", ["Min. DCF (g=e-4)", "Min. DCF (g=e-3)", "Min. DCF (g=e-2)", "Min. DCF (g=e-1)"],
            ["DCF (g=e-4)", "DCF (g=e-3)", "DCF (g=e-2)", "DCF (g=e-1)"], PLOT_PATH_SVM, SVM_EVALUATION_RESULTS[-1], "png"
        )

    return {
        "tasks": task_names,
        "results": best_results
    }