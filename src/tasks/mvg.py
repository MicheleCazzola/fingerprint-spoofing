"""
    MVG task module.
    
    Handles training and evaluation of various Gaussian models including standard MVG, Naive Bayes MVG, and Tied MVG.
    Supports PCA preprocessing and application-specific prior adjustments.
    Executes classification analysis and saves results accordingly.
"""

import numpy as np

from src.config.config import APPLICATIONS, FILE_PATH_MVG, GAUSSIAN, GAUSSIAN_APPLICATION_PRIORS, GAUSSIAN_BAYES_ERROR, GAUSSIAN_ERROR_RATES, GAUSSIAN_EVALUATION_RESULTS, GAUSSIAN_MODELS, LABEL_NAMES, PLOT_PATH_MVG, SAVE
from src.dimred.pca import PCA
from src.evaluator.evaluator import Evaluator
from src.models.mvg import MVG, NaiveBayesMVG, TiedMVG
from src.utils.plot import plot_bayes_errors
from src.utils.utils import effective_prior, relative_mis_calibration, save

def build_table(errors):
    result = ""
    result += f"{'Method':^16s}{'Error rate':^14s}{'Error rate (%)':^14s}\n"
    for (alg, err) in errors.items():
        result += f"{alg:^16s}{err:^14.4f}{100 * err:^14.2f}\n"
    result += "\n"

    return result


def write_gaussian_classification_results(
    error_rates,
    corr_matrices,
    error_rates_1_4,
    error_rates_1_2,
    error_rates_3_4,
    error_rates_pca
):

    result = ""
    result += "--All features--\n"
    result += build_table(error_rates)

    result += "--Correlation matrices--\n"
    for (corr_matrix, label) in zip(corr_matrices, LABEL_NAMES.keys()):
        result += f"{label} class\n"
        for line in corr_matrix:
            for element in line:
                result += f"{element: .2f}\t"
            result += "\n"
        result += "\n"

    result += "--Using subsets of features--\n"
    result += "Features 1-4\n"
    result += build_table(error_rates_1_4)

    result += "Features 1-2\n"
    result += build_table(error_rates_1_2)

    result += "Features 3-4\n"
    result += build_table(error_rates_3_4)

    result += "--PCA preprocessing--\n"
    result += "Error rates\n"
    result += f"{'PCA dimensions':<16s}{'Standard MVG':^14s}{'Tied MVG':^12s}{'Naive Bayes MVG':^17s}\n"
    for (m, err) in error_rates_pca.items():
        result += (f"{m:^16d}"
                   f"{err['Standard MVG']:^14.4f}"
                   f"{err['Tied MVG']:^12.4f}"
                   f"{err['Naive Bayes MVG']:^17.4f}\n")
    result += "\n"

    result += "Error rates (%)\n"
    result += f"{'PCA dimensions':<16s}{'Standard MVG':^14s}{'Tied MVG':^12s}{'Naive Bayes MVG':^17s}\n"
    for (m, err) in error_rates_pca.items():
        result += (f"{m:^16d}"
                   f"{100 * err['Standard MVG']:^14.2f}"
                   f"{100 * err['Tied MVG']:^12.2f}"
                   f"{100 * err['Naive Bayes MVG']:^17.2f}\n")

    return result


def save_gaussian_classification_results(
    error_rates,
    corr_matrices,
    error_rates_1_4,
    error_rates_1_2,
    error_rates_3_4,
    error_rates_pca,
    path_root,
    file_name
):
    print_string = write_gaussian_classification_results(
        error_rates,
        corr_matrices,
        error_rates_1_4,
        error_rates_1_2,
        error_rates_3_4,
        error_rates_pca
    )

    with open(f"{path_root}{file_name}", mode="w", encoding="utf-8") as fout:
        fout.write(print_string)


def write_application_priors(applications, eff_priors):
    result = ""
    result += "--Applications and associated effective priors--\n"
    result += (f"{'Prior':<6s}"
               f"{'False negative cost (C_fn)':^28s}"
               f"{'False positive cost (C_fp)':^28s}"
               f"{'Effective prior':^17s}\n")

    for ([prior, c_fn, c_fp], eff_prior) in zip(applications, eff_priors):
        result += (f"{prior:^6.1f}"
                   f"{c_fn:^28.1f}"
                   f"{c_fp:^28.1f}"
                   f"{eff_prior:^17.1f}\n")

    return result


def save_application_priors(applications, eff_priors, path_root, file_name):
    print_string = write_application_priors(applications, eff_priors)
    with open(f"{path_root}{file_name}", mode="w", encoding="utf-8") as fout:
        fout.write(print_string)


def print_DCFs(result, key, m):
    return (f"{str(m) if m is not None else 'Not applied':^16s}"
            f"{result['Standard MVG'][key]:^14.3f}"
            f"{result['Tied MVG'][key]:^12.3f}"
            f"{result['Naive Bayes MVG'][key]:^17.3f}\n")


def print_mis_calibrations(result, m="Not applied"):
    return (f"{str(m):^16s}"
            f"{relative_mis_calibration(result['Standard MVG']):^14.2f}"
            f"{relative_mis_calibration(result['Tied MVG']):^12.2f}"
            f"{relative_mis_calibration(result['Naive Bayes MVG']):^17.2f}\n")


def write_tables(results):
    print_string = "Minimum DCF\n"
    print_string += f"{'PCA dimensions':<16s}{'Standard MVG':^14s}{'Tied MVG':^12s}{'Naive Bayes MVG':^17s}\n"
    for (m, result) in sorted(results.items(), key=lambda x: x[0] if x[0] != "Not applied" else np.inf):
        print_string += print_DCFs(result, "min_dcf", m)
    print_string += "\n"

    print_string += "Actual DCF\n"
    print_string += f"{'PCA dimensions':<16s}{'Standard MVG':^14s}{'Tied MVG':^12s}{'Naive Bayes MVG':^17s}\n"
    for (m, result) in sorted(results.items(), key=lambda x: x[0] if x[0] != "Not applied" else np.inf):
        print_string += print_DCFs(result, "dcf", m)
    print_string += "\n"

    print_string += "Relative miscalibration loss (%)\n"
    print_string += f"{'PCA dimensions':<16s}{'Standard MVG':^14s}{'Tied MVG':^12s}{'Naive Bayes MVG':^17s}\n"
    for (m, result) in sorted(results.items(), key=lambda x: x[0] if x[0] != "Not applied" else np.inf):
        print_string += print_mis_calibrations(result, m)
    print_string += "\n"

    return print_string


def write_gaussian_results(eval_results):
    print_string = ""
    for (eff_prior, results) in sorted(eval_results.items(), key=lambda x: x[0]):
        print_string += f"--Effective prior: {eff_prior}--\n"
        print_string += write_tables(results)

    return print_string


def classification_analysis(models, train_data, train_labels, val_data, val_labels, application_priors, pca=None, evaluate=False, eval_results=None):
    """
    Perform classification analysis using different Gaussian models.
    
    :param models: list of Gaussian models to use
    :param train_data: training data
    :param train_labels: training labels
    :param val_data: validation data
    :param val_labels: validation labels
    :param application_priors: list of application priors
    :param evaluate: whether to evaluate the results
    :param eval_results: evaluation results dictionary
    :return err_rates: error rates
    """
    
    err_rates = {}
    for name, model in models.items():
        model.fit(train_data, train_labels)
        if model.get_name() == "Tied MVG":
            err_rates[name] = model.predict(train_data, train_labels, val_data, val_labels, application_priors, pca=pca, evaluate=evaluate, eval_results=eval_results)
        else:
            err_rates[name] = model.predict(val_data, val_labels, application_priors, pca=pca, evaluate=evaluate, eval_results=eval_results)
            
    return err_rates   

def classification_PCA_preprocessing(models, train_data, train_labels, val_data, val_labels, application_priors, evaluate=False, eval_results=None):
    """
    Perform classification analysis with PCA preprocessing.
    
    :param train_data: training data
    :param train_labels: training labels
    :param val_data: validation data
    :param val_labels: validation labels
    :param application_priors: list of application priors
    :param evaluate: whether to evaluate the results
    :param eval_results: evaluation results dictionary
    :return error_rates_pca: error rates
    """
    
    error_rates_pca = {}
    pca = PCA()
    for m in range(2, train_data.shape[0]):
        
        DTR_pca = pca.fit_transform(train_data, n_components=m)
        DVAL_pca = pca.transform(val_data)
        error_rates_pca[m] = classification_analysis(
            models,
            DTR_pca,
            train_labels,
            DVAL_pca,
            val_labels,
            application_priors,
            pca,
            evaluate,
            eval_results
        )
        
    return error_rates_pca
            
def MVG_task(trainset, validset, app_prior, effective_prior_log_odds):
    """
    Execute MVG training and evaluation for various Gaussian models.
    Autonomously handles classification analysis, PCA preprocessing, and result saving.
    
    :param trainset: Training dataset.
    :param validset: Validation dataset.
    :param app_prior: Prior probability of the positive class in the application data.
    :param effective_prior_log_odds: Effective prior log-odds for Bayes error computation.
    """
    
    DTR, LTR = trainset.get_data(), trainset.get_labels()
    DVAL, LVAL = validset.get_data(), validset.get_labels()
    
    full_mvg = MVG()
    naive_mvg = NaiveBayesMVG()
    tied_mvg = TiedMVG()

    application_priors = [effective_prior(application) for application in APPLICATIONS]
    system_applications = sorted(set(application_priors))
    
    models = dict(zip(GAUSSIAN_MODELS, [full_mvg, tied_mvg, naive_mvg]))

    # Classification with features 1-6
    # 1: 7.65 %
    # 2: 9.5 % (same as LDA)
    # 3: 7.85 %
    eval_results = dict(zip(system_applications, [dict() for _ in system_applications]))
    error_rates = classification_analysis(
        models,
        DTR,
        LTR,
        DVAL,
        LVAL,
        system_applications,
        evaluate=True,
        eval_results=eval_results
    )

    # 4: low correlation, but not null -> Indeed Naive is good, but little worse than MVG (used full_mvg, but could use any model to compute correlations)
    corr_matrices = full_mvg.compute_correlations(DTR, LTR)

    # 5: features 5 and 6 does not fit well with Gaussian assumption

    # 6: repeat analysis, but only with features 1-4
    # MVG: 7.95 %
    # Tied: 9.50 %
    # Naive: 7.65 %
    error_rates_1_4 = classification_analysis(
        models,
        DTR[0:4, :],
        LTR,
        DVAL[0:4, :],
        LVAL,
        system_applications[1:2]
    )

    # 7: repeat classification, but only with features (1-2) and then (3-4)

    # Features 1-2
    # MVG: 36.50 %
    # Tied: 49.45 %
    # Naive: 36.30 %
    error_rates_1_2 = classification_analysis(
        models,
        DTR[0:2, :],
        LTR,
        DVAL[0:2, :],
        LVAL,
        system_applications[1:2]
    )

    # Features 3-4
    # 9.45 %
    # 9.40 %
    # 9.45 %
    error_rates_3_4 = classification_analysis(
        models,
        DTR[2:4, :],
        LTR,
        DVAL[2:4, :],
        LVAL,
        system_applications[1:2]
    )

    # 8: repeat classification, by applying PCA preprocessing
    error_rates_pca = classification_PCA_preprocessing(
        models,
        DTR,
        LTR,
        DVAL,
        LVAL,
        system_applications,
        evaluate=True,
        eval_results=eval_results
    )

    if SAVE:      
        save(
            FILE_PATH_MVG,
            GAUSSIAN_ERROR_RATES,
            write_gaussian_classification_results,
            error_rates,
            corr_matrices,
            error_rates_1_4,
            error_rates_1_2,
            error_rates_3_4,
            error_rates_pca
        )

    # Get the best configuration for each evaluator
    best_configurations = Evaluator.best_configuration(eval_results, GAUSSIAN, app_prior)

    # Compute bayes errors
    bayes_errors = {
        model_name: (
            config["pca"],
            (
                effective_prior_log_odds,
                Evaluator.bayes_error(config["llr"], config["LVAL"], effective_prior_log_odds)
            )
        )
        for (model_name, config) in best_configurations.items()
    }

    # Save application prior log-odd considered in bayes error plot
    best_prior_log_odd = np.log(app_prior / (1 - app_prior))
    
    if SAVE:
        # Save application priors
        save_application_priors(
            APPLICATIONS,
            application_priors,
            FILE_PATH_MVG,
            GAUSSIAN_APPLICATION_PRIORS
        )

        # Save classification results
        save(
            FILE_PATH_MVG,
            GAUSSIAN_EVALUATION_RESULTS,
            write_gaussian_results,
            eval_results
        )

        # Plot bayes error plots
        for (model_name, model_best_info) in bayes_errors.items():
            plot_bayes_errors(
                model_best_info[1][0],
                [model_best_info[1][1]["min_dcf"]],
                [model_best_info[1][1]["dcf"]],
                best_prior_log_odd,
                f"Bayes error plot - {model_name}",
                f'''PCA {'not applied' if model_best_info[0] is None else
                f'with {model_best_info[0]} components'}''',
                "Prior log-odds",
                "DCF value",
                PLOT_PATH_MVG,
                f"{GAUSSIAN_BAYES_ERROR}_{model_name.replace(' ', '_')}",
                "png"
            )