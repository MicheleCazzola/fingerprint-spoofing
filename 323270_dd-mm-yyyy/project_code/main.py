from sys import argv

import numpy as np

from constants import (APPLICATIONS, FILE_PATH_GENERATIVE_GAUSSIAN, PLOT_PATH_GENERATIVE_GAUSSIAN, PLOT_PATH_CMP,
                       FILE_PATH_LOGISTIC_REGRESSION, FILE_PATH_SVM, FILE_PATH_GMM, GMM_EVALUATION, LR, SVM, GMM,
                       FILE_PATH_CMP)
from fio import save_application_priors, save_gaussian_evaluation_results, save_LR_evaluation_results, \
    save_SVM_evaluation_results, save_GMM_results, save_best_results
from plot import plot_bayes_errors
from gaussian import gaussian_classification
from gmm import gmm_task
from discriminative.logreg import LR_task
from discriminative.svm import svm_task
from src.dimred import lda
from src.dimred.pca import PCA
from src.io import fio, constants, plot
from src.utilities import utilities
from src.fitting import fitting
from evaluation.evaluation import Evaluator

if __name__ == "__main__":

    # Load data with exception handling
    try:
        features, labels = fio.load_csv(argv[1])
    except IndexError:
        exit("Missing argument: name of training data file")
    except FileNotFoundError:
        exit(f"File {argv[1]} not found")

    (features_tr, labels_tr), (features_val, labels_val) = utilities.split_db_2to1(features, labels)

    pca = PCA()

    # Plot distributions of the features
    plot.plot_feature_distributions(features, labels,
                                    constants.PLOT_PATH,
                                    "Feature",
                                    "Feature",
                                    "histogram",
                                    "scatter",
                                    "pdf")

    # Compute and print mean and variance per class for each feature
    statistics = utilities.compute_statistics(features, labels,
                                              Mean=lambda array, ax, labels: (
                                                  array[:, labels == 0].mean(axis=ax),
                                                  array[:, labels == 1].mean(axis=ax)),
                                              Variance=lambda array, ax, labels: (
                                                  array[:, labels == 0].var(axis=ax),
                                                  array[:, labels == 1].var(axis=ax)))

    fio.save_statistics(statistics, constants.FILE_PATH, "feature_statistics.txt")

    features_projected_PCA = pca.fit_transform(features, n_components=6)

    plot.plot_feature_distributions(features_projected_PCA, labels, constants.PLOT_PATH_PCA,
                                    "PCA feature",
                                    "PCA feature",
                                    "PCA_histogram",
                                    "PCA_scatter",
                                    "pdf")

    features_projected_LDA = lda.apply(features, labels)

    plot.plot_hist(features_projected_LDA[:, labels == 0],
                   features_projected_LDA[:, labels == 1],
                   0,
                   f"{constants.PLOT_PATH_LDA}histograms/",
                   f"LDA direction",
                   f"LDA direction",
                   f"LDA_histogram",
                   "pdf")

    PVAL, error_rate, threshold_default = lda.classify(features, labels)

    error_rate_trend, red_error_rate_trend = lda.classify_best_threshold(features, labels)

    plot.plot_line(error_rate_trend[0], error_rate_trend[1],
                   f"{constants.PLOT_PATH_LDA}lines/",
                   "Error rate vs. threshold",
                   "Threshold",
                   "Error rate",
                   "error_rate_threshold",
                   "pdf",
                   (threshold_default, error_rate))

    plot.plot_line(red_error_rate_trend[0], red_error_rate_trend[1],
                   f"{constants.PLOT_PATH_LDA}lines/",
                   "Error rate vs. threshold",
                   "Threshold",
                   "Error rate",
                   "error_rate_threshold_compact",
                   "pdf",
                   (threshold_default, error_rate))

    PCA_preprocessing_dimensions, error_rates = lda.classify_PCA_preprocess(features, labels)

    fio.save_LDA_errors(error_rate, PCA_preprocessing_dimensions, error_rates,
                        constants.FILE_PATH_LDA, "error_rates.txt")

    x_domain, y_estimations, features_per_class = fitting.gaussian_estimation(features, labels)

    plot.plot_estimated_features(x_domain, y_estimations, features_per_class)

    application_priors, evaluation_results, bayes_errors, eff_prior_log_odd = gaussian_classification(features, labels)

    # Save application priors
    save_application_priors(APPLICATIONS, application_priors,
                            FILE_PATH_GENERATIVE_GAUSSIAN, "gaussian_application_priors.txt")

    # Save classification results
    save_gaussian_evaluation_results(evaluation_results,
                                     FILE_PATH_GENERATIVE_GAUSSIAN,
                                     "gaussian_evaluation_results.txt")

    # Plot bayes error plots
    _ = [plot.plot_bayes_errors(model_best_info[1][0],
                                [model_best_info[1][1]["min_dcf"]],
                                [model_best_info[1][1]["dcf"]],
                                eff_prior_log_odd,
                                f"Bayes error plot - {model_name}",
                                f'''PCA {'not applied' if model_best_info[0] is None else
                                f'with {model_best_info[0]} components'}''',
                                "Prior log-odds",
                                "DCF value",
                                PLOT_PATH_GENERATIVE_GAUSSIAN,
                                f"bayes_error_{model_name.replace(' ', '_')}",
                                "pdf")
         for (model_name, model_best_info) in bayes_errors.items()]

    # np.random.seed(0)
    # idx = np.random.permutation(features.shape[1])[0:features.shape[1] // 10]
    # reduced_features, reduced_labels = features[:, idx], labels[idx]
    # _, (red_feat_val, red_lab_val) = utilities.split_db_2to1(reduced_features, reduced_labels)

    lr_results = LR_task(features, labels)
    best_LR = Evaluator.best_configuration(lr_results, LR)
    save_LR_evaluation_results(lr_results, FILE_PATH_LOGISTIC_REGRESSION, "LR_evaluation_results.txt")

    svm_results = svm_task(features, labels)
    best_svm = Evaluator.best_configuration(svm_results["results"], SVM)
    save_SVM_evaluation_results(svm_results, FILE_PATH_SVM, "SVM_evaluation_results.txt")

    gmm_results = gmm_task(features, labels)
    best_gmm = Evaluator.best_configuration(gmm_results, GMM)
    save_GMM_results(gmm_results, FILE_PATH_GMM, GMM_EVALUATION)

    app_prior = 0.1
    model_results = {
        LR: best_LR,
        SVM: best_svm,
        GMM: best_gmm
    }
    save_best_results(model_results, FILE_PATH_CMP, "best_results.txt")
    best_model = Evaluator.best_model(model_results, "min_dcf")

    eff_prior_log_odds = np.linspace(-4, 4, 33)
    bayes_errors = list(map(Evaluator.bayes_error,
                            [result["llr"] for result in model_results.values()],
                            [labels_val for i in range(len(model_results))],
                            [eff_prior_log_odds for i in range(len(model_results))]))

    min_dcfs = [error["min_dcf"] for error in bayes_errors]
    dcfs = [error["dcf"] for error in bayes_errors]
    log_odd_application = np.log(app_prior / (1 - app_prior))
    plot_bayes_errors(eff_prior_log_odds, min_dcfs, dcfs, log_odd_application,
                      "Bayes error plots comparison",
                      "",
                      "Prior log-odds",
                      "DCF value",
                      PLOT_PATH_CMP,
                      "bayes_error_comparison",
                      "pdf",
                      model_results.keys())
