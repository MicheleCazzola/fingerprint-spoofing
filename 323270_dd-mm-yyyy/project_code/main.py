from pprint import pprint
from sys import argv

import numpy as np

from calibration import calibration_task
from constants import (APPLICATIONS, FILE_PATH_GENERATIVE_GAUSSIAN, PLOT_PATH_GENERATIVE_GAUSSIAN, PLOT_PATH_CMP,
                       FILE_PATH_LOGISTIC_REGRESSION, FILE_PATH_SVM, FILE_PATH_GMM, GMM_EVALUATION_RESULT, LR, SVM, GMM,
                       FILE_PATH_CMP, PLOT_PATH_CAL_FUS, BEST_RESULTS_CAL, FUSION, REDUCED, SAVE, LOG,
                       FILE_PATH_FEATURES, FEATURE_STATISTICS, PLOT_PATH_FEATURES, PLOT_PATH_PCA,
                       FEATURE_PREFIX_HISTOGRAM, FEATURE_PREFIX_SCATTERPLOT, PCA_PREFIX_HISTOGRAM,
                       PCA_PREFIX_SCATTERPLOT, PLOT_SUBPATH_HISTOGRAM_LDA, LDA_HISTOGRAM, PLOT_SUBPATH_LINES_LDA,
                       LDA_ERROR_RATE_TH, PLOT_PATH_LDA, LDA_ERROR_RATE_TH_COMPACT, FILE_PATH_LDA, LDA_ERROR_RATES,
                       GAUSSIAN_APPLICATION_PRIORS, GAUSSIAN_EVALUATION_RESULTS, GAUSSIAN_BAYES_ERROR,
                       CMP_BAYES_ERROR_RAW, BEST_RESULTS_RAW, SVM_EVALUATION_RESULT, LR_EVALUATION_RESULT,
                       CMP_BAYES_ERROR_CAL, CMP_BAYES_ERROR_FUSION)
from fio import save_application_priors, save_gaussian_evaluation_results, save_LR_evaluation_results, \
    save_SVM_evaluation_results, save_GMM_evaluation_results, save_best_results, save_statistics, save_LDA_errors
from fusion import fusion_task
from plot import plot_bayes_errors, plot_estimated_features, plot_feature_distributions, plot_hist, plot_line
from gaussian import MVG_task
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

    app_prior = 0.1
    log_odd_application = np.log(app_prior / (1 - app_prior))
    eff_prior_log_odds = np.linspace(-4, 4, 101)
    (features_tr, labels_tr), (features_val, labels_val) = utilities.split_db_2to1(features, labels)

    if REDUCED:
        np.random.seed(0)
        idx = np.random.permutation(features.shape[1])[0:features.shape[1] // 20]
        features, labels = features[:, idx], labels[idx]
        (features_tr, labels_tr), (features_val, labels_val) = utilities.split_db_2to1(features, labels)

    pca = PCA()

    # Compute mean and variance per class for each feature
    statistics = utilities.compute_statistics(
        features,
        labels,
        mean=lambda array, ax, labels: (
          array[:, labels == 0].mean(axis=ax),
          array[:, labels == 1].mean(axis=ax)
        ),
        variance=lambda array, ax, labels: (
          array[:, labels == 0].var(axis=ax),
          array[:, labels == 1].var(axis=ax)
        )
    )

    if SAVE:
        # Plot distributions of the features
        plot_feature_distributions(
            features,
            labels,
            PLOT_PATH_FEATURES,
            "Feature",
            "Feature",
            FEATURE_PREFIX_HISTOGRAM,
            FEATURE_PREFIX_SCATTERPLOT,
            "pdf"
        )

        # Print mean and variance per class for each feature
        save_statistics(
            statistics,
            FILE_PATH_FEATURES,
            FEATURE_STATISTICS
        )

    features_projected_PCA = pca.fit_transform(features, n_components=6)

    features_projected_LDA = lda.apply(features, labels)

    if SAVE:
        plot_feature_distributions(
            features_projected_PCA,
            labels,
            PLOT_PATH_PCA,
            "PCA feature",
            "PCA feature",
            PCA_PREFIX_HISTOGRAM,
            PCA_PREFIX_SCATTERPLOT,
            "pdf"
        )

        plot_hist(
            features_projected_LDA[:, labels == 0],
            features_projected_LDA[:, labels == 1],
            0,
            f"{constants.PLOT_PATH_LDA}{PLOT_SUBPATH_HISTOGRAM_LDA}",
            f"LDA direction",
            f"LDA direction",
            LDA_HISTOGRAM,
            "pdf"
        )

    PVAL, error_rate, threshold_default = lda.classify(
        features_tr,
        labels_tr,
        features_val,
        labels_val
    )

    error_rate_trend, red_error_rate_trend = lda.classify_best_threshold(
        features_tr,
        labels_tr,
        features_val,
        labels_val
    )

    PCA_preprocessing_dimensions, error_rates = lda.classify_PCA_preprocess(
        features_tr,
        labels_tr,
        features_val,
        labels_val
    )

    if SAVE:
        plot_line(
            error_rate_trend[0],
            error_rate_trend[1],
            f"{PLOT_PATH_LDA}{PLOT_SUBPATH_LINES_LDA}",
            "Error rate vs. threshold",
            "Threshold",
            "Error rate",
            LDA_ERROR_RATE_TH,
            "pdf",
            (threshold_default, error_rate)
        )

        plot_line(
            red_error_rate_trend[0],
            red_error_rate_trend[1],
            f"{PLOT_PATH_LDA}{PLOT_SUBPATH_LINES_LDA}",
            "Error rate vs. threshold",
            "Threshold",
            "Error rate",
            LDA_ERROR_RATE_TH_COMPACT,
            "pdf",
            (threshold_default, error_rate)
        )

        save_LDA_errors(
            error_rate,
            PCA_preprocessing_dimensions,
            error_rates,
            FILE_PATH_LDA,
            LDA_ERROR_RATES
        )

    x_domain, y_estimations, features_per_class = fitting.gaussian_estimation(features, labels)

    if SAVE:
        plot_estimated_features(x_domain, y_estimations, features_per_class)

    application_priors, evaluation_results, bayes_errors, eff_prior_log_odd = MVG_task(
        features_tr,
        labels_tr,
        features_val,
        labels_val,
        app_prior,
        eff_prior_log_odds
    )

    if SAVE:
        # Save application priors
        save_application_priors(
            APPLICATIONS,
            application_priors,
            FILE_PATH_GENERATIVE_GAUSSIAN,
            GAUSSIAN_APPLICATION_PRIORS
        )

        # Save classification results
        save_gaussian_evaluation_results(
            evaluation_results,
            FILE_PATH_GENERATIVE_GAUSSIAN,
            GAUSSIAN_EVALUATION_RESULTS
        )

        # Plot bayes error plots
        for (model_name, model_best_info) in bayes_errors.items():
            plot.plot_bayes_errors(
                model_best_info[1][0],
                [model_best_info[1][1]["min_dcf"]],
                [model_best_info[1][1]["dcf"]],
                eff_prior_log_odd,
                f"Bayes error plot - {model_name}",
                f'''PCA {'not applied' if model_best_info[0] is None else
                f'with {model_best_info[0]} components'}''',
                "Prior log-odds",
                "DCF value",
                PLOT_PATH_GENERATIVE_GAUSSIAN,
                f"{GAUSSIAN_BAYES_ERROR}_{model_name.replace(' ', '_')}",
                "pdf"
            )

    lr_results = LR_task(features_tr, labels_tr, features_val, labels_val, app_prior)
    best_LR = Evaluator.best_configuration(lr_results, LR)

    svm_results = svm_task(features_tr, labels_tr, features_val, labels_val, app_prior)
    best_svm = Evaluator.best_configuration(svm_results["results"], SVM)

    gmm_results = gmm_task(features_tr, labels_tr, features_val, labels_val, app_prior)
    best_gmm = Evaluator.best_configuration(gmm_results, GMM)

    model_results = {
        LR: best_LR,
        SVM: best_svm,
        GMM: best_gmm
    }

    if SAVE:
        save_LR_evaluation_results(lr_results, FILE_PATH_LOGISTIC_REGRESSION, LR_EVALUATION_RESULT)
        save_SVM_evaluation_results(svm_results, FILE_PATH_SVM, SVM_EVALUATION_RESULT)
        save_GMM_evaluation_results(gmm_results, FILE_PATH_GMM, GMM_EVALUATION_RESULT)
        save_best_results(model_results, FILE_PATH_CMP, BEST_RESULTS_RAW)

    if LOG:
        print("Model classification results (no calibration)")
        for (method, result) in model_results.items():
            print(f"Method: {method}")
            print(f"Minimum DCF: {result['min_dcf']:.5f}")
            print(f"Actual DCF: {result['act_dcf']:.5f}")
            print(
                f"LLR: shape = {result['llr'].shape} "
                f"mean = {result['llr'].mean():.5f}, "
                f"max = {result['llr'].max():.5f}, "
                f"min = {result['llr'].min():.5f}, "
                f"devstd = {result['llr'].std():.5f}"
            )
            pprint(result['llr'])
            print(f"Method parameters:")
            pprint(result['params'])
            print()
        print()

    # best_model = Evaluator.best_model(model_results, "min_dcf")

    bayes_errors = list(map(
        Evaluator.bayes_error,
        [result["llr"] for result in model_results.values()],
        [labels_val for i in range(len(model_results))],
        [eff_prior_log_odds for i in range(len(model_results))]
    ))

    min_dcfs = [error["min_dcf"] for error in bayes_errors]
    dcfs = [error["dcf"] for error in bayes_errors]

    if SAVE:
        plot_bayes_errors(
            eff_prior_log_odds,
            min_dcfs,
            dcfs,
            log_odd_application,
            "Bayes error plots comparison",
            "Raw scores",
            "Prior log-odds",
            "DCF value",
            PLOT_PATH_CMP,
            CMP_BAYES_ERROR_RAW,
            "pdf",
            model_results.keys()
        )

    bayes_errors_raw = dict(zip(model_results.keys(), zip(min_dcfs, dcfs)))

    calibration_result, labels_val_unfolded = calibration_task(
        model_results,
        labels_val,
        app_prior,
        bayes_errors_raw,
        eff_prior_log_odds,
        log_odd_application
    )

    bayes_errors = list(map(
        Evaluator.bayes_error,
        [result["llr"] for result in calibration_result.values()],
        [label_val for label_val in labels_val_unfolded.values()],
        [eff_prior_log_odds for i in range(len(calibration_result))]
    ))

    if SAVE:
        min_dcfs = [error["min_dcf"] for error in bayes_errors]
        dcfs = [error["dcf"] for error in bayes_errors]
        plot_bayes_errors(
            eff_prior_log_odds,
            min_dcfs,
            dcfs,
            log_odd_application,
            "Bayes error plots comparison",
            "Calibrated scores",
            "Prior log-odds",
            "DCF value",
            PLOT_PATH_CMP,
            CMP_BAYES_ERROR_CAL,
            "pdf",
            calibration_result.keys()
        )

    scores = [model["llr"] for model in model_results.values()]
    fusion_result, labels_val_unfolded = fusion_task(scores, labels_val, app_prior)

    cal_fus_result = calibration_result | {FUSION: fusion_result}

    if LOG:
        print("Results after calibration / fusion: ")
        for (method, result) in cal_fus_result.items():
            print(f"Method: {method}")
            print(f"Minimum DCF: {result['min_dcf']:.5f}")
            print(f"Actual DCF: {result['act_dcf']:.5f}")
            print(
                f"LLR: shape = {result['llr'].shape}, "
                f"mean = {result['llr'].mean():.5f}, "
                f"max = {result['llr'].max():.5f}, "
                f"min = {result['llr'].min():.5f}, "
                f"devstd = {result['llr'].std():.5f}"
            )
            pprint(result['llr'])
            print(f"Method parameters:")
            pprint(result['params'])
            print()

    if SAVE:
        save_best_results(cal_fus_result, FILE_PATH_CMP, BEST_RESULTS_CAL)

        min_dcf_fus, act_dcf_fus = map(
            Evaluator.bayes_error(
                fusion_result["llr"],
                labels_val_unfolded,
                eff_prior_log_odds).get,
            ["min_dcf", "dcf"]
        )

        plot_bayes_errors(
            eff_prior_log_odds,
            [min_dcf_fus],
            [act_dcf_fus],
            log_odd_application,
            "Bayes error plots",
            "Fused scores",
            "Prior log-odds",
            "DCF value",
            PLOT_PATH_CAL_FUS,
            CMP_BAYES_ERROR_FUSION,
            "pdf"
        )
