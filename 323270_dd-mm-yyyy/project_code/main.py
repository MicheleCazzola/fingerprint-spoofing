from sys import argv
from src.dimred import lda, pca
from src.io import fio, constants, plot
from src.utilities import utilities
from src.fitting import fitting

if __name__ == "__main__":

    # Load data with exception handling
    try:
        features, labels = fio.load_csv(argv[1])
    except IndexError:
        exit("Missing argument: name of training data file")
    except FileNotFoundError:
        exit(f"File {argv[1]} not found")

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

    features_projected_PCA = pca.apply(features, 6)

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