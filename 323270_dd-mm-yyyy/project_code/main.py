from sys import argv

import matplotlib.pyplot as plt

import lab2_loading_plots as lab2
import lab3_dimensionality_reduction as lab3

if __name__ == "__main__":

    # Load data with exception handling
    try:
        features, labels = lab2.load_csv(argv[1])
    except IndexError:
        exit("Missing argument: name of training data file")
    except FileNotFoundError:
        exit(f"File {argv[1]} not found")

    # Plot distributions of the features
    lab2.plot_feature_distributions(features, labels,
                                    lab2.PLOT_PATH,
                                    "Feature",
                                    "Feature",
                                    "histogram",
                                    "scatter",
                                    "pdf")

    # Compute and print mean and variance per class
    # for each of the first four features
    statistics = lab2.compute_statistics(features, labels,
                                         Mean=lambda array, ax, labels: (
                                             array[:, labels == 0].mean(axis=ax), array[:, labels == 1].mean(axis=ax)),
                                         Variance=lambda array, ax, labels: (
                                             array[:, labels == 0].var(axis=ax), array[:, labels == 1].var(axis=ax)))

    lab2.plot_statistics(statistics, lab2.FILE_PATH, "feature_statistics.txt")

    features_projected_PCA = lab3.PCA(features, 6)

    lab2.plot_feature_distributions(features_projected_PCA, labels, lab2.PLOT_PATH_PCA,
                                    "PCA feature",
                                    "PCA feature",
                                    "PCA_histogram",
                                    "PCA_scatter",
                                    "pdf")

    features_projected_LDA = lab3.LDA_apply(features, labels)

    print(features_projected_LDA.shape)

    lab2.print_hist(features_projected_LDA[:, labels == 0],
                    features_projected_LDA[:, labels == 1],
                    0,
                    f"{lab2.PLOT_PATH_LDA}histograms\\",
                    f"LDA direction",
                    f"LDA direction",
                    f"LDA_histogram",
                    "pdf")

    PVAL, error_rate, threshold_default = lab3.LDA_classify(features, labels)
    print(f"{error_rate}")

    error_rate_trend, red_error_rate_trend = lab3.classification_best_threshold(features, labels)

    lab2.print_line_plot(error_rate_trend[0], error_rate_trend[1],
                         f"{lab2.PLOT_PATH_LDA}lines\\",
                         "Error rate vs. threshold",
                         "Threshold",
                         "Error rate",
                         "error_rate_threshold",
                         "pdf",
                         (threshold_default, error_rate))

    lab2.print_line_plot(red_error_rate_trend[0], red_error_rate_trend[1],
                         f"{lab2.PLOT_PATH_LDA}lines\\",
                         "Error rate vs. threshold",
                         "Threshold",
                         "Error rate",
                         "error_rate_threshold_compact",
                         "pdf",
                         (threshold_default, error_rate))

    for m in range(5, 1, -1):
        PVAL, error_rate, _ = lab3.LDA_classify(features, labels, m, True)
        print(f"{m}: {error_rate}")
