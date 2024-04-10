from sys import argv
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

    lab2.save_statistics(statistics, lab2.FILE_PATH, "feature_statistics.txt")

    features_projected_PCA = lab3.PCA(features, 6)

    lab2.plot_feature_distributions(features_projected_PCA, labels, lab2.PLOT_PATH_PCA,
                                    "PCA feature",
                                    "PCA feature",
                                    "PCA_histogram",
                                    "PCA_scatter",
                                    "pdf")

    features_projected_LDA = lab3.LDA_apply(features, labels)

    lab2.print_hist(features_projected_LDA[:, labels == 0],
                    features_projected_LDA[:, labels == 1],
                    0,
                    f"{lab2.PLOT_PATH_LDA}histograms\\",
                    f"LDA direction",
                    f"LDA direction",
                    f"LDA_histogram",
                    "pdf")

    PVAL, error_rate, threshold_default = lab3.LDA_classify(features, labels)

    error_rate_trend, red_error_rate_trend = lab3.LDA_classification_best_threshold(features, labels)

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

    PCA_preprocessing_dimensions, error_rates = lab3.LDA_classification_PCA(features, labels)

    lab2.save_LDA_errors(error_rate, PCA_preprocessing_dimensions, error_rates,
                         lab2.FILE_PATH_LDA, "error_rates.txt")
