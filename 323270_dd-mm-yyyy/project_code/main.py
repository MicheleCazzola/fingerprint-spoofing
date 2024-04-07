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

    features_projected = lab3.PCA(features, 6)

    lab2.plot_feature_distributions(features_projected, labels, lab3.PLOT_PATH_PCA,
                                    "PCA feature",
                                    "PCA feature",
                                    "PCA_histogram",
                                    "PCA_scatter",
                                    "pdf")

    LDA_object = lab3.LDA(features, labels, 1)
    features_projected_LDA = LDA_object.projection(LDA_object.D, LDA_object.W)

    print(features_projected_LDA.shape)

    lab2.print_hist(features_projected_LDA[:, labels == 0],
                    features_projected_LDA[:, labels == 1],
                    0,
                    f"{lab3.PLOT_PATH_LDA}histograms\\",
                    f"LDA direction",
                    f"LDA direction",
                    f"LDA_histogram",
                    "pdf")

    predicted_labels, real_labels, error_rate, threshold = LDA_object.classification()

    print(predicted_labels, real_labels, error_rate, threshold, sep = "\n")

    error_rate_trend = LDA_object.classification_best_threshold()

    plt.figure(1, figsize=(12.8, 9.6), dpi=400)
    plt.plot(error_rate_trend[0], error_rate_trend[1], linewidth=1)
    plt.vlines(x = threshold, ymin = 0.09, ymax = 0.1, colors = "b")
    plt.hlines(y = error_rate, xmin = -0.25, xmax = 0.25, colors = "r")
    plt.savefig(lab3.PLOT_PATH_LDA + "trend.png")
    plt.show()