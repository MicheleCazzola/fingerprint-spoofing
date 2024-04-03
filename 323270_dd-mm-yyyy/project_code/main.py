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

    lab2.plot_statistics(statistics, lab2.FILE_PATH, "feature_statistics.txt")

    features_projected = lab3.PCA(features, 6)

    lab2.plot_feature_distributions(features_projected, labels, lab3.PLOT_PATH,
                                    "PCA feature",
                                    "PCA feature",
                                    "PCA_histogram",
                                    "PCA_scatter",
                                    "pdf")
