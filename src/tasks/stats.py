"""
    Statistics task module.
    
    Handles statistical analysis of datasets, including computation of feature means and variances,
    plotting feature distributions, and saving statistical results.
"""

from src.config.config import FEATURE_PREFIX_HISTOGRAM, FEATURE_PREFIX_SCATTERPLOT, FEATURE_STATISTICS, FILE_PATH_FEATURES, LABEL_NAMES, PLOT_PATH_EVAL_FEATURES, PLOT_PATH_FEATURES, SAVE
from src.dataset.dataset import Dataset
from src.utils.plot import plot_feature_distributions
from src.utils.utils import compute_statistics, save


def write_statistics(statistics):
    print_string = ""
    for (name, stat) in statistics.items():
        print_string += f"--{name} values--\n"
        for i in range(len(stat[0])):
            print_string += (f"Feature {i + 1}:\n"
                             f"\t{LABEL_NAMES[False]}: {stat[0][i]:.3f}\n"
                             f"\t{LABEL_NAMES[True]}: {stat[1][i]:.3f}\n")
        print_string += "\n"

    return print_string


def stats_task(trainset: Dataset, testset: Dataset):
    """
    Perform statistical analysis on the training and test datasets.
    Handles analysis, plotting, and saving of feature statistics.
    
    :param trainset: Training dataset.
    :param testset: Test dataset.
    """
    # Compute mean and variance per class for each feature
    statistics = compute_statistics(
        trainset.features,
        trainset.labels,
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
            trainset.features,
            trainset.labels,
            PLOT_PATH_FEATURES,
            "Feature",
            "Feature",
            FEATURE_PREFIX_HISTOGRAM,
            FEATURE_PREFIX_SCATTERPLOT,
            "png"
        )

        plot_feature_distributions(
            testset.features,
            testset.labels,
            PLOT_PATH_EVAL_FEATURES,
            "Feature",
            "Feature",
            FEATURE_PREFIX_HISTOGRAM,
            FEATURE_PREFIX_SCATTERPLOT,
            "png"
        )

        # Print mean and variance per class for each feature
        save(
            FILE_PATH_FEATURES,
            FEATURE_STATISTICS,
            write_statistics,
            statistics
        )