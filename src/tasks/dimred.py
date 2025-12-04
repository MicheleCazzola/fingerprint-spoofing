"""
    Dimensionality Reduction Task Module
    
    This module performs dimensionality reduction using PCA and LDA, visualizes the results,
    and evaluates classification performance on the reduced data.
"""

from src.config.config import FILE_PATH_LDA, LDA_ERROR_RATE_TH, LDA_ERROR_RATE_TH_COMPACT, LDA_ERROR_RATES, LDA_HISTOGRAM, PCA_PREFIX_HISTOGRAM, PCA_PREFIX_SCATTERPLOT, PLOT_PATH_LDA, PLOT_PATH_PCA, PLOT_SUBPATH_HISTOGRAM_LDA, PLOT_SUBPATH_LINES_LDA, SAVE
from src.dataset.dataset import Dataset
from src.dimred.lda import LDA
from src.dimred.pca import PCA
from src.utils.plot import plot_feature_distributions, plot_hist, plot_line
from src.utils.utils import save

def write_LDA_errors(base_error_rate, dimensions, error_rates):
    print_string = ""
    print_string += ("Classification error rate without PCA preprocessing " +
                     f"{base_error_rate:.4f} ({100 * base_error_rate:.2f} %)\n\n")
    print_string += f"--Classification error rates with PCA preprocessing--\n"
    print_string += f"PCA dimensions\tError rate\tError rate (%)\n"
    for (dim, err) in zip(dimensions, error_rates):
        print_string += f"{dim:^14d}\t{err:^10.4f}\t{100 * err:^13.2f}\n"

    return print_string


def dimred_task(
    wholeset: Dataset, trainset: Dataset, validset: Dataset,
):
    """Performs dimensionality reduction using PCA and LDA, visualizes results,
    and evaluates classification performance.
    
    :param wholeset: complete dataset features and labels
    :param trainset: training dataset features and labels
    :param validset: validation dataset features and labels
    """

    pca = PCA()
    features_projected_PCA = pca.fit_transform(wholeset.features, n_components=6)

    lda = LDA()
    features_projected_LDA = lda.fit_transform(wholeset.features, wholeset.labels)

    if SAVE:
        plot_feature_distributions(
            features_projected_PCA,
            wholeset.labels,
            PLOT_PATH_PCA,
            "PCA feature",
            "PCA feature",
            PCA_PREFIX_HISTOGRAM,
            PCA_PREFIX_SCATTERPLOT,
            "png"
        )

        plot_hist(
            features_projected_LDA[:, wholeset.labels == 0],
            features_projected_LDA[:, wholeset.labels == 1],
            0,
            f"{PLOT_PATH_LDA}{PLOT_SUBPATH_HISTOGRAM_LDA}",
            f"LDA direction",
            f"LDA direction",
            LDA_HISTOGRAM,
            "png"
        )

    PVAL, error_rate, threshold_default = lda.classify(
        trainset.features,
        trainset.labels,
        validset.features,
        validset.labels
    )

    error_rate_trend, red_error_rate_trend = lda.classify_generalized_threshold(
        trainset.features,
        trainset.labels,
        validset.features,
        validset.labels
    )

    PCA_preprocessing_dimensions, error_rates = lda.classify_generalized_pca(
        trainset.features,
        trainset.labels,
        validset.features,
        validset.labels,
        pca_maxdim=5
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
            "png",
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
            "png",
            (threshold_default, error_rate)
        )
        
        save(
            FILE_PATH_LDA, 
            LDA_ERROR_RATES,
            write_LDA_errors,
            error_rate,
            PCA_preprocessing_dimensions,
            error_rates
        )