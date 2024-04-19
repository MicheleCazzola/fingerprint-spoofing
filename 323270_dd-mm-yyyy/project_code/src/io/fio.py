import numpy as np
from .constants import LABEL_NAMES


def load_csv(filename):
    """
    Reads a csv file with comma separator and saves features and labels
    into a two distinct numpy arrays

    :param filename: name of csv file with dataset
    :return: a tuple, with features as horizontal stacked column objects and labels as an array
    """
    features = []
    labels = []
    with open(filename, mode="r") as fin:
        for line in fin:
            fields = line.strip().split(",")
            v = np.array([float(f.strip()) for f in fields[:-1]])
            features.append(v.reshape((len(v), 1)))
            labels.append(int(fields[-1].strip()))

    return np.hstack(features), np.array(labels, dtype=np.int32)


def save_statistics(statistics, path_root, file_name):
    """
    Prints some pre-computed statistics about features and labels at the specified path

    :param statistics: dictionary with statistics to print
    :param path_root: folder to store the printed statistics
    :param file_name: file to store the printed statistics
    :return: None
    """
    with open(f"{path_root}{file_name}", mode="w", encoding="utf-8") as fout:
        for (name, stat) in statistics.items():
            fout.write(f"--{name} values--\n")
            for i in range(len(stat[0])):
                fout.write(f"Feature {i + 1}:\n"
                           f"\t{LABEL_NAMES[False]}: {stat[0][i]:.3f}\n"
                           f"\t{LABEL_NAMES[True]}: {stat[1][i]:.3f}\n")
            fout.write("\n")


def save_LDA_errors(base_error_rate, dimensions, error_rates, path_root, file_name):
    """
    Prints LDA error rates at the specified path

    :param base_error_rate: LDA error rate without PCA preprocessing
    :param dimensions: list of dimensions after PCA preprocessing
    :param error_rates: list of error rates after PCA preprocessing
    :param path_root: folder to store the printed statistics
    :param file_name: file to store the printed statistics
    :return: None
    """
    with open(f"{path_root}{file_name}", mode="w", encoding="utf-8") as fout:
        fout.write("Classification error rate without PCA preprocessing " +
                   f"{base_error_rate:.4f} ({100 * base_error_rate:.2f} %)\n\n")
        fout.write(f"--Classification error rates with PCA preprocessing--\n")
        fout.write(f"PCA dimensions\tError rate\tError rate (%)\n")
        for (dim, err) in zip(dimensions, error_rates):
            fout.write(f"{dim:^14d}\t{err:^10.4f}\t{100 * err:^13.2f}\n")
