import numpy as np
import matplotlib.pyplot as plt

LABEL_NAMES = {
    False: "Fake",
    True: "Genuine"
}

PLOT_PATH = "output\\plots\\original_features\\"
FILE_PATH = "output\\files\\"


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


def plot_feature_distributions(features, labels, path_root, title_prefix, axes_prefix, name_prefix_hist,
                               name_prefix_scatter, extension):
    """
    Plots features selected into histograms and scatter plots

    :param features: features of the dataset to print
    :param labels: labels associated with the features in the training set
    :param path_root: root directory to save the plots
    :param title_prefix: name of the title for the plots
    :param axes_prefix: name of the axes for the plots
    :param name_prefix_hist: prefix for name of histogram plots
    :param name_prefix_scatter: prefix for name of scatter plots
    :param extension: extension of the plot files
    :return: None
    """
    features0 = features[:, labels == 0]
    features1 = features[:, labels == 1]

    # Histogram plot
    for c in range(features.shape[0]):
        print_hist(features0, features1, c,
                   f"{path_root}histograms\\",
                   f"{title_prefix} {c + 1}",
                   f"{axes_prefix} {c + 1}",
                   f"{name_prefix_hist}_{c + 1}",
                   extension)

    # Scatter plots
    for i in range(features.shape[0]):
        for j in range(i + 1, features.shape[0]):
            print_scatter(features0, features1, i, j,
                          f"{path_root}scatterplots\\",
                          f"{title_prefix}s {i + 1}, {j + 1}",
                          f"{axes_prefix} {i + 1}",
                          f"{axes_prefix} {j + 1}",
                          f"{name_prefix_scatter}_{i + 1}_{j + 1}",
                          extension)


def compute_statistics(features, labels, **functions):
    """
    Computes some statistics about features and labels and store them in a dictionary

    :param features: features to compute statistics for
    :param labels: labels to compute statistics for
    :param functions: dictionary of functions with statistics to compute
    :return: a dictionary with statistics about features and labels
    """
    r = {}
    for (name, func) in functions.items():
        result = func(features, 1, labels)
        r[name] = result

    return r


def plot_statistics(statistics, path_root, file_name):
    """
    Prints some pre-computed statistics about features and labels at the specified path

    :param statistics: dictionary with statistics to print
    :param path_root: folder to store the printed statistics
    :param file_name: file to store the printed statistics
    :return: None
    """
    with open(f"{path_root}\\{file_name}", mode="w") as fout:
        for (name, stat) in statistics.items():
            fout.write(f"--{name} values--\n")
            for i in range(len(stat[0])):
                fout.write(f"Feature {i + 1}:\n"
                           f"\t{LABEL_NAMES[False]}: {stat[0][i]:.3f}\n"
                           f"\t{LABEL_NAMES[True]}: {stat[1][i]:.3f}\n")
            fout.write("\n")


def print_hist(features_false, features_true, n, path, title, axis_label, name, extension):
    """
    Prints a histogram of the features, with specified parameters

    :param features_false: features for false class
    :param features_true: features for true class
    :param n: index of feature to print
    :param path: path to store the plots
    :param title: plot title
    :param axis_label: x-axis label
    :param name: name of the plot in the file system
    :param extension: file extension of the plot
    :return: None
    """
    plt.figure(name)
    plt.hist(features_false[n, :], bins=20, density=True, alpha=0.4, label=LABEL_NAMES[False])
    plt.hist(features_true[n, :], bins=20, density=True, alpha=0.4, label=LABEL_NAMES[True])
    plt.xlabel(axis_label)
    plt.legend()
    plt.title(f"{title} histogram")
    plt.savefig(f"{path}{name}.{extension}")
    plt.close(name)


def print_scatter(features_false, features_true, n1, n2, path, title, x_label, y_label, name, extension):
    """
    Prints a scatter plot of the pair of features, with specified parameters

    :param features_false: features for false class
    :param features_true: features for true class
    :param n1: index of the first feature to print
    :param n2: index of the second feature to print
    :param path: path to store the plots
    :param title: plot title
    :param x_label: x-axis label
    :param y_label: y-axis label
    :param name: name of the plot in the file system
    :param extension: file extension of the plot
    :return: None
    """
    plt.figure(name)
    plt.scatter(features_false[n1:n1 + 1, :], features_false[n2:n2 + 1, :], alpha=0.4, label=LABEL_NAMES[False])
    plt.scatter(features_true[n1:n1 + 1, :], features_true[n2:n2 + 1, :], alpha=0.4, label=LABEL_NAMES[True])
    plt.xlabel(f"Feature {n1 + 1}")
    plt.ylabel(f"Feature {n2 + 1}")
    plt.legend()
    plt.title(f"{title} scatter plot")
    plt.savefig(f"{path}{name}.{extension}")
    plt.close(name)
