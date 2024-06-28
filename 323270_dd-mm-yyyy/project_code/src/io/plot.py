import matplotlib.pyplot as plt
from src.io.constants import LABEL_NAMES, PLOT_PATH_ESTIMATIONS


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
        plot_hist(features0, features1, c,
                  f"{path_root}histograms/",
                  f"{title_prefix} {c + 1}",
                  f"{axes_prefix} {c + 1}",
                  f"{name_prefix_hist}_{c + 1}",
                  extension)

    # Scatter plots
    for i in range(features.shape[0]):
        for j in range(i + 1, features.shape[0]):
            plot_scatter(features0, features1, i, j,
                         f"{path_root}scatterplots/",
                         f"{title_prefix}s {i + 1}, {j + 1}",
                         f"{axes_prefix} {i + 1}",
                         f"{axes_prefix} {j + 1}",
                         f"{name_prefix_scatter}_{i + 1}_{j + 1}",
                         extension)


def plot_hist(features_false, features_true, n, path, title, axis_label, name, extension):
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


def plot_scatter(features_false, features_true, n1, n2, path, title, x_label, y_label, name, extension):
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
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.title(f"{title} scatter plot")
    plt.savefig(f"{path}{name}.{extension}")
    plt.close(name)


def plot_line(x, y, path, title, x_label, y_label, name, extension, cross_center=None):
    """
    Prints a line plot of y vs. x, with specified parameters

    :param x: x vector
    :param y: y vector
    :param path: path to store the plots
    :param title: plot title
    :param x_label: x-axis label
    :param y_label: y-axis label
    :param name: name of the plot in the file system
    :param extension: file extension of the plot
    :param cross_center: center (x, y) of the cross (optional)
    :return: None
    """

    plt.figure(name)
    plt.plot(x, y, linewidth=2)
    plt.grid()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"{title}")
    plt.vlines(x=cross_center[0], ymin=cross_center[1] - 0.01, ymax=cross_center[1] + 0.01, colors="k")
    plt.hlines(y=cross_center[1], xmin=cross_center[0] - 0.1, xmax=cross_center[0] + 0.1, colors="k")
    plt.savefig(f"{path}{name}.{extension}")
    plt.close(name)


def plot_line_hist(x, y_false, y_true, features_false, features_true,
                   path, title, axis_label, name, extension):
    """
    Plots a line chart over a histogram

    :param x: the domain where plot the charts
    :param y_false: ordinate values for line chart, false class
    :param y_true: ordinate values for line chart, true class
    :param features_false: feature values for histogram, false class
    :param features_true: feature values for histogram, true class
    :param path: path to store the plots
    :param title: plot title
    :param axis_label: x-axis label
    :param name: name of the plot in the file system
    :param extension: file extension of the plot
    :return: None
    """
    plt.figure(name)
    plt.hist(features_false.ravel(), bins=50, density=True, alpha=0.4, label=LABEL_NAMES[False], color="dodgerblue")
    plt.hist(features_true.ravel(), bins=50, density=True, alpha=0.4, label=LABEL_NAMES[True], color="orange")
    plt.plot(x, y_false, color="blue", label=f"Est. {LABEL_NAMES[False]}")
    plt.plot(x, y_true, color="red", label=f"Est. {LABEL_NAMES[True]}")
    plt.xlabel(axis_label)
    plt.legend()
    plt.title(f"{title}")
    plt.savefig(f"{path}{name}.{extension}")
    plt.close(name)


def plot_estimated_features(x, y, features):
    """
    Plots estimated features, by overlapping estimated graph over the corresponding histogram

    :param x: features domain
    :param y: estimated values for the features
    :param features: dataset features, divided by class
    :return: None
    """
    i = 0
    for ((y_est_false, y_est_true), (f_false, f_true)) in zip(y, features):
        plot_line_hist(x, y_est_false, y_est_true, f_false, f_true,
                       PLOT_PATH_ESTIMATIONS,
                       f"Feature {i + 1} estimate",
                       f"Estimated feature {i + 1}",
                       f"estimated_feature_{i + 1}",
                       "pdf")
        i += 1


def plot_bayes_errors(eff_prior_log_odds, min_dcf, actual_dcf, eff_prior_log_odd, title, subtitle, x_label, y_label,
                      path, name, extension, models=None):
    if models is None:
        models = [None] * len(min_dcf)

    plt.figure(name)
    for (min_dcf_val, act_dcf_val, model) in zip(min_dcf, actual_dcf, models):
        model_lab = "" if model is None else f" - {model}"
        plt.plot(eff_prior_log_odds, min_dcf_val, label=f"Minimum DCF{model_lab}")
        plt.plot(eff_prior_log_odds, act_dcf_val, label=f"Actual DCF{model_lab}")
    plt.vlines(x=eff_prior_log_odd, ymin=plt.axis()[2], label="System application", ymax=plt.axis()[3], color="black",
               linewidth=2, linestyle="dashed")
    plt.grid()
    plt.legend()
    plt.margins(0)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.suptitle(title)
    plt.title(subtitle, fontsize="medium", fontweight=400)
    plt.savefig(f"{path}{name}.{extension}")
    plt.close(name)


def plot_log_double_line(x, y1, y2, title, x_label, y_label, legend1, legend2, path, name, extension, subtitle=None):
    plt.figure(name)
    plt.xscale('log', base=10)
    plt.plot(x, y1, label=legend1)
    plt.plot(x, y2, label=legend2)
    plt.grid()
    plt.margins(0.1)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    if subtitle:
        plt.suptitle(title, fontsize="large")
        plt.title(subtitle, fontsize="medium")
    else:
        plt.title(title)
    plt.savefig(f"{path}{name}.{extension}")
    plt.close(name)


def plot_log_N_double_lines(x, ys1, ys2, title, x_label, y_label, legends1, legends2, path, name, extension):
    plt.figure(name)
    plt.xscale('log', base=10)
    for (y1, y2, legend1, legend2) in zip(ys1, ys2, legends1, legends2):
        plt.plot(x, ys1[y1], label=legend1)
        plt.plot(x, ys2[y2], label=legend2)
    plt.grid()
    plt.margins(0.1)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.title(title)
    plt.savefig(f"{path}{name}.{extension}")
    plt.close(name)