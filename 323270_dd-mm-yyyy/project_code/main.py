from sys import argv
import numpy as np
import matplotlib.pyplot as plt

label_names = {
    False: "Fake",
    True: "Genuine"
}


def load_csv(filename):
    features = []
    labels = []
    with open(filename, mode="r") as fin:
        for line in fin:
            fields = line.strip().split(",")
            v = np.array([float(f.strip()) for f in fields[:-1]])
            features.append(v.reshape((len(v), 1)))
            labels.append(int(fields[-1].strip()))

    return np.hstack(features), np.array(labels, dtype=np.int32)


def plot_feature_distributions(features, labels):
    features0 = features[:, labels == 0]
    features1 = features[:, labels == 1]

    # Histogram plot
    for c in range(features.shape[0]):
        print_hist(features0, features1, c)

    # Scatter plots
    for i in range(features.shape[0]):
        for j in range(i + 1, features.shape[0]):
            print_scatter(features0, features1, i, j)

    plt.show()


def compute_statistics(features, labels, **functions):
    for (name, func) in functions.items():
        result = func(features, 1, labels)
        print(f"--{name} values--")
        for i in range(0, 4):
            print(f"Feature {i}:\n"
                  f"\t{label_names[False]}: {result[0][i]:.3f}\n"
                  f"\t{label_names[True]}: {result[1][i]:.3f}")
        print()


def print_hist(features_false, features_true, n):
    plt.figure(f"Histogram for feature {n + 1}")
    plt.hist(features_false[n, :], bins=20, density=True, alpha=0.5, label=label_names[False])
    plt.hist(features_true[n, :], bins=20, density=True, alpha=0.5, label=label_names[True])
    plt.xlabel(f"Feature {n + 1}")
    plt.legend()
    plt.title(f"Feature {n + 1} histogram")


def print_scatter(features_false, features_true, n1, n2):
    plt.figure(f"Scatter plot for features {n1 + 1}, {n2 + 1}")
    plt.scatter(features_false[n1:n1 + 1, :], features_false[n2:n2 + 1, :], alpha=0.2, label=label_names[False])
    plt.scatter(features_true[n1:n1 + 1, :], features_true[n2:n2 + 1, :], alpha=0.2, label=label_names[True])
    plt.xlabel(f"Feature {n1 + 1}")
    plt.ylabel(f"Feature {n2 + 1}")
    plt.legend()
    plt.title(f"Features {n1 + 1}, {n2 + 1} scatter plot")


if __name__ == "__main__":

    # Load data with exception handling
    try:
        features, labels = load_csv(argv[1])
    except IndexError:
        exit("Missing argument: name of training data file")
    except FileNotFoundError:
        exit(f"File {argv[1]} not found")

    # Plot distributions of the features
    plot_feature_distributions(features, labels)

    # Compute and print mean and variance per class
    # for each of the first four features
    compute_statistics(features, labels,
                       Mean=lambda array, ax, labels: (
                           array[:, labels == 0].mean(axis=ax), array[:, labels == 1].mean(axis=ax)),
                       Variance=lambda array, ax, labels: (
                           array[:, labels == 0].var(axis=ax), array[:, labels == 1].var(axis=ax)))
