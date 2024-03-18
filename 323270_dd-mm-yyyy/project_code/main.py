from sys import argv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class Entry:
    def __init__(self, line):
        fields = line.strip().split(",")
        self.features = [float(f.strip()) for f in fields[:-1]]
        self.label = float(fields[-1].strip())


def load_train_data(train_data_file):
    training_set = []
    with open(train_data_file, mode="r") as fin:
        for line in fin:
            training_set.append(Entry(line))

    return training_set


def print_hist(features, class_masks, start, end):
    for c in range(start, end):
        plt.figure(f"Histogram for feature {c}")
        plt.hist(features[c, class_masks[0]], bins=20, density=True, alpha=0.5, label=f"False class")
        plt.hist(features[c, class_masks[1]], bins=20, density=True, alpha=0.5, label=f"True class")
        plt.xlabel(f"Feature {c}")
        plt.legend()

        # Only to test
        plt.plot(np.linspace(features[c].min(), features[c].max(), 100),
                 norm.pdf(np.linspace(features[c].min(), features[c].max(), 100), features[c].mean(), features[c].std()),
                 'k', linewidth=2)

        plt.title(f"Feature {c} histogram")



def print_scatter(features, class_masks, first, second):
    plt.figure(f"Scatter plot for features {first}, {second}")
    plt.scatter(features[first, class_masks[0]], features[second, class_masks[0]], alpha=0.2, label="False class")
    plt.scatter(features[first, class_masks[1]], features[second, class_masks[1]], alpha=0.2, label="True class")
    plt.xlabel(f"Feature {first}")
    plt.ylabel(f"Feature {second}")
    plt.legend()
    plt.title(f"Feature {first}, {second} scatter plot")


if __name__ == "__main__":

    if len(argv) < 2:
        exit("Missing arguments: name of training data file")
    dataset_raw = np.array(load_train_data(argv[1]))
    features = np.array([entry.features for entry in dataset_raw]).T
    labels = np.array([entry.label for entry in dataset_raw]).reshape(1, len(dataset_raw))
    label_masks = [np.array([entry.label == t for entry in dataset_raw]) for t in [0.0, 1.0]]

    # Features 0, 1
    print_hist(features, label_masks, 0, 2)
    print_scatter(features, label_masks, 0, 1)

    # Features 2, 3
    print_hist(features, label_masks, 2, 4)
    print_scatter(features, label_masks, 2, 3)

    # Features 4, 5
    print_hist(features, label_masks, 4, 6)
    print_scatter(features, label_masks, 4, 5)

    # Means
    means_f = features[:, label_masks[0]].mean(axis=1)
    means_t = features[:, label_masks[1]].mean(axis=1)
    print("--Mean values--")
    for i in range(0, 4):
        print(f"Feature {i}:\n\tFalse class: {means_f[i]:.3f}\n\tTrue class: {means_t[i]:.3f}")
    print()

    # Variances
    variance_f = features[:, label_masks[0]].var(axis=1)
    variance_t = features[:, label_masks[1]].var(axis=1)
    print("--Variance values--")
    for i in range(0, 4):
        print(f"Feature {i}:\n\tFalse class: {variance_f[i]:.3f}\n\tTrue class: {variance_t[i]:.3f}")

    #print(features, features.shape, labels, labels.shape, sep="\n\n")

    plt.show()
