import numpy as np
from sklearn.model_selection import train_test_split


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


def vcol(array: np.ndarray):
    """
    Converts a 1D-ndarray into a column 2D-ndarray

    :param array: 1D-ndarray
    :return: column 2D-ndarray
    """
    return array.reshape(array.size, 1)


def vrow(array: np.ndarray):
    """
    Converts a 1D-ndarray into a row 2D-ndarray

    :param array: 1D-ndarray
    :return: row 2D-ndarray
    """
    return array.reshape(1, array.size)

def project(D: np.ndarray, M: np.ndarray):
    """
    Project data over basis spanned by columns of matrix M

    :param D: dataset
    :param M: transformation matrix
    :return: projected dataset
    """
    return M.T @ D

def effective_prior(application):
    """
    Computes the effective prior of an application

    :param application: application triplet (prior, false negative cost, false positive cost)
    :return: effective prior
    """
    return application[0] * application[1] / (application[0] * application[1] + (1 - application[0]) * application[2])

def save(path_root: str, file_name: str, function, *args):
    """
    Saves data to a file using a specified function

    :param path_root: root path to save the file
    :param file_name: name of the file
    :param function: function to use for saving
    :param args: arguments to pass to the saving function
    :return: None
    """
    
    content = function(*args)
    with open(f"{path_root}{file_name}", mode="w", encoding="utf-8") as fout:
        fout.write(content)