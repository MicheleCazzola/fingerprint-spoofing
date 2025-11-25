import os
import numpy as np


def compute_statistics(features: np.ndarray, labels: np.ndarray, **functions) -> dict:
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


def vcol(array: np.ndarray) -> np.ndarray:
    """
    Converts a 1D-ndarray into a column 2D-ndarray

    :param array: 1D-ndarray
    :return: column 2D-ndarray
    """
    return array.reshape(array.size, 1)


def vrow(array: np.ndarray) -> np.ndarray:
    """
    Converts a 1D-ndarray into a row 2D-ndarray

    :param array: 1D-ndarray
    :return: row 2D-ndarray
    """
    return array.reshape(1, array.size)

def project(D: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    Project data over basis spanned by columns of matrix M

    :param D: dataset
    :param M: transformation matrix
    :return: projected dataset
    """
    return M.T @ D

def effective_prior(application: tuple) -> float:
    """
    Computes the effective prior of an application

    :param application: application triplet (prior, false negative cost, false positive cost)
    :return: effective prior
    """
    return application[0] * application[1] / (application[0] * application[1] + (1 - application[0]) * application[2])

def expand(DTR):
    expanded = []
    for i in range(DTR.shape[1]):
        arr = np.concatenate([(DTR[:, i:i + 1] @ DTR[:, i:i + 1].T).ravel(), DTR[:, i]])
        expanded.append(arr)
    return np.array(expanded).T

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
        
def relative_mis_calibration(dcfs: dict) -> float:
    """
    Computes the relative mis-calibration given the DCFs
    
    :param dcfs: dictionary containing "dcf" and "min_dcf"
    :return: relative mis-calibration as a percentage
    """
    return 100 * (dcfs["dcf"] - dcfs["min_dcf"]) / dcfs["min_dcf"]

def delete_all(filepaths: list):
    """
    Deletes all files in the provided list of file paths.
    
    :param filepaths: list of file paths to delete"""
    
    for filepath in filepaths:
        if os.path.exists(filepath):
            os.remove(filepath)
        else:
            print(f"Warning: file {filepath} does not exist.")

def print_model_result(method, result):
    print(f"Method: {method}")
    print(f"Minimum DCF: {result['min_dcf']:.5f}")
    print(f"Actual DCF: {result['act_dcf']:.5f}")
    print(
        f"LLR: shape = {result['llr'].shape} "
        f"mean = {result['llr'].mean():.5f}, "
        f"max = {result['llr'].max():.5f}, "
        f"min = {result['llr'].min():.5f}, "
        f"devstd = {result['llr'].std():.5f}"
    )
    print(f"Method parameters:")
    print(result['params'])
    print()
    
    
def print_scores_stats(scores, names):
    for (score, name) in zip(scores, names):
        print(f"Score type: {name}")
        print(
            f"LLR: shape = {score.shape} "
            f"mean = {score.mean():.5f}, "
            f"max = {score.max():.5f}, "
            f"min = {score.min():.5f}, "
            f"devstd = {score.std():.5f}"
        )