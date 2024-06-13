import numpy as np

from evaluation.evaluation import relative_mis_calibration
from src.io.constants import LABEL_NAMES


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


def build_table(errors):
    result = ""
    result += f"{'Method':^16s}{'Error rate':^14s}{'Error rate (%)':^14s}\n"
    for (alg, err) in errors.items():
        result += f"{alg:^16s}{err:^14.4f}{100 * err:^14.2f}\n"
    result += "\n"

    return result


def save_gaussian_classification_results(error_rates, corr_matrices, error_rates_1_4, error_rates_1_2, error_rates_3_4,
                                         error_rates_pca, path_root, file_name):
    result = ""
    result += "--All features--\n"
    result += build_table(error_rates)

    result += "--Correlation matrices--\n"
    for (corr_matrix, label) in zip(corr_matrices, LABEL_NAMES.keys()):
        result += f"{label} class\n"
        for line in corr_matrix:
            for element in line:
                result += f"{element: .2f}\t"
            result += "\n"
        result += "\n"

    result += "--Using subsets of features--\n"
    result += "Features 1-4\n"
    result += build_table(error_rates_1_4)

    result += "Features 1-2\n"
    result += build_table(error_rates_1_2)

    result += "Features 3-4\n"
    result += build_table(error_rates_3_4)

    result += "--PCA preprocessing--\n"
    result += "Error rates\n"
    result += f"{'PCA dimensions':<16s}{'MVG':^10s}{'Tied MVG':^12s}{'Naive Bayes MVG':^17s}\n"
    for (m, err) in error_rates_pca.items():
        result += (f"{m:^16d}"
                   f"{err['MVG']:^10.4f}"
                   f"{err['Tied MVG']:^12.4f}"
                   f"{err['Naive Bayes MVG']:^17.4f}\n")
    result += "\n"

    result += "Error rates (%)\n"
    result += f"{'PCA dimensions':<16s}{'MVG':^10s}{'Tied MVG':^12s}{'Naive Bayes MVG':^17s}\n"
    for (m, err) in error_rates_pca.items():
        result += (f"{m:^16d}"
                   f"{100 * err['MVG']:^10.2f}"
                   f"{100 * err['Tied MVG']:^12.2f}"
                   f"{100 * err['Naive Bayes MVG']:^17.2f}\n")

    with open(f"{path_root}{file_name}", mode="w", encoding="utf-8") as fout:
        fout.write(result)


def save_application_priors(applications, eff_priors, path_root, file_name):
    result = ""
    result += "--Applications and associated effective priors--\n"
    result += (f"{'Prior':<6s}"
               f"{'False negative cost (C_fn)':^28s}"
               f"{'False positive cost (C_fp)':^28s}"
               f"{'Effective prior':^17s}\n")

    for ([prior, c_fn, c_fp], eff_prior) in zip(applications, eff_priors):
        result += (f"{prior:^6.1f}"
                   f"{c_fn:^28.1f}"
                   f"{c_fp:^28.1f}"
                   f"{eff_prior:^17.1f}\n")

    with open(f"{path_root}{file_name}", mode="w", encoding="utf-8") as fout:
        fout.write(result)


def print_DCFs(result, key, m):
    return (f"{str(m) if m is not None else 'Not applied':^16s}"
            f"{result['MVG'][key]:^10.3f}"
            f"{result['Tied MVG'][key]:^12.3f}"
            f"{result['Naive Bayes MVG'][key]:^17.3f}\n")


def print_mis_calibrations(result, m="Not applied"):
    return (f"{str(m):^16s}"
            f"{relative_mis_calibration(result['MVG']):^10.2f}"
            f"{relative_mis_calibration(result['Tied MVG']):^12.2f}"
            f"{relative_mis_calibration(result['Naive Bayes MVG']):^17.2f}\n")


def write_tables(results):
    print_string = "Minimum DCF\n"
    print_string += f"{'PCA dimensions':<16s}{'MVG':^10s}{'Tied MVG':^12s}{'Naive Bayes MVG':^17s}\n"
    for (m, result) in sorted(results.items(), key=lambda x: x[0] if x[0] != "Not applied" else np.inf):
        print_string += print_DCFs(result, "min_dcf", m)
    print_string += "\n"

    print_string += "Actual DCF\n"
    print_string += f"{'PCA dimensions':<16s}{'MVG':^10s}{'Tied MVG':^12s}{'Naive Bayes MVG':^17s}\n"
    for (m, result) in sorted(results.items(), key=lambda x: x[0] if x[0] != "Not applied" else np.inf):
        print_string += print_DCFs(result, "dcf", m)
    print_string += "\n"

    print_string += "Relative calibration loss (%)\n"
    print_string += f"{'PCA dimensions':<16s}{'MVG':^10s}{'Tied MVG':^12s}{'Naive Bayes MVG':^17s}\n"
    for (m, result) in sorted(results.items(), key=lambda x: x[0] if x[0] != "Not applied" else np.inf):
        print_string += print_mis_calibrations(result, m)
    print_string += "\n"

    return print_string


def write_gaussian_results(eval_results):
    print_string = ""
    for (eff_prior, results) in sorted(eval_results.items(), key=lambda x: x[0]):
        print_string += f"--Effective prior: {eff_prior}--\n"
        print_string += write_tables(results)

    return print_string


def save_gaussian_evaluation_results(results, path_root, file_name):
    print_string = write_gaussian_results(results)

    with open(f"{path_root}{file_name}", mode="w", encoding="utf-8") as fout:
        fout.write(print_string)


def write_LR_results(eval_results):
    print_string = "--Minimum DCFs--\n"
    for [min_dcf, reg_coeff, task_name] in eval_results:
        print_string += f"{task_name:<70s}: {min_dcf:.3f} (λ = {reg_coeff:.2f})\n"

    return print_string


def save_LR_evaluation_results(results, path_root, file_name):
    print_string = write_LR_results(results)

    with open(f"{path_root}{file_name}", mode="w", encoding="utf-8") as fout:
        fout.write(print_string)


def write_SVM_results(results):
    task_names = results["tasks"]
    best_results = results["results"]

    print_string = "--Minimum DCFs--\n"
    for (task_name, best_result) in zip(task_names[:-1], best_results[:-1]):
        print_string += f"{task_name}: {best_result[1]:.3f} (C = {best_result[0]:.3f}, K = {best_result[2]:.1f})\n"

    print_string += f"{task_names[-1]}:\n"
    print_string += f"{'Scale':^7s}{'Minimum DCF':^13s}{'C':^7s}{'K':^5s}\n"
    for (rbf, best_rbf) in best_results[-1].items():
        print_string += f"{rbf:^7.3f}{best_rbf[1]:^13.3f}{best_rbf[0]:^7.3f}{best_rbf[2]:^5.1f}\n"

    return print_string


def save_SVM_evaluation_results(results, path_root, file_name):
    print_string = write_SVM_results(results)

    with open(f"{path_root}{file_name}", mode="w", encoding="utf-8") as fout:
        fout.write(print_string)


def write_GMM_results(results):
    print_string = ""
    for (variant, results) in results.items():
        print_string += f"--{'Full covariance' if variant == 'full' else 'Diagonal covariance'} matrices--\n"
        print_string += f"{'GMM components':<15s}{'Minimum DCF':^13s}{'Actual DCF':^12s}\n"
        for (num_components, result) in results.items():
            min_dcf, dcf = map(result.get, ["min_dcf", "dcf"])
            print_string += f"{num_components:^15d}{min_dcf:^13.4f}{dcf:^12.4f}\n"
        print_string += "\n"

    return print_string


def save_GMM_results(results, path_root, file_name):
    print_string = write_GMM_results(results)

    with open(f"{path_root}{file_name}", mode="w", encoding="utf-8") as fout:
        fout.write(print_string)
