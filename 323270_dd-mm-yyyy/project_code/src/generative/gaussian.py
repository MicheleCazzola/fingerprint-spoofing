import numpy as np

from constants import FILE_PATH_GENERATIVE_GAUSSIAN, GAUSSIAN_ERROR_RATES, GAUSSIAN_MODELS, APPLICATIONS
from evaluation.evaluation import Evaluator
from fitting.fitting import logpdf_GAU_ND, compute_estimators
from pca import reduce
from src.io.fio import save_gaussian_classification_results
from utilities.utilities import vcol, split_db_2to1, vrow, project, effective_prior


def estimate(D, L):
    mu_c, cov_c = [], []
    for c in np.unique(L):
        Dc = D[:, L == c]
        mu, cov = compute_estimators(Dc, np.mean(Dc, axis=1))
        mu_c.append(vcol(mu))
        cov_c.append(cov)

    return np.array(mu_c), np.array(cov_c)


def set_prior():
    return vcol(np.array([1 / 2, 1 / 2]))


def set_threshold(prior_false, prior_true):
    return -np.log(prior_true / prior_false)


def compute_cov_naive_approx(cov_c):
    return cov_c * np.eye(cov_c[0].shape[1])


def class_conditional(DTE, mu_c, cov_c):
    return logpdf_GAU_ND(DTE, mu_c[0], cov_c[0]), logpdf_GAU_ND(DTE, mu_c[1], cov_c[1])


def within_class_covariance(cov_c, DTR, LTR):
    return (DTR[:, LTR == 0].shape[1] * cov_c[0] + DTR[:, LTR == 1].shape[1] * cov_c[1]) / DTR.shape[1]


def compute_llr(DTE, mu_c, cov_c):
    cc_false, cc_true = class_conditional(DTE, mu_c, cov_c)
    return cc_true - cc_false


def predict(LTE, llr, threshold):
    LPR = np.zeros(LTE.shape, dtype=np.int32)
    LPR[llr >= threshold] = 1
    LPR[llr < threshold] = 0

    return LPR


def compute_error_rate(LPR, LTE):
    return np.sum(LPR != LTE) / LTE.shape[0]


def compute_predictions(DTE, LTE, mu_c, cov_c, thresholds):
    llr = compute_llr(DTE, mu_c, cov_c)
    LPRs = [predict(LTE, llr, t) for t in thresholds]
    return llr, LPRs


def classify(DTE, LTE, mu_c, cov_c, eff_priors, m_pca, evaluator=None):
    thresholds = [set_threshold(1 - eff_prior, eff_prior) for eff_prior in eff_priors]

    llr, LPRs = compute_predictions(DTE, LTE, mu_c, cov_c, thresholds)

    if evaluator:
        for (eff_prior, LPR) in zip(eff_priors, LPRs):
            evaluator.evaluate(llr, LPR, LTE, eff_prior, m_pca)

    LPR_uniform = LPRs[1] if evaluator else LPRs[0]

    return compute_error_rate(LPR_uniform, LTE)


def MVG(DTE, LTE, mu_c, cov_c, eff_priors, m_pca, evaluator):
    return classify(DTE, LTE, mu_c, cov_c, eff_priors, m_pca, evaluator)


def TiedMVG(DTE, LTE, mu_c, cov_c, DTR, LTR, eff_priors, m_pca, evaluator):
    cov = within_class_covariance(cov_c, DTR, LTR)
    return classify(DTE, LTE, mu_c, np.array([cov] * 2), eff_priors, m_pca, evaluator)


def Naive_BayesMVG(DTE, LTE, mu_c, cov_c, eff_priors, m_pca, evaluator):
    return classify(DTE, LTE, mu_c, compute_cov_naive_approx(cov_c), eff_priors, m_pca, evaluator)


def compute_correlations(DTR, LTR):
    _, cov_c = estimate(DTR, LTR)

    return [C / (vcol(C.diagonal() ** 0.5) * vrow(C.diagonal() ** 0.5)) for C in cov_c]


def classification_analysis(DTR, LTR, DTE, LTE, eff_priors, m_pca=None, evaluators=None):
    mu_c, cov_c = estimate(DTR, LTR)

    return {
        GAUSSIAN_MODELS[0]: MVG(DTE, LTE, mu_c, cov_c, eff_priors, m_pca,
                                evaluators[GAUSSIAN_MODELS[0]] if evaluators is not None else None),
        GAUSSIAN_MODELS[1]: TiedMVG(DTE, LTE, mu_c, cov_c, DTR, LTR, eff_priors, m_pca,
                                    evaluators[GAUSSIAN_MODELS[1]] if evaluators is not None else None),
        GAUSSIAN_MODELS[2]: Naive_BayesMVG(DTE, LTE, mu_c, cov_c, eff_priors, m_pca,
                                           evaluators[GAUSSIAN_MODELS[2]] if evaluators is not None else None)
    }


def classification_PCA_preprocessing(DTR, LTR, DTE, LTE, eff_priors, evaluators):
    error_rates_pca = {}
    for m in range(2, DTR.shape[0]):
        P = reduce(DTR, m)
        DTR_pca, DTE_pca = project(DTR, P), project(DTE, P)
        error_rates_pca[m] = classification_analysis(DTR_pca, LTR, DTE_pca, LTE, eff_priors, m, evaluators)

    return error_rates_pca


def gaussian_classification(D, L):
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

    application_priors = [effective_prior(application) for application in APPLICATIONS]
    evaluators = {model_name: Evaluator(model_name) for model_name in GAUSSIAN_MODELS}

    # Classification with features 1-6
    # 1: 7 %
    # 2: 9.3 % (same as LDA)
    # 3: 7.2 %
    error_rates = classification_analysis(DTR, LTR, DTE, LTE, sorted(set(application_priors)), evaluators=evaluators)

    # 4: low correlation, but not null -> Indeed Naive is good, but little worse than MVG
    corr_matrices = compute_correlations(DTR, LTR)

    # 5: features 5 and 6 does not fit well with Gaussian assumption

    # 6: repeat analysis, but only with features 1-4
    # MVG: 7.95 %
    # Tied: 9.50 %
    # Naive: 7.65 %
    error_rates_1_4 = classification_analysis(DTR[0:4, :], LTR, DTE[0:4, :], LTE, application_priors[1:2])

    # 7: repeat classification, but only with features (1-2) and then (3-4)

    # Features 1-2
    # MVG: 36.50 %
    # Tied: 49.45 %
    # Naive: 36.30 %
    error_rates_1_2 = classification_analysis(DTR[0:2, :], LTR, DTE[0:2, :], LTE, application_priors[1:2])

    # Features 3-4
    # 9.45 %
    # 9.40 %
    # 9.45 %
    error_rates_3_4 = classification_analysis(DTR[2:4, :], LTR, DTE[2:4, :], LTE, application_priors[1:2])

    # 8: repeat classification, by applying PCA preprocessing

    error_rates_pca = classification_PCA_preprocessing(DTR, LTR, DTE, LTE, application_priors, evaluators)

    save_gaussian_classification_results(error_rates, corr_matrices, error_rates_1_4, error_rates_1_2, error_rates_3_4,
                                         error_rates_pca, FILE_PATH_GENERATIVE_GAUSSIAN, GAUSSIAN_ERROR_RATES)

    # Evaluate model
    evaluation_results = {model_name: evaluator.get_results() for (model_name, evaluator) in evaluators.items()}

    # Get the best configuration for each evaluator
    eff_prior_conf = 0.1
    best_configurations = {model_name: evaluator.best_configuration(0.1) for (model_name, evaluator) in evaluators.items()}

    # Compute bayes errors
    bayes_errors = {model_name: (config["pca"], Evaluator.bayes_error(config["llr"], config["LTE"]))
                    for (model_name, config) in best_configurations.items()}

    # Save application prior log-odd considered in bayes error plot
    best_prior_log_odd = np.log(eff_prior_conf / (1 - eff_prior_conf))

    return application_priors, evaluation_results, bayes_errors, best_prior_log_odd
