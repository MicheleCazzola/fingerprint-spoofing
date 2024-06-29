# LABELS
LABEL_NAMES = {
    False: "Fake",
    True: "Genuine"
}
GAUSSIAN_MODELS = ["MVG", "Tied MVG", "Naive Bayes MVG"]
LR = "LR"
LR_STANDARD = "LR_STD"
PRIOR_WEIGHTED_LR = "PWLR"
SVM = "SVM"
GMM = "GMM"

# PATHS
PLOT_PATH_ESTIMATIONS = "output/plots/feature_estimation/"
PLOT_PATH_PCA = "output/plots/PCA_features/"
PLOT_PATH_LDA = "output/plots/LDA/"
PLOT_PATH = "output/plots/original_features/"
PLOT_PATH_GENERATIVE_GAUSSIAN = "output/plots/generative_models/gaussian/"
PLOT_PATH_LOGISTIC_REGRESSION = "output/plots/discriminative_models/logistic_regression/"
PLOT_PATH_SVM = "output/plots/discriminative_models/svm/"
PLOT_PATH_GMM = "output/plots/generative_models/gmm/"
PLOT_PATH_CMP = "output/plots/comparisons/"
PLOT_PATH_CAL_FUS = "output/plots/calibration_fusion/"

FILE_PATH = "output/files/original_features/"
FILE_PATH_LDA = "output/files/LDA/"
FILE_PATH_GENERATIVE_GAUSSIAN = "output/files/generative_models/gaussian/"
FILE_PATH_LOGISTIC_REGRESSION = "output/files/discriminative_models/logistic_regression/"
FILE_PATH_SVM = "output/files/discriminative_models/svm/"
FILE_PATH_GMM = "output/files/generative_models/gmm/"
FILE_PATH_CMP = "output/files/comparisons/"


# FILE NAMES
FEATURES_STATISTICS = "feature_statistics"
LDA_ERROR_RATES = "LDA_error_rates"
GAUSSIAN_ERROR_RATES = "gaussian_error_rates.txt"
GMM_EVALUATION = "gmm_evaluation.txt"
BEST_RESULTS_CAL = "best_results_cal.txt"

# NUMERIC
APPLICATIONS = [
    (0.5, 1.0, 1.0),
    (0.9, 1.0, 1.0),
    (0.1, 1.0, 1.0),
    (0.5, 1.0, 9.0),
    (0.5, 9.0, 1.0)
]

# MODELS
GAUSSIAN = 0
LOG_REG = 1
