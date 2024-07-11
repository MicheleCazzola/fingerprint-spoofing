# LABELS
LABEL_NAMES = {
    False: "Fake",
    True: "Genuine"
}
GAUSSIAN_MODELS = ["Standard MVG", "Tied MVG", "Naive Bayes MVG"]
GAUSSIAN = "MVG"
LR = "LR"
LR_STANDARD = "Standard LR"
LR_RED_DATA = "Standard LR (reduced dataset)"
PRIOR_WEIGHTED_LR = "PWLR"
QUADRATIC_LR = "Quadratic LR"
PRIOR_WEIGHTED_LR_PREPROCESS = "PWLR - Preprocess"
SVM = "SVM"
SVM_LINEAR = "Linear"
SVM_LINEAR_PREPROCESS = "Linear - Data centering"
SVM_POLYNOMIAL = "Polynomial"
SVM_RBF = "RBF"
GMM = "GMM"
FUSION = "Fusion"

# Plot paths
PLOT_PATH_FEATURES = "output/plots/original_features/"
PLOT_PATH_EVAL_FEATURES = "output/plots/eval_features/"
PLOT_SUBPATH_HISTOGRAM_FEATURES = "histograms/"
PLOT_SUBPATH_SCATTERPLOTS_FEATURES = "scatterplots/"
PLOT_PATH_ESTIMATIONS = "output/plots/feature_estimation/"
PLOT_PATH_PCA = "output/plots/PCA_features/"
PLOT_PATH_LDA = "output/plots/LDA/"
PLOT_SUBPATH_HISTOGRAM_LDA = "histograms/"
PLOT_SUBPATH_LINES_LDA = "lines/"
PLOT_PATH_GENERATIVE_GAUSSIAN = "output/plots/generative_models/gaussian/"
PLOT_PATH_LOGISTIC_REGRESSION = "output/plots/discriminative_models/logistic_regression/"
PLOT_PATH_SVM = "output/plots/discriminative_models/svm/"
PLOT_PATH_GMM = "output/plots/generative_models/gmm/"
PLOT_PATH_CMP = "output/plots/comparisons/"
PLOT_PATH_CAL_FUS = "output/plots/calibration_fusion/"
PLOT_PATH_EVAL_CAL = "output/plots/evaluation/calibrated/"
PLOT_PATH_EVAL_CMP = "output/plots/evaluation/comparisons/"
PLOT_PATH_EVAL_LR = "output/plots/evaluation/logistic_regression/"

# File paths
FILE_PATH_FEATURES = "output/files/original_features/"
FILE_PATH_LDA = "output/files/LDA/"
FILE_PATH_GENERATIVE_GAUSSIAN = "output/files/generative_models/gaussian/"
FILE_PATH_LOGISTIC_REGRESSION = "output/files/discriminative_models/logistic_regression/"
FILE_PATH_SVM = "output/files/discriminative_models/svm/"
FILE_PATH_GMM = "output/files/generative_models/gmm/"
FILE_PATH_CMP = "output/files/comparisons/"
FILE_PATH_EVAL = "output/files/evaluation/"


# File names
FEATURE_STATISTICS = "feature_statistics.txt"
LDA_ERROR_RATES = "error_rates.txt"
GAUSSIAN_APPLICATION_PRIORS = "gaussian_application_priors.txt"
GAUSSIAN_ERROR_RATES = "gaussian_error_rates.txt"
GAUSSIAN_EVALUATION_RESULTS = "gaussian_evaluation_results.txt"
LR_EVALUATION_RESULT = "LR_evaluation_results.txt"
SVM_EVALUATION_RESULT = "SVM_evaluation_results.txt"
GMM_EVALUATION_RESULT = "gmm_evaluation_results.txt"
BEST_RESULTS_RAW = "best_results_raw.txt"
BEST_RESULTS_CAL = "best_results_cal.txt"
APP_EVAL_RESULTS = "app_eval_results.txt"
APP_EVAL_LR_RESULTS = "LR_app_eval_results.txt"

# Plot names
FEATURE_PREFIX_HISTOGRAM = "histogram"
FEATURE_PREFIX_SCATTERPLOT = "scatter"
PCA_PREFIX_HISTOGRAM = "PCA_histogram"
PCA_PREFIX_SCATTERPLOT = "PCA_scatter"
LDA_HISTOGRAM = "LDA_histogram"
LDA_ERROR_RATE_TH = "error_rate_threshold"
LDA_ERROR_RATE_TH_COMPACT = "error_rate_threshold_compact"
ESTIMATED_FEATURE = "estimated_feature"
GAUSSIAN_BAYES_ERROR = "bayes_error"
LR_EVALUATION_RESULTS = [
    "LR_standard_non_weighted",
    "LR_standard_non_weighted_filtered_dataset",
    "Prior_weighted_LR",
    "Prior_weighted_LR_expanded_feature_space",
    "Prior_weighted_LR_data_centering"
]
SVM_EVALUATION_RESULTS = [
    "linear_no_preprocessing",
    "linear_data_centering",
    "quadratic_no_preprocessing",
    "RBF_kernel_bias1"
]
CMP_BAYES_ERROR_RAW = "bayes_error_comparison_raw"
CMP_BAYES_ERROR_CAL = "bayes_error_comparison_calibrated"
CMP_BAYES_ERROR_FUSION = "bayes_error_fusion"
EVAL_BAYES_ERR_LR = "bayes_error_LR"
EVAL_BAYES_ERR_SVM = "bayes_error_SVM"
EVAL_BAYES_ERR_GMM = "bayes_error_GMM"
EVAL_BAYES_ERR_FUSION = "bayes_error_fusion"
EVAL_BAYES_ERR_ALL = "eval_bayes_error_cmp"
EVAL_BAYES_ERR_ALL_ACT_DCF = "eval_bayes_error_cmp_act_dcf"

# NUMERIC
APPLICATIONS = [
    (0.5, 1.0, 1.0),
    (0.9, 1.0, 1.0),
    (0.1, 1.0, 1.0),
    (0.5, 1.0, 9.0),
    (0.5, 9.0, 1.0)
]

# EXECUTION MODES
SAVE = 1
LOG = 1
REDUCED = 0
REDUCE_FACTOR = 100
