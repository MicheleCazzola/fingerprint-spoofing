import joblib
import numpy as np
from src.calfus.calibration import calibration_task
from src.calfus.fusion import fusion_task, write_best_results
from src.dataset.dataset import Dataset
from src.evaluator.application import app_evaluation, write_app_results
from src.evaluator.evaluator import Evaluator
from src.tasks.gmm import gmm_task, write_GMM_results
from src.tasks.logreg import LR_task, write_LR_results
from src.tasks.mvg import MVG_task
from src.tasks.stats import stats_task
from src.config.config import APP_EVAL_LR_RESULTS, APP_EVAL_RESULTS, APPLICATION_PRIOR, BEST_RESULTS_CAL, BEST_RESULTS_RAW, CMP_BAYES_ERROR_CAL, CMP_BAYES_ERROR_FUSION, CMP_BAYES_ERROR_RAW, EFF_PRIOR_LOG_ODDS_PARAMS, EVAL_BAYES_ERR_ALL, EVAL_BAYES_ERR_ALL_ACT_DCF, FILE_PATH_CMP, FILE_PATH_EVAL, FILE_PATH_GMM, FILE_PATH_LR, FILE_PATH_SVM, FUSION, GMM, GMM_EVALUATION_RESULT, LOG, LR, LR_EVALUATION_RESULT, MODEL_PATH_RESULTS, PLOT_PATH_CAL_FUS, PLOT_PATH_CMP, RANDOM_SEED, REDUCE_FACTOR, REDUCED, SAVE, SVM, SVM_EVALUATION_RESULT, SVM_EVALUATION_RESULTS, TRAINSET_SIZE, EXECUTE
from src.tasks.dimred import dimred_task
from src.tasks.svm import svm_task, write_SVM_results
from src.utils.fitting import gaussian_estimation
from src.utils.plot import plot_bayes_errors, plot_estimated_features
from src.utils.utils import print_model_result, save

def main():

    print("Hello, World!")
    
    trainset_orig = Dataset.create("data/trainData.csv")
    testset = Dataset.create("data/evalData.csv")
    
    
    
    if EXECUTE["stats"]:
        stats_task(trainset_orig, testset)
    
    if REDUCED:
        trainset_orig = trainset_orig.reduce(ratio=REDUCE_FACTOR, seed=RANDOM_SEED)
        testset = testset.reduce(ratio=REDUCE_FACTOR, seed=RANDOM_SEED)
    
    trainset, validset = trainset_orig.split(TRAINSET_SIZE)
    
    print(f"Training set size: {trainset.size}, dimension: {trainset.dim}")
    print(f"Validation set size: {validset.size}, dimension: {validset.dim}")
    print(f"Test set size: {testset.size}, dimension: {testset.dim}")
    
    if EXECUTE["dimred"]:
        dimred_task(trainset_orig, trainset, validset)
        
    if EXECUTE["fitting"]:
        x_domain, y_estimations, features_per_class = gaussian_estimation(trainset_orig)

        if SAVE:
            plot_estimated_features(x_domain, y_estimations, features_per_class)
    
    eff_prior_log_odds = np.linspace(*EFF_PRIOR_LOG_ODDS_PARAMS)
    log_odd_application = np.log(APPLICATION_PRIOR / (1 - APPLICATION_PRIOR))        
    
    if EXECUTE["MVG"]:
        MVG_task(trainset, validset, APPLICATION_PRIOR, eff_prior_log_odds)
    
    model_results = {}
    if EXECUTE["LR"]:
        lr_results = LR_task(trainset, validset, APPLICATION_PRIOR, target="validation")
        best_lr = Evaluator.best_configuration(lr_results, LR)
        model_results[LR] = best_lr
        
        if SAVE:
            save(FILE_PATH_LR, LR_EVALUATION_RESULT, write_LR_results, lr_results)
        
    if EXECUTE["SVM"]:
        svm_results = svm_task(trainset, validset, APPLICATION_PRIOR)     # to run with complete dataset
        best_svm = Evaluator.best_configuration(svm_results["results"], SVM)
        model_results[SVM] = best_svm
        
        if SAVE:
            save(FILE_PATH_SVM, SVM_EVALUATION_RESULT, write_SVM_results, svm_results)
        
    if EXECUTE["GMM"]:
        gmm_results = gmm_task(trainset, validset, APPLICATION_PRIOR)
        best_gmm = Evaluator.best_configuration(gmm_results, GMM)
        model_results[GMM] = best_gmm
        
        if SAVE:
            save(FILE_PATH_GMM, GMM_EVALUATION_RESULT, write_GMM_results, gmm_results)
    
    if model_results:
        joblib.dump(model_results, MODEL_PATH_RESULTS)
        print(f"Saved model results to '{MODEL_PATH_RESULTS}'")
        
        if SAVE:
            save(FILE_PATH_CMP, BEST_RESULTS_RAW, write_best_results, model_results)
            
        if LOG:
            print("Model classification results (no calibration)")
            for (method, result) in model_results.items():
                print_model_result(method, result)
            print()
            
    if EXECUTE["comparison"]:
        
        # Load previous results if not executed in this run (all or nothing)
        if not model_results:
            model_results = joblib.load("./models/model_results.pkl")
        
        bayes_errors = list(map(
            Evaluator.bayes_error,
            [result["llr"] for result in model_results.values()],
            [validset.get_labels() for _ in range(len(model_results))],
            [eff_prior_log_odds for _ in range(len(model_results))]
        ))

        min_dcfs = [error["min_dcf"] for error in bayes_errors]
        dcfs = [error["dcf"] for error in bayes_errors]

        if SAVE:
            plot_bayes_errors(
                eff_prior_log_odds, min_dcfs, dcfs, log_odd_application, "Bayes error plots comparison", "Raw scores",
                "Prior log-odds", "DCF value", PLOT_PATH_CMP, CMP_BAYES_ERROR_RAW, "png", model_results.keys()
            )
        
        bayes_errors_raw = dict(zip(model_results.keys(), zip(min_dcfs, dcfs)))

        calibration_result, labels_val_unfolded = calibration_task(
            model_results, validset.get_labels(), APPLICATION_PRIOR, bayes_errors_raw, eff_prior_log_odds, log_odd_application
        )

        bayes_errors = list(map(
            Evaluator.bayes_error,
            [result["llr"] for result in calibration_result.values()],
            [label_val for label_val in labels_val_unfolded.values()],
            [eff_prior_log_odds for i in range(len(calibration_result))]
        ))

        if SAVE:
            min_dcfs = [error["min_dcf"] for error in bayes_errors]
            dcfs = [error["dcf"] for error in bayes_errors]
            plot_bayes_errors(
                eff_prior_log_odds, min_dcfs, dcfs, log_odd_application, "Bayes error plots comparison", "Calibrated scores",
                "Prior log-odds", "DCF value", PLOT_PATH_CMP, CMP_BAYES_ERROR_CAL, "png", calibration_result.keys()
            )
        
        scores = {model_name: model_result["llr"] for (model_name, model_result) in model_results.items()}
        fusion_result, labels_val_unfolded = fusion_task(list(scores.values()), validset.get_labels(), APPLICATION_PRIOR)

        cal_fus_result = calibration_result | {FUSION: fusion_result}

        best_model = min(cal_fus_result, key=lambda k: cal_fus_result[k]["act_dcf"])

        if LOG:
            print("Results after calibration / fusion: ")
            for (method, result) in cal_fus_result.items():
                print_model_result(method, result)
            print()

        if SAVE:
            save(FILE_PATH_CMP, BEST_RESULTS_CAL, write_best_results, cal_fus_result)

            min_dcf_fus, act_dcf_fus = map(
                Evaluator.bayes_error(
                    fusion_result["llr"],
                    labels_val_unfolded,
                    eff_prior_log_odds).get,
                ["min_dcf", "dcf"]
            )

            plot_bayes_errors(
                eff_prior_log_odds, [min_dcf_fus], [act_dcf_fus], log_odd_application, "Bayes error plots", "Fused scores",
                "Prior log-odds", "DCF value", PLOT_PATH_CAL_FUS, CMP_BAYES_ERROR_FUSION, "png"
            )
        
        app_result = {
            model_name: app_evaluation(
                model_name, model_result["params"], model_result["id"], trainset, model_result["llr"], validset.get_labels(),
                testset, APPLICATION_PRIOR, cal_fus_result[model_name]["params"]["training_prior"], eff_prior_log_odds,
                log_odd_application, model_name == best_model
            ) for (model_name, model_result) in model_results.items()
        }
            
        fusion_params = {
            model_name: model_result["params"]
            for (model_name, model_result) in model_results.items()
        }
        
        fusion_id = {
            model_name: model_result["id"]
            for (model_name, model_result) in model_results.items()
        }

        fus_result = app_evaluation(
            FUSION, fusion_params, fusion_id, trainset, np.vstack([model_result["llr"] for model_result in model_results.values()]),
            validset.get_labels(), testset, APPLICATION_PRIOR, cal_fus_result[FUSION]["params"]["training_prior"], eff_prior_log_odds,
            log_odd_application, FUSION == best_model
        )

        app_result[FUSION] = fus_result

        app_result_cal = {model_name: app_res["cal"] for (model_name, app_res) in app_result.items()}
        app_bayes_err_min_dcf = [res["bayes_err_min_dcf"] for res in app_result_cal.values()]
        app_bayes_err_act_dcf = [res["bayes_err_act_dcf"] for res in app_result_cal.values()]

        if SAVE:
            save(FILE_PATH_EVAL, APP_EVAL_RESULTS, write_app_results, app_result)

            # ActDCF for each model (and their fusion)
            plot_bayes_errors(
                eff_prior_log_odds, None, app_bayes_err_act_dcf, log_odd_application, "Bayes error calibrated models",
                "Evaluation dataset - Actual DCF", "Prior log-odds", "DCF value", PLOT_PATH_CMP, EVAL_BAYES_ERR_ALL_ACT_DCF,
                "png", app_result_cal.keys()
            )

            # MinDCF and actDCF of all models (and fusion)
            plot_bayes_errors(
                eff_prior_log_odds, app_bayes_err_min_dcf, app_bayes_err_act_dcf, log_odd_application, "Bayes error calibrated models",
                "Evaluation dataset", "Prior log-odds", "DCF value", PLOT_PATH_CMP, EVAL_BAYES_ERR_ALL, "png", app_result_cal.keys()
            )

        lr_eval_results = LR_task(trainset, testset, APPLICATION_PRIOR, target="evaluation")

        if SAVE:
            save(FILE_PATH_EVAL, APP_EVAL_LR_RESULTS, write_LR_results, lr_eval_results)
    
    print("Goodbye, World!")
    
if __name__ == "__main__":
    main()