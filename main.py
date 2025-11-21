import numpy as np
from src.dataset.dataset import Dataset
from src.evaluator.evaluator import Evaluator
from src.tasks.logreg import LR_task, write_LR_results
from src.tasks.mvg import MVG_task
from src.tasks.stats import stats_task
from src.config.config import APPLICATION_PRIOR, EFF_PRIOR_LOG_ODDS_PARAMS, FILE_PATH_LR, FILE_PATH_SVM, LR, LR_EVALUATION_RESULT, RANDOM_SEED, REDUCE_FACTOR, REDUCED, SAVE, SVM, SVM_EVALUATION_RESULT, SVM_EVALUATION_RESULTS, TRAINSET_SIZE, EXECUTE
from src.tasks.dimred import dimred_task
from src.tasks.svm import svm_task, write_SVM_results
from src.utils.fitting import gaussian_estimation
from src.utils.plot import plot_estimated_features
from src.utils.utils import save

def main():

    print("Hello, World!")
    
    trainset_orig = Dataset.create("data/trainData.csv")
    testset = Dataset.create("data/evalData.csv")
    
    print(f"Trainset size: {trainset_orig.size}, dimension: {trainset_orig.dim}")
    print(f"Testset size: {testset.size}, dimension: {testset.dim}")
    
    if EXECUTE["stats"]:
        stats_task(trainset_orig, testset)
    
    if REDUCED:
        trainset_orig = trainset_orig.reduce(ratio=REDUCE_FACTOR, seed=RANDOM_SEED)
        testset = testset.reduce(ratio=REDUCE_FACTOR, seed=RANDOM_SEED)
    
    trainset, validset = trainset_orig.split(TRAINSET_SIZE)
    
    if EXECUTE["dimred"]:
        dimred_task(trainset_orig, trainset, validset)
        
    if EXECUTE["fitting"]:
        x_domain, y_estimations, features_per_class = gaussian_estimation(trainset_orig)

        if SAVE:
            plot_estimated_features(x_domain, y_estimations, features_per_class)
    
    eff_prior_log_odds = np.linspace(*EFF_PRIOR_LOG_ODDS_PARAMS)        
    
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
        pass  # Placeholder for GMM task
    if EXECUTE["comparison"]:
        pass  # Placeholder for comparison task
    
    print("Goodbye, World!")
    
if __name__ == "__main__":
    main()