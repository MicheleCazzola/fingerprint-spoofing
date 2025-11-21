import numpy as np
from src.dataset.dataset import Dataset
from src.tasks.mvg import MVG_task
from src.tasks.stats import stats_task
from src.config.config import APPLICATION_PRIOR, EFF_PRIOR_LOG_ODDS_PARAMS, RANDOM_SEED, REDUCE_FACTOR, REDUCED, SAVE, TRAINSET_SIZE, EXECUTE
from src.tasks.dimred import dimred_task
from src.utils.fitting import gaussian_estimation
from src.utils.plot import plot_estimated_features

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
    
    if EXECUTE["LR"]:
        pass  # Placeholder for LR task
    if EXECUTE["SVM"]:
        pass  # Placeholder for SVM task
    if EXECUTE["GMM"]:
        pass  # Placeholder for GMM task
    if EXECUTE["comparison"]:
        pass  # Placeholder for comparison task
    
    print("Goodbye, World!")
    
if __name__ == "__main__":
    main()