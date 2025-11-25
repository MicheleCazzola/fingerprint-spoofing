from src.config.config import LOG, MODEL_PATH_GMM
from src.dataset.dataset import Dataset
from src.evaluator.evaluator import Evaluator
from src.models.gmm import GaussianMixtureModel

def write_GMM_results(results):
    print_string = ""
    for (variant, results) in results.items():
        print_string += f"--{'Full covariance' if variant == 'full' else 'Diagonal covariance'} matrices--\n"
        print_string += f"{'Components(False)':<18s}{'Components(True)':^19s}{'Minimum DCF':^13s}{'Actual DCF':^12s}\n"
        for (num_components, result) in results.items():
            min_dcf, dcf = map(result.get, ["min_dcf", "dcf"])
            nc_false, nc_true = num_components
            print_string += f"{nc_false:^18d}{nc_true:^19d}{min_dcf:^13.3f}{dcf:^12.3f}\n"
        print_string += "\n"

    return print_string


def gmm_variant(trainset: Dataset, validset: Dataset, app_prior, variant, components, gmm: GaussianMixtureModel):
    
    DTR, LTR = trainset.get_data(), trainset.get_labels()
    DVAL, LVAL = validset.get_data(), validset.get_labels()
    
    result = {}

    if LOG:
        str_variant = "Full" if variant == "full" else "Diagonal"
        print(f"--{str_variant} covariance matrices--")

    gmm.set_params(variant=variant)
    for nc_false in components:
        for nc_true in components:

            if LOG:
                print(f"Components: F = {nc_false}, T = {nc_true}")

            gmm.set_params(components=(nc_false, nc_true))
            gmm.fit(DTR, LTR)
            id = gmm.save_state_dict(MODEL_PATH_GMM)
            llr = gmm.scores(DVAL, LVAL)
            LPR = gmm.predict(DVAL, LVAL, app_prior)

            min_dcf, dcf, llr = map(
                Evaluator.evaluate(llr, LPR, LVAL, app_prior)["results"].get,
                ["min_dcf", "dcf", "llr"]
            )

            result[(nc_false, nc_true)] = {
                "min_dcf": min_dcf,
                "dcf": dcf,
                "llr": llr,
                "id": id
            }

    return result


def gmm_task(trainset, validset, app_prior):
    steps = 6
    gmm_components = [2 ** i for i in range(steps)]
    gmm_results = {
        "full": {},
        "diag": {}
    }

    gmm = GaussianMixtureModel()

    for variant in gmm_results:
        gmm_results[variant] = gmm_variant(trainset, validset, app_prior, variant, gmm_components, gmm)

    return gmm_results