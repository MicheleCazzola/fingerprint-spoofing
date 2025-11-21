from src.config.config import LOG
from src.evaluator.evaluator import Evaluator
from src.models.gmm import GaussianMixtureModel


def gmm_variant(DTR, LTR, DVAL, LVAL, app_prior, variant, components, gmm: GaussianMixtureModel):
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
            llr = gmm.scores(DVAL, LVAL)
            LPR = gmm.predict(DVAL, LVAL, app_prior)

            min_dcf, dcf, llr = map(
                Evaluator.evaluate(llr, LPR, LVAL, app_prior)["results"].get,
                ["min_dcf", "dcf", "llr"]
            )

            result[(nc_false, nc_true)] = {
                "min_dcf": min_dcf,
                "dcf": dcf,
                "llr": llr
            }

    return result


def gmm_task(DTR, LTR, DVAL, LVAL, app_prior):
    steps = 6
    gmm_components = [2 ** i for i in range(steps)]
    gmm_results = {
        "full": {},
        "diag": {}
    }

    gmm = GaussianMixtureModel()

    for variant in gmm_results:
        gmm_results[variant] = gmm_variant(DTR, LTR, DVAL, LVAL, app_prior, variant, gmm_components, gmm)

    return gmm_results