"""
    File for console I/O
    Usage: logging
"""


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
    print(result["llr"][:, 5], result["llr"][:, -5:], sep="\n")
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
        print(score[:, :5], score[:, -5:], sep="\n")
        print()
