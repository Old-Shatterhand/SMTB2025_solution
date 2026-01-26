import numpy as np
from scipy.optimize import curve_fit


def _compute_id_2NN(mus):
    """
    Compute the id using the 2NN algorithm.
    Helper of return return_id_2NN.

    Args:
        mus (np.ndarray(float)): ratio of the distances of first- and second-nearest neighbours

    Returns:
        intrinsic_dim (float): the estimation of the intrinsic dimension
    """
    N = mus.shape[0]
    n_eff = int(N * 0.9)
    log_mus_reduced = np.sort(np.log(mus))[:n_eff]

    y = -np.log(1 - np.arange(1, n_eff + 1) / N)

    def func(x, m):
        return m * x

    return curve_fit(func, log_mus_reduced, y)[0][0]


def compute_id_2NN(
    distances,
):
    """Compute intrinsic dimension using the 2NN algorithm.

    Args:
        algorithm (str): 'base' to perform the linear fit, 'ml' to perform maximum likelihood
        mu_fraction (float): fraction of mus that will be considered for the estimate (discard highest mus)
        data_fraction (float): fraction of randomly sampled points used to compute the id
        n_iter (int): number of times the ID is computed on data subsets (useful when decimation < 1)
        set_attr (bool): whether to change the class attributes as a result of the computation

    Returns:
        id (float): the estimated intrinsic dimension
        id_err (float): the standard error on the id estimation
        rs (float): the average nearest neighbor distance (rs)
    """
    return _compute_id_2NN(distances[:, 2] / distances[:, 1])


def return_data_overlap(
    dist_indices_base=None,
    dist_indices_other=None,
    k: int = 10,
):
    assert dist_indices_base.shape[0] == dist_indices_other.shape[0]
    ndata = dist_indices_base.shape[0]

    overlaps = -np.ones(ndata)
    for i in range(ndata):
        overlaps[i] = (
            len(
                np.intersect1d(
                    dist_indices_base[i, 1 : k + 1],
                    dist_indices_other[i, 1 : k + 1],
                )
            )
            / k
        )
    return np.mean(overlaps)
