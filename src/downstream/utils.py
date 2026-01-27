import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import matthews_corrcoef


def compute_id_2NN(distances: np.ndarray) -> float:
    """Compute intrinsic dimension using the 2NN algorithm.
    Based on the implementation from DADApy

    Args:
        algorithm (str): 'base' to perform the linear fit, 'ml' to perform maximum likelihood
        mu_fraction (float): fraction of mus that will be considered for the estimate (discard highest mus)
        data_fraction (float): fraction of randomly sampled points used to compute the id
        n_iter (int): number of times the ID is computed on data subsets (useful when decimation < 1)
        set_attr (bool): whether to change the class attributes as a result of the computation

    Returns:
        intrinsic_dim (float): the estimated intrinsic dimension
    """
    # with np.errstate(divide='ignore', invalid='ignore'):
    mus = distances[:, 2] / distances[:, 1]
    N = mus.shape[0]
    n_eff = int(N * 0.9)
    log_mus_reduced = np.sort(np.log(mus))[:n_eff]

    y = -np.log(1 - np.arange(1, n_eff + 1) / N)

    def func(x, m):
        return m * x

    return curve_fit(func, log_mus_reduced, y)[0][0]


def return_data_overlap(dist_indices_base: np.ndarray, dist_indices_other: np.ndarray, k: int = 10) -> np.float32:
    """
    Compute the data overlap between two distance index matrices.
    Based on the implementation from DADApy

    Args:
        dist_indices_base (np.ndarray): distance indices matrix for the base data
        dist_indices_other (np.ndarray): distance indices matrix for the other data
        k (int): number of nearest neighbors to consider for overlap computation

    Returns:
        overlap (float): average data overlap between the two datasets
    """
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


def multioutput_mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the average Matthews Correlation Coefficient (MCC) for a multi-output task.

    Args:
        y_true: np.ndarray of shape (n_samples, n_outputs)
        y_pred: np.ndarray of shape (n_samples, n_outputs)

    Returns:
        float: average MCC across outputs
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    assert y_true.shape == y_pred.shape, "Shapes of y_true and y_pred must match"
    
    mccs = []
    for i in range(y_true.shape[1]):
        try:
            mcc = matthews_corrcoef(y_true[:, i], y_pred[:, i] > 0.5)
        except ValueError:
            # Handle cases where MCC is undefined (e.g., only one class present)
            mcc = 0.0
        mccs.append(mcc)
    
    return float(np.mean(mccs))
