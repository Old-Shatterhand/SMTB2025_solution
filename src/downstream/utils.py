import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import matthews_corrcoef
import pickle


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
    distances = distances[distances[:, 1] > 0]
    N = distances.shape[0]
    if N <= 1:
        return 0.0
    mus = distances[:, 2] / distances[:, 1]
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
    
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )
    
    mccs = []
    for i in range(y_true.shape[1]):
        try:
            mcc = matthews_corrcoef(y_true[:, i], y_pred[:, i] > 0.5)
        except ValueError:
            # Handle cases where MCC is undefined (e.g., only one class present)
            mcc = 0.0
        mccs.append(mcc)
    
    return float(np.mean(mccs))


def multiclass_mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the Matthews Correlation Coefficient (MCC) for a multiclass classification task.

    Args:
        y_true: np.ndarray of shape (n_samples,)
        y_pred: np.ndarray of shape (n_samples,)
    
    Returns:
        float: MCC for the multiclass classification
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    classes = np.unique(y_true)
    mcc_scores = []

    for k in classes:
        # Binarise: class k → 1 (positive), everything else → 0 (negative)
        bt = (y_true == k).astype(int)
        bp = (y_pred == k).astype(int)

        tp = int(np.sum((bt == 1) & (bp == 1)))
        tn = int(np.sum((bt == 0) & (bp == 0)))
        fp = int(np.sum((bt == 0) & (bp == 1)))
        fn = int(np.sum((bt == 1) & (bp == 0)))

        denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

        mcc_k = (tp * tn - fp * fn) / denom if denom > 0 else 0.0
        mcc_scores.append(mcc_k)

    return float(np.mean(mcc_scores))
