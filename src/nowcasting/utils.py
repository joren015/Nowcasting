import numpy as np
from typing import Tuple


def sliding_window_expansion(
        X: np.ndarray,
        y: np.ndarray,
        input_window_size: int = 3,
        target_window_size: int = 1,
        target_offset: int = 1,
        step: int = 2,
        sample_ratio: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    sliding_window_expansion Generates input and target pairs from time series like dataset

    Parameters
    ----------
    X : np.ndarray
        Input timeseries array
    y : np.ndarray
        Target timeseries array
    input_window_size : int, optional
        Number of timesteps in each input, by default 3
    target_window_size : int, optional
        Number of timesteps in each target, by default 1
    target_offset : int, optional
        Number of timesteps the start of each target should be from the end of each input, by default 1
    step : int, optional
        Number of timesteps to use to determine the start of the next input and target pair, by default 2
    sample_ratio : int, optional
        Percentage of results to return, by default 1

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Input and target timeseries pairs
    """
    max_idx = X.shape[
        0] - input_window_size - target_window_size - target_offset
    x_idx = (np.expand_dims(np.arange(input_window_size), 0) +
             np.expand_dims(np.arange(max_idx, step=step), 0).T)
    y_idx = (np.expand_dims(np.arange(target_window_size), 0) +
             np.expand_dims(np.arange(max_idx, step=step),
                            0).T) + input_window_size + target_offset

    if sample_ratio < 1:
        sample_mask = (np.random.random(size=(len(x_idx))) <= sample_ratio)
        x_idx = x_idx[sample_mask]
        y_idx = y_idx[sample_mask]

    X_new = X[x_idx]
    y_new = y[y_idx]

    return X_new, y_new