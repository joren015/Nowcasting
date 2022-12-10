import os
import random
from functools import partial
from shutil import rmtree
from typing import List, Tuple

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


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


class CustomGenerator(keras.utils.Sequence):
    """
    CustomGenerator Data loader/generator

    Custom data loader/generator used to load inputs from disk into RAM and GPU VRAM during training

    Parameters
    ----------
    keras : keras.utils.Sequence
        Inherited keras Sequence class
    """

    def __init__(self,
                 input_paths: List[str],
                 batch_size: int,
                 shuffle: bool = True):
        """
        __init__ Class constructor

        Parameters
        ----------
        input_paths : List[str]
            List of file paths to each input (files should contain a single sample)
        batch_size : int
            Batch size to use when retrieving input
        shuffle : bool, optional
            Option to shuffle input samples, by default True
        """
        self.input_paths = input_paths
        self.batch_size = batch_size

        if shuffle:
            random.shuffle(self.input_paths)

    def __len__(self) -> int:
        """
        __len__ Get number of batches based on batch size

        Returns
        -------
        int
            Total number of batches
        """
        return len(self.input_paths) // int(self.batch_size)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        __getitem__ Get item

        Returns a batch based on index argument

        Parameters
        ----------
        idx : int
            Index of batch to return

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (Input, label) pair
        """
        batch_x = self.input_paths[idx * self.batch_size:(idx + 1) *
                                   self.batch_size]

        X = []
        y = []
        for i in range(self.batch_size):
            arr = np.load(batch_x[i], allow_pickle=True)
            X.append(arr[0])
            y.append(arr[1])

        X = np.array(X)
        y = np.array(y)

        return X, y


def KGMeanSquaredErrorBase(y_true, y_pred, alpha: float = 0.1):
    """
    KGMeanSquaredErrorBase Knowledge guided mse

    Knowledge guided mean squared error that penalizes prediction with a weighted percentage of negative values

    Parameters
    ----------
    y_true : any
        Ground truth
    y_pred : any
        Predicted value
    alpha : float, optional
        Weight to apply to knowledge guided error, by default 0.1

    Returns
    -------
    float
        loss/error value
    """
    mse = tf.reduce_mean(tf.math.square(y_pred - y_true), axis=-1)
    lt_zero_count = tf.cast(tf.math.count_nonzero(y_pred < 0), tf.int32)
    lt_zero_err = tf.cast(tf.divide(lt_zero_count, tf.size(y_pred)),
                          tf.float32)
    lt_zero_weighted_err = tf.cast(tf.multiply(alpha, lt_zero_err), tf.float16)
    err = tf.add(mse, lt_zero_weighted_err)
    return err


def KGMeanSquaredError(alpha: float = 0.1):
    """
    KGMeanSquaredError Partial wrapper for Knowledge guided mse

    Knowledge guided mean squared error that penalizes prediction with a weighted percentage of negative values

    Parameters
    ----------
    alpha : float, optional
        Weight to apply to knowledge guided error, by default 0.1

    Returns
    -------
    float
        loss/error value
    """
    return partial(KGMeanSquaredErrorBase, alpha=alpha)


def recreate_directory(directory: str):
    try:
        rmtree(directory)
    except Exception as e:
        print(e)

    os.makedirs(directory)


def plot_samples(X: np.ndarray, y: np.ndarray, y_hat: np.ndarray,
                 output_dir: str, file_name: str):
    y_hat[y_hat < 0] = 0
    for x in ["input", "expected_output", "predicted_output", "comparison"]:
        os.makedirs(f"{output_dir}/{x}", exist_ok=True)

    # Input figures
    fig, axs = plt.subplots(2, 6, figsize=(30, 5))
    for i in range(2):
        for j in range(6):
            axs[i, j].imshow(X[i * 6 + j, :, :, 0])
            axs[i, j].set_xlabel(f"t={i * 6 + j}")

    fig.tight_layout()
    fig.savefig(f"{output_dir}/input/{file_name}.png")

    # Expected output figures
    fig, axs = plt.subplots(2, 4, figsize=(25, 5))
    for i in range(2):
        for j in range(4):
            axs[i, j].imshow(y[i * 4 + j, :, :])
            axs[i, j].set_xlabel(f"t={i * 4 + j}")

    fig.tight_layout()
    fig.savefig(f"{output_dir}/expected_output/{file_name}.png")

    # Predicted figures
    fig, axs = plt.subplots(2, 4, figsize=(25, 5))

    for i in range(2):
        for j in range(4):
            axs[i, j].imshow(y_hat[i * 4 + j, :, :])
            axs[i, j].set_xlabel(f"t={i * 4 + j}")

    fig.tight_layout()
    fig.savefig(f"{output_dir}/predicted_output/{file_name}.png")

    # Comparison figures
    fig, axs = plt.subplots(2, 4, figsize=(25, 5))
    axs[0, 0].set_ylabel("y true")
    axs[1, 0].set_ylabel("y predicted")
    for j in range(4):
        axs[0, j].imshow(y[j, :, :])
        axs[0, j].set_xlabel(f"t={j}")
        axs[1, j].imshow(y_hat[j, :, :])
        axs[1, j].set_xlabel(f"t={j}")

    fig.tight_layout()
    fig.savefig(f"{output_dir}/comparison/{file_name}.png")


def csi(y: np.ndarray,
        y_pred: np.ndarray,
        threshold: float = 0.125,
        axis: Tuple[int] = (0, 2, 3)):
    y_th = y > threshold
    y_pred_th = y_pred > threshold
    tp = np.count_nonzero(np.logical_and(y_th, y_pred_th), axis=axis)
    fp_fn = np.count_nonzero(np.logical_xor(y_th, y_pred_th), axis=axis)
    return tp / (tp + fp_fn)