import json
import os
import random
import re
from functools import partial
from shutil import rmtree
from typing import List, Tuple

import keras
import matplotlib.pyplot as plt
import numpy as np
import scipy
import tensorflow as tf
from tqdm import tqdm

CB_color_cycle = [
    '#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3',
    '#999999', '#e41a1c', '#dede00'
]


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


def KGLossBase(y_true, y_pred, alpha: float = 0.1, beta: float = 0.1) -> float:
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

    y_th = y_true > 10
    y_pred_th = y_pred > 10
    tp = tf.math.count_nonzero(tf.math.logical_and(y_th, y_pred_th))
    fp_fn = tf.math.count_nonzero(tf.math.logical_xor(y_th, y_pred_th))
    csi = tf.divide(tp, tf.add(tf.add(tp, fp_fn), 1))
    csi_err = tf.subtract(1.0, tf.cast(csi, tf.float32))
    csi_weighted_err = tf.cast(tf.multiply(csi_err, beta), tf.float16)

    err = tf.add(tf.add(mse, lt_zero_weighted_err), csi_weighted_err)
    return err


def KGLoss(alpha: float = 0.1, beta: float = 0.1) -> partial:
    """
    KGMeanSquaredError Partial wrapper for Knowledge guided mse

    Knowledge guided mean squared error that penalizes prediction with a weighted percentage of negative values

    Parameters
    ----------
    alpha : float, optional
        Weight to apply to knowledge guided error, by default 0.1

    Returns
    -------
    partial
        Partial function used to compute loss in neural network training
    """
    return partial(KGLossBase, alpha=alpha, beta=beta)


def recreate_directory(directory: str) -> None:
    """
    recreate_directory Function used to delete and recreate a directory

    Parameters
    ----------
    directory : str
        Path to target directory
    """
    try:
        rmtree(directory)
    except Exception as e:
        print(e)

    os.makedirs(directory)


def plot_samples(X: np.ndarray, y: np.ndarray, y_hat: np.ndarray,
                 output_dir: str, file_name: str) -> None:
    """
    plot_samples Use to plot several samples

    Plot a set of input, expected output, predicted output, and comparrison samples 

    Parameters
    ----------
    X : np.ndarray
        Input
    y : np.ndarray
        Expected output
    y_hat : np.ndarray
        Predicted output
    output_dir : str
        Directory to save plots
    file_name : str
        Name to use for each file
    """
    y[y < 0] = 0
    y_hat[y_hat < 0] = 0
    vmin = np.min(np.hstack([y, y_hat]))
    vmax = np.max(np.hstack([y, y_hat]))
    for x in ["input", "expected_output", "predicted_output", "comparison"]:
        os.makedirs(f"{output_dir}/{x}", exist_ok=True)

    # Input figures
    fig, axs = plt.subplots(2, 6, figsize=(30, 5))
    for i in range(2):
        for j in range(6):
            axs[i, j].imshow(X[i * 6 + j, :, :, 0], vmin=vmin, vmax=vmax)
            axs[i, j].set_xlabel(f"t={i * 6 + j}")

    fig.tight_layout()
    fig.savefig(f"{output_dir}/input/{file_name}.png", bbox_inches="tight")

    # Expected output figures
    fig, axs = plt.subplots(2, 4, figsize=(25, 5))
    for i in range(2):
        for j in range(4):
            axs[i, j].imshow(y[i * 4 + j, :, :], vmin=vmin, vmax=vmax)
            axs[i, j].set_xlabel(f"t={i * 4 + j}")

    fig.tight_layout()
    fig.savefig(f"{output_dir}/expected_output/{file_name}.png",
                bbox_inches="tight")

    # Predicted figures
    fig, axs = plt.subplots(2, 4, figsize=(25, 5))

    for i in range(2):
        for j in range(4):
            axs[i, j].imshow(y_hat[i * 4 + j, :, :], vmin=vmin, vmax=vmax)
            axs[i, j].set_xlabel(f"t={i * 4 + j}")

    fig.tight_layout()
    fig.savefig(f"{output_dir}/predicted_output/{file_name}.png",
                bbox_inches="tight")

    # Comparison figures
    fig, axs = plt.subplots(2, 4, figsize=(25, 5))
    axs[0, 0].set_ylabel("y true")
    axs[1, 0].set_ylabel("y predicted")
    for j in range(4):
        axs[0, j].imshow(y[j, :, :], vmin=vmin, vmax=vmax)
        axs[0, j].set_xlabel(f"t={j}")
        axs[1, j].imshow(y_hat[j, :, :], vmin=vmin, vmax=vmax)
        axs[1, j].set_xlabel(f"t={j}")

    fig.tight_layout()
    fig.savefig(f"{output_dir}/comparison/{file_name}.png",
                bbox_inches="tight")


def csi(y: np.ndarray,
        y_pred: np.ndarray,
        threshold: float = 0.125,
        axis: Tuple[int] = (0, 2, 3)) -> float:
    """
    csi Critical success index

    Compute the critical success index for a given output and prediction along one or more axes

    Parameters
    ----------
    y : np.ndarray
        Output
    y_pred : np.ndarray
        Prediction
    threshold : float, optional
        Threshold to use when calculating csi, by default 0.125
    axis : Tuple[int], optional
        Axis to use when calculating csi, by default (0, 2, 3)

    Returns
    -------
    float
        Critical success index
    """
    y_th = y > threshold
    y_pred_th = y_pred > threshold
    tp = np.count_nonzero(np.logical_and(y_th, y_pred_th), axis=axis)
    fp_fn = np.count_nonzero(np.logical_xor(y_th, y_pred_th), axis=axis)
    return tp / (tp + fp_fn)


def create_samples_from_mat_files(
        mat_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    create_samples_from_mat_files Create a set of input, output, and gfs samples from mat files

    Parameters
    ----------
    mat_path : str
        Path to directory containing mat files

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        input, output, and gfs samples as numpy arrays
    """
    mat_files = [
        f"{mat_path}/{x}" for x in os.listdir(mat_path)
        if re.match(r"20.*-S.*\.mat", x)
    ]
    mat_files.sort()

    Xs = []
    ys = []
    gfss = []
    print("Loading .mat files")
    for mat_file in tqdm(mat_files):
        mat = scipy.io.loadmat(mat_file)
        mat_shape = mat["X"]["imerg"][0][0].shape
        x = np.array([
            mat["X"][x][0][0] for x in ["imerg", "gfs_v", "gfs_tpw", "gfs_u"]
        ])
        x = np.moveaxis(x, 0, 2)
        Xs.append(x)
        ys.append(mat["X"]["imerg"][0][0].reshape(
            (mat_shape[0], mat_shape[1], 1)))
        gfss.append(mat["X"]["gfs_pr"][0][0].reshape(
            (mat_shape[0], mat_shape[1], 1)))

    Xs = np.array(Xs)
    ys = np.array(ys)
    gfss = np.array(gfss)

    return Xs, ys, gfss


def write_samples_to_npy(X: np.ndarray, y: np.ndarray,
                         write_directory: str) -> None:
    """
    write_samples_to_npy Join and write samples of input and outputs to .npy files

    Parameters
    ----------
    X : np.ndarray
        Input array
    y : np.ndarray
        Output array
    write_directory : str
        Directory where files will be written
    """
    for i in tqdm(range(X.shape[0])):
        arr = np.array([X[i], y[i]], dtype=object)
        np.save(f"{write_directory}/{i}.npy", arr)


def model_analysis(model, results_dir: str, dataset_directory: str):
    metrics = {}
    for split in ["train", "val", "test"]:
        os.makedirs(f"{results_dir}/{split}")
        for x in ["distribution", "mse", "csi"]:
            os.makedirs(f"{results_dir}/{split}/{x}")

        split_directory = f"{dataset_directory}/{split}"
        gfs_split_directory = f"{dataset_directory}/gfs/{split}"
        eval_paths = [
            f"{split_directory}/{x}" for x in os.listdir(split_directory)
        ]
        gfs_paths = [
            f"{gfs_split_directory}/{x}"
            for x in os.listdir(gfs_split_directory)
        ]
        eval_dataset = CustomGenerator(eval_paths, 1, shuffle=False)
        gfs_dataset = CustomGenerator(gfs_paths, 1, shuffle=False)

        # Getting inputs, outputs, and predictions (including gfs and persist) for all samples
        y_pred = model.predict(eval_dataset)
        y_pred[y_pred < 0] = 0
        Xs = []
        ys = []
        gfss = []
        y_persists = []
        for k in tqdm(range(y_pred.shape[0])):
            X, y = eval_dataset.__getitem__(k)
            _, gfs = gfs_dataset.__getitem__(k)
            X = X[0]
            y = y[0][:, :, :, 0]
            gfs = gfs[0][:, :, :, 0]
            y_persist = np.array([X[-1, :, :, 0] for i in range(8)])

            Xs.append(X)
            ys.append(y)
            gfss.append(gfs)
            y_persists.append(y_persist)

        X = np.array(Xs)
        y = np.array(ys)
        y_gfs = np.array(gfss)
        y_persist = np.array(y_persists)

        predictions = {
            "y": y,
            "y_pred": y_pred,
            "y_gfs": y_gfs,
            "y_persist": y_persist
        }

        # Plotting prediction distributions by pixel
        for k, v in predictions.items():
            fig, axs = plt.subplots(3, figsize=(8, 19))
            vmin = np.min(v)
            vmax = np.max(v)

            pos0 = axs[0].imshow(np.mean(v, axis=(0, 1)), vmin=vmin, vmax=vmax)
            axs[0].set_title(f"Average Rainfall ({k} {split})")
            fig.colorbar(pos0,
                         ax=axs[0],
                         location="right",
                         shrink=0.35,
                         label="mm/hr")

            pos1 = axs[1].imshow(np.std(v, axis=(0, 1)), vmin=vmin, vmax=vmax)
            axs[1].set_title(f"Standard Deviation of Rainfall ({k} {split})")
            fig.colorbar(pos1,
                         ax=axs[1],
                         location="right",
                         shrink=0.35,
                         label="mm/hr")

            pos2 = axs[2].imshow(np.max(v, axis=(0, 1)), vmin=vmin, vmax=vmax)
            axs[2].set_title(f"Maximum Rainfall ({k} {split})")
            fig.colorbar(pos2,
                         ax=axs[2],
                         location="right",
                         shrink=0.35,
                         label="mm/hr")

            fig.tight_layout(pad=-20)
            fig.savefig(
                f"{results_dir}/{split}/distribution/{split}_{k}_distribution_by_pixel.png",
                bbox_inches="tight")

        # Calculating metrics along all axes
        for k, v in predictions.items():
            metrics[f"{split}_{k}_mae"] = float(
                np.mean(np.abs(y - v)).reshape(-1)[0])
            metrics[f"{split}_{k}_mse"] = float(
                np.mean((y - v)**2).reshape(-1)[0])

            y_flat = y.flatten()
            v_flat = v.flatten()
            flat_nonzero_mask = y_flat > 0
            metrics[f"{split}_{k}_mse_nonzero"] = float(
                np.mean((y_flat[flat_nonzero_mask] -
                         v_flat[flat_nonzero_mask])**2).reshape(-1)[0])

            for threshold in [0.125, 2, 5, 10]:
                metrics[f"{split}_{k}_csi_{threshold}"] = csi(
                    y=y, y_pred=v, threshold=threshold, axis=(0, 1, 2, 3))

        # Finding representative examples to plot
        y_mse_sample = np.mean((y - y_pred)**2, axis=(1, 2, 3)).reshape(-1)
        y_mse_sample_argsort = np.argsort(y_mse_sample)
        n = y_mse_sample_argsort.shape[0]

        err_samples = {
            "min_err": y_mse_sample_argsort[0],
            "low_err": y_mse_sample_argsort[n // 4],
            "mid_err": y_mse_sample_argsort[n // 2],
            "high_err": y_mse_sample_argsort[(n // 4) * 3],
            "max_err": y_mse_sample_argsort[-1]
        }
        for k, v in err_samples.items():
            plot_samples(X[v], y[v], y_pred[v],
                         f"{results_dir}/{split}/examples", f"{split}_{k}")

        # Calculating mse metrics by pixel
        mse_px = {}
        for k, v in predictions.items():
            mse_px[k] = np.mean((y - v)**2, axis=(0, 1))

        vmin = np.min([x for x in mse_px.values()])
        vmax = np.max([x for x in mse_px.values()])

        for k, v in mse_px.items():
            fig, ax = plt.subplots(1, figsize=(8, 19))
            pos = ax.imshow(v, vmin=vmin, vmax=vmax)
            ax.set_title(f"Mean Squared Error by Pixel ({k} {split})")
            fig.colorbar(pos,
                         ax=ax,
                         location="right",
                         shrink=0.13,
                         label="MSE")
            fig.tight_layout()
            fig.savefig(
                f"{results_dir}/{split}/mse/{split}_{k}_mse_by_pixel.png",
                bbox_inches="tight")

        # Calculating csi metrics by pixel
        for threshold in [0.125, 2, 5, 10]:
            csi_px = {}
            for k, v in predictions.items():
                csi_px = csi(y=y, y_pred=v, threshold=threshold, axis=(0, 1))

                fig, ax = plt.subplots(1, figsize=(8, 19))
                pos = ax.imshow(csi_px, vmin=0, vmax=1)
                ax.set_title(
                    f"Critical Success Index by Pixel at {threshold} mm/hr ({k} {split})"
                )
                fig.colorbar(pos,
                             ax=ax,
                             location="right",
                             shrink=0.13,
                             label="CSI")
                fig.tight_layout()
                fig.savefig(
                    f"{results_dir}/{split}/csi/{split}_{k}_csi_by_pixel_{threshold}.png",
                    bbox_inches="tight")

        # Calculating mse metrics by lead time
        mse_lt = {}
        for k, v in predictions.items():
            mse = np.mean((y - v)**2, axis=(0, 2, 3)).reshape(-1)
            mse_nonzeros = []
            for i in range(8):
                yi = y[:, i, :, :].flatten()
                vi = v[:, i, :, :].flatten()
                flat_nonzero_mask = yi > 0
                mse_nonzeros.append(
                    float(
                        np.mean((yi[flat_nonzero_mask] -
                                 vi[flat_nonzero_mask])**2).reshape(-1)[0]))

            mse_lt[k] = {"zeros": mse, "nonzeros": np.array(mse_nonzeros)}

        x_plt = np.arange(mse_lt["y_pred"]["zeros"].shape[0])
        fig, ax = plt.subplots(1, figsize=(10, 5))

        ax.plot(x_plt,
                mse_lt["y_pred"]["zeros"],
                label="Our model",
                marker="s",
                c=CB_color_cycle[0])
        ax.plot(x_plt,
                mse_lt["y_gfs"]["zeros"],
                label="GFS",
                marker="s",
                c=CB_color_cycle[1])
        ax.plot(x_plt,
                mse_lt["y_persist"]["zeros"],
                label="Persistence",
                marker="s",
                c=CB_color_cycle[2])
        ax.set_title(f"Mean Squared Error by Lead Time ({split})")
        ax.set_xlabel("Lead Time")
        ax.set_ylabel("MSE")
        ax.legend()

        fig.savefig(f"{results_dir}/{split}/mse/{split}_mse_by_lead_time.png",
                    bbox_inches="tight")

        fig, ax = plt.subplots(1, figsize=(10, 5))
        ax.plot(x_plt,
                mse_lt["y_pred"]["nonzeros"],
                label="Our model",
                marker="s",
                c=CB_color_cycle[0])
        ax.plot(x_plt,
                mse_lt["y_gfs"]["nonzeros"],
                label="GFS",
                marker="s",
                c=CB_color_cycle[1])
        ax.plot(x_plt,
                mse_lt["y_persist"]["nonzeros"],
                label="Persistence",
                marker="s",
                c=CB_color_cycle[2])
        ax.set_title(
            f"Mean Squared Error by Lead Time for Nonzero Values ({split})")
        ax.set_xlabel("Lead Time")
        ax.set_ylabel("MSE")
        ax.legend()

        fig.savefig(
            f"{results_dir}/{split}/mse/{split}_mse_by_lead_time_nonzero.png",
            bbox_inches="tight")

        # Calculating csi metrics my lead time
        for threshold in [0.125, 2, 5, 10]:
            y_csi = csi(y=y,
                        y_pred=y_pred,
                        threshold=threshold,
                        axis=(0, 2, 3))
            y_gfs_csi = csi(y=y,
                            y_pred=y_gfs,
                            threshold=threshold,
                            axis=(0, 2, 3))
            y_persist_csi = csi(y=y,
                                y_pred=y_persist,
                                threshold=threshold,
                                axis=(0, 2, 3))

            x_plt = np.arange(y_csi.shape[0])
            fig, ax = plt.subplots(1, figsize=(10, 5))

            ax.plot(x_plt,
                    y_csi,
                    label="Our model",
                    marker="s",
                    c=CB_color_cycle[0])

            ax.plot(x_plt,
                    y_gfs_csi,
                    label="GFS",
                    marker="s",
                    c=CB_color_cycle[1])

            ax.plot(x_plt,
                    y_persist_csi,
                    label="Persistence",
                    marker="s",
                    c=CB_color_cycle[2])

            ax.set_title(
                f"Critical Success Index by Lead Time at {threshold} mm/hr ({split})"
            )
            ax.set_xlabel("Lead Time")
            ax.set_ylabel("CSI")
            ax.legend()

            fig.savefig(
                f"{results_dir}/{split}/csi/{split}_csi_by_lead_time_{threshold}.png",
                bbox_inches="tight")

        for threshold in [0.125, 2, 5, 10]:
            y_csi = csi(y=y, y_pred=y_pred, threshold=threshold, axis=(2, 3))
            y_plt_mean = np.mean(y_csi, axis=(0))
            y_plt_std = np.std(y_csi, axis=(0))

            y_gfs_csi = csi(y=y,
                            y_pred=y_gfs,
                            threshold=threshold,
                            axis=(2, 3))
            y_gfs_plt_mean = np.mean(y_gfs_csi, axis=(0))
            y_gfs_plt_std = np.std(y_gfs_csi, axis=(0))

            y_persist_csi = csi(y=y,
                                y_pred=y_pred,
                                threshold=threshold,
                                axis=(2, 3))
            y_persist_plt_mean = np.mean(y_persist_csi, axis=(0))
            y_persist_plt_std = np.std(y_persist_csi, axis=(0))

            x_plt = np.arange(y_plt_mean.shape[0])
            fig, ax = plt.subplots(1, figsize=(10, 5))

            ax.errorbar(x_plt,
                        y_plt_mean,
                        y_plt_std,
                        label="Our model",
                        marker="s",
                        c=CB_color_cycle[0])

            ax.errorbar(x_plt,
                        y_gfs_plt_mean,
                        y_gfs_plt_std,
                        label="GFS",
                        marker="s",
                        c=CB_color_cycle[1])

            ax.errorbar(x_plt,
                        y_persist_plt_mean,
                        y_persist_plt_std,
                        label="Persistence",
                        marker="s",
                        c=CB_color_cycle[2])

            ax.set_title(
                f"Mean Critical Success Index by Lead Time at {threshold} mm/hr ({split})"
            )
            ax.set_xlabel("Lead Time")
            ax.set_ylabel("CSI")
            ax.legend()

            fig.savefig(
                f"{results_dir}/{split}/csi/{split}_mean_csi_by_lead_time_{threshold}.png",
                bbox_inches="tight")

    with open(f"{results_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics