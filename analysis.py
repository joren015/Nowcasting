import json
import os
import random

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from nowcasting.unet import res2
from nowcasting.utils import (CustomGenerator, csi, plot_samples,
                              recreate_directory)

seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

CB_color_cycle = [
    '#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3',
    '#999999', '#e41a1c', '#dede00'
]

if __name__ == "__main__":
    # Absolute paths
    mlruns_path = "/panfs/jay/groups/6/csci8523/joren015/repos/Nowcasting/mlruns"
    dataset_directory = "/panfs/jay/groups/6/csci8523/joren015/repos/Nowcasting/data/datasets/12_8_0_20_1.0"

    # Relative paths
    mlruns_path = "mlruns"
    dataset_directory = "data/datasets/12_8_0_20_1.0"

    # Set mlflow tracking uri so we can find the run results
    mlflow.set_tracking_uri(mlruns_path)

    # Get dataframe of mlflow runs by experiment name
    experiment_name = "hpo_res_mse_12_8_0_20_1.0"
    experiment_id = mlflow.get_experiment_by_name(
        experiment_name).experiment_id

    df = mlflow.search_runs(experiment_id)

    # Get run by name
    # run = df[df["tags.mlflow.runName"] == "receptive-sheep-254"].iloc[0]

    # Get run with lowest validation mse
    run = df.sort_values(by="metrics.val_mse").iloc[0]

    # Create path to saved weights
    run_id = run["run_id"]
    experiment_id = run["experiment_id"]
    weights_path = f"{mlruns_path}/{experiment_id}/{run_id}/artifacts/script_n1.h5"

    # Get number of base filters used so that model has the correct number of parameters
    num_filters_base = int(run["params.hpo_num_filters_base"])

    # Create model and load weights
    model = res2((12, 256, 620, 4),
                 num_filters_base=num_filters_base,
                 dropout_rate=0)
    model.load_weights(weights_path)
    model.compile(loss="mean_absolute_error", metrics=["mae", "mse"])

    results_dir = f"data/analysis/{experiment_name}"
    recreate_directory(results_dir)

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

        metrics[split] = {}
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
            metrics[split][f"{k}_mae"] = float(
                np.mean(np.abs(y - v)).reshape(-1)[0])
            metrics[split][f"{k}_mse"] = float(
                np.mean((y - v)**2).reshape(-1)[0])

            for threshold in [0.125, 2, 5, 10]:
                metrics[split][f"{k}_csi_{threshold}"] = csi(
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
        y_mse = np.mean((y - y_pred)**2, axis=(0, 2, 3)).reshape(-1)
        y_gfs_mse = np.mean((y - y_gfs)**2, axis=(0, 2, 3)).reshape(-1)
        y_persist_mse = np.mean((y - y_persist)**2, axis=(0, 2, 3)).reshape(-1)

        x_plt = np.arange(y_mse.shape[0])
        fig, ax = plt.subplots(1, figsize=(10, 5))

        ax.plot(x_plt,
                y_mse,
                label="Our model",
                marker="s",
                c=CB_color_cycle[0])
        ax.plot(x_plt, y_gfs_mse, label="GFS", marker="s", c=CB_color_cycle[1])
        ax.plot(x_plt,
                y_persist_mse,
                label="Persistence",
                marker="s",
                c=CB_color_cycle[2])
        ax.set_title(f"Mean Squared Error by Lead Time ({split})")
        ax.set_xlabel("Lead Time")
        ax.set_ylabel("MSE")
        ax.legend()

        fig.savefig(f"{results_dir}/{split}/mse/{split}_mse_by_lead_time.png",
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

    with open(f"{results_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
