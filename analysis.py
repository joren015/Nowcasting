import json
import os
import random

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from nowcasting.unet import res2
from nowcasting.utils import CustomGenerator, plot_samples, recreate_directory

seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

if __name__ == "__main__":
    # Absolute paths
    mlruns_path = "/panfs/jay/groups/6/csci8523/joren015/repos/Nowcasting/mlruns"
    train_path = "/panfs/jay/groups/6/csci8523/joren015/repos/Nowcasting/data/datasets/12_8_0_20_1.0/train/"
    val_path = "/panfs/jay/groups/6/csci8523/joren015/repos/Nowcasting/data/datasets/12_8_0_20_1.0/val/"
    test_path = "/panfs/jay/groups/6/csci8523/joren015/repos/Nowcasting/data/datasets/12_8_0_20_1.0/test/"

    # Relative paths
    mlruns_path = "mlruns"
    train_path = "data/datasets/12_8_0_20_1.0/train/"
    val_path = "data/datasets/12_8_0_20_1.0/val/"
    test_path = "data/datasets/12_8_0_20_1.0/test/"

    datasets = {"train": train_path, "val": val_path, "test": test_path}

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
    for dataset_split, dataset_directory in datasets.items():
        eval_paths = [
            f"{dataset_directory}/{x}" for x in os.listdir(dataset_directory)
        ]
        random.shuffle(eval_paths)
        eval_dataset = CustomGenerator(eval_paths, 1, shuffle=False)

        mae, _, mse = model.evaluate(eval_dataset)
        metrics[dataset_split] = {"mae": mae, "mse": mse}

        y_pred = model.predict(eval_dataset)

        Xs = []
        ys = []
        y_persists = []
        for k in tqdm(range(y_pred.shape[0])):
            X, y = eval_dataset.__getitem__(k)
            X = X[0]
            y = y[0][:, :, :, 0]

            Xs.append(X)
            ys.append(y)
            y_persists.append(np.array([X[-1, :, :, 0] for i in range(8)]))

        X = np.array(Xs)
        y = np.array(ys)
        y_persist = np.array(y_persists)

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
            plot_samples(X[v], y[v], y_pred[v], results_dir,
                         f"{dataset_split}_{k}")

        # Calculating mse metrics by lead time
        y_mse = np.mean((y - y_pred)**2, axis=(0, 2, 3)).reshape(-1)
        y_persist_mse = np.mean((y - y_persist)**2, axis=(0, 2, 3)).reshape(-1)

        x_plt = np.arange(y_mse.shape[0])
        fig, ax = plt.subplots(1, figsize=(15, 5))

        ax.plot(x_plt, y_mse, label="Our model", marker="s")
        ax.plot(x_plt, y_persist_mse, label="Persistence", marker="s")
        ax.set_title(f"Mean Squared Error by Lead Time ({dataset_split})")
        ax.set_xlabel("Lead Time")
        ax.set_ylabel("MSE")
        ax.legend()

        fig.savefig(f"{results_dir}/{dataset_split}_mse_by_lead_time.png")

        # Calculating csi metrics my lead time
        for threshold in [0.125, 2, 5, 10]:
            y_th = y > threshold
            y_pred_th = y_pred > threshold
            tp = np.count_nonzero(np.logical_and(y_th, y_pred_th),
                                  axis=(0, 2, 3))
            fp_fn = np.count_nonzero(np.logical_xor(y_th, y_pred_th),
                                     axis=(0, 2, 3))
            y_csi = tp / (tp + fp_fn)

            y_persist_th = y_persist > threshold
            tp = np.count_nonzero(np.logical_and(y_th, y_persist_th),
                                  axis=(0, 2, 3))
            fp_fn = np.count_nonzero(np.logical_xor(y_th, y_persist_th),
                                     axis=(0, 2, 3))
            y_persist_csi = tp / (tp + fp_fn)

            x_plt = np.arange(y_csi.shape[0])
            fig, ax = plt.subplots(1, figsize=(15, 5))

            ax.plot(x_plt, y_csi, label="Our model", marker="s")

            ax.plot(x_plt, y_persist_csi, label="Persistence", marker="s")

            ax.set_title(
                f"Critical Success Index by Lead Time at {threshold} mm per hr ({dataset_split})"
            )
            ax.set_xlabel("Lead Time")
            ax.set_ylabel("CSI")
            ax.legend()

            fig.savefig(
                f"{results_dir}/{dataset_split}_csi_by_lead_time_{threshold}.png"
            )

    with open(f"{results_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
