import json
import os
import random
from shutil import rmtree

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import tensorflow as tf

from nowcasting.unet import res2
from nowcasting.utils import CustomGenerator

seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


def recreate_directory(directory):
    try:
        rmtree(directory)
    except Exception as e:
        print(e)

    os.makedirs(directory)


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

        for x in [
                "input", "expected_output", "predicted_output",
                "comparison_output"
        ]:
            recreate_directory(f"{results_dir}/{dataset_split}/{x}")

        eval_paths = [
            f"{dataset_directory}/{x}" for x in os.listdir(dataset_directory)
        ]
        random.shuffle(eval_paths)
        eval_dataset = CustomGenerator(eval_paths, 1, shuffle=False)
        eval_subset = CustomGenerator(eval_paths[:10], 1, shuffle=False)

        mae, _, mse = model.evaluate(eval_subset)
        metrics[dataset_split] = {"mae": mae, "mse": mse}

        y_pred = model.predict(eval_subset)
        for k in range(y_pred.shape[0]):
            X, y = eval_subset.__getitem__(k)
            X = X[0]
            y = y[0]
            y_hat = y_pred[k]

            # Input figures
            fig, axs = plt.subplots(2, 6, figsize=(30, 5))
            for i in range(2):
                for j in range(6):
                    axs[i, j].imshow(X[i * 6 + j, :, :, 0])

            fig.tight_layout()
            fig.savefig(f"{results_dir}/{dataset_split}/input/{k}.png")

            # Expected output figures
            fig, axs = plt.subplots(2, 4, figsize=(25, 5))
            for i in range(2):
                for j in range(4):
                    axs[i, j].imshow(y[i * 4 + j, :, :, 0])

            fig.tight_layout()
            fig.savefig(
                f"{results_dir}/{dataset_split}/expected_output/{k}.png")

            # Predicted figures
            fig, axs = plt.subplots(2, 4, figsize=(25, 5))

            for i in range(2):
                for j in range(4):
                    axs[i, j].imshow(y_hat[i * 4 + j, :, :])

            fig.tight_layout()
            fig.savefig(
                f"{results_dir}/{dataset_split}/predicted_output/{k}.png")

            # Comparison figures
            fig, axs = plt.subplots(2, 4, figsize=(25, 5))
            for j in range(4):
                axs[0, j].imshow(y[j, :, :, 0])
                axs[1, j].imshow(y_hat[j, :, :])

            fig.tight_layout()
            fig.savefig(
                f"{results_dir}/{dataset_split}/comparison_output/{k}.png")

    with open(f"{results_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
