import random

import mlflow
import numpy as np
import tensorflow as tf

from nowcasting.unet import res2
from nowcasting.utils import model_analysis, recreate_directory

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
    model.compile(loss="mean_square_error", metrics=["mae", "mse"])

    results_dir = f"data/analysis/{experiment_name}"
    recreate_directory(results_dir)

    metrics = model_analysis(model, results_dir, dataset_directory)
    print(metrics)
    print("Done")
