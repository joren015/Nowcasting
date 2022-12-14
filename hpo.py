import argparse
import gc
import os
import random
import tempfile
import time

import keras
import mlflow
import numpy as np
import optuna
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import mixed_precision

from nowcasting.unet import res2
from nowcasting.unet_conv3d import unet_conv3d
from nowcasting.utils import CustomGenerator, KGLoss, model_analysis

seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


def objective(trial):
    time.sleep(5)
    num_filters_base = trial.suggest_int("num_filters_base", 4, 8, step=2)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.75, step=0.05)
    learning_rate = trial.suggest_float("learning_rate", 1e-10, 1e-1, log=True)
    batch_size = trial.suggest_int("batch_size", 4, 8, step=4)
    # kgl_alpha = trial.suggest_float("kgl_alpha", 0.0, 1.0, step=0.25)
    # kgl_beta = trial.suggest_float("kgl_beta", 0.0, 1.0, step=0.25)

    train_directory = f"{args.dataset_directory}/train"
    val_directory = f"{args.dataset_directory}/val"

    train_paths = [
        f"{train_directory}/{x}" for x in os.listdir(train_directory)
    ]
    val_paths = [f"{val_directory}/{x}" for x in os.listdir(val_directory)]

    train_dataset = CustomGenerator(train_paths, batch_size)
    val_dataset = CustomGenerator(val_paths, batch_size)

    experiment = mlflow.get_experiment_by_name(args.experiment_name)
    if experiment is None:
        mlflow.create_experiment(args.experiment_name)
        experiment = mlflow.get_experiment_by_name(args.experiment_name)

    mlflow.tensorflow.autolog(log_models=False)

    with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
        try:
            params = {
                "hpo_num_filters_base": num_filters_base,
                "hpo_dropout_rate": dropout_rate,
                "hpo_learning_rate": learning_rate,
                "hpo_batch_size": batch_size,
                # "hpo_kgl_alpha": kgl_alpha,
                # "hpo_kgl_beta": kgl_beta
            }
            print(params)
            mlflow.log_params(params)
        except Exception as e:
            print(e)

        model = unet_conv3d((12, 256, 620, 4),
                            num_filters_base=num_filters_base,
                            dropout_rate=dropout_rate)
        model.summary()

        # loss = KGLoss(alpha=kgl_alpha, beta=kgl_beta)
        loss = "mean_squared_error"

        model.compile(
            loss=loss,
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=["mae", "mse"])

        checkpoint_directory = f"data/checkpoints/{run.info.run_id}"
        os.makedirs(checkpoint_directory)
        checkpoint_filepath = f"{checkpoint_directory}/script_n1.h5"
        callbacks = [
            EarlyStopping(patience=25, verbose=1),
            ReduceLROnPlateau(factor=0.1, patience=10, min_lr=1e-16,
                              verbose=1),
            ModelCheckpoint(filepath=checkpoint_filepath,
                            verbose=1,
                            save_best_only=True,
                            save_weights_only=True)
        ]

        try:
            print("Starting fit")
            results = model.fit(train_dataset,
                                batch_size=batch_size,
                                epochs=128,
                                callbacks=callbacks,
                                verbose=1,
                                validation_data=val_dataset)

            val_loss = np.min(results.history["val_loss"])

            model.load_weights(checkpoint_filepath)
            mlflow.log_artifact(checkpoint_filepath)

            os.makedirs("data/tmp", exist_ok=True)
            with tempfile.TemporaryDirectory(dir="data/tmp") as tmpdirname:
                metrics = model_analysis(
                    model,
                    results_dir=tmpdirname,
                    dataset_directory=args.dataset_directory)
                mlflow.log_artifacts(tmpdirname, "analysis")
                mlflow.log_metrics(metrics)

            return val_loss
        except Exception as e:
            print(e)

        del model
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="hpo.py",
        description="Runs a set of trials performing hyperparameter tuning")

    parser.add_argument(
        "--dataset_directory",
        type=str,
        default="data/datasets/12_8_0_20_1.0",
        help=
        "Path where training, testing, and validation dataset are stored. By default data/datasets/12_8_0_20_1.0"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="hpo_unet_conv3d_12_8_0_20_1.0",
        help=
        "Named used for mlflow experiment, by default hpo_unet_conv3d_12_8_0_20_1.0"
    )

    args = parser.parse_args()

    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    tf.config.optimizer.set_experimental_options({"layout_optimizer": False})

    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

    policy = mixed_precision.Policy("mixed_float16")
    mixed_precision.set_global_policy(policy)

    gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    print(tf.config.list_physical_devices("GPU"))

    storage = optuna.storages.RDBStorage(url="sqlite:///optuna.db",
                                         heartbeat_interval=60,
                                         grace_period=120)

    search_space = {
        "num_filters_base": [4, 8],
        "dropout_rate": [0, 0.25, 0.5],
        "learning_rate": [1e-4, 1e-2, 1e-1],
        "batch_size": [4, 8],
        # "kgl_alpha": [0.0, 0.25, 0.5, 0.75, 1.0],
        # "kgl_beta": [0.0, 0.25, 0.5, 0.75, 1.0]
    }

    study = optuna.create_study(
        study_name=args.experiment_name,
        storage=storage,
        sampler=optuna.samplers.GridSampler(search_space),
        direction="minimize",
        load_if_exists=True)

    print("Starting trials")
    study.optimize(objective, n_trials=30)
