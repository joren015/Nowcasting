import argparse
import gc
import os
import time

import keras
import mlflow
import numpy as np
import optuna
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import mixed_precision

from nowcasting.unet import res2
from nowcasting.utils import CustomGenerator


def objective(trial):
    time.sleep(5)
    num_filters_base = trial.suggest_int("num_filters_base", 4, 8, step=2)
    dropout_rate = trial.suggest_float("dropout_rate", 0.05, 0.5, step=0.05)
    learning_rate = trial.suggest_float("learning_rate", 1e-10, 1e-1, log=True)
    batch_size = trial.suggest_int("batch_size", 2, 16, step=2)

    train_directory = f"data/datasets/{args.dataset_directory}/train"
    val_directory = f"data/datasets/{args.dataset_directory}/val"
    # test_directory = f"data/datasets/{args.dataset_directory}/test"

    train_paths = [
        f"{train_directory}/{x}" for x in os.listdir(train_directory)
    ]
    val_paths = [f"{val_directory}/{x}" for x in os.listdir(val_directory)]
    # test_paths = [f"{test_directory}/{x}" for x in os.listdir(test_directory)]

    train_dataset = CustomGenerator(train_paths, batch_size)
    val_dataset = CustomGenerator(val_paths, batch_size)
    # test_dataset = CustomGenerator(test_paths, batch_size)

    experiment = mlflow.get_experiment_by_name(study_experiment)
    if experiment is None:
        mlflow.create_experiment(study_experiment)
        experiment = mlflow.get_experiment_by_name(study_experiment)

    mlflow.tensorflow.autolog(log_models=False)

    with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
        try:
            params = {
                "hpo_num_filters_base": num_filters_base,
                "hpo_dropout_rate": dropout_rate,
                "hpo_learning_rate": learning_rate,
                "hpo_batch_size": batch_size
            }
            print(params)
            mlflow.log_params(params)
        except Exception as e:
            print(e)

        model = res2((12, 256, 620, 4),
                     num_filters_base=num_filters_base,
                     dropout_rate=dropout_rate)
        model.summary()

        model.compile(
            loss="mean_absolute_error",
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=["mae", "mse"])

        checkpoint_directory = f"data/checkpoints/{run.info.run_id}"
        os.makedirs(checkpoint_directory)
        checkpoint_filepath = f"{checkpoint_directory}/script_n1.h5"
        callbacks = [
            EarlyStopping(patience=10, verbose=1),
            ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-16, verbose=1),
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
            # test_results = model.evaluate(test_dataset)
            # print(f"TEST RESULTS: {test_results}")
            mlflow.log_artifact(checkpoint_filepath)
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
        default="12_8_0_20_1.0",
        help=
        "Subdirectory in data/datasets to use for training, testing, and validation. By default 12_8_0_20_1.0"
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

    study_experiment = f"res2_{args.dataset_directory}"
    storage = optuna.storages.RDBStorage(url="sqlite:///optuna.db",
                                         heartbeat_interval=60,
                                         grace_period=120)

    study = optuna.create_study(study_name=study_experiment,
                                storage=storage,
                                sampler=optuna.samplers.TPESampler(),
                                direction="minimize",
                                load_if_exists=True)

    print("Starting trials")
    study.optimize(objective, n_trials=30)
