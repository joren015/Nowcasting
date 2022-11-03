import gc
import os
import time

import keras
import mat73
import mlflow
import numpy as np
import optuna
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from tensorflow.keras import mixed_precision
from tqdm import tqdm

from nowcasting.unet import res1


def objective(trial):
    time.sleep(5)
    num_filters_base = trial.suggest_int("num_filters_base", 4, 12, step=4)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.3, step=0.1)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_int("batch_size", 4, 12, step=4)

    mlflow.tensorflow.autolog(log_models=False)

    with mlflow.start_run() as run:
        try:
            mlflow.log_params({
                "hpo_num_filters_base": num_filters_base,
                "hpo_dropout_rate": dropout_rate,
                "hpo_learning_rate": learning_rate,
                "hpo_batch_size": batch_size
            })
        except Exception as e:
            print(e)

        model = res1((12, 120, 120, 3),
                     num_filters_base=num_filters_base,
                     dropout_rate=dropout_rate)

        model.compile(
            loss="mean_absolute_error",
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=["mae", "mse"])

        checkpoint_filepath = "script_n1.h5"
        callbacks = [
            EarlyStopping(patience=10, verbose=1),
            ReduceLROnPlateau(factor=0.1,
                              patience=5,
                              min_lr=0.00001,
                              verbose=1),
            ModelCheckpoint(filepath=checkpoint_filepath,
                            verbose=1,
                            save_best_only=True,
                            save_weights_only=True)
        ]

        try:
            results = model.fit(X_train,
                                y_train,
                                batch_size=batch_size,
                                epochs=128,
                                callbacks=callbacks,
                                verbose=1,
                                validation_data=(X_test, y_test))

            val_loss = np.min(results.history["val_loss"])
        except Exception as e:
            print(e)

        del model
        gc.collect()

        return val_loss


if __name__ == "__main__":
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    tf.config.optimizer.set_experimental_options({"layout_optimizer": False})

    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

    policy = mixed_precision.Policy("mixed_float16")
    mixed_precision.set_global_policy(policy)

    print(tf.config.list_physical_devices('GPU'))

    mat = mat73.loadmat("data/GD/1Deg_800Sample.mat")  # 8 time step estimation
    X_1 = mat[
        "X_train"]  # (sample, time sequence, latitude, longitude, channel) here channels are 1: precipitation, 2: wind velocity in x direction, 3: wind velocity in y direction
    y_1 = mat["y_train"]  # (sample, time sequence, lat, lon)

    X_test = mat["X_test"]
    y_test = mat["y_test"]
    GFS = mat["GFS_test"]

    X_train, X_val, y_train, y_val = train_test_split(X_1,
                                                      y_1,
                                                      test_size=0.15,
                                                      random_state=42)
    print("Train feature", X_train.shape, "Train label", y_train.shape)
    print("Validation feature", X_test.shape, "Validation label", y_val.shape)

    del X_1, y_1, mat, X_val, y_val, GFS
    gc.collect()

    storage = optuna.storages.RDBStorage(url="sqlite:///optuna.db",
                                         heartbeat_interval=60,
                                         grace_period=120)

    search_space = {
        "num_filters_base": [4, 8, 12],
        "dropout_rate": [0.1, 0.2, 0.3],
        "learning_rate": [1e-4, 1e-3, 1e-2],
        "batch_size": [4, 8, 12]
    }
    study = optuna.create_study(
        storage=storage,
        sampler=optuna.samplers.GridSampler(search_space),
        direction="minimize",
        load_if_exists=True)

    study.optimize(objective)
