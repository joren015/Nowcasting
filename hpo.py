import gc
import os
import re
import time

import keras
import mat73
import mlflow
import numpy as np
import optuna
import scipy
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import mixed_precision
from tqdm import tqdm

from nowcasting.unet import res1, res2
from nowcasting.utils import sliding_window_expansion


def objective(trial):
    time.sleep(5)
    num_filters_base = trial.suggest_int("num_filters_base", 4, 8, step=2)
    dropout_rate = trial.suggest_float("dropout_rate", 0.05, 0.5, step=0.05)
    learning_rate = trial.suggest_float("learning_rate", 1e-10, 1e-3, log=True)
    batch_size = trial.suggest_int("batch_size", 4, 24, step=4)

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

        model = res2((12, 256, 620, 3),
                     num_filters_base=num_filters_base,
                     dropout_rate=dropout_rate)

        model.compile(
            loss="mean_absolute_error",
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=["mae", "mse"])

        checkpoint_filepath = "script_n1.h5"
        callbacks = [
            EarlyStopping(patience=10, verbose=1),
            ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-16, verbose=1),
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
                                validation_data=(X_val, y_val))

            val_loss = np.min(results.history["val_loss"])
            return val_loss
        except Exception as e:
            print(e)

        del model
        gc.collect()


if __name__ == "__main__":
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    tf.config.optimizer.set_experimental_options({"layout_optimizer": False})

    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

    policy = mixed_precision.Policy("mixed_float16")
    mixed_precision.set_global_policy(policy)

    print(tf.config.list_physical_devices('GPU'))

    # mat = mat73.loadmat("data/GD/1Deg_800Sample.mat")  # 8 time step estimation

    # X_1 = mat[
    #     "X_train"]  # (sample, time sequence, latitude, longitude, channel) here channels are 1: precipitation, 2: wind velocity in x direction, 3: wind velocity in y direction
    # y_1 = mat["y_train"]  # (sample, time sequence, lat, lon)

    mat_path = "/panfs/jay/groups/6/csci8523/rahim035"
    mat_files = [
        f"{mat_path}/{x}" for x in os.listdir(mat_path)
        if re.match(r'20.*-S.*\.mat', x)
    ]
    mat_files.sort()

    Xs = []
    ys = []
    for mat_file in tqdm(mat_files):
        mat = scipy.io.loadmat(mat_file)
        mat_shape = mat["X"]["imerg"][0][0].shape
        Xs.append(
            np.array([
                mat["X"][x][0][0] for x in ["imerg", "gfs_v", "gfs_tpw"]
            ]).reshape((mat_shape[0], mat_shape[1], 3)))
        ys.append(mat["X"]["gfs_pr"][0][0].reshape(
            (mat_shape[0], mat_shape[1], 1)))

    Xs = np.array(Xs)
    ys = np.array(ys)

    X, y = sliding_window_expansion(Xs,
                                    ys,
                                    input_window_size=12,
                                    target_window_size=8,
                                    target_offset=0,
                                    step=8,
                                    sample_ratio=1)

    train_cutoff = int(X.shape[0] * 0.9)
    X_train = X[:train_cutoff]
    y_train = y[:train_cutoff]
    X_val = X[train_cutoff:]
    y_val = y[train_cutoff:]

    print("Train feature", X_train.shape, "Train label", y_train.shape)
    print("Validation feature", X_val.shape, "Validation label", y_val.shape)

    del Xs, ys, mat
    gc.collect()

    storage = optuna.storages.RDBStorage(url="sqlite:///optuna.db",
                                         heartbeat_interval=60,
                                         grace_period=120)

    # search_space = {
    #     "num_filters_base": [4, 8, 12],
    #     "dropout_rate": [0.1, 0.2, 0.3],
    #     "learning_rate": [1e-4, 1e-3, 1e-2],
    #     "batch_size": [4, 8, 12]
    # }
    study = optuna.create_study(study_name="res2",
                                storage=storage,
                                sampler=optuna.samplers.TPESampler(),
                                direction="minimize",
                                load_if_exists=True)

    study.optimize(objective, n_trials=30)
