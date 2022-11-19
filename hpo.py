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

train_directory = "data/train"
val_directory = "data/val"


def objective(trial):
    time.sleep(5)
    num_filters_base = trial.suggest_int("num_filters_base", 4, 8, step=2)
    dropout_rate = trial.suggest_float("dropout_rate", 0.05, 0.5, step=0.05)
    learning_rate = trial.suggest_float("learning_rate", 1e-10, 1e-3, log=True)
    batch_size = trial.suggest_int("batch_size", 1, 2, step=1)

    train_paths = [
        f"{train_directory}/{x}" for x in os.listdir(train_directory)
    ]
    val_paths = [f"{val_directory}/{x}" for x in os.listdir(val_directory)]

    train_dataset = CustomGenerator(train_paths, batch_size)
    val_dataset = CustomGenerator(val_paths, batch_size)

    mlflow.tensorflow.autolog(log_models=False)

    with mlflow.start_run() as run:
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

        model = res2((12, 256, 620, 3),
                     num_filters_base=num_filters_base,
                     dropout_rate=dropout_rate)
        model.summary()

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
            print("Starting fit")
            results = model.fit(train_dataset,
                                batch_size=batch_size,
                                epochs=128,
                                callbacks=callbacks,
                                verbose=1,
                                validation_data=val_dataset)

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

    gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    print(tf.config.list_physical_devices("GPU"))

    storage = optuna.storages.RDBStorage(url="sqlite:///optuna.db",
                                         heartbeat_interval=60,
                                         grace_period=120)

    study = optuna.create_study(study_name="res2",
                                storage=storage,
                                sampler=optuna.samplers.TPESampler(),
                                direction="minimize",
                                load_if_exists=True)

    print("Starting trials")
    study.optimize(objective, n_trials=30)
