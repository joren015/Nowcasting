import argparse
import gc
import os

import keras
import mlflow
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import mixed_precision

from nowcasting.unet import res2
from nowcasting.utils import CustomGenerator

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
    parser.add_argument(
        "--num_filters_base",
        type=int,
        default=4,
        help="Number of base filters to use when creating model. By default 4")
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.5,
        help="Dropout rate used throughout model. By default 0.5")
    parser.add_argument("--learning_rate",
                        type=float,
                        default=1e-6,
                        help="Starting learning rate. By default 1e-6")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size used when training/evaluating. By default 8")
    parser.add_argument(
        "--experiment_prefix",
        type=str,
        default="res2",
        help="Prefix used to identify mlflow experiment, by default res2")
    parser.add_argument("--early_stopping",
                        type=bool,
                        default=False,
                        help="Option to use early stopping, by default True")

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

    study_experiment = f"{args.experiment_prefix}_{args.dataset_directory}"

    train_directory = f"data/datasets/{args.dataset_directory}/train"
    val_directory = f"data/datasets/{args.dataset_directory}/val"
    # test_directory = f"data/datasets/{args.dataset_directory}/test"

    train_paths = [
        f"{train_directory}/{x}" for x in os.listdir(train_directory)
    ]
    val_paths = [f"{val_directory}/{x}" for x in os.listdir(val_directory)]
    # test_paths = [f"{test_directory}/{x}" for x in os.listdir(test_directory)]

    train_dataset = CustomGenerator(train_paths, args.batch_size)
    val_dataset = CustomGenerator(val_paths, args.batch_size)
    # test_dataset = CustomGenerator(test_paths, batch_size)

    experiment = mlflow.get_experiment_by_name(study_experiment)
    if experiment is None:
        mlflow.create_experiment(study_experiment)
        experiment = mlflow.get_experiment_by_name(study_experiment)

    mlflow.tensorflow.autolog(log_models=False)

    with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
        try:
            params = {
                "selected_num_filters_base": args.num_filters_base,
                "selected_dropout_rate": args.dropout_rate,
                "selected_learning_rate": args.learning_rate,
                "selected_batch_size": args.batch_size
            }
            print(params)
            mlflow.log_params(params)
        except Exception as e:
            print(e)

        model = res2((12, 256, 620, 4),
                     num_filters_base=args.num_filters_base,
                     dropout_rate=args.dropout_rate)
        model.summary()

        model.compile(
            loss="mean_absolute_error",
            optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
            metrics=["mae", "mse"])

        checkpoint_directory = f"data/checkpoints/{run.info.run_id}"
        os.makedirs(checkpoint_directory)
        checkpoint_filepath = f"{checkpoint_directory}/script_n1.h5"
        callbacks = [
            ReduceLROnPlateau(factor=0.1, patience=5, min_lr=1e-16, verbose=1),
            ModelCheckpoint(filepath=checkpoint_filepath,
                            verbose=1,
                            save_best_only=True,
                            save_weights_only=True)
        ]
        if args.early_stopping:
            callbacks.append(EarlyStopping(patience=20, verbose=1), )

        try:
            print("Starting fit")
            results = model.fit(train_dataset,
                                batch_size=args.batch_size,
                                epochs=128,
                                callbacks=callbacks,
                                verbose=1,
                                validation_data=val_dataset)

            print(results)
            val_loss = np.min(results.history["val_loss"])
            print(f"Min val loss: {val_loss}")

            model.load_weights(checkpoint_filepath)
            mlflow.log_artifact(checkpoint_filepath)
        except Exception as e:
            print(e)

        del model
        gc.collect()
