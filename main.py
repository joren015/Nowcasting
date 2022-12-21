import argparse
import os
import random
import tempfile

import keras
import mlflow
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import mixed_precision

from nowcasting.unet import res2
from nowcasting.utils import CustomGenerator, KGLoss, model_analysis

seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

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
        "--num_filters_base",
        type=int,
        default=4,
        help="Number of base filters to use when creating model. By default 4")
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0,
        help="Dropout rate used throughout model. By default 0")
    parser.add_argument("--learning_rate",
                        type=float,
                        default=1e-4,
                        help="Starting learning rate. By default 1e-4")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size used when training/evaluating. By default 4")
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="unet_conv3d_12_8_0_20_1.0",
        help=
        "Named used for mlflow experiment, by default hpo_unet_conv3d_12_8_0_20_1.0"
    )
    parser.add_argument("--early_stopping",
                        type=bool,
                        default=True,
                        help="Option to use early stopping, by default True")
    parser.add_argument(
        "--kgl_alpha",
        type=float,
        default=0.0,
        help=
        "Weight apply to knowledge guided loss (negative values error), by default 0.0"
    )
    parser.add_argument(
        "--kgl_beta",
        type=float,
        default=1.0,
        help="Weight apply to knowledge guided loss (csi error), by default 1.0"
    )

    args = parser.parse_args()

    # Setting GPU related environment variables
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    tf.config.optimizer.set_experimental_options({"layout_optimizer": False})

    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

    policy = mixed_precision.Policy("mixed_float16")
    mixed_precision.set_global_policy(policy)

    gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    print(tf.config.list_physical_devices("GPU"))

    # Creating train/test/val datasets
    train_directory = f"{args.dataset_directory}/train"
    val_directory = f"{args.dataset_directory}/val"

    train_paths = [
        f"{train_directory}/{x}" for x in os.listdir(train_directory)
    ]
    val_paths = [f"{val_directory}/{x}" for x in os.listdir(val_directory)]
    # test_paths = [f"{test_directory}/{x}" for x in os.listdir(test_directory)]

    train_dataset = CustomGenerator(train_paths, args.batch_size)
    val_dataset = CustomGenerator(val_paths, args.batch_size)
    # test_dataset = CustomGenerator(test_paths, batch_size)

    # Creating mlflow experiment if it does not already exist
    experiment = mlflow.get_experiment_by_name(args.experiment_name)
    if experiment is None:
        mlflow.create_experiment(args.experiment_name)
        experiment = mlflow.get_experiment_by_name(args.experiment_name)

    mlflow.tensorflow.autolog(log_models=False)

    with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
        try:
            params = {
                "hpo_num_filters_base": args.num_filters_base,
                "hpo_dropout_rate": args.dropout_rate,
                "hpo_learning_rate": args.learning_rate,
                "hpo_batch_size": args.batch_size,
                "hpo_kgl_alpha": args.kgl_alpha,
                "hpo_kgl_beta": args.kgl_beta
            }
            print(params)
            mlflow.log_params(params)
        except Exception as e:
            print(e)

        model = res2((12, 256, 620, 4),
                     num_filters_base=args.num_filters_base,
                     dropout_rate=args.dropout_rate)
        model.summary()

        loss = KGLoss(alpha=args.kgl_alpha, beta=args.kgl_beta)

        model.compile(
            loss=loss,
            optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
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
                                batch_size=args.batch_size,
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

        except Exception as e:
            raise e
