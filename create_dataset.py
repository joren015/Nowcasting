import argparse
import os
import re
from shutil import rmtree

import numpy as np
import scipy
from tqdm import tqdm

from nowcasting.utils import sliding_window_expansion

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="create_dataset.py",
        description=
        "Builds a train, test, and validation dataset based on the sliding window expansion function",
        epilog=
        "Resulting datasets are written to data/datasets in a directory following the pattern input_window_size_target_window_size_target_offset_step_sample_ratio"
    )

    parser.add_argument(
        "--input_window_size",
        type=int,
        default=12,
        help="Number of timesteps in each input. By default 12")
    parser.add_argument(
        "--target_window_size",
        type=int,
        default=8,
        help="Number of timesteps in each target. By default 8")
    parser.add_argument(
        "--target_offset",
        type=int,
        default=0,
        help=
        "Number of timesteps the start of each target should be from the end of each input. By default 0"
    )
    parser.add_argument(
        "--step",
        type=int,
        default=20,
        help=
        "Number of timesteps to use to determine the start of the next input and target pair. By default 20"
    )
    parser.add_argument("--sample_ratio",
                        type=float,
                        default=1.0,
                        help="Percentage of results to return. By default 1.0")

    args = parser.parse_args()

    mat_path = "/panfs/jay/groups/6/csci8523/rahim035"
    mat_files = [
        f"{mat_path}/{x}" for x in os.listdir(mat_path)
        if re.match(r"20.*-S.*\.mat", x)
    ]
    mat_files.sort()

    Xs = []
    ys = []
    print("Loading .mat files")
    for mat_file in tqdm(mat_files):
        mat = scipy.io.loadmat(mat_file)
        mat_shape = mat["X"]["imerg"][0][0].shape
        x = np.array([
            mat["X"][x][0][0] for x in ["imerg", "gfs_v", "gfs_tpw", "gfs_u"]
        ])
        x = np.moveaxis(x, 0, 2)
        Xs.append(x)
        ys.append(mat["X"]["imerg"][0][0].reshape(
            (mat_shape[0], mat_shape[1], 1)))

    Xs = np.array(Xs)
    ys = np.array(ys)

    X, y = sliding_window_expansion(Xs,
                                    ys,
                                    input_window_size=args.input_window_size,
                                    target_window_size=args.target_window_size,
                                    target_offset=args.target_offset,
                                    step=args.step,
                                    sample_ratio=args.sample_ratio)

    train_val_cutoff = int(X.shape[0] * 0.9)

    X_train = X[:train_val_cutoff]
    y_train = y[:train_val_cutoff]
    X_val = X[train_val_cutoff:]
    y_val = y[train_val_cutoff:]

    print("Train features", X_train.shape, "Train labels", y_train.shape)
    print("Validation features", X_val.shape, "Validation labels", y_val.shape)

    sub_directory = f"{args.input_window_size}_{args.target_window_size}_{args.target_offset}_{args.step}_{args.sample_ratio}"
    train_directory = f"data/datasets/{sub_directory}/train"
    val_directory = f"data/datasets/{sub_directory}/val"

    for directory in [train_directory, val_directory]:
        try:
            rmtree(directory)
        except Exception as e:
            print(e)

        os.makedirs(directory)

    print("Writing training dataset to disk")
    for i in tqdm(range(X_train.shape[0])):
        arr = np.array([X_train[i], y_train[i]], dtype=object)
        np.save(f"{train_directory}/{i}.npy", arr)

    print("Writing validation dataset to disk")
    for i in tqdm(range(X_val.shape[0])):
        arr = np.array([X_val[i], y_val[i]], dtype=object)
        np.save(f"{val_directory}/{i}.npy", arr)
