import argparse
import os
from shutil import rmtree

import numpy as np

from nowcasting.utils import (create_samples_from_mat_files,
                              recreate_directory, sliding_window_expansion,
                              write_samples_to_npy)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="create_dataset.py",
        description=
        "Builds a train and validation dataset based on the sliding window expansion function",
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

    args = parser.parse_args()

    # Train and validation dataset
    mat_path = "/panfs/jay/groups/6/csci8523/rahim035"
    Xs, ys, gfss = create_samples_from_mat_files(mat_path)

    X, y = sliding_window_expansion(Xs,
                                    ys,
                                    input_window_size=args.input_window_size,
                                    target_window_size=args.target_window_size,
                                    target_offset=args.target_offset,
                                    step=args.step,
                                    sample_ratio=1)
    X_gfs, gfs = sliding_window_expansion(
        Xs,
        gfss,
        input_window_size=args.input_window_size,
        target_window_size=args.target_window_size,
        target_offset=args.target_offset,
        step=args.step,
        sample_ratio=1)

    swe_test = np.all(X == X_gfs)
    assert swe_test
    print(swe_test)

    train_val_cutoff = int(X.shape[0] * 0.9)

    X_train = X[:train_val_cutoff]
    y_train = y[:train_val_cutoff]
    gfs_train = gfs[:train_val_cutoff]
    X_val = X[train_val_cutoff:]
    y_val = y[train_val_cutoff:]
    gfs_val = gfs[train_val_cutoff:]

    mu = np.mean(X_train)
    s = np.std(X_train)

    X_train = (X_train - mu) / s
    X_val = (X_val - mu) / s

    print("Train features", X_train.shape, "Train labels", y_train.shape)
    print("Validation features", X_val.shape, "Validation labels", y_val.shape)

    sub_directory = f"{args.input_window_size}_{args.target_window_size}_{args.target_offset}_{args.step}_1.0"
    train_directory = f"data/datasets/{sub_directory}/train"
    val_directory = f"data/datasets/{sub_directory}/val"
    gfs_train_directory = f"data/datasets/{sub_directory}/gfs/train"
    gfs_val_directory = f"data/datasets/{sub_directory}/gfs/val"

    for directory in [
            train_directory, val_directory, gfs_train_directory,
            gfs_val_directory
    ]:
        try:
            rmtree(directory)
        except Exception as e:
            print(e)

        os.makedirs(directory)

    with open(f"data/datasets/{sub_directory}/mean.txt", "w") as f:
        f.write(str(mu))
    with open(f"data/datasets/{sub_directory}/std.txt", "w") as f:
        f.write(str(s))

    print("Writing train dataset to disk")
    write_samples_to_npy(X=X_train, y=y_train, write_directory=train_directory)

    print("Writing validation dataset to disk")
    write_samples_to_npy(X=X_val, y=y_val, write_directory=val_directory)

    print("Writing gfs train dataset to disk")
    write_samples_to_npy(X=X_train,
                         y=gfs_train,
                         write_directory=gfs_train_directory)

    print("Writing gfs validation dataset to disk")
    write_samples_to_npy(X=X_val, y=gfs_val, write_directory=gfs_val_directory)

    # Test dataset
    mat_path = "/panfs/jay/groups/6/csci8523/rahim035/testset"
    Xs, ys, gfss = create_samples_from_mat_files(mat_path)

    X, y = sliding_window_expansion(Xs,
                                    ys,
                                    input_window_size=args.input_window_size,
                                    target_window_size=args.target_window_size,
                                    target_offset=args.target_offset,
                                    step=args.step,
                                    sample_ratio=1)
    X_gfs, gfs = sliding_window_expansion(
        Xs,
        gfss,
        input_window_size=args.input_window_size,
        target_window_size=args.target_window_size,
        target_offset=args.target_offset,
        step=args.step,
        sample_ratio=1)

    swe_test = np.all(X == X_gfs)
    assert swe_test
    print(swe_test)

    sub_directory = f"{args.input_window_size}_{args.target_window_size}_{args.target_offset}_{args.step}_1.0"
    test_directory = f"data/datasets/{sub_directory}/test"
    gfs_directory = f"data/datasets/{sub_directory}/gfs/test"

    with open(f"data/datasets/{sub_directory}/mean.txt", "r") as f:
        mu = float(f.read())
    with open(f"data/datasets/{sub_directory}/std.txt", "r") as f:
        s = float(f.read())

    X = (X - mu) / s

    print("Features", X.shape, "Labels", X.shape)

    for dir in [test_directory, gfs_directory]:
        recreate_directory(dir)

    print("Writing test dataset to disk")
    write_samples_to_npy(X=X, y=y, write_directory=test_directory)

    print("Writing gfs test dataset to disk")
    write_samples_to_npy(X=X, y=gfs, write_directory=gfs_directory)
