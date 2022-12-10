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
        "Builds a test dataset based on the sliding window expansion function",
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

    mat_path = "/panfs/jay/groups/6/csci8523/rahim035/testset"
    mat_files = [
        f"{mat_path}/{x}" for x in os.listdir(mat_path)
        if re.match(r"20.*-S.*\.mat", x)
    ]
    mat_files.sort()

    Xs = []
    ys = []
    gfss = []
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
        gfss.append(mat["X"]["gfs_pr"][0][0].reshape(
            (mat_shape[0], mat_shape[1], 1)))

    Xs = np.array(Xs)
    ys = np.array(ys)
    gfss = np.array(gfss)

    X, y = sliding_window_expansion(
        Xs,
        ys,
        input_window_size=args.input_window_size,
        target_window_size=args.target_window_size,
        target_offset=args.target_offset,
        step=args.step,
        # sample_ratio=args.sample_ratio)
        sample_ratio=1)
    X_gfs, gfs = sliding_window_expansion(
        Xs,
        gfss,
        input_window_size=args.input_window_size,
        target_window_size=args.target_window_size,
        target_offset=args.target_offset,
        step=args.step,
        # sample_ratio=args.sample_ratio)
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
        try:
            rmtree(dir)
        except Exception as e:
            print(e)

        os.makedirs(dir)

    print("Writing test dataset to disk")
    for i in tqdm(range(X.shape[0])):
        arr = np.array([X[i], y[i]], dtype=object)
        np.save(f"{test_directory}/{i}.npy", arr)

    print("Writing gfs test dataset to disk")
    for i in tqdm(range(X.shape[0])):
        arr = np.array([X[i], gfs[i]], dtype=object)
        np.save(f"{gfs_directory}/{i}.npy", arr)
