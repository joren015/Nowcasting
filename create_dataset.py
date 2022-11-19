import os
import re
from shutil import rmtree

import numpy as np
import scipy
from tqdm import tqdm

from nowcasting.utils import sliding_window_expansion

train_directory = "data/train"
val_directory = "data/val"

if __name__ == "__main__":
    mat_path = "data/full_sample"
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

    try:
        rmtree(train_directory)
        rmtree(val_directory)
    except Exception as e:
        print(e)

    os.makedirs(train_directory)
    os.makedirs(val_directory)

    print("Writing training dataset to disk")
    for i in tqdm(range(X_train.shape[0])):
        arr = np.array([X_train[i], y_train[i]], dtype=object)
        np.save(f"{train_directory}/{i}.npy", arr)

    print("Writing validation dataset to disk")
    for i in tqdm(range(X_val.shape[0])):
        arr = np.array([X_val[i], y_val[i]], dtype=object)
        np.save(f"{val_directory}/{i}.npy", arr)
