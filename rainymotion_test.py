import mat73
import numpy as np
import cv2

# TODO: make rainymotion import work
# from rainymotion.models import Dense

# TODO: run on slurm script since it will be too big when I run on target dataset
# target dataset: IMERG Early Run

import sys

# sys.path.append("/Users/yashveerbika/class/csci8523/nowcasting_project/pkgs")

from rainymotion.models import Dense

# def greyscale(img_stream):
#     return 

# print("Y")

#%% load the data
mat = mat73.loadmat('/Users/yashveerbika/class/csci8523/nowcasting-project/data/1Deg_800Sample.mat')  # 8 time step estimation
X_1 = mat[
    'X_train']  # (sample, time sequence, latitude, longitude, channel) here channels are 1: precipitation, 2: wind velocity in x direction, 3: wind velocity in y direction
y_1 = mat['y_train']  # (sample, time sequence, lat, lon)

# print("Z")

X_test = mat['X_test']
y_test = mat['y_test']
GFS = mat['GFS_test']

print("[X] All data loaded")

# initialize the model
model = Dense()

# upload data to the model instance
print(f"training data: {X_1.shape}")
print(f"training data: {( (np.sum(X_1, axis=4) / 3.0).reshape((9600, 120, 120, 1))[-2:] ).shape}")

model.input_data = (np.sum(X_1, axis=4) / 3.0).reshape((9600, 120, 120, 1))[-2:]

# run the model with default parameters
nowcast = model.run()
print(f"nowcast: {nowcast.shape}")
for i in range(nowcast.shape[0]):
    # print(f"nowcast[i]: {nowcast[i]}")
    cv2.imwrite(f"nowcast[{i}].png", nowcast[i]*255/(np.max(nowcast[i])) )